"""
DepositProof AI — hackathon starter application.

LLM integration is intentionally isolated in llm_integration.py.
This app can run in deterministic demo mode without an LLM, but demo findings
are clearly marked and must never be represented as real AI analysis.
"""
from __future__ import annotations

import inspect
import json
import html
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from document_processing import extract_inspection_fields, extract_pdf
from evidence_engine import (
    classify_deduction,
    evaluate_locality_benchmark,
    extract_invoice_fields,
    find_relevant_clauses,
    deduction_groups,
    groups_for,
    money_text,
    retrieve_topic_passages,
    terms_for,
    verify_quote,
)
from llm_integration import (
    analyze_image_pair_with_llm,
    assess_deduction_with_llm,
    get_configuration_error,
    get_last_error,
    llm_is_configured,
    validate_image_file,
)
from report_generator import generate_audit_pdf
from repair_price_intelligence import (
    DATASET_PATH,
    DATASET_NAME,
    SAMPLE_DATA_WARNING,
    RepairRateDataError,
    compare_repair_price,
    load_repair_rates,
)

UPLOADS = ROOT / "uploads"
REPORTS = ROOT / "reports"
DB_PATH = ROOT / "data" / "depositproof.sqlite3"
LOCALITY_DATA = pd.read_csv(DATASET_PATH)
LOCALITY_REQUIRED_COLUMNS = {
    "City", "Locality / Area", "Category", "Min Price (₹)", "Max Price (₹)",
}
missing_locality_columns = LOCALITY_REQUIRED_COLUMNS - set(LOCALITY_DATA.columns)
if missing_locality_columns:
    raise RepairRateDataError(
        "Repair-rate dataset is missing required columns: "
        + ", ".join(sorted(missing_locality_columns))
    )
LOCALITY_DATA["Min Price (₹)"] = pd.to_numeric(
    LOCALITY_DATA["Min Price (₹)"], errors="raise"
)
LOCALITY_DATA["Max Price (₹)"] = pd.to_numeric(
    LOCALITY_DATA["Max Price (₹)"], errors="raise"
)
LOCALITY_BY_LABEL = {
    f"{locality}, {city}": (city, locality)
    for city, locality in LOCALITY_DATA[["City", "Locality / Area"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
}
LOCALITY_CHOICES = sorted(LOCALITY_BY_LABEL, key=str.casefold)
CITY_CHOICES = sorted(
    {city for city, _ in LOCALITY_BY_LABEL.values()},
    key=str.casefold,
)
LOCALITY_CHOICES_BY_CITY = {
    city: sorted(
        {
            locality
            for selected_city, locality in LOCALITY_BY_LABEL.values()
            if selected_city == city
        },
        key=str.casefold,
    )
    for city in CITY_CHOICES
}
DEFAULT_CITY = CITY_CHOICES[0] if CITY_CHOICES else None
DEFAULT_LOCALITIES = (
    LOCALITY_CHOICES_BY_CITY[DEFAULT_CITY] if DEFAULT_CITY is not None else []
)
DEFAULT_AREA = DEFAULT_LOCALITIES[0] if DEFAULT_LOCALITIES else None
UPLOADS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
DB_PATH.parent.mkdir(exist_ok=True)


@contextmanager
def db() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
    except Exception:
        con.rollback()
        raise
    else:
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, deposit REAL NOT NULL,
            claimed_deductions REAL NOT NULL, proposed_refund REAL,
            move_in TEXT, move_out TEXT, currency TEXT NOT NULL DEFAULT 'INR',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
            filename TEXT NOT NULL, category TEXT NOT NULL, stored_path TEXT NOT NULL,
            extracted_text TEXT, page_text_json TEXT, status TEXT NOT NULL,
            invoice_json TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS deductions (
            id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
            category TEXT NOT NULL, description TEXT NOT NULL, amount REAL NOT NULL,
            invoice_document_id TEXT, photo_pair_id TEXT, status TEXT,
            assessment_json TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS photo_pairs (
            id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
            item_name TEXT NOT NULL, move_in_path TEXT, move_out_path TEXT,
            observation_json TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS findings (
            id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
            deduction_id TEXT REFERENCES deductions(id) ON DELETE CASCADE,
            finding_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS repair_benchmarks (
            deduction_id TEXT PRIMARY KEY REFERENCES deductions(id) ON DELETE CASCADE,
            case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
            deduction_description TEXT NOT NULL,
            deduction_amount REAL NOT NULL,
            benchmark_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)


init_db()


def money(value: Any, currency: str = "INR") -> str:
    try:
        n = float(value or 0)
    except (TypeError, ValueError):
        n = 0.0
    symbol = "₹" if currency == "INR" else currency + " "
    return f"{symbol}{n:,.2f}".replace(".00", "")


def get_locality_benchmarks(selection: str | None) -> dict[str, Any] | None:
    """Return all sample rate categories for one selected locality."""
    if not selection:
        return None
    selected = LOCALITY_BY_LABEL.get(selection)
    if selected is None:
        raise ValueError(f"Unknown locality selection: {selection}")
    city, locality = selected
    rows = LOCALITY_DATA.loc[
        (LOCALITY_DATA["City"] == city)
        & (LOCALITY_DATA["Locality / Area"] == locality)
    ]
    rates = [
        {
            "category": str(category),
            "minimum": float(category_rows["Min Price (₹)"].min()),
            "maximum": float(category_rows["Max Price (₹)"].max()),
        }
        for category, category_rows in rows.groupby("Category", sort=True)
    ]
    return {
        "city": city,
        "locality": locality,
        "source_dataset": DATASET_NAME,
        "sample_data_warning": SAMPLE_DATA_WARNING,
        "rates": rates,
    }


def update_locality_choices(city):
    localities = LOCALITY_CHOICES_BY_CITY.get(city, [])
    return gr.update(
        choices=localities,
        value=localities[0] if localities else None,
    )


def locality_selection_from_values(city, area):
    if not city and not area:
        return None
    if not city or not area:
        raise gr.Error("Select both a city and an area/locality.")
    selection = f"{area}, {city}"
    if LOCALITY_BY_LABEL.get(selection) != (city, area):
        raise gr.Error("Select an area/locality listed for the selected city.")
    return selection


def _assessment_method(finding: dict) -> str:
    mode = finding.get("mode")
    if mode == "llm":
        if finding.get("explanation_source") == "rules (model explanation rejected)":
            return "LLM response validated; its explanation was rejected, rules explanation shown, classification determined by evidence rules"
        if finding.get("explanation_source") != "model":
            return "Hosted LLM assessment returned; explanation source not recorded, classification determined by evidence rules"
        return "Validated LLM explanation; classification determined by evidence rules"
    if mode == "fallback_after_llm_error":
        return "Rules-based fallback after hosted LLM assessment failed"
    return "Rules-based assessment; hosted LLM assessment not used"


def _assessment_modes_summary(findings: list[dict]) -> str:
    counts: dict[str, int] = {}
    labels = {
        "llm": "LLM explanation with rules-engine classification",
        "fallback_after_llm_error": "rules-based fallback after LLM failure",
        "rules_fallback_not_llm": "rules-only (LLM not used)",
    }
    for finding in findings:
        mode = finding.get("mode", "rules_fallback_not_llm")
        label = labels.get(mode, "rules-based assessment")
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "No deduction findings."
    return "; ".join(f"{count} {label}" for label, count in counts.items())


def _markdown_cell(value: Any, fallback: str) -> str:
    text = str(value or fallback).replace("\r", " ").replace("\n", " ").replace("|", r"\|")
    return text or fallback


def _photo_comparison_markdown(
    photo: dict, item_name: str | None, description: str | None
) -> list[str]:
    if photo.get("mode") != "llm_analyzed":
        lines = ["The pair was saved, but no successful vision-model observation is stored."]
        if photo.get("error"):
            lines.append(f"- **Analysis error:** {photo['error']}")
        if not description:
            lines.append("- This photo pair is not associated with a deduction.")
        return lines

    lines = []
    summary = photo.get("observation")
    if summary:
        lines.append(f"- **Summary:** {summary}")
    lines.append(
        f"- **Comparability:** {photo.get('comparable') or 'not stated'}; "
        f"same area visible in both: {photo.get('same_area_visible_in_both', 'not stated')}."
    )
    move_in = photo.get("move_in_condition")
    move_out = photo.get("move_out_condition")
    if move_in:
        lines.append(f"- **Move-in condition:** {move_in}")
    if move_out:
        lines.append(f"- **Move-out condition:** {move_out}")

    changes = photo.get("observed_changes") or []
    if changes:
        lines.extend([
            "",
            "| Observed feature / location | Move-in condition | Move-out condition | Interpretation |",
            "|---|---|---|---|",
        ])
        for change in changes:
            feature = _markdown_cell(change.get("feature"), "Visible feature")
            location = change.get("location")
            if location:
                feature += f" ({_markdown_cell(location, '')})"
            lines.append(
                f"| {feature} | {_markdown_cell(change.get('move_in_condition'), 'Not described')} "
                f"| {_markdown_cell(change.get('move_out_condition'), 'Not described')} "
                f"| {_markdown_cell(change.get('interpretation'), 'No interpretation provided')} |"
            )
    elif photo.get("visible_differences"):
        lines.append("- **Observed changes:** " + "; ".join(photo["visible_differences"]))
        lines.append("- The model did not provide separate move-in/move-out descriptions or locations for these changes.")
    else:
        lines.append("- **Observed changes:** No specific visible differences were reported.")
    if photo.get("possible_damage_or_deterioration"):
        lines.append(
            "- **Possible damage (uncertain observation, not a finding):** "
            + "; ".join(photo["possible_damage_or_deterioration"])
        )

    uncertainty = photo.get("model_uncertainty")
    lines.append(
        f"- **Evidence strength / uncertainty:** Listed changes are model-reported visual comparisons; "
        f"possible-damage items are uncertain observations. Qualitative model uncertainty: "
        f"{uncertainty or 'not provided'}; this is not a calibrated confidence score."
    )
    if photo.get("not_reliably_comparable"):
        lines.append("- **Not reliably comparable:** " + "; ".join(photo["not_reliably_comparable"]))
    if photo.get("limitations"):
        lines.append(f"- **Model-stated limitations:** {photo['limitations']}")
    if description:
        lines.append(
            f"- **Relevance to deduction:** This separate photo comparison concerns "
            f"{_markdown_cell(item_name, 'the linked photo item')}; it was not a deduction assessment. "
            f"It may provide visual context for “{_markdown_cell(description, 'the claim')}”, but does not "
            "establish that the claimed repair was required or its cost."
        )
    else:
        lines.append(
            "- **Deduction association:** This photo pair is not linked to a deduction; do not treat it as "
            "evidence for any particular claim."
        )
    lines.append(
        "- Photographs can show visible differences, but cannot independently establish when a change "
        "occurred, who caused it, actual repair cost, or legal responsibility."
    )
    return lines


def _photo_upload_result_markdown(observation: dict | None, error: str | None) -> str:
    if not observation or observation.get("mode") != "llm_analyzed":
        message = "Automated image comparison was not completed."
        if error:
            message += f"\n\n> Photo analysis failed: {html.escape(error)}"
        return message

    lines = ["**Vision-model comparison completed.**"]
    if observation.get("observation"):
        lines.append(f"\n{observation['observation']}")
    if observation.get("move_in_condition"):
        lines.append(f"\n- **Move-in:** {observation['move_in_condition']}")
    if observation.get("move_out_condition"):
        lines.append(f"- **Move-out:** {observation['move_out_condition']}")
    changes = observation.get("observed_changes") or []
    if changes:
        lines.append("- **Observed changes:**")
        for change in changes:
            feature = change.get("feature") or "Visible difference"
            location = f" ({change['location']})" if change.get("location") else ""
            interpretation = change.get("interpretation")
            lines.append(f"  - {feature}{location}" + (f": {interpretation}" if interpretation else ""))
    elif observation.get("visible_differences"):
        lines.append("- **Observed changes:** " + "; ".join(observation["visible_differences"]))
    else:
        lines.append("- **Observed changes:** No specific visible differences were reported.")
    if observation.get("model_uncertainty"):
        lines.append(f"- **Qualitative uncertainty:** {observation['model_uncertainty']} (not a calibrated score)")
    if observation.get("limitations"):
        lines.append(f"- **Limitations:** {observation['limitations']}")
    return "\n".join(lines)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:120] or "upload"


def _normalize_case_dates(move_in, move_out):
    def normalize(value, label):
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        try:
            parsed = datetime.strptime(str(value), "%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} must be a valid date in YYYY-MM-DD format."
            ) from exc
        normalized = parsed.strftime("%Y-%m-%d")
        if normalized != str(value):
            raise ValueError(
                f"{label} must be a valid date in YYYY-MM-DD format."
            )
        return normalized

    move_in_date = normalize(move_in, "Move-in date")
    move_out_date = normalize(move_out, "Move-out date")
    if move_in_date and move_out_date and move_out_date < move_in_date:
        raise ValueError("Move-out date cannot be earlier than move-in date.")
    return move_in_date, move_out_date


def validate_case_dates(move_in, move_out):
    try:
        _normalize_case_dates(move_in, move_out)
    except ValueError as exc:
        return gr.update(value=str(exc), visible=True)
    return gr.update(value="", visible=False)


def handle_move_in_change(move_in, move_out):
    try:
        normalized_move_in, _ = _normalize_case_dates(move_in, None)
        normalized_move_out = _normalize_case_dates(None, move_out)[1]
    except ValueError as exc:
        return gr.update(), gr.update(value=str(exc), visible=True)

    if (
        normalized_move_in
        and normalized_move_out
        and normalized_move_out < normalized_move_in
    ):
        return gr.update(value=None), gr.update(value="", visible=False)
    return gr.update(), gr.update(value="", visible=False)


def save_upload(src: str, case_id: str) -> str:
    src_path = Path(src)
    destination = UPLOADS / case_id
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{uuid.uuid4().hex[:8]}_{safe_name(src_path.name)}"
    target.write_bytes(src_path.read_bytes())
    return str(target)


def create_case(name, deposit, claimed, proposed, move_in, move_out):
    if not name or not str(name).strip():
        raise gr.Error("Please enter a case or property name.")
    try:
        move_in, move_out = _normalize_case_dates(move_in, move_out)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    try:
        deposit = float(deposit)
        claimed = float(claimed)
        proposed = None if proposed in (None, "") else float(proposed)
    except (TypeError, ValueError):
        raise gr.Error("Enter valid monetary amounts.")
    if deposit < 0 or claimed < 0 or (proposed is not None and proposed < 0):
        raise gr.Error("Amounts cannot be negative.")
    case_id = uuid.uuid4().hex[:10].upper()
    now = datetime.now().isoformat(timespec="seconds")
    with db() as con:
        con.execute(
            """INSERT INTO cases(id,name,deposit,claimed_deductions,proposed_refund,
               move_in,move_out,created_at) VALUES(?,?,?,?,?,?,?,?)""",
            (case_id, str(name).strip(), deposit, claimed, proposed,
             move_in or None, move_out or None, now)
        )
    calculated = deposit - claimed
    msg = (
        f"<h3>Case created: <span class=\"dp-case-id\">{case_id}</span></h3>\n\n"
        f"- **Security deposit:** {money(deposit)}\n"
        f"- **Claimed deductions:** {money(claimed)}\n"
        f"- **Calculated refund:** {money(calculated)}\n"
        f"- **Proposed refund entered:** {money(proposed) if proposed is not None else 'Not provided'}\n\n"
        f"{'⚠️ The proposed refund differs from the calculated refund.' if proposed is not None and abs(proposed-calculated) > 0.01 else 'Financial arithmetic checked.'}\n\n"
        f"{ai_status_markdown()}"
    )
    return case_id, msg


def ai_status_markdown() -> str:
    """Show AI availability and required key names without exposing credentials."""
    if llm_is_configured():
        return "**AI status:** AI-assisted analysis is available."
    return (
        "**AI status:** AI-assisted analysis is unavailable. "
        + html.escape(get_configuration_error())
        + " rules-based assessment remains available."
    )


def upload_documents(case_id, primary_file, additional_files, additional_category):
    if not case_id:
        raise gr.Error("Create a case first.")
    if not primary_file:
        raise gr.Error("Upload the primary lease/agreement document.")

    files = [(primary_file, "Rental agreement")]
    if additional_files:
        if not isinstance(additional_files, (list, tuple)):
            additional_files = [additional_files]
        files.extend((item, additional_category) for item in additional_files)

    results = []
    for item, category in files:
        src = item if isinstance(item, str) else getattr(item, "name", None)
        if not src or not os.path.isfile(src):
            continue
        stored = save_upload(src, case_id)
        filename = Path(src).name
        doc_id = uuid.uuid4().hex[:12]
        text, page_text, status = "", {}, "uploaded"
        invoice_data = {}
        suffix = Path(src).suffix.lower()
        if suffix == ".pdf":
            try:
                text, page_text, status = extract_pdf(stored)
                if text and category in ("Invoice / receipt", "Deduction statement"):
                    invoice_data = extract_invoice_fields(text)
                elif text and category == "Inspection report":
                    invoice_data = extract_inspection_fields(text)
            except Exception as exc:
                status = f"extraction_error: {type(exc).__name__}"
        elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
            status = "image_uploaded"
        else:
            status = "unsupported_for_extraction"
        with db() as con:
            con.execute(
                """INSERT INTO documents(id,case_id,filename,category,stored_path,
                   extracted_text,page_text_json,status,invoice_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (doc_id, case_id, filename, category, stored, text,
                 json.dumps(page_text, ensure_ascii=False), status,
                 json.dumps(invoice_data, ensure_ascii=False), datetime.now().isoformat(timespec="seconds"))
            )
        ok = status.startswith("extracted")
        flag = "✅" if ok and "partial" not in status else "⚠️"
        results.append(f"- {flag} **{filename}** ({category}) — {status}; {len(text)} characters extracted.")
    _auto_link_case_evidence(case_id)
    return ("### Upload results\n\n" + ("\n".join(results) if results else "No valid files were processed.")
            + "\n\nFiles that could not be read are stored but **not** analysed; the audit will say so rather than guess."), \
        refresh_deductions(case_id)


def documents_for_case(case_id):
    with db() as con:
        return con.execute("SELECT * FROM documents WHERE case_id=? ORDER BY created_at", (case_id,)).fetchall()


def _choice_id(choice: str | None) -> str | None:
    """'abc123 — label' -> 'abc123'; 'None'/empty -> None."""
    if not choice or choice == "None":
        return None
    return choice.split(" — ", 1)[0].strip()


def deduction_choices(case_id):
    if not case_id:
        return []
    with db() as con:
        rows = con.execute("SELECT id,description,amount FROM deductions WHERE case_id=? ORDER BY created_at", (case_id,)).fetchall()
    return [f"{r['id']} — {r['description']} ({money(r['amount'])})" for r in rows]


_EVIDENCE_LOCATION_WORDS = {
    "apartment", "bath", "bathroom", "bed", "bedroom", "entry", "hall", "kitchen",
    "living", "lobby", "room", "study",
}


def _evidence_match_score(description, category, evidence_text):
    description = str(description or "")
    evidence_text = str(evidence_text or "")
    shared_groups = deduction_groups(description, category) & groups_for(evidence_text)
    description_terms = terms_for(description) - _EVIDENCE_LOCATION_WORDS
    evidence_terms = terms_for(evidence_text) - _EVIDENCE_LOCATION_WORDS
    shared_terms = description_terms & evidence_terms
    return len(shared_groups) * 5 + len(shared_terms) * 2


def _invoice_match_text(document):
    invoice_fields = json.loads(document["invoice_json"] or "{}")
    service_description = invoice_fields.get("service_description")
    invoice_text = service_description or (document["extracted_text"] or "")[:2000]
    return " ".join((document["filename"], invoice_text))


def _best_evidence_match(deduction, candidates, used_ids, label_for):
    scored = [
        (
            _evidence_match_score(
                deduction["description"],
                deduction["category"],
                label_for(candidate),
            ),
            candidate,
        )
        for candidate in candidates
        if candidate["id"] not in used_ids
    ]
    scored = [(score, candidate) for score, candidate in scored if score > 0]
    if not scored:
        return None
    best_score = max(score for score, _ in scored)
    best = [candidate for score, candidate in scored if score == best_score]
    return best[0] if len(best) == 1 else None


def _auto_link_case_evidence(case_id):
    """Persist unique keyword matches while leaving ambiguous evidence unlinked."""
    with db() as con:
        deductions = con.execute(
            "SELECT * FROM deductions WHERE case_id=? ORDER BY created_at,id", (case_id,)
        ).fetchall()
        invoices = con.execute(
            """SELECT * FROM documents WHERE case_id=?
               AND category IN ('Invoice / receipt','Deduction statement')
               ORDER BY created_at,id""",
            (case_id,),
        ).fetchall()
        photos = con.execute(
            "SELECT * FROM photo_pairs WHERE case_id=? ORDER BY created_at,id", (case_id,)
        ).fetchall()
        used_invoice_ids = {row["invoice_document_id"] for row in deductions if row["invoice_document_id"]}
        used_photo_ids = {row["photo_pair_id"] for row in deductions if row["photo_pair_id"]}

        for row in deductions:
            updates = {}
            deduction = dict(row)
            if not row["invoice_document_id"]:
                invoice = _best_evidence_match(
                    deduction,
                    invoices,
                    used_invoice_ids,
                    _invoice_match_text,
                )
                if invoice:
                    updates["invoice_document_id"] = invoice["id"]
                    used_invoice_ids.add(invoice["id"])
            if not row["photo_pair_id"]:
                photo = _best_evidence_match(
                    deduction,
                    photos,
                    used_photo_ids,
                    lambda pair: pair["item_name"],
                )
                if photo:
                    updates["photo_pair_id"] = photo["id"]
                    used_photo_ids.add(photo["id"])
            if updates:
                assignments = ", ".join(f"{column}=?" for column in updates)
                con.execute(
                    f"UPDATE deductions SET {assignments}, status=NULL, assessment_json=NULL "
                    "WHERE id=? AND case_id=?",
                    (*updates.values(), row["id"], case_id),
                )


REPAIR_RATE_CATEGORY_BY_DEDUCTION = {
    "painting / repainting": "Painting / Repainting",
    "cleaning": "Cleaning",
    "furniture damage": "Furniture / Carpentry",
    "electrical repairs": "Electrical Repairs",
    "plumbing": "Plumbing",
}


def _automatic_repair_price_benchmark(
    deduction, locality_benchmarks, repair_rate_data, repair_rate_error, currency,
):
    city = locality_benchmarks["city"]
    locality = locality_benchmarks["locality"]
    category = REPAIR_RATE_CATEGORY_BY_DEDUCTION.get(
        str(deduction["category"]).strip().casefold()
    )
    benchmark = {
        "city": city,
        "locality": locality,
        "area": locality,
        "damage_cause": deduction["description"],
        "source_dataset": DATASET_NAME,
        "sample_data_warning": SAMPLE_DATA_WARNING,
        "matched_records": [],
        "record_count": 0,
    }
    if currency != "INR":
        return {
            **benchmark,
            "status": "unavailable",
            "explanation": "Repair-rate benchmarks are available only for INR claims.",
        }
    if not category:
        return {
            **benchmark,
            "status": "no_match",
            "category": deduction["category"],
            "explanation": (
                f"No sample-rate category is mapped to the {deduction['category']} deduction type."
            ),
        }
    if repair_rate_error:
        return {
            **benchmark,
            "status": "unavailable",
            "category": category,
            "explanation": f"Repair-rate data could not be loaded: {repair_rate_error}",
        }

    try:
        result = compare_repair_price(
            repair_rate_data["records"],
            city=city,
            locality=locality,
            category=category,
            claimed_amount=deduction["amount"],
        )
    except ValueError as exc:
        return {
            **benchmark,
            "status": "unavailable",
            "category": category,
            "explanation": str(exc),
        }
    return {
        **result,
        "area": locality,
        "damage_cause": deduction["description"],
        "source_dataset": repair_rate_data["dataset_name"],
        "sample_data_warning": SAMPLE_DATA_WARNING,
        "dataset_quality": {
            "valid_row_count": len(repair_rate_data["records"]),
            "issues": repair_rate_data["issues"],
            "duplicate_count": repair_rate_data["duplicate_count"],
        },
    }


def _owned_id(con, table: str, item_id: str | None, case_id: str) -> str | None:
    """Return item_id only if that record belongs to this case (never link another case's files)."""
    if not item_id:
        return None
    row = con.execute(f"SELECT id FROM {table} WHERE id=? AND case_id=?", (item_id, case_id)).fetchone()
    if not row:
        raise gr.Error("That item does not belong to the active case. Refresh the lists and try again.")
    return item_id


def add_deduction(case_id, category, description, amount):
    if not case_id:
        raise gr.Error("Create a case first.")
    if not description or not str(description).strip():
        raise gr.Error("Enter a deduction description.")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise gr.Error("Enter a valid amount.")
    if amount < 0:
        raise gr.Error("Deduction amount cannot be negative.")
    deduction_id = uuid.uuid4().hex[:12]
    with db() as con:
        con.execute(
            """INSERT INTO deductions(id,case_id,category,description,amount,
               invoice_document_id,photo_pair_id,created_at) VALUES(?,?,?,?,?,?,?,?)""",
            (deduction_id, case_id, category, str(description).strip(), amount,
             None, None, datetime.now().isoformat(timespec="seconds"))
        )
    _auto_link_case_evidence(case_id)
    return (f"Added deduction `{deduction_id}`: **{description} — {money(amount)}**",
            refresh_deductions(case_id))


def remove_deduction(case_id, deduction_choice):
    if not case_id:
        raise gr.Error("Create a case first.")
    deduction_id = _choice_id(deduction_choice)
    if not deduction_id:
        raise gr.Error("Select a deduction to remove.")
    with db() as con:
        _owned_id(con, "deductions", deduction_id, case_id)
        con.execute("DELETE FROM deductions WHERE id=? AND case_id=?", (deduction_id, case_id))
    return "Deduction removed.", refresh_deductions(case_id), gr.update(choices=deduction_choices(case_id), value=None)


def refresh_deductions(case_id):
    if not case_id:
        return "Create a case first."
    with db() as con:
        rows = con.execute("SELECT * FROM deductions WHERE case_id=? ORDER BY created_at", (case_id,)).fetchall()
        case = con.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    if not rows:
        return "No deductions added yet."
    total = sum(r["amount"] for r in rows)
    with db() as con:
        doc_names = {d["id"]: d["filename"] for d in con.execute("SELECT id,filename FROM documents WHERE case_id=?", (case_id,))}
        pair_names = {p["id"]: p["item_name"] for p in con.execute("SELECT id,item_name FROM photo_pairs WHERE case_id=?", (case_id,))}
    lines = [f"### Deductions for {case['name']}", "",
             "| Description | Amount | Invoice | Photo pair | Status |", "|---|---:|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['description']} | {money(r['amount'], case['currency'])} | "
                     f"{doc_names.get(r['invoice_document_id'], '—')} | {pair_names.get(r['photo_pair_id'], '—')} | "
                     f"{r['status'] or 'Not audited'} |")
    lines += ["", f"**Entered deduction total:** {money(total, case['currency'])}",
              f"**Landlord-stated total:** {money(case['claimed_deductions'], case['currency'])}",
              "**Check:** Totals match." if abs(total-case["claimed_deductions"]) < .01 else "**⚠️ Check:** Entered deductions do not match the landlord-stated total."]
    return "\n".join(lines)


def run_audit(case_id, locality_selection=None):
    if not case_id:
        raise gr.Error("Create a case first.")
    locality_benchmarks = get_locality_benchmarks(locality_selection)
    _auto_link_case_evidence(case_id)
    with db() as con:
        case = con.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        deductions = con.execute("SELECT * FROM deductions WHERE case_id=?", (case_id,)).fetchall()
        docs = con.execute("SELECT * FROM documents WHERE case_id=?", (case_id,)).fetchall()
        photos = con.execute("SELECT * FROM photo_pairs WHERE case_id=?", (case_id,)).fetchall()
        benchmarks = {
            row["deduction_id"]: row
            for row in con.execute("SELECT * FROM repair_benchmarks WHERE case_id=?", (case_id,)).fetchall()
        }
    if not deductions:
        raise gr.Error("Add at least one deduction before running the audit.")

    agreement_docs = [d for d in docs if d["category"] == "Rental agreement"]
    agreement_texts = []
    for d in docs:
        if d["category"] == "Rental agreement" and d["extracted_text"]:
            agreement_texts.append((d["filename"], d["extracted_text"], json.loads(d["page_text_json"] or "{}")))
    if not agreement_docs:
        agreement_problem = "No rental agreement was uploaded, so contract terms could not be checked."
    elif not agreement_texts:
        agreement_problem = "The uploaded rental agreement could not be extracted, so contract terms could not be checked."
    else:
        agreement_problem = None
    repair_rate_data = None
    repair_rate_error = None
    if (
        locality_benchmarks
        and case["currency"] == "INR"
        and any(
            REPAIR_RATE_CATEGORY_BY_DEDUCTION.get(
                str(deduction["category"]).strip().casefold()
            )
            for deduction in deductions
        )
    ):
        try:
            repair_rate_data = load_repair_rates()
        except RepairRateDataError as exc:
            repair_rate_error = str(exc)

    report_findings = []
    for row in deductions:
        deduction = dict(row)
        clauses = []
        for filename, text, pages in agreement_texts:
            for clause in find_relevant_clauses(
                deduction["description"], text, filename, pages, category=deduction["category"]
            ):
                clauses.append({
                    **clause,
                    "quote_verified": verify_quote(clause.get("quote", ""), text),
                })

        invoice = {}
        if deduction.get("invoice_document_id"):
            matched = next((d for d in docs if d["id"] == deduction["invoice_document_id"]), None)
            if matched:
                invoice = json.loads(matched["invoice_json"] or "{}")
                invoice["filename"] = matched["filename"]
                invoice["extracted_text"] = matched["extracted_text"] or ""

        related_photo = next((p for p in photos if p["id"] == deduction.get("photo_pair_id")), None)
        photo_observation = json.loads(related_photo["observation_json"]) if related_photo and related_photo["observation_json"] else None

        deduction.update({
            "invoice": invoice,
            "photo_observation": photo_observation,
            "photo_item_name": related_photo["item_name"] if related_photo else None,
        })
        payload = {
            **deduction,
            "currency": case["currency"],
            "case_dates": {"move_in": case["move_in"], "move_out": case["move_out"]},
            "clauses": clauses,
            "agreement_available": bool(agreement_texts),
            "agreement_problem": agreement_problem,
            "locality_benchmarks": locality_benchmarks,
            "inspection_documents": [
                {"filename": d["filename"], "text": (d["extracted_text"] or "")[:5000]}
                for d in docs if d["category"] == "Inspection report"
            ],
        }
        if llm_is_configured():
            assessment = assess_deduction_with_llm(payload)
            if not assessment:
                model_error = get_last_error()
                assessment = classify_deduction(payload)
                assessment["mode"] = "fallback_after_llm_error"
                if model_error:
                    assessment["ai_error"] = model_error
            else:
                assessment["mode"] = "llm"
        else:
            assessment = classify_deduction(payload)
            assessment["mode"] = "rules_fallback_not_llm"
        assessment["locality_benchmark"] = evaluate_locality_benchmark(payload)
        assessment.setdefault("rules_classification", assessment.get("classification"))
        assessment["assessment_method"] = _assessment_method(assessment)
        assessment["clauses"] = clauses
        assessment["invoice"] = invoice
        assessment["photo_observation"] = photo_observation
        assessment["photo_item_name"] = related_photo["item_name"] if related_photo else None
        assessment["photo_pair_id"] = related_photo["id"] if related_photo else None
        assessment["description"] = deduction["description"]
        assessment["amount"] = deduction["amount"]
        assessment["source_note"] = "Only uploaded and successfully extracted evidence was considered."
        if locality_benchmarks:
            assessment["repair_price_benchmark"] = _automatic_repair_price_benchmark(
                deduction,
                locality_benchmarks,
                repair_rate_data,
                repair_rate_error,
                case["currency"],
            )
        else:
            benchmark_row = benchmarks.get(deduction["id"])
            if (benchmark_row
                    and benchmark_row["deduction_description"] == deduction["description"]
                    and abs(float(benchmark_row["deduction_amount"]) - float(deduction["amount"])) < 0.01
                    and case["currency"] == "INR"):
                benchmark = json.loads(benchmark_row["benchmark_json"])
                if benchmark.get("status") == "matched":
                    assessment["repair_price_benchmark"] = benchmark
        report_findings.append(assessment)
        with db() as con:
            con.execute(
                "UPDATE deductions SET status=?, assessment_json=? WHERE id=?",
                (assessment.get("classification", "Insufficiently Documented"),
                 json.dumps(assessment, ensure_ascii=False), deduction["id"])
            )
            con.execute("DELETE FROM findings WHERE case_id=? AND deduction_id=?",
                        (case_id, deduction["id"]))
            con.execute(
                "INSERT INTO findings(id,case_id,deduction_id,finding_json,created_at) VALUES(?,?,?,?,?)",
                (uuid.uuid4().hex[:12], case_id, deduction["id"],
                 json.dumps(assessment, ensure_ascii=False), datetime.now().isoformat(timespec="seconds"))
            )

    total_by_status = {"Supported": 0.0, "Inconsistent": 0.0, "Insufficiently Documented": 0.0}
    for f in report_findings:
        status = f.get("classification", "Insufficiently Documented")
        total_by_status[status] = total_by_status.get(status, 0.0) + float(f.get("amount", 0) or 0)

    lines = [
        f"# DepositProof AI audit — {case['name']}",
        f"**Case ID:** {case_id}",
        f"**Assessment modes by deduction:** {_assessment_modes_summary(report_findings)}",
        "",
        f"- Security deposit: {money(case['deposit'], case['currency'])}",
        f"- Landlord-claimed deductions: {money(case['claimed_deductions'], case['currency'])}",
        f"- Proposed refund: {money(case['proposed_refund'], case['currency']) if case['proposed_refund'] is not None else 'Not provided'}",
        f"- Supported-classification amount: {money(total_by_status.get('Supported', 0), case['currency'])}",
        f"- Inconsistent-classification amount: {money(total_by_status.get('Inconsistent', 0), case['currency'])}",
        f"- Insufficiently documented amount: {money(total_by_status.get('Insufficiently Documented', 0), case['currency'])}",
        "",
        "> These are evidence classifications, not a determination of legal liability or the amount legally recoverable.",
    ]
    for i, f in enumerate(report_findings, 1):
        lines += [
            "", f"## {i:02d} — {f['description']}",
            f"**Claimed amount:** {money(f['amount'], case['currency'])}",
            "",
            "### Final classification",
            '<div class="dp-report-classification"><span>Final classification</span>'
            f"<strong>{html.escape(str(f.get('classification', 'Insufficiently Documented')))}</strong></div>",
            "",
            "### Assessment details",
            f"- **Assessment method:** {f.get('assessment_method', _assessment_method(f))}",
            f"- **Rules-engine classification:** {f.get('rules_classification', f.get('classification', 'Insufficiently Documented'))}",
            "",
            "### Explanation",
        ]
        explanation = str(f.get("explanation") or "No explanation available.")
        explanation_paragraphs = re.split(r"\n\s*\n", explanation.strip())
        lines.extend(
            f"<p>{html.escape(paragraph).replace(chr(10), '<br>')}</p>"
            for paragraph in explanation_paragraphs if paragraph.strip()
        )
        if f.get("clauses"):
            lines.extend(["", "### Relevant agreement passages"])
            for c in f["clauses"][:5]:
                verified = "Verified" if c.get("quote_verified") else "Not verified"
                lines.extend([
                    "",
                    '<div class="dp-agreement-passage">',
                    f'<p><strong>Source document:</strong> {html.escape(str(c.get("filename", "Agreement")))}'
                    f' &nbsp; <strong>Page:</strong> {html.escape(str(c.get("page", "unknown")))}</p>',
                    f'<p><strong>Clause text</strong><br>{html.escape(str(c.get("quote", ""))).replace(chr(10), "<br>")}</p>',
                    f'<p><strong>Quotation verification:</strong> {verified}</p>',
                    "</div>",
                ])
        lines.extend([
            "",
            "### Limitations",
            f.get("limitations", "This assessment is limited to the uploaded evidence."),
        ])
        if f.get("photo_observation"):
            photo = f["photo_observation"]
            lines.append("")
            lines.extend(_photo_comparison_markdown(photo, f.get("photo_item_name"), f["description"]))
        else:
            lines.extend([
                "",
                "No linked usable photo observation is available for this deduction.",
            ])
        benchmark = f.get("repair_price_benchmark")
        if benchmark:
            lines.extend(["", "### Repair-price benchmark — context only"])
            lines.append(
                f"- Damage cause: {html.escape(str(benchmark.get('damage_cause', f['description'])))}."
            )
            if benchmark.get("area"):
                lines.append(
                    f"- Area: {html.escape(str(benchmark['area']))}, "
                    f"{html.escape(str(benchmark.get('city', '')))}."
                )
            if benchmark.get("status") == "matched":
                lines.extend([
                    f"- Comparison: {html.escape(str(benchmark.get('match_description', 'Comparable sample-rate records')))} "
                    f"for {html.escape(str(benchmark.get('category', 'the selected category')))}; "
                    f"{benchmark.get('record_count', 0)} sample record(s).",
                    f"- Claimed amount: {money(benchmark['claimed_amount'])}.",
                    f"- Sample range: {money(benchmark['lower'])}–{money(benchmark['upper'])}.",
                    f"- Claim position: {benchmark.get('position', 'not stated')}; difference from nearest boundary: "
                    f"{money(benchmark.get('difference', 0))}."
                    + (f" ({benchmark['difference_percent']:.1f}%)"
                       if benchmark.get("difference_percent") is not None and benchmark.get("position") != "within"
                       else ""),
                    "- Matched sample rates:",
                ])
                for sample in benchmark.get("matched_records", [])[:8]:
                    lines.append(
                        f"  - {html.escape(str(sample['city']))}, "
                        f"{html.escape(str(sample['locality']))}: "
                        f"{html.escape(str(sample['category']))} — "
                        f"{money(sample['minimum'])}–{money(sample['maximum'])}."
                    )
                lines.append(
                    f"- Source: {benchmark.get('source_dataset', DATASET_NAME)}. "
                    f"{benchmark.get('sample_data_warning', 'Sample data; not verified quotations.')}"
                )
            else:
                lines.append(
                    "- Cost comparison unavailable: "
                    + html.escape(str(benchmark.get(
                        "explanation", "No comparable sample-rate data was available."
                    )))
                )
        locality_benchmark = f.get("locality_benchmark")
        if locality_benchmark:
            lines.extend(["", "### Selected locality benchmark — context only"])
            if locality_benchmark.get("status") == "matched":
                lines.extend([
                    f"- Comparison: {locality_benchmark['category']} in "
                    f"{locality_benchmark['locality']}, {locality_benchmark['city']}.",
                    f"- Sample range: {money(locality_benchmark['lower'])}–"
                    f"{money(locality_benchmark['upper'])}.",
                    f"- Claim position: {locality_benchmark['position']}; difference from nearest boundary: "
                    f"{money(locality_benchmark['difference'])}."
                    + (f" ({locality_benchmark['difference_percent']:.1f}%)"
                       if locality_benchmark.get("difference_percent") is not None
                       and locality_benchmark["position"] != "within" else ""),
                ])
            else:
                if locality_benchmark.get("status") == "no_amount":
                    lines.append("- The deduction amount was unavailable for comparison.")
                else:
                    lines.append(
                        f"- No matching locality price category was available for "
                        f"{locality_benchmark.get('locality', 'the selected locality')}, "
                        f"{locality_benchmark.get('city', '')}."
                    )
            lines.extend([
                f"- Source: {locality_benchmark['source_dataset']}. "
                f"{locality_benchmark['sample_data_warning']}",
                "- This sample comparison is contextual only and does not determine the evidence classification.",
            ])
        if f.get("ai_error"):
            lines.extend(["", f"**Hosted AI unavailable:** {f['ai_error']} (rules-based fallback used.)"])
    linked_photo_ids = {f.get("photo_pair_id") for f in report_findings if f.get("photo_pair_id")}
    unlinked_photo_pairs = [p for p in photos if p["id"] not in linked_photo_ids]
    if unlinked_photo_pairs:
        lines.extend(["", "## Saved photo comparisons not linked to a deduction"])
        for pair in unlinked_photo_pairs:
            photo = json.loads(pair["observation_json"]) if pair["observation_json"] else {
                "mode": "not_analyzed",
                "limitations": "No photo observation is stored.",
            }
            lines.extend([
                "",
                f"### {pair['item_name']} — status: {pair['status']}",
                *_photo_comparison_markdown(photo, pair["item_name"], None),
            ])
    lines += [
        "",
        "---",
        "**Disclaimer:** This report combines rules-based evidence classifications with hosted-model "
        "explanations or photo comparisons only where explicitly indicated. It is not legal advice or a guarantee of refund.",
    ]
    markdown = "\n".join(lines)
    pdf_path = generate_audit_pdf(
        case, report_findings, total_by_status, REPORTS, photo_pairs=unlinked_photo_pairs
    )
    return markdown, pdf_path, refresh_deductions(case_id)


def upload_photo_pair(case_id, item_name, move_in_file, move_out_file):
    if not case_id:
        raise gr.Error("Create a case first.")
    if not item_name or not str(item_name).strip():
        raise gr.Error("Enter the room or item name.")
    if not move_in_file or not move_out_file:
        raise gr.Error("Upload both move-in and move-out photographs.")
    for label, path in (("Move-in", move_in_file), ("Move-out", move_out_file)):
        valid, message = validate_image_file(path)
        if not valid:
            raise gr.Error(f"{label} photo is invalid: {message}")
    move_in_path = save_upload(move_in_file, case_id)
    move_out_path = save_upload(move_out_file, case_id)
    observation = None
    mode = "uploaded_not_analyzed"
    analysis_error = None
    if llm_is_configured():
        observation = analyze_image_pair_with_llm(move_in_path, move_out_path, item_name)
        if observation:
            mode = "llm_analyzed"
        else:
            analysis_error = get_last_error() or "The hosted vision model returned no usable result."
    else:
        analysis_error = get_last_error() or get_configuration_error()
    if not observation:
        observation = {
            "observation": "Photographs uploaded. Vision-language model analysis is not configured or did not return a result.",
            "limitations": "No automated visual comparison was performed.",
            "mode": "not_analyzed",
            "error": analysis_error,
        }
    pair_id = uuid.uuid4().hex[:12]
    with db() as con:
        con.execute(
            """INSERT INTO photo_pairs(id,case_id,item_name,move_in_path,move_out_path,
               observation_json,status,created_at) VALUES(?,?,?,?,?,?,?,?)""",
            (pair_id, case_id, str(item_name).strip(), move_in_path, move_out_path,
             json.dumps(observation, ensure_ascii=False), mode, datetime.now().isoformat(timespec="seconds"))
        )
    _auto_link_case_evidence(case_id)
    detail = f"Photo pair saved for **{item_name}**. Status: `{mode}`. Pair ID: `{pair_id}`"
    detail += "\n\n" + _photo_upload_result_markdown(
        observation if mode == "llm_analyzed" else None,
        analysis_error,
    )
    return detail, refresh_deductions(case_id)


def _workflow_visibility(active_step: int):
    if active_step not in range(1, 5):
        raise ValueError("Workflow step must be between 1 and 4.")
    return (
        *(gr.update(visible=index == active_step) for index in range(1, 5)),
        *(
            gr.update(
                variant="secondary",
                elem_classes=[
                    "dp-step",
                    "dp-step--active" if index == active_step
                    else "dp-step--upcoming",
                ],
            )
            for index in range(1, 5)
        ),
    )


def _navigate_to_step(target_step: int, case_id: str | None, report_markdown: str | None):
    if target_step not in range(1, 5):
        raise ValueError("Workflow step must be between 1 and 4.")
    message = ""
    if target_step in (2, 3):
        problem = _active_case_problem(case_id)
        if problem:
            message = (
                f"**{problem} You can still explore this section without an active case.**"
            )
    elif target_step == 4 and (
        not report_markdown
        or not report_markdown.strip()
        or report_markdown.strip() == "Your report will appear here after the audit completes."
    ):
        message = "**Run the evidence audit from Review findings to generate a report.**"
    return (*_workflow_visibility(target_step), message)


def _enter_application():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        *_workflow_visibility(1),
        "",
    )


def _active_case_problem(case_id):
    if not case_id:
        return "Create an audit case before continuing."
    with db() as con:
        if not con.execute("SELECT 1 FROM cases WHERE id=?", (case_id,)).fetchone():
            return "The audit case could not be found. Create it again."
    return None


def _continue_from_case(case_id):
    return _navigate_to_step(2, case_id, None)


def _continue_from_evidence(case_id):
    return _navigate_to_step(3, case_id, None)


def _continue_from_review(report_markdown):
    return _navigate_to_step(4, None, report_markdown)


def toggle_additional_documents(is_visible):
    visible = not bool(is_visible)
    button_label = (
        "- Hide Optional Additional Documents"
        if visible else "+ Add Optional Additional Documents"
    )
    return visible, gr.update(visible=visible), gr.update(value=button_label)


def _run_audit_for_workflow(case_id, city=None, area=None):
    locality_selection = locality_selection_from_values(city, area)
    markdown, pdf_path, deductions = run_audit(case_id, locality_selection)
    return markdown, markdown, pdf_path, deductions


SCROLL_TO_TOP_JS = (
    "(...args) => { window.scrollTo({ top: 0, behavior: 'smooth' }); return args; }"
)
SHOW_LOADING_JS = """(...args) => {
    const overlay = document.getElementById("dp-loading-overlay");
    if (overlay) {
        overlay.classList.add("dp-loading-overlay-visible");
        overlay.setAttribute("aria-hidden", "false");
    }
    return args;
}"""
HIDE_LOADING_JS = """(...args) => {
    const overlay = document.getElementById("dp-loading-overlay");
    if (overlay) {
        overlay.classList.remove("dp-loading-overlay-visible");
        overlay.setAttribute("aria-hidden", "true");
    }
    return args;
}"""


DESIGN_CSS = """
:root {
  --dp-ivory: #f5f7fa;
  --dp-paper: #ffffff;
  --dp-paper-muted: #edf3f8;
  --dp-navy: #14263d;
  --dp-ink: #1b2939;
  --dp-muted: #5f6f82;
  --dp-blue: #355c7d;
  --dp-blue-dark: #294b69;
  --dp-line: #dfe6ed;
  --dp-line-strong: #ccd6e0;
  --dp-success: #376b57;
  --dp-warning: #8a6835;
  --dp-danger: #9a514e;
  --dp-radius: 10px;
  --dp-shadow: 0 8px 24px rgba(20, 38, 61, .045);
}
*, *:before, *:after { box-sizing: border-box; }
html, body, .gradio-container {
  background: var(--dp-ivory) !important;
  color: var(--dp-ink) !important;
  font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}
.gradio-container {
  --body-text-color: var(--dp-ink);
  --body-background-fill: var(--dp-ivory);
  --background-fill-primary: var(--dp-paper);
  --background-fill-secondary: var(--dp-paper-muted);
  --border-color-primary: var(--dp-line);
  max-width: 100% !important;
  padding: 0 !important;
  color-scheme: light;
}
.dp-app-shell { --block-label-text-color: var(--dp-navy); }
.dp-welcome, .dp-app-shell { width: min(1120px, calc(100% - 48px)); margin: 0 auto !important; }
.dp-welcome { min-height: 100vh; display: flex; flex-direction: column; justify-content: center; padding: 54px 0; }
.dp-welcome__frame { position: relative; overflow: hidden; background: var(--dp-paper); border: 1px solid var(--dp-line); border-radius: 18px; padding: clamp(34px, 7vw, 86px); box-shadow: var(--dp-shadow); }
.dp-welcome__frame:after { content: ""; position: absolute; z-index: 0; width: 360px; height: 360px; right: -160px; bottom: -190px; border: 1px solid #d9e2e7; border-radius: 50%; box-shadow: 0 0 0 30px #f8faf9, 0 0 0 31px #e3e8e9, 0 0 0 65px #fff, 0 0 0 66px #e3e8e9; pointer-events: none; }
.dp-welcome__frame > * { position: relative; z-index: 1; }
.dp-brand-lockup { display: flex; align-items: center; gap: 13px; color: var(--dp-navy); font-weight: 650; letter-spacing: -.025em; }
.dp-brand-lockup svg { width: 42px; height: 46px; flex: none; }
.dp-brand-lockup strong { display: block; color: var(--dp-navy) !important; font-size: 18px; }
.dp-brand-lockup small { display: block; margin-top: 3px; color: var(--dp-muted) !important; font-size: 11px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; }
.dp-welcome__eyebrow, .dp-eyebrow { color: var(--dp-blue); font-size: 11px; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
.dp-welcome__eyebrow { margin-top: clamp(48px, 9vw, 96px); }
.dp-welcome h1 { max-width: 780px; margin: 16px 0 18px; color: var(--dp-navy); font-size: clamp(38px, 6.4vw, 72px); font-weight: 570; letter-spacing: -.055em; line-height: 1.04; }
.dp-welcome__tagline { max-width: 700px; color: var(--dp-blue-dark); font-family: Georgia, "Times New Roman", serif; font-size: clamp(20px, 3vw, 29px); line-height: 1.35; }
.dp-welcome__description { max-width: 620px; margin: 20px 0 31px; color: var(--dp-muted); font-size: 16px; line-height: 1.75; }
.dp-welcome__fineprint { margin-top: 25px; color: #768493; font-size: 12px; line-height: 1.6; }
.dp-enter { width: max-content !important; min-width: 220px !important; }
.dp-app-shell { padding-bottom: 48px; }
.dp-topbar { position: sticky; z-index: 20; top: 0; display: flex; align-items: center; justify-content: space-between; gap: 22px; margin-bottom: 24px; padding: 12px 0; border-bottom: 1px solid var(--dp-line); background: rgba(245,247,250,.96); backdrop-filter: blur(12px); }
.dp-topbar > .block,
.dp-topbar > .row { width: auto !important; min-width: 0; flex: none !important; }
.dp-topbar .dp-brand-lockup svg { width: 34px; height: 38px; }
.dp-header-quote { margin-left: auto; color: var(--dp-blue-dark); font-family: Georgia, "Times New Roman", serif; font-size: 14px; font-style: italic; line-height: 1.4; text-align: right; white-space: nowrap; }
.dp-page-heading { margin: 0 0 18px; }
.dp-page-heading h1 { margin: 6px 0 8px; color: var(--dp-navy); font-size: clamp(27px, 4vw, 38px); font-weight: 620; letter-spacing: -.04em; }
.dp-page-heading p { max-width: 780px; margin: 0; color: var(--dp-muted); font-size: 14px; line-height: 1.65; }
.dp-stepper { display: flex; align-items: stretch; gap: 22px; margin: 18px 0 24px; border-bottom: 1px solid var(--dp-line); }
.dp-step { position: relative; display: flex; flex: 1 1 0; width: auto !important; min-width: 0 !important; min-height: 52px !important; align-items: center; justify-content: flex-start; padding: 8px 0 14px !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; color: #34485c !important; font-size: 13px !important; font-weight: 550 !important; text-align: left; box-shadow: none !important; transition: color .16s ease !important; }
.dp-step:hover { background: transparent !important; color: var(--dp-navy) !important; }
.dp-step:focus-visible { outline: 2px solid #72869a !important; outline-offset: 2px; }
.dp-step:after { position: absolute; right: 0; bottom: -1px; left: 0; height: 2px; background: transparent; content: ""; }
.dp-step--active:after { background: var(--dp-navy); }
.dp-step--active { color: var(--dp-navy) !important; font-weight: 700 !important; }
.dp-workflow-message:has(.md:empty) { display: none !important; }
.dp-workflow-message { margin: 12px 0; padding: 10px 13px; border-left: 2px solid #8ba3b8; background: #f5f8fa; color: #34485c; font-size: 12px; line-height: 1.55; }
.dp-workflow-panel { min-width: 0; padding: clamp(20px, 3vw, 32px); border: 1px solid var(--dp-line); border-radius: var(--dp-radius); background: var(--dp-paper); box-shadow: var(--dp-shadow); }
.dp-section-title { margin: 0 0 7px; color: var(--dp-navy); font-size: 20px; font-weight: 620; letter-spacing: -.02em; }
.dp-section-copy { margin: 0 0 21px; color: var(--dp-muted); font-size: 13px; line-height: 1.65; }
.dp-subsection { margin: 20px 0 10px; padding-top: 16px; border-top: 1px solid var(--dp-line); color: var(--dp-navy); font-size: 15px; font-weight: 650; }
.dp-inline-note { padding: 13px 15px; border-left: 2px solid #8ba3b8; background: #f5f8fa; color: #536678; font-size: 12px; line-height: 1.6; }
.dp-workflow-panel label,
.dp-workflow-panel [data-testid="block-info"] { color: #34485c !important; opacity: 1 !important; font-size: 12px !important; font-weight: 620 !important; }
.dp-workflow-panel [data-testid="block-label"] { color: var(--dp-ink) !important; opacity: 1 !important; }
.dp-workflow-panel h1,
.dp-workflow-panel h2,
.dp-workflow-panel h3,
.dp-workflow-panel h4,
.dp-workflow-panel .prose h1,
.dp-workflow-panel .prose h2,
.dp-workflow-panel .prose h3,
.dp-workflow-panel .prose h4 { color: var(--dp-navy) !important; }
.dp-workflow-panel input, .dp-workflow-panel textarea, .dp-workflow-panel select, .dp-workflow-panel [role="listbox"] { border-color: var(--dp-line-strong) !important; border-radius: 7px !important; background: #fff !important; color: var(--dp-ink) !important; box-shadow: none !important; }
.dp-workflow-panel input:focus, .dp-workflow-panel textarea:focus, .dp-workflow-panel select:focus { border-color: var(--dp-blue) !important; box-shadow: 0 0 0 3px rgba(73,107,140,.12) !important; }
.dp-workflow-panel input:not([type="checkbox"]):not([type="radio"]):not([type="file"]),
.dp-workflow-panel select, .dp-workflow-panel [role="combobox"] { min-height: 44px; }
.dp-workflow-panel .wrap,
.dp-workflow-panel .wrap-inner,
.dp-workflow-panel .secondary-wrap,
.dp-workflow-panel .icon-wrap { background-color: #fff !important; color: var(--dp-ink) !important; }
.dp-workflow-panel .wrap { border: 1px solid var(--dp-line-strong) !important; border-radius: 8px !important; }
.dp-workflow-panel .wrap:focus-within { border-color: var(--dp-blue) !important; box-shadow: 0 0 0 3px rgba(53,92,125,.12) !important; }
.dp-workflow-panel .dropdown-arrow,
.dp-workflow-panel .icon-wrap svg { color: var(--dp-muted) !important; fill: currentColor !important; }
.dp-workflow-panel .block { background-color: transparent !important; }
.dp-workflow-panel .gr-accordion { border: 1px solid var(--dp-line) !important; border-radius: 9px !important; background: #fff !important; }
.dp-workflow-panel .gr-accordion > button { background: #fff !important; color: var(--dp-navy) !important; }
.dp-workflow-panel input::placeholder,
.dp-workflow-panel textarea::placeholder { color: #7a8998 !important; opacity: 1 !important; }
.dp-workflow-panel input[type="file"] { min-height: 0; }
.dp-case-layout { align-items: stretch !important; gap: 18px !important; }
.dp-case-card { min-width: 0; padding: 16px 18px; border: 1px solid var(--dp-line); border-radius: 9px; background: #fff; }
.dp-case-card .dp-subsection { margin-top: 0; padding-top: 0; border-top: 0; }
.dp-document-intro, .dp-photo-instructions { display: flex; align-items: flex-start; gap: 11px; margin: 0 0 12px; color: var(--dp-muted); font-size: 12px; line-height: 1.55; }
.dp-upload-icon { width: 22px; height: 22px; flex: none; color: var(--dp-blue); }
.dp-photo-grid { align-items: stretch !important; gap: 16px !important; }
.dp-photo-card { min-width: 0; padding: 14px; border: 1px dashed #b9c8d6; border-radius: 9px; background: #f8fafc; }
.dp-photo-card .dp-photo-instructions { color: var(--dp-ink); font-weight: 600; }
.dp-photo-card .dp-photo-instructions small { display: block; margin-top: 3px; color: var(--dp-muted); font-size: 11px; font-weight: 400; }
.dp-photo-card .wrap,
.dp-document-upload .wrap { min-height: 88px; border: 1px dashed #b9c8d6 !important; background: #fff !important; }
.dp-photo-card .wrap:hover,
.dp-document-upload .wrap:hover { border-color: var(--dp-blue) !important; background: #f7fafc !important; }
.dp-photo-card [data-testid="block-label"],
.dp-document-upload [data-testid="block-label"] { border: 0 !important; background: transparent !important; color: var(--dp-ink) !important; }
.dp-photo-card .upload-container,
.dp-document-upload .upload-container { border: 1px dashed #b9c8d6 !important; border-radius: 7px !important; background: #fff !important; color: var(--dp-ink) !important; }
.dp-photo-card .block { height: auto !important; }
.dp-photo-card .upload-container { height: 124px !important; min-height: 124px !important; padding: 10px !important; }
.dp-document-upload .upload-container { height: 124px !important; min-height: 124px !important; padding: 10px !important; }
.dp-photo-card .upload-container:hover,
.dp-document-upload .upload-container:hover { background: #f7fafc !important; }
.dp-photo-card .dp-photo-instructions span { color: var(--dp-ink) !important; }
.dp-photo-card .dp-photo-instructions small { color: var(--dp-muted) !important; }
.dp-photo-card .dp-photo-instructions .dp-upload-icon { color: var(--dp-blue) !important; }
.dp-photo-card [data-testid="upload-text"] .or,
.dp-document-upload [data-testid="upload-text"] .or { color: var(--dp-muted) !important; }
.dp-photo-card img { max-height: 190px; object-fit: contain; }
.dp-photo-preview img { max-height: 180px; object-fit: contain !important; }
.dp-photo-preview [data-testid="image"] { background: #f8fafc !important; }
.dp-document-upload { margin-top: 4px; }
.dp-workflow-panel .block,
.dp-workflow-panel .form { min-width: 0; }
.dp-workflow-panel .prose,
.dp-workflow-panel .markdown { overflow-wrap: anywhere; color: var(--dp-ink); opacity: 1; }
.dp-workflow-panel button.primary, .dp-welcome button.primary { min-height: 44px !important; padding: 0 20px !important; border: 1px solid var(--dp-blue-dark) !important; border-radius: 7px !important; background: var(--dp-blue-dark) !important; color: #fff !important; font-weight: 650 !important; box-shadow: none !important; transition: background .16s ease, transform .16s ease !important; }
.dp-workflow-panel button.primary:hover, .dp-welcome button.primary:hover { background: var(--dp-navy) !important; transform: translateY(-1px); }
.dp-workflow-panel button:not(.primary) { border: 1px solid var(--dp-line-strong) !important; border-radius: 7px !important; background: #fff !important; color: var(--dp-ink) !important; box-shadow: none !important; }
.dp-action-row { display: flex; justify-content: space-between; gap: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid var(--dp-line); }
.dp-action-row .dp-action-spacer { flex: 1; }
.dp-action-row button { flex: 0 0 auto !important; width: auto !important; min-width: 150px; }
.dp-upload-help { margin: 3px 0 15px; color: var(--dp-muted); font-size: 12px; line-height: 1.65; }
.dp-date-validation { margin: 4px 0 8px; color: #b42318 !important; font-size: 13px; }
.dp-date-validation .prose { color: #b42318 !important; }
.dp-result-block { min-width: 0; margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--dp-line); overflow-wrap: anywhere; color: var(--dp-ink); opacity: 1; }
.dp-case-summary .prose .dp-case-id { display: inline-block; max-width: 100%; padding: 8px 11px; border: 1px solid var(--dp-line-strong); border-radius: 6px; background: #f8fafc; color: var(--dp-navy); font: 600 15px/1.4 "Segoe UI", Arial, sans-serif; overflow-wrap: anywhere; user-select: text; }
.dp-document-upload .file-preview .file,
.dp-photo-upload .file-preview .file { background: #fff !important; color: var(--dp-ink) !important; }
.dp-document-upload .file-preview .filename,
.dp-document-upload .file-preview .filename .stem,
.dp-document-upload .file-preview .filename .ext,
.dp-photo-upload .file-preview .filename,
.dp-photo-upload .file-preview .filename .stem,
.dp-photo-upload .file-preview .filename .ext { color: var(--dp-ink) !important; -webkit-text-fill-color: var(--dp-ink) !important; opacity: 1 !important; overflow-wrap: anywhere; }
.dp-document-upload .file-preview .download,
.dp-photo-upload .file-preview .download,
.dp-document-upload .file-preview .download-link,
.dp-photo-upload .file-preview .download-link { color: var(--dp-muted) !important; -webkit-text-fill-color: var(--dp-muted) !important; opacity: 1 !important; }
.dp-document-upload .file-preview .file:hover,
.dp-photo-upload .file-preview .file:hover { background: #f8fafc !important; }
.dp-document-upload .file-preview .icon-button,
.dp-photo-upload .file-preview .icon-button { color: var(--dp-ink) !important; }
.dp-report .prose { max-width: 100%; color: var(--dp-ink); opacity: 1; font-size: 15px; line-height: 1.65; overflow-wrap: anywhere; }
.dp-report .prose h1, .dp-report .prose h2, .dp-report .prose h3 { color: var(--dp-navy); letter-spacing: -.02em; overflow-wrap: anywhere; }
.dp-report .prose h2 { margin: 30px 0 8px; font-size: 26px; line-height: 1.3; }
.dp-report .prose h3 { margin: 22px 0 9px; font-size: 18px; line-height: 1.4; }
.dp-report .prose p { margin: 8px 0; }
.dp-report .prose .dp-report-classification { display: flex; flex-wrap: wrap; align-items: baseline; gap: 7px 12px; box-sizing: border-box; margin: 10px 0 16px; padding: 12px 15px; border: 1px solid #f2c66d; border-radius: 7px; background: #fff7e6; color: #593b08; overflow-wrap: anywhere; }
.dp-report .prose .dp-report-classification { font-size: 15px; }
.dp-report .prose .dp-report-classification strong { font-size: 17px; }
.dp-report .prose .dp-report-classification span { font-weight: 650; }
.dp-report .prose .dp-agreement-passage { box-sizing: border-box; margin: 9px 0; padding: 10px 13px; border: 1px solid #dce4ee; border-left: 3px solid #9aabc0; border-radius: 5px; background: #f8fafc; overflow-wrap: anywhere; }
.dp-report .prose .dp-agreement-passage p { margin: 3px 0; }
.dp-report .prose table { width: 100%; border-collapse: collapse; }
.dp-report .prose th, .dp-report .prose td { padding: 10px 12px; border-bottom: 1px solid var(--dp-line); text-align: left; vertical-align: top; }
.dp-report .prose th { background: var(--dp-paper-muted); color: var(--dp-navy); font-size: 12px; }
.dp-footer-note { margin: 28px 0 0; padding-top: 17px; border-top: 1px solid var(--dp-line); color: #72808c; font-size: 11px; line-height: 1.65; }
.dp-shell-hidden { display: none !important; }
@media (max-width: 900px) {
  .dp-welcome, .dp-app-shell { width: min(100% - 28px, 680px); }
  .dp-welcome { padding: 20px 0; }
  .dp-welcome__frame { padding: 30px 24px 38px; }
  .dp-topbar { align-items: flex-start; flex-direction: column; gap: 10px; padding: 12px 0; }
  .dp-header-quote { max-width: 100%; margin-left: 0; white-space: normal; text-align: left; }
  .dp-stepper { gap: 14px; }
  .dp-step { min-height: 48px !important; font-size: 11px !important; }
  .dp-workflow-panel { min-height: 0; padding: 20px 16px; }
  .dp-case-layout, .dp-photo-grid { flex-wrap: wrap !important; }
  .dp-case-card, .dp-photo-card { flex: 1 1 100% !important; }
  .dp-action-row { margin-top: 22px; }
}
@media (max-width: 480px) {
  .dp-stepper { gap: 8px; }
  .dp-step { justify-content: center; font-size: 10px !important; text-align: center; }
  .dp-action-row { flex-wrap: wrap; }
  .dp-welcome, .dp-app-shell { width: calc(100% - 24px); }
  .dp-workflow-panel { padding: 18px 14px; }
  .dp-case-card { padding: 14px; }
  .dp-workflow-panel .row { gap: 8px !important; }
  .dp-action-row button { flex: 0 1 auto !important; }
  .dp-report .prose table { display: block; overflow-x: auto; }
}
@media print {
  body, .gradio-container { background: #fff !important; }
  .dp-topbar, .dp-stepper, .dp-action-row, .dp-footer-note { display: none !important; }
  .dp-app-shell, .dp-workflow-panel { width: 100% !important; max-width: none !important; padding: 0 !important; border: 0 !important; box-shadow: none !important; }
}
"""

APP_CSS = DESIGN_CSS + """
footer { visibility: hidden !important; display: none !important; }
.gradio-container .footer { display: none !important; }
#dp-loading-overlay {
  display: none !important;
  position: fixed !important;
  inset: 0 !important;
  z-index: 9999 !important;
  width: 100vw !important;
  height: 100vh !important;
  max-width: none !important;
  margin: 0 !important;
  padding: 24px !important;
  align-items: center !important;
  justify-content: center !important;
  background: rgba(15, 23, 42, .48) !important;
  backdrop-filter: blur(4px);
  pointer-events: auto !important;
}
#dp-loading-overlay.dp-loading-overlay-visible { display: flex !important; }
#dp-loading-overlay .dp-loading-overlay__content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  max-width: min(440px, 100%);
  padding: 28px 34px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  color: #1b2939;
  text-align: center;
  box-shadow: 0 16px 48px rgba(15, 23, 42, .2);
}
#dp-loading-overlay .dp-loading-overlay__content p { margin: 0; font-size: 15px; font-weight: 650; line-height: 1.5; }
#dp-loading-overlay .dp-loading-spinner {
  width: 44px;
  height: 44px;
  border: 4px solid #dbe5ef;
  border-top-color: #2563eb;
  border-radius: 50%;
  animation: dp-loading-spin .8s linear infinite;
}
@keyframes dp-loading-spin { to { transform: rotate(360deg); } }
"""

with gr.Blocks(
    title="DepositProof AI",
) as demo:
    with gr.Column(visible=True, elem_classes=["dp-welcome"]) as welcome_screen:
        with gr.Column(elem_classes=["dp-welcome__frame"]):
            gr.HTML(
                """
                <div class="dp-welcome__content">
                  <div class="dp-brand-lockup" aria-label="DepositProof AI">
                    <svg viewBox="0 0 48 52" role="img" aria-label="Shield and document mark">
                      <path d="M24 3.5 42 10v13.2c0 11.4-7.1 20.2-18 25.3C13.1 43.4 6 34.6 6 23.2V10l18-6.5Z" fill="none" stroke="#315675" stroke-width="2"/>
                      <path d="M17 16.5h14v20H17zM20.5 22h7M20.5 26h7M20.5 30h4" fill="none" stroke="#496b8c" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.7"/>
                    </svg>
                    <span><strong>DepositProof AI</strong><small>Evidence-led deposit review</small></span>
                  </div>
                  <div class="dp-welcome__eyebrow">A clearer view of your move-out statement</div>
                  <h1>Keep the evidence.<br>Question the deduction.</h1>
                  <div class="dp-welcome__tagline">Rental deposit protection, backed by evidence.</div>
                  <p class="dp-welcome__description">Review rental deductions using move-in and move-out photographs, rental agreement passages, invoices, and repair-price comparisons where available. Keep each finding connected to the evidence you provide.</p>
                </div>
                """
            )
            enter_app_btn = gr.Button("Enter DepositProof AI", variant="primary", elem_classes=["dp-enter"])
            gr.HTML(
                '<p class="dp-welcome__fineprint">A practical evidence-review tool, not legal advice. '
                'Your files are stored locally by this application.</p>'
            )

    with gr.Column(visible=False, elem_classes=["dp-app-shell"]) as app_shell:
        with gr.Row(elem_classes=["dp-topbar"]):
            gr.HTML(
                """
                <div class="dp-brand-lockup">
                  <svg viewBox="0 0 48 52" aria-hidden="true">
                    <path d="M24 3.5 42 10v13.2c0 11.4-7.1 20.2-18 25.3C13.1 43.4 6 34.6 6 23.2V10l18-6.5Z" fill="none" stroke="#315675" stroke-width="2"/>
                    <path d="M17 16.5h14v20H17zM20.5 22h7M20.5 26h7M20.5 30h4" fill="none" stroke="#496b8c" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.7"/>
                  </svg>
                  <span><strong>DepositProof AI</strong><small>Evidence-led deposit review</small></span>
                </div>
                """
            )
            gr.HTML(
                '<div class="dp-header-quote">'
                '"Your deposit, Your evidence, Your peace of mind"'
                "</div>"
            )

        gr.HTML(
            '<header class="dp-page-heading"><div class="dp-eyebrow">Your deposit review</div>'
            '<h1>Build a clear evidence record</h1>'
            '<p>Move through each step at your own pace. Uploaded records and model observations are shown only after the application reports them.</p></header>'
        )
        with gr.Row(elem_classes=["dp-stepper"]):
            step_case = gr.Button("Audit case", elem_classes=["dp-step", "dp-step--active"])
            step_evidence = gr.Button("Evidence", elem_classes=["dp-step", "dp-step--upcoming"])
            step_findings = gr.Button("Review findings", elem_classes=["dp-step", "dp-step--upcoming"])
            step_report = gr.Button("Final report", elem_classes=["dp-step", "dp-step--upcoming"])
        workflow_message = gr.Markdown(elem_classes=["dp-workflow-message"])

        with gr.Column(visible=True, elem_classes=["dp-workflow-panel"]) as case_step:
            gr.HTML(
                '<h2 class="dp-section-title">1. Set up your audit case</h2>'
                '<p class="dp-section-copy">Add the deposit and property details, then create your audit case.</p>'
            )
            with gr.Column(elem_classes=["dp-case-card"]):
                gr.HTML('<h3 class="dp-subsection">Property and deposit details</h3>')
                case_name = gr.Textbox(label="Case / property name", placeholder="e.g. My Apartment")
                with gr.Row():
                    deposit = gr.Number(label="Security deposit (₹)", value=30000, minimum=0)
                    claimed = gr.Number(label="Landlord-claimed deductions (₹)", value=12000, minimum=0)
                proposed = gr.Number(label="Proposed refund (₹, optional)", value=18000, minimum=0)
                with gr.Row():
                    move_in = gr.DateTime(
                        label="Move-In Date",
                        include_time=False,
                        type="string",
                        info="Select a date or leave blank.",
                    )
                    move_out = gr.DateTime(
                        label="Move-Out Date",
                        include_time=False,
                        type="string",
                        info="Select a date or leave blank.",
                    )
                case_date_validation = gr.Markdown(
                    visible=False,
                    elem_classes=["dp-date-validation"],
                )
                with gr.Row():
                    city_dropdown = gr.Dropdown(
                        choices=CITY_CHOICES,
                        value=DEFAULT_CITY,
                        label="City",
                        filterable=True,
                    )
                    area_dropdown = gr.Dropdown(
                        choices=DEFAULT_LOCALITIES,
                        value=DEFAULT_AREA,
                        label="Area / Locality",
                        filterable=True,
                    )
                gr.HTML(
                    '<p class="dp-upload-help">The audit uses this locality for repair-price comparisons '
                    'where eligible deductions have matching sample data.</p>'
                )
                create_btn = gr.Button("Create audit case", variant="primary")
                active_case = gr.Textbox(label="Active case ID", interactive=False)
                case_message = gr.Markdown(
                    elem_classes=["dp-result-block", "dp-case-summary"]
                )
            with gr.Row(elem_classes=["dp-action-row"]):
                gr.HTML('<span class="dp-action-spacer"></span>')
                case_continue = gr.Button("Continue to evidence", variant="primary")

        with gr.Column(visible=False, elem_classes=["dp-workflow-panel"]) as evidence_step:
            gr.HTML(
                '<h2 class="dp-section-title">2. Add the supporting evidence</h2>'
                '<p class="dp-section-copy">Add the lease, optional supporting documents, move-in and move-out photos, and itemised claims.</p>'
            )
            gr.HTML('<h3 class="dp-subsection">Rental documents</h3>')
            gr.HTML(
                '<p class="dp-upload-help">Text-based PDFs are extracted locally. Scanned PDFs are not OCR-processed. '
                'The interface reports extraction and model status after processing; selecting a file alone does not mean it was analysed.</p>'
            )
            gr.HTML(
                '<div class="dp-document-intro"><svg class="dp-upload-icon" viewBox="0 0 24 24" '
                'fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">'
                '<path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>'
                '</svg><span>Upload the primary lease or agreement<small>One required file. PDFs and images are accepted; only text-based PDFs are extracted.</small></span></div>'
            )
            document_files = gr.File(
                label="Primary lease/agreement document",
                file_count="single", type="filepath",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
                elem_classes=["dp-document-upload"],
            )
            optional_documents_state = gr.State(False)
            additional_documents_toggle = gr.Button("+ Add Optional Additional Documents")
            with gr.Column(visible=False, elem_classes=["dp-optional-documents"]) as additional_documents_panel:
                additional_document_category = gr.Dropdown(
                    ["Invoice / receipt", "Deduction statement", "Inspection report", "Other document"],
                    value="Other document", label="Additional document category",
                )
                additional_document_files = gr.File(
                    label="Optional additional documents",
                    file_count="multiple", type="filepath",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
                    elem_classes=["dp-document-upload"],
                )
            with gr.Row():
                upload_btn = gr.Button("Upload documents and extract", variant="primary")
            upload_output = gr.Markdown(elem_classes=["dp-result-block"])

            gr.HTML('<h3 class="dp-subsection">Move-in and move-out photos</h3>')
            gr.HTML('<p class="dp-upload-help">Provide both photos of the same room or item. The vision model compares only the images provided and reports when it cannot analyse them.</p>')
            photo_item = gr.Textbox(label="Room or item", placeholder="e.g. Living room wall")
            with gr.Row(elem_classes=["dp-photo-grid"]):
                with gr.Column(elem_classes=["dp-photo-card"]):
                    gr.HTML(
                        '<div class="dp-photo-instructions"><svg class="dp-upload-icon" viewBox="0 0 24 24" '
                        'fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">'
                        '<path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>'
                        '</svg><span>Upload move-in photo<small>Drag an image here or browse files</small></span></div>'
                    )
                    photo_in = gr.File(
                        label="Choose move-in photo", type="filepath", file_count="single",
                        file_types=[".png", ".jpg", ".jpeg", ".webp"],
                        elem_classes=["dp-photo-upload"],
                    )
                    photo_in_preview = gr.Image(
                        label="Selected move-in preview", interactive=False,
                        visible=False, height=180, elem_classes=["dp-photo-preview"],
                    )
                with gr.Column(elem_classes=["dp-photo-card"]):
                    gr.HTML(
                        '<div class="dp-photo-instructions"><svg class="dp-upload-icon" viewBox="0 0 24 24" '
                        'fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">'
                        '<path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>'
                        '</svg><span>Upload move-out photo<small>Drag an image here or browse files</small></span></div>'
                    )
                    photo_out = gr.File(
                        label="Choose move-out photo", type="filepath", file_count="single",
                        file_types=[".png", ".jpg", ".jpeg", ".webp"],
                        elem_classes=["dp-photo-upload"],
                    )
                    photo_out_preview = gr.Image(
                        label="Selected move-out preview", interactive=False,
                        visible=False, height=180, elem_classes=["dp-photo-preview"],
                    )
            photo_btn = gr.Button("Save and compare photo pair", variant="primary")
            photo_output = gr.Markdown(elem_classes=["dp-result-block"])

            gr.HTML('<h3 class="dp-subsection">Itemised deductions</h3>')
            gr.HTML('<p class="dp-upload-help">Enter each landlord claim separately. Uploaded invoices and photo pairs are matched to deductions using the description and evidence keywords; uncertain matches are left unlinked.</p>')
            with gr.Row():
                deduction_category = gr.Dropdown(
                    ["Painting / repainting", "Cleaning", "Furniture damage", "Appliance damage",
                     "Plumbing", "Electrical repairs", "Structural damage", "Other"],
                    value="Other", label="Deduction category",
                )
                deduction_amount = gr.Number(label="Claimed amount (₹)", value=1000, minimum=0)
            deduction_description = gr.Textbox(label="Deduction description", placeholder="e.g. Repainting the living room")
            add_deduction_btn = gr.Button("Add deduction", variant="primary")
            deduction_output = gr.Markdown(elem_classes=["dp-result-block"])
            deductions_view = gr.Markdown(elem_classes=["dp-result-block"])
            add_deduction_btn.click(
                add_deduction,
                [active_case, deduction_category, deduction_description, deduction_amount],
                [deduction_output, deductions_view],
            )

            with gr.Row(elem_classes=["dp-action-row"]):
                gr.HTML('<span class="dp-action-spacer"></span>')
                evidence_continue = gr.Button("Continue to review", variant="primary")

        with gr.Column(visible=False, elem_classes=["dp-workflow-panel"]) as review_step:
            gr.HTML(
                '<h2 class="dp-section-title">3. Review the findings</h2>'
                '<p class="dp-section-copy">Run the audit to see rules classifications, agreement passages, linked photo observations, invoice evidence, and any saved price benchmark. Hosted explanations are labelled separately from evidence-rule results.</p>'
            )
            gr.Markdown(ai_status_markdown())
            gr.HTML(
                '<div class="dp-inline-note">Rules remain the classification authority. '
                'When hosted assessment fails, the result is labelled rules-based fallback.</div>'
            )
            with gr.Row():
                audit_btn = gr.Button("Run evidence audit", variant="primary")
            audit_output = gr.Markdown(elem_classes=["dp-result-block", "dp-report"])
            with gr.Row(elem_classes=["dp-action-row"]):
                review_back = gr.Button("Back to evidence")
                gr.HTML('<span class="dp-action-spacer"></span>')
                review_continue = gr.Button("Continue to report", variant="primary")

        with gr.Column(visible=False, elem_classes=["dp-workflow-panel"]) as report_step:
            gr.HTML(
                '<h2 class="dp-section-title">4. Final audit report</h2>'
                '<p class="dp-section-copy">Review the generated report and download its PDF. If you update evidence or deductions, return to the review step and run the audit again.</p>'
            )
            report_output = gr.Markdown(
                "Your report will appear here after the audit completes.",
                elem_classes=["dp-result-block", "dp-report"],
            )
            report_file = gr.DownloadButton(
                label="Download Audit Report",
                variant="primary",
                elem_classes=["dp-download-button"]
            )
            with gr.Row(elem_classes=["dp-action-row"]):
                report_back = gr.Button("Back to findings")
                gr.HTML('<span class="dp-action-spacer"></span>')

        gr.HTML(
            '<p class="dp-footer-note">DepositProof AI is an evidence-review aid, not legal advice. '
            'Photographs show visible differences but cannot independently establish timing, cause, repair cost, or legal responsibility. '
            'Local files and database records require appropriate privacy and retention controls.</p>'
        )

    gr.HTML(
        """
        <div class="dp-loading-overlay__content" role="status" aria-live="assertive">
          <div class="dp-loading-spinner" aria-hidden="true"></div>
          <p>Analyzing evidence &amp; generating report... Please wait</p>
        </div>
        """,
        elem_id="dp-loading-overlay",
    )

    workflow_components = [
        case_step, evidence_step, review_step, report_step,
        step_case, step_evidence, step_findings, step_report, workflow_message,
    ]

    enter_app_btn.click(
        _enter_application,
        outputs=[welcome_screen, app_shell, *workflow_components],
        show_progress="hidden",
        js=SCROLL_TO_TOP_JS,
    )
    case_continue.click(
        _continue_from_case, [active_case],
        workflow_components,
        js=SCROLL_TO_TOP_JS,
    )
    evidence_continue.click(
        _continue_from_evidence,
        [active_case],
        workflow_components,
        js=SCROLL_TO_TOP_JS,
    )
    review_back.click(
        lambda case_id: _navigate_to_step(2, case_id, None),
        [active_case], workflow_components,
        js=SCROLL_TO_TOP_JS,
    )
    review_continue.click(
        _continue_from_review,
        [report_output],
        workflow_components,
        js=SCROLL_TO_TOP_JS,
    )
    report_back.click(
        lambda case_id: _navigate_to_step(3, case_id, None),
        [active_case], workflow_components,
        js=SCROLL_TO_TOP_JS,
    )
    for step_number, step_button in enumerate(
        (step_case, step_evidence, step_findings, step_report), start=1,
    ):
        step_button.click(
            lambda case_id, report, target=step_number: _navigate_to_step(target, case_id, report),
            [active_case, report_output],
            workflow_components,
            show_progress="hidden",
            js=SCROLL_TO_TOP_JS,
        )

    create_btn.click(
        create_case, [case_name, deposit, claimed, proposed, move_in, move_out],
        [active_case, case_message],
    )
    move_in.change(
        handle_move_in_change,
        [move_in, move_out],
        [move_out, case_date_validation],
        show_progress="hidden",
    )
    move_out.change(
        validate_case_dates,
        [move_in, move_out],
        case_date_validation,
        show_progress="hidden",
    )
    additional_documents_toggle.click(
        toggle_additional_documents,
        [optional_documents_state],
        [optional_documents_state, additional_documents_panel, additional_documents_toggle],
        show_progress="hidden",
    )
    upload_event = upload_btn.click(
        upload_documents,
        [active_case, document_files, additional_document_files, additional_document_category],
        [upload_output, deductions_view],
        show_progress="full",
        js=SHOW_LOADING_JS,
    )
    upload_event.then(
        fn=None, inputs=[], outputs=[], js=HIDE_LOADING_JS, show_progress="hidden",
    )
    photo_event = photo_btn.click(
        upload_photo_pair, [active_case, photo_item, photo_in, photo_out],
        [photo_output, deductions_view], show_progress="full", js=SHOW_LOADING_JS,
    )
    photo_event.then(
        fn=None, inputs=[], outputs=[], js=HIDE_LOADING_JS, show_progress="hidden",
    )
    photo_in.change(
        lambda path: gr.update(value=path, visible=bool(path)),
        [photo_in], [photo_in_preview], show_progress="hidden",
    )
    photo_out.change(
        lambda path: gr.update(value=path, visible=bool(path)),
        [photo_out], [photo_out_preview], show_progress="hidden",
    )
    city_dropdown.change(
        update_locality_choices,
        [city_dropdown],
        [area_dropdown],
        show_progress="hidden",
    )
    audit_event = audit_btn.click(
        _run_audit_for_workflow, [active_case, city_dropdown, area_dropdown],
        [audit_output, report_output, report_file, deductions_view],
        show_progress="full", js=SHOW_LOADING_JS,
    )
    audit_event = audit_event.then(
        fn=None, inputs=[], outputs=[], js=HIDE_LOADING_JS, show_progress="hidden",
    ).then(
        lambda case_id, report: _navigate_to_step(4, case_id, report),
        [active_case, report_output],
        workflow_components,
        js=SCROLL_TO_TOP_JS,
    )
    gr.HTML(
        '<noscript><p>JavaScript is required to navigate this interactive audit application.</p></noscript>'
    )


def _launch_kwargs() -> dict[str, Any]:
    configured_port = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT")
    hosted = "RENDER" in os.environ or configured_port is not None
    launch_kwargs = {
        "theme": gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        "css": APP_CSS,
        "server_name": "0.0.0.0" if hosted else "127.0.0.1",
        "server_port": int(configured_port) if configured_port else None,
    }
    if not hosted:
        launch_kwargs["inbrowser"] = True
    if "show_api" in inspect.signature(gr.Blocks.launch).parameters:
        launch_kwargs["show_api"] = False
    return launch_kwargs


if __name__ == "__main__":
    demo.launch(**_launch_kwargs())