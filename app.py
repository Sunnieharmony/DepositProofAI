"""
DepositProof AI — hackathon starter application.

LLM integration is intentionally isolated in llm_integration.py.
This app can run in deterministic demo mode without an LLM, but demo findings
are clearly marked and must never be represented as real AI analysis.
"""
from __future__ import annotations

import json
import html
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import gradio as gr
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

from document_processing import extract_inspection_fields, extract_pdf
from evidence_engine import (
    classify_deduction,
    embedding_status,
    extract_invoice_fields,
    find_relevant_clauses,
    money_text,
    retrieve_topic_passages,
    verify_quote,
)
from llm_integration import (
    analyze_agreement_with_llm,
    analyze_image_pair_with_llm,
    assess_deduction_with_llm,
    get_last_error,
    explain_repair_benchmark_with_llm,
    llm_is_configured,
    text_model,
    validate_image_file,
    vision_model,
)
from report_generator import generate_audit_pdf
from repair_price_intelligence import (
    DATASET_NAME,
    RepairRateDataError,
    compare_repair_price,
    load_repair_rates,
)

UPLOADS = ROOT / "uploads"
REPORTS = ROOT / "reports"
DB_PATH = ROOT / "data" / "depositproof.sqlite3"
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
        lines = ["**Photo comparison:** The pair was saved, but no successful vision-model observation is stored."]
        if photo.get("error"):
            lines.append(f"- **Analysis error:** {photo['error']}")
        if not description:
            lines.append("- This photo pair is not associated with a deduction.")
        return lines

    lines = ["**Photo comparison (separate vision-language observation):**"]
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
            message += f"\n\n> Photo analysis failed: {error}"
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
        f"### Case created: `{case_id}`\n\n"
        f"- **Security deposit:** {money(deposit)}\n"
        f"- **Claimed deductions:** {money(claimed)}\n"
        f"- **Calculated refund:** {money(calculated)}\n"
        f"- **Proposed refund entered:** {money(proposed) if proposed is not None else 'Not provided'}\n\n"
        f"{'⚠️ The proposed refund differs from the calculated refund.' if proposed is not None and abs(proposed-calculated) > 0.01 else 'Financial arithmetic checked.'}\n\n"
        f"{ai_status_markdown()}"
    )
    return case_id, msg, gr.update(choices=list_cases())


def ai_status_markdown() -> str:
    """Plain statement of which AI components are active, so nothing is overstated."""
    emb = embedding_status()
    if llm_is_configured():
        hosted = f"Hosted models enabled — vision: `{vision_model()}`, language: `{text_model()}`."
    else:
        hosted = "Hosted models **not configured** (no API key for the selected provider): photos are saved but not analysed, and assessments use the rules-based fallback only."
    retrieval = ("Clause retrieval uses semantic search (`sentence-transformers`) when installed; otherwise a labelled keyword fallback."
                 if not emb["error"] else f"Clause retrieval is using the keyword fallback ({emb['error']}).")
    return f"**AI status:** {hosted} {retrieval}"


def list_cases():
    with db() as con:
        rows = con.execute("SELECT id,name,created_at FROM cases ORDER BY created_at DESC").fetchall()
    return [f"{r['id']} — {r['name']} ({r['created_at']})" for r in rows]


def parse_case_choice(choice: str | None) -> str | None:
    if not choice:
        return None
    return choice.split(" — ", 1)[0].strip()


def upload_documents(case_id, files, category):
    if not case_id:
        raise gr.Error("Create a case first.")
    if not files:
        return "No files selected.", gr.update()
    results = []
    for item in files:
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
    return ("### Upload results\n\n" + ("\n".join(results) if results else "No valid files were processed.")
            + "\n\nFiles that could not be read are stored but **not** analysed; the audit will say so rather than guess."), \
        gr.update(choices=document_choices(case_id), value="None")


def documents_for_case(case_id):
    with db() as con:
        return con.execute("SELECT * FROM documents WHERE case_id=? ORDER BY created_at", (case_id,)).fetchall()


def _choice_id(choice: str | None) -> str | None:
    """'abc123 — label' -> 'abc123'; 'None'/empty -> None."""
    if not choice or choice == "None":
        return None
    return choice.split(" — ", 1)[0].strip()


def photo_choices(case_id):
    if not case_id:
        return ["None"]
    with db() as con:
        rows = con.execute("SELECT id,item_name,status FROM photo_pairs WHERE case_id=? ORDER BY created_at", (case_id,)).fetchall()
    return ["None"] + [f"{r['id']} — {r['item_name']} ({r['status']})" for r in rows]


def deduction_choices(case_id):
    if not case_id:
        return []
    with db() as con:
        rows = con.execute("SELECT id,description,amount FROM deductions WHERE case_id=? ORDER BY created_at", (case_id,)).fetchall()
    return [f"{r['id']} — {r['description']} ({money(r['amount'])})" for r in rows]


def prefill_benchmark_deduction(case_id, deduction_choice):
    deduction_id = _choice_id(deduction_choice)
    if not case_id or not deduction_id:
        return gr.update(), None
    with db() as con:
        row = con.execute(
            "SELECT category,description,amount FROM deductions WHERE id=? AND case_id=?",
            (deduction_id, case_id),
        ).fetchone()
    if not row:
            return gr.update(), None
    categories = list(dict.fromkeys([
        "Cleaning", "Electrical Repairs", "Furniture / Carpentry",
        "Painting / Repainting", "Plumbing", row["category"],
    ]))
    return (
        gr.update(choices=categories, value=row["category"]),
        row["amount"],
    )


def _benchmark_deterministic_explanation(result):
    if result["status"] == "no_match":
        return result["explanation"]
    amount = money(result["difference"], "INR")
    percent = (
        f" ({result['difference_percent']:.1f}%)"
        if result.get("difference_percent") is not None and result["position"] != "within"
        else ""
    )
    if result["position"] == "above":
        return f"The claim is above the matched sample range by {amount}{percent}."
    if result["position"] == "below":
        return f"The claim is below the matched sample range by {amount}{percent}."
    return "The claim falls within the matched sample range; this is not a finding that the price is fair or legally recoverable."


def _benchmark_error_html(message):
    return (
        '<div class="dp-price"><div class="dp-alert"><strong>Could not compare repair rates</strong><br>'
        + html.escape(str(message))
        + "</div></div>"
    )


def render_repair_benchmark(result):
    styles = """
    <style>
      .dp-price{font-family:Arial,sans-serif;color:#172554;padding:4px 0 18px}
      .dp-price .dp-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0}
      .dp-price .dp-card{background:#f8fafc;border:1px solid #dbe4ef;border-radius:12px;padding:14px}
      .dp-price .dp-card small,.dp-price .dp-muted{color:#64748b}
      .dp-price .dp-value{font-size:1.35rem;font-weight:700;margin-top:5px}
      .dp-price .dp-alert{background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:14px;color:#7c2d12}
      .dp-price .dp-good{background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;padding:14px}
      .dp-price .dp-chart{width:100%;height:auto;max-height:185px}
      .dp-price table{width:100%;border-collapse:collapse;margin:10px 0;font-size:.88rem}
      .dp-price th,.dp-price td{border:1px solid #dbe4ef;text-align:left;padding:8px;vertical-align:top}
      .dp-price th{background:#eff6ff}
      .dp-price h3{margin:20px 0 8px}
      @media(max-width:600px){.dp-price table{font-size:.76rem}.dp-price th,.dp-price td{padding:5px}}
    </style>
    """
    if result["status"] == "no_match":
        return styles + (
            '<div class="dp-price"><h2>Repair Price Intelligence</h2>'
            '<div class="dp-alert">' + html.escape(result["explanation"]) + "</div>"
            '<p class="dp-muted">' + html.escape(result["sample_data_warning"]) + "</p></div>"
        )

    low, high = result["lower"], result["upper"]
    claim = result["claimed_amount"]
    difference = result["difference"]
    variance = (
        f"{result['difference_percent']:.1f}%"
        if result["position"] != "within" and result.get("difference_percent") is not None
        else ("Within range" if result["position"] == "within" else "Not available")
    )
    max_axis = max(high, claim, 1) * 1.12
    chart_left, chart_right = 75, 755
    x = lambda value: chart_left + value / max_axis * (chart_right - chart_left)
    chart = (
        '<svg class="dp-chart" viewBox="0 0 820 150" role="img" aria-label="Claim compared with sample repair-rate range">'
        '<text x="75" y="19" fill="#475569" font-size="13">Claim compared with sample range (INR)</text>'
        '<line x1="75" y1="76" x2="755" y2="76" stroke="#cbd5e1" stroke-width="5"/>'
        f'<line x1="{x(low):.1f}" y1="76" x2="{x(high):.1f}" y2="76" stroke="#2563eb" stroke-width="18" stroke-linecap="round"/>'
        f'<text x="{x(low):.1f}" y="112" text-anchor="middle" fill="#334155" font-size="12">{html.escape(money(low))}</text>'
        f'<text x="{x(high):.1f}" y="112" text-anchor="middle" fill="#334155" font-size="12">{html.escape(money(high))}</text>'
        f'<circle cx="{x(claim):.1f}" cy="76" r="9" fill="#e11d48" stroke="white" stroke-width="3"/>'
        f'<text x="{x(claim):.1f}" y="48" text-anchor="middle" fill="#9f1239" font-size="12">Claim {html.escape(money(claim))}</text>'
        '<text x="75" y="137" fill="#64748b" font-size="11">₹0</text>'
        f'<text x="755" y="137" text-anchor="end" fill="#64748b" font-size="11">{html.escape(money(max_axis))}</text>'
        "</svg>"
    )
    cards = (
        f'<div class="dp-grid"><div class="dp-card"><small>Landlord claim</small><div class="dp-value">{html.escape(money(claim))}</div></div>'
        f'<div class="dp-card"><small>Matched sample range</small><div class="dp-value">{html.escape(money(low))} – {html.escape(money(high))}</div></div>'
        f'<div class="dp-card"><small>Difference from nearest boundary</small><div class="dp-value">{html.escape(money(difference))}</div></div>'
        f'<div class="dp-card"><small>Position</small><div class="dp-value">{html.escape(result["position"].replace("_", " ").title())}</div><small>{html.escape(variance)}</small></div></div>'
    )
    rows = []
    for row in result["matched_records"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['city'])}<br>{html.escape(row['locality'])}</td>"
            f"<td>{html.escape(row['category'])}</td>"
            f"<td>{html.escape(money(row['minimum']))} – {html.escape(money(row['maximum']))}</td>"
            "</tr>"
        )
    quality = result.get("dataset_quality") or {}
    issue_note = ""
    if quality.get("issues") or quality.get("duplicate_count"):
        issue_note = (
            f"<p class='dp-muted'>Data quality: {len(quality.get('issues', []))} invalid row(s) skipped; "
            f"{quality.get('duplicate_count', 0)} duplicate row(s) removed.</p>"
        )
    dataset_note = (
        f"<p class='dp-muted'>Dataset: {quality.get('valid_row_count', 'unknown')} usable sample row(s) "
        f"across the source file; {result['record_count']} matched row(s) used for this calculation.</p>"
    )
    comparison = (
        '<div class="dp-good"><strong>Comparison level:</strong> '
        + html.escape(result["match_description"])
        + f". {result['record_count']} sample record(s) included.</div>"
    )
    explanation_mode = result.get("explanation_mode", "deterministic")
    if explanation_mode == "hosted_model":
        explanation = (
            '<p><strong>Optional hosted-model explanation (does not change the calculation):</strong> '
            + html.escape(result["llm_explanation"])
            + "</p>"
        )
    else:
        explanation = (
            '<p><strong>Deterministic explanation:</strong> '
            + html.escape(result.get("deterministic_explanation", ""))
            + "</p>"
        )
        if result.get("llm_error"):
            explanation += (
                '<p class="dp-muted">Hosted-model explanation was not used: '
                + html.escape(result["llm_error"])
                + "</p>"
            )
    attach_note = result.get("report_attachment_note")
    if attach_note:
        explanation += '<p class="dp-muted">' + html.escape(attach_note) + "</p>"
    return (
        styles + '<div class="dp-price"><h2>Repair Price Intelligence</h2>'
        + comparison + cards + chart + explanation
        + "<p><strong>Matching method:</strong> " + html.escape(result["match_description"]) + ".</p>"
        + "<p>" + html.escape(result["calculation"]) + "</p>"
        + "<p class='dp-muted'>" + html.escape(result["sample_data_warning"]) + "</p>"
        + dataset_note + issue_note
        + "<h3>Matched sample records</h3><div style='overflow-x:auto'><table><thead><tr>"
        + "<th>City / locality</th><th>Category</th><th>Listed range (INR)</th>"
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></div>"
    )


def run_repair_price_comparison(
    case_id, deduction_choice, category, claimed_amount, city, locality, ask_hosted_model
):
    try:
        dataset = load_repair_rates()
        result = compare_repair_price(
            dataset["records"],
            city=city,
            locality=locality,
            category=category,
            claimed_amount=claimed_amount,
        )
    except (RepairRateDataError, ValueError) as exc:
        return _benchmark_error_html(str(exc))

    result["dataset_quality"] = {
        "valid_row_count": len(dataset["records"]),
        "issues": dataset["issues"],
        "duplicate_count": dataset["duplicate_count"],
    }
    result["deterministic_explanation"] = _benchmark_deterministic_explanation(result)
    result["explanation_mode"] = "deterministic"
    deduction_id = _choice_id(deduction_choice)
    if deduction_id:
        if not case_id:
            return _benchmark_error_html("Select an active case before attaching a comparison to a deduction.")
        with db() as con:
            selected_deduction = con.execute(
                """SELECT d.description,d.amount,c.currency
                   FROM deductions d JOIN cases c ON c.id=d.case_id
                   WHERE d.id=? AND d.case_id=?""",
                (deduction_id, case_id),
            ).fetchone()
        if not selected_deduction:
            return _benchmark_error_html("The selected deduction is not part of the active case. Refresh the deduction list.")
    if ask_hosted_model and result["status"] == "matched":
        result["llm_explanation"] = explain_repair_benchmark_with_llm(result)
        if result["llm_explanation"]:
            result["explanation_mode"] = "hosted_model"
        else:
            result["llm_error"] = get_last_error() or "No valid hosted-model explanation was returned."

    if deduction_id:
        with db() as con:
            deduction = con.execute(
                """SELECT d.description,d.amount,c.currency
                   FROM deductions d JOIN cases c ON c.id=d.case_id
                   WHERE d.id=? AND d.case_id=?""",
                (deduction_id, case_id),
            ).fetchone()
            if not deduction:
                return _benchmark_error_html("The selected deduction is no longer available; refresh the deduction list.")
            con.execute("DELETE FROM repair_benchmarks WHERE deduction_id=?", (deduction_id,))
            amount_matches = abs(float(deduction["amount"]) - float(result["claimed_amount"])) < 0.01
            if (result["status"] == "matched" and amount_matches
                    and deduction["currency"] == "INR"):
                con.execute(
                    """INSERT INTO repair_benchmarks(
                        deduction_id,case_id,deduction_description,deduction_amount,
                        benchmark_json,created_at
                    ) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(deduction_id) DO UPDATE SET
                        case_id=excluded.case_id,
                        deduction_description=excluded.deduction_description,
                        deduction_amount=excluded.deduction_amount,
                        benchmark_json=excluded.benchmark_json,
                        created_at=excluded.created_at""",
                    (
                        deduction_id, case_id, deduction["description"], deduction["amount"],
                        json.dumps(result, ensure_ascii=False),
                        datetime.now().isoformat(timespec="seconds"),
                    ),
                )
                result["report_attachment_note"] = (
                    "Saved to this deduction. The next audit report will include it as a separate, "
                    "non-classifying benchmark section."
                )
            elif deduction["currency"] != "INR":
                result["report_attachment_note"] = "Not attached to the audit because the case currency is not INR."
            elif not amount_matches:
                result["report_attachment_note"] = (
                    "Not attached to the audit because the comparison amount does not match the saved deduction amount."
                )
            else:
                result["report_attachment_note"] = "No matched estimate was saved to the audit."
    elif result["status"] == "matched":
        result["report_attachment_note"] = "Dashboard-only comparison; select a deduction to include it in the next audit report."
    return render_repair_benchmark(result)


def _owned_id(con, table: str, item_id: str | None, case_id: str) -> str | None:
    """Return item_id only if that record belongs to this case (never link another case's files)."""
    if not item_id:
        return None
    row = con.execute(f"SELECT id FROM {table} WHERE id=? AND case_id=?", (item_id, case_id)).fetchone()
    if not row:
        raise gr.Error("That item does not belong to the active case. Refresh the lists and try again.")
    return item_id


def add_deduction(case_id, category, description, amount, invoice_choice, photo_choice=None):
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
        invoice_id = _owned_id(con, "documents", _choice_id(invoice_choice), case_id)
        photo_id = _owned_id(con, "photo_pairs", _choice_id(photo_choice), case_id)
        con.execute(
            """INSERT INTO deductions(id,case_id,category,description,amount,
               invoice_document_id,photo_pair_id,created_at) VALUES(?,?,?,?,?,?,?,?)""",
            (deduction_id, case_id, category, str(description).strip(), amount,
             invoice_id, photo_id, datetime.now().isoformat(timespec="seconds"))
        )
    return (f"Added deduction `{deduction_id}`: **{description} — {money(amount)}**",
            refresh_deductions(case_id),
            gr.update(choices=deduction_choices(case_id), value=None))


def link_evidence(case_id, deduction_choice, invoice_choice, photo_choice):
    """Attach (or change) the invoice and/or photo pair of an EXISTING deduction. Persisted in SQLite."""
    if not case_id:
        raise gr.Error("Create a case first.")
    deduction_id = _choice_id(deduction_choice)
    if not deduction_id:
        raise gr.Error("Select a deduction first.")
    with db() as con:
        _owned_id(con, "deductions", deduction_id, case_id)
        invoice_id = _owned_id(con, "documents", _choice_id(invoice_choice), case_id)
        photo_id = _owned_id(con, "photo_pairs", _choice_id(photo_choice), case_id)
        con.execute("UPDATE deductions SET invoice_document_id=?, photo_pair_id=?, status=NULL, assessment_json=NULL WHERE id=? AND case_id=?",
                    (invoice_id, photo_id, deduction_id, case_id))
    return ("Evidence links saved. Invoice: "
            f"{invoice_choice if invoice_id else 'none'}; photo pair: {photo_choice if photo_id else 'none'}. Re-run the audit to refresh results."), \
        refresh_deductions(case_id)


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


def document_choices(case_id):
    if not case_id:
        return ["None"]
    with db() as con:
        rows = con.execute(
            "SELECT id,filename FROM documents WHERE case_id=? AND category IN ('Invoice / receipt','Deduction statement')",
            (case_id,)
        ).fetchall()
    return ["None"] + [f"{r['id']} — {r['filename']}" for r in rows]


def refresh_deductions(case_id):
    if not case_id:
        return "Create or select a case first."
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


def run_audit(case_id):
    if not case_id:
        raise gr.Error("Create a case first.")
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

        payload = {
            "description": deduction["description"],
            "category": deduction["category"],
            "amount": deduction["amount"],
            "currency": case["currency"],
            "case_dates": {"move_in": case["move_in"], "move_out": case["move_out"]},
            "clauses": clauses,
            "agreement_available": bool(agreement_texts),
            "agreement_problem": agreement_problem,
            "invoice": invoice,
            "photo_observation": photo_observation,
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
            "", f"## {i}. {f['description']} — {money(f['amount'], case['currency'])}",
            f"**Final classification:** {f.get('classification', 'Insufficiently Documented')}",
            f"**Assessment method:** {f.get('assessment_method', _assessment_method(f))}",
            f"**Rules-engine classification:** {f.get('rules_classification', f.get('classification', 'Insufficiently Documented'))}",
            f"**Explanation:** {f.get('explanation', 'No explanation available.')}",
            f"**Evidence reviewed:** {f.get('evidence_reviewed', 'See evidence details below.')}",
            f"**Missing or conflicting evidence:** {f.get('missing_or_conflicting_evidence', 'Not specified.')}",
            f"**Limitations:** {f.get('limitations', 'This assessment is limited to the uploaded evidence.')}",
        ]
        if f.get("clauses"):
            lines.append("**Relevant agreement passages:**")
            for c in f["clauses"][:5]:
                lines.append(f"- {c.get('filename', 'Agreement')}, page {c.get('page', 'unknown')}: “{c.get('quote', '')}” (quotation verified: {c.get('quote_verified', False)})")
        if f.get("invoice"):
            lines.append(f"**Invoice source:** {f['invoice'].get('filename', 'Selected document')}")
            lines.append(f"**Invoice extraction:** {json.dumps({k:v for k,v in f['invoice'].items() if k != 'extracted_text'}, ensure_ascii=False)}")
        if f.get("photo_observation"):
            photo = f["photo_observation"]
            lines.extend(_photo_comparison_markdown(photo, f.get("photo_item_name"), f["description"]))
        else:
            lines.append("**Photo comparison:** No linked usable photo observation is available for this deduction.")
        benchmark = f.get("repair_price_benchmark")
        if benchmark:
            lines.extend([
                "",
                "**Repair-price benchmark (context only; does not change the classification):**",
                f"- Comparison: {benchmark.get('match_description', 'Comparable sample-rate records')} "
                f"for {benchmark.get('category', 'the selected category')} in "
                f"{benchmark.get('city', 'the selected city')}"
                + (f", {benchmark['locality']}" if benchmark.get("locality") else "")
                + f"; {benchmark.get('record_count', 0)} sample record(s).",
                f"- Sample range: {money(benchmark['lower'])}–{money(benchmark['upper'])}.",
                f"- Claim position: {benchmark.get('position', 'not stated')}; difference from nearest boundary: "
                f"{money(benchmark.get('difference', 0))}."
                + (f" ({benchmark['difference_percent']:.1f}%)"
                   if benchmark.get("difference_percent") is not None and benchmark.get("position") != "within"
                   else ""),
                f"- Source: {benchmark.get('source_dataset', DATASET_NAME)}. "
                f"{benchmark.get('sample_data_warning', 'Sample data; not verified quotations.')}",
            ])
            if benchmark.get("llm_explanation"):
                lines.append(
                    f"- Optional hosted-model explanation (does not change the calculation): {benchmark['llm_explanation']}"
                )
        if f.get("ai_error"):
            lines.append(f"**Hosted AI unavailable:** {f['ai_error']} (rules-based fallback used.)")
        if f.get("llm_classification_suggestion"):
            lines.append(
                f"**LLM suggestion (not authoritative):** {f['llm_classification_suggestion']}"
            )
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
        analysis_error = get_last_error() or "Hosted AI is not configured; check the selected provider's API key and LLM_PROVIDER in .env."
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
    detail = f"Photo pair saved for **{item_name}**. Status: `{mode}`. Pair ID: `{pair_id}`"
    detail += "\n\n" + _photo_upload_result_markdown(
        observation if mode == "llm_analyzed" else None,
        analysis_error,
    )
    return detail, gr.update(choices=photo_choices(case_id), value=f"{pair_id} — {str(item_name).strip()} ({mode})")


def analyze_agreement(case_id):
    if not case_id:
        raise gr.Error("Create a case first.")
    docs = documents_for_case(case_id)
    agreements = [d for d in docs if d["category"] == "Rental agreement" and d["extracted_text"]]
    if not agreements:
        return "No readable rental agreement PDF found for this case."
    if not llm_is_configured():
        return "Hosted language models are not configured. Add an API key for the selected provider to the local `.env` file; the extracted agreement text remains available for rules-based clause retrieval."
    outputs = []
    for d in agreements:
        result = analyze_agreement_with_llm(
            d["extracted_text"],
            json.loads(d["page_text_json"] or "{}"),
            d["filename"],
        )
        if result is None:
            outputs.append(f"## {d['filename']}\n\nHosted analysis failed: {get_last_error() or 'No usable model response was returned.'}")
        else:
            outputs.append(f"## {d['filename']}\n\n```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```")
    return "\n\n".join(outputs)


def load_selected_case(choice):
    case_id = parse_case_choice(choice)
    return (
        case_id or "",
        refresh_deductions(case_id),
        gr.update(choices=document_choices(case_id), value="None"),
        gr.update(choices=photo_choices(case_id), value="None"),
        gr.update(choices=deduction_choices(case_id), value=None),
        f"Loaded case `{case_id}`." if case_id else "Select a saved case first.",
    )

_WORKFLOW_STEPS = ("Audit case", "Evidence", "Review findings", "Final report")


def _render_workflow_stepper(active_step: int) -> str:
    items = []
    for index, label in enumerate(_WORKFLOW_STEPS, start=1):
        state = "complete" if index < active_step else "active" if index == active_step else "upcoming"
        items.append(
            f'<div class="dp-step dp-step--{state}" aria-current="{"step" if state == "active" else "false"}">'
            f'<span class="dp-step__number">{index}</span><span>{html.escape(label)}</span></div>'
        )
    return '<nav class="dp-stepper" aria-label="Audit progress">' + "".join(items) + "</nav>"


def _workflow_visibility(active_step: int):
    if active_step not in range(1, 5):
        raise ValueError("Workflow step must be between 1 and 4.")
    return (
        *(gr.update(visible=index == active_step) for index in range(1, 5)),
        _render_workflow_stepper(active_step),
    )


def _navigate_to_step(label: str):
    if label not in _WORKFLOW_STEPS:
        raise ValueError(f"Unknown workflow step: {label}")
    return (*_workflow_visibility(_WORKFLOW_STEPS.index(label) + 1)[:5], "")


def _enter_application():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        *_workflow_visibility(1),
        "",
    )


def _active_case_problem(case_id):
    if not case_id:
        return "Create or select an audit case before continuing."
    with db() as con:
        if not con.execute("SELECT 1 FROM cases WHERE id=?", (case_id,)).fetchone():
            return "The selected audit case could not be found. Create or select it again."
    return None


def _continue_from_case(case_id):
    problem = _active_case_problem(case_id)
    if problem:
        return (*_workflow_visibility(1)[:5], f"**{problem}**")
    return (*_workflow_visibility(2)[:5], "")


def _continue_from_evidence(case_id):
    problem = _active_case_problem(case_id)
    if problem:
        return (*_workflow_visibility(2)[:5], f"**{problem}**")
    return (*_workflow_visibility(3)[:5], "")


def _continue_from_review(report_markdown):
    if not report_markdown or not str(report_markdown).strip():
        return (
            *_workflow_visibility(3)[:5],
            "**Run the evidence audit before continuing to the final report.**",
        )
    return (*_workflow_visibility(4)[:5], "")


def _run_audit_for_workflow(case_id):
    markdown, pdf_path, deductions = run_audit(case_id)
    return markdown, markdown, pdf_path, deductions


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
.dp-topbar .dp-brand-lockup svg { width: 34px; height: 38px; }
.dp-nav { display: flex; justify-content: flex-end; gap: 5px; flex-wrap: wrap; }
.dp-nav button { min-height: 38px !important; padding: 0 12px !important; border: 1px solid transparent !important; border-radius: 7px !important; background: transparent !important; color: var(--dp-muted) !important; font-size: 12px !important; font-weight: 600 !important; box-shadow: none !important; transition: color .16s ease, background .16s ease, border-color .16s ease !important; }
.dp-nav button:hover { border-color: var(--dp-line) !important; background: #fff !important; color: var(--dp-navy) !important; }
.dp-mobile-nav { display: none !important; }
.dp-page-heading { margin: 0 0 18px; }
.dp-page-heading h1 { margin: 6px 0 8px; color: var(--dp-navy); font-size: clamp(27px, 4vw, 38px); font-weight: 620; letter-spacing: -.04em; }
.dp-page-heading p { max-width: 780px; margin: 0; color: var(--dp-muted); font-size: 14px; line-height: 1.65; }
.dp-stepper { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 18px 0 24px; border-bottom: 1px solid var(--dp-line); }
.dp-step { position: relative; display: flex; align-items: center; gap: 10px; min-height: 52px; padding: 8px 12px 14px 0; color: #82909c; font-size: 12px; font-weight: 600; }
.dp-step:after { position: absolute; right: 10px; bottom: -1px; left: 0; height: 2px; background: transparent; content: ""; }
.dp-step--active { color: var(--dp-navy); }
.dp-step--active:after { background: var(--dp-blue); }
.dp-step--complete { color: var(--dp-success); }
.dp-step__number { display: grid; width: 25px; height: 25px; flex: none; place-items: center; border: 1px solid var(--dp-line-strong); border-radius: 50%; color: inherit; font-size: 11px; }
.dp-step > span:last-child { color: inherit !important; }
.dp-step--active .dp-step__number { border-color: var(--dp-blue); background: var(--dp-blue); color: #fff; }
.dp-step--active .dp-step__number { color: #fff !important; }
.dp-step--complete .dp-step__number { border-color: #c9ddd2; background: #edf5f0; }
.dp-workflow-message:empty { display: none !important; }
.dp-workflow-message { margin: 12px 0; padding: 10px 13px; border-left: 2px solid var(--dp-danger); background: #fbf4f2; color: #754541; font-size: 12px; line-height: 1.55; }
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
.dp-photo-card [data-testid="file-upload"],
.dp-document-upload [data-testid="file-upload"] { color: var(--dp-ink) !important; }
.dp-photo-card [data-testid="file-upload"] *,
.dp-document-upload [data-testid="file-upload"] * { color: inherit; }
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
.dp-result-block { min-width: 0; margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--dp-line); overflow-wrap: anywhere; color: var(--dp-ink); opacity: 1; }
.dp-report .prose { max-width: 100%; color: var(--dp-ink); opacity: 1; line-height: 1.75; }
.dp-report .prose h1, .dp-report .prose h2, .dp-report .prose h3 { color: var(--dp-navy); letter-spacing: -.025em; }
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
  .dp-nav { justify-content: flex-start; width: 100%; }
  .dp-nav button { padding: 0 9px !important; font-size: 11px !important; }
  .dp-step { align-items: flex-start; flex-direction: row; gap: 7px; min-width: 0; padding-right: 5px; font-size: 10px; }
  .dp-workflow-panel { min-height: 0; padding: 20px 16px; }
  .dp-case-layout, .dp-photo-grid { flex-wrap: wrap !important; }
  .dp-case-card, .dp-photo-card { flex: 1 1 100% !important; }
  .dp-action-row { margin-top: 22px; }
}
@media (max-width: 480px) {
  .dp-step { flex-direction: column; gap: 5px; font-size: 9px; }
  .dp-step__number { width: 22px; height: 22px; }
  .dp-nav { display: none !important; }
  .dp-mobile-nav { display: block !important; width: 100% !important; }
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
  .dp-topbar, .dp-nav, .dp-mobile-nav, .dp-stepper, .dp-action-row, .dp-footer-note { display: none !important; }
  .dp-app-shell, .dp-workflow-panel { width: 100% !important; max-width: none !important; padding: 0 !important; border: 0 !important; box-shadow: none !important; }
}
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
            with gr.Row(elem_classes=["dp-nav"]):
                nav_case = gr.Button("1  Case", elem_classes=["dp-nav-button"])
                nav_evidence = gr.Button("2  Evidence", elem_classes=["dp-nav-button"])
                nav_findings = gr.Button("3  Findings", elem_classes=["dp-nav-button"])
                nav_report = gr.Button("4  Report", elem_classes=["dp-nav-button"])
            mobile_nav = gr.Dropdown(
                label="Navigate to a section",
                choices=list(_WORKFLOW_STEPS),
                value=_WORKFLOW_STEPS[0],
                interactive=True,
                elem_classes=["dp-mobile-nav"],
            )

        gr.HTML(
            '<header class="dp-page-heading"><div class="dp-eyebrow">Your deposit review</div>'
            '<h1>Build a clear evidence record</h1>'
            '<p>Move through each step at your own pace. Uploaded records and model observations are shown only after the application reports them.</p></header>'
        )
        stepper = gr.HTML(_render_workflow_stepper(1))
        workflow_message = gr.Markdown(elem_classes=["dp-workflow-message"])

        with gr.Column(visible=True, elem_classes=["dp-workflow-panel"]) as case_step:
            gr.HTML(
                '<h2 class="dp-section-title">1. Set up your audit case</h2>'
                '<p class="dp-section-copy">Add the deposit details, then create a new case or return to one you saved earlier.</p>'
            )
            with gr.Row(elem_classes=["dp-case-layout"]):
                with gr.Column(scale=3, elem_classes=["dp-case-card"]):
                    gr.HTML('<h3 class="dp-subsection">Property and deposit details</h3>')
                    case_name = gr.Textbox(label="Case / property name", placeholder="e.g. My Apartment")
                    with gr.Row():
                        deposit = gr.Number(label="Security deposit (₹)", value=30000, minimum=0)
                        claimed = gr.Number(label="Landlord-claimed deductions (₹)", value=12000, minimum=0)
                    proposed = gr.Number(label="Proposed refund (₹, optional)", value=18000, minimum=0)
                    with gr.Row():
                        move_in = gr.Textbox(label="Move-in date (optional)", placeholder="YYYY-MM-DD")
                        move_out = gr.Textbox(label="Move-out date (optional)", placeholder="YYYY-MM-DD")
                    create_btn = gr.Button("Create audit case", variant="primary")
                    active_case = gr.Textbox(label="Active case ID", interactive=False)
                    case_message = gr.Markdown(elem_classes=["dp-result-block"])
                with gr.Column(scale=2, elem_classes=["dp-case-card"]):
                    gr.HTML('<h3 class="dp-subsection">Continue a saved case</h3>')
                    case_dropdown = gr.Dropdown(label="Saved cases", choices=list_cases(), interactive=True)
                    load_btn = gr.Button("Load selected case")
                    load_result = gr.Markdown(elem_classes=["dp-result-block"])
                    gr.HTML(
                        '<div class="dp-inline-note">Case data and uploads are stored locally in SQLite and the project upload folder.</div>'
                    )
            with gr.Row(elem_classes=["dp-action-row"]):
                gr.HTML('<span class="dp-action-spacer"></span>')
                case_continue = gr.Button("Continue to evidence", variant="primary")

        with gr.Column(visible=False, elem_classes=["dp-workflow-panel"]) as evidence_step:
            gr.HTML(
                '<h2 class="dp-section-title">2. Add the supporting evidence</h2>'
                '<p class="dp-section-copy">Upload readable documents, compare a matched photo pair, enter itemised claims, and optionally compare a repair price with the sample dataset.</p>'
            )
            gr.HTML('<h3 class="dp-subsection">Rental documents</h3>')
            gr.HTML(
                '<p class="dp-upload-help">Text-based PDFs are extracted locally. Scanned PDFs are not OCR-processed. '
                'The interface reports extraction and model status after processing; selecting a file alone does not mean it was analysed.</p>'
            )
            upload_category = gr.Dropdown(
                ["Rental agreement", "Invoice / receipt", "Deduction statement", "Inspection report", "Other document"],
                value="Rental agreement", label="Document category",
            )
            gr.HTML(
                '<div class="dp-document-intro"><svg class="dp-upload-icon" viewBox="0 0 24 24" '
                'fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">'
                '<path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>'
                '</svg><span>Upload supporting documents<small>Drag files here or browse. PDFs and images are accepted; only text-based PDFs are extracted.</small></span></div>'
            )
            document_files = gr.File(
                label="Choose agreement, invoice, deduction statement, or inspection PDF",
                file_count="multiple", type="filepath", elem_classes=["dp-document-upload"],
            )
            with gr.Row():
                upload_btn = gr.Button("Upload and extract", variant="primary")
                llm_extract_btn = gr.Button("Review agreement details with hosted model")
            upload_output = gr.Markdown(elem_classes=["dp-result-block"])
            llm_extract_output = gr.Markdown(elem_classes=["dp-result-block"])
            llm_extract_btn.click(
                analyze_agreement, [active_case], [llm_extract_output],
                show_progress="full",
            )

            with gr.Accordion("Itemised deductions and evidence links", open=True):
                gr.HTML('<p class="dp-upload-help">Enter each landlord claim separately. Link the invoice and photo comparison that relate to that specific deduction.</p>')
                with gr.Row():
                    deduction_category = gr.Dropdown(
                        ["Painting / repainting", "Cleaning", "Furniture damage", "Appliance damage",
                         "Plumbing", "Electrical repairs", "Structural damage", "Other"],
                        value="Other", label="Deduction category",
                    )
                    deduction_amount = gr.Number(label="Claimed amount (₹)", value=1000, minimum=0)
                deduction_description = gr.Textbox(label="Deduction description", placeholder="e.g. Repainting the living room")
                with gr.Row():
                    invoice_dropdown = gr.Dropdown(label="Associated invoice (optional)", choices=["None"], value="None")
                    photo_dropdown = gr.Dropdown(label="Associated photo pair (optional)", choices=["None"], value="None")
                link_deduction_dropdown = gr.Dropdown(label="Deduction to update", choices=[], value=None)
                refresh_invoice_choices = gr.Button("Refresh evidence lists")
                refresh_invoice_choices.click(
                    lambda c: (
                        gr.update(choices=document_choices(c), value="None"),
                        gr.update(choices=photo_choices(c), value="None"),
                        gr.update(choices=deduction_choices(c), value=None),
                    ),
                    [active_case], [invoice_dropdown, photo_dropdown, link_deduction_dropdown],
                )
                add_deduction_btn = gr.Button("Add deduction", variant="primary")
                deduction_output = gr.Markdown(elem_classes=["dp-result-block"])
                deductions_view = gr.Markdown(elem_classes=["dp-result-block"])
                add_deduction_btn.click(
                    add_deduction,
                    [active_case, deduction_category, deduction_description, deduction_amount,
                     invoice_dropdown, photo_dropdown],
                    [deduction_output, deductions_view, link_deduction_dropdown],
                )
                gr.Markdown("**Attach evidence to an existing deduction**")
                link_evidence_btn = gr.Button("Save evidence links")
                link_evidence_output = gr.Markdown(elem_classes=["dp-result-block"])
                link_evidence_btn.click(
                    link_evidence, [active_case, link_deduction_dropdown, invoice_dropdown, photo_dropdown],
                    [link_evidence_output, deductions_view],
                )

            with gr.Accordion("Move-in and move-out photographs", open=True):
                gr.HTML('<p class="dp-upload-help">Choose photos of the same room or item. The vision model compares only the images provided and reports when it cannot analyse them.</p>')
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
                            label="Choose move-in photo", type="filepath",
                            file_types=[".png", ".jpg", ".jpeg", ".webp"],
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
                            label="Choose move-out photo", type="filepath",
                            file_types=[".png", ".jpg", ".jpeg", ".webp"],
                        )
                        photo_out_preview = gr.Image(
                            label="Selected move-out preview", interactive=False,
                            visible=False, height=180, elem_classes=["dp-photo-preview"],
                        )
                photo_btn = gr.Button("Save and compare photo pair", variant="primary")
                photo_output = gr.Markdown(elem_classes=["dp-result-block"])

            with gr.Accordion("Repair Price Intelligence (optional)", open=False):
                gr.HTML(
                    '<p class="dp-upload-help">Compare the selected city, locality, and category with the 10-metro sample ranges. '
                    'Leave locality blank for a city-wide range aggregated across its listed localities. '
                    'This is contextual only, not a finding about legal responsibility; a saved comparison never changes the audit classification.</p>'
                )
                benchmark_deduction = gr.Dropdown(
                    label="Attach to a saved deduction (optional)", choices=["None"], value="None",
                )
                with gr.Row():
                    refresh_benchmark_deductions = gr.Button("Refresh saved deductions")
                    benchmark_category = gr.Dropdown(
                        ["Cleaning", "Electrical Repairs", "Furniture / Carpentry",
                         "Painting / Repainting", "Plumbing"],
                        value="Painting / Repainting", label="Repair category",
                    )
                benchmark_amount = gr.Number(label="Claimed amount (₹)", minimum=0)
                with gr.Row():
                    benchmark_city = gr.Textbox(label="City", placeholder="e.g. Bengaluru")
                    benchmark_locality = gr.Textbox(label="Locality (optional)", placeholder="e.g. HSR Layout")
                benchmark_ask_hosted_model = gr.Checkbox(
                    label="Ask the configured hosted model for an optional explanation (uses API quota; malformed replies may retry once)",
                    value=False,
                )
                benchmark_compare_btn = gr.Button("Compare with sample rates", variant="primary")
                benchmark_output = gr.HTML()
                refresh_benchmark_deductions.click(
                    lambda active: gr.update(choices=["None"] + deduction_choices(active), value="None"),
                    [active_case], [benchmark_deduction],
                )
                benchmark_deduction.change(
                    prefill_benchmark_deduction,
                    [active_case, benchmark_deduction],
                    [benchmark_category, benchmark_amount],
                )
                benchmark_compare_btn.click(
                    run_repair_price_comparison,
                    [
                        active_case, benchmark_deduction, benchmark_category, benchmark_amount,
                        benchmark_city, benchmark_locality, benchmark_ask_hosted_model,
                    ],
                    [benchmark_output], show_progress="full",
                )

            with gr.Row(elem_classes=["dp-action-row"]):
                evidence_back = gr.Button("Back to case")
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
                show_deductions_btn = gr.Button("Refresh deduction list")
            audit_output = gr.Markdown(elem_classes=["dp-result-block", "dp-report"])
            show_deductions_btn.click(refresh_deductions, [active_case], [deductions_view])
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
            report_file = gr.File(label="Download the generated audit report PDF")
            with gr.Row(elem_classes=["dp-action-row"]):
                report_back = gr.Button("Back to findings")
                gr.HTML('<span class="dp-action-spacer"></span>')

        gr.HTML(
            '<p class="dp-footer-note">DepositProof AI is an evidence-review aid, not legal advice. '
            'Photographs show visible differences but cannot independently establish timing, cause, repair cost, or legal responsibility. '
            'Local files and database records require appropriate privacy and retention controls.</p>'
        )

    enter_app_btn.click(
        _enter_application,
        outputs=[welcome_screen, app_shell, case_step, evidence_step, review_step, report_step, stepper, workflow_message],
        show_progress="hidden",
    )
    case_continue.click(
        _continue_from_case, [active_case],
        [case_step, evidence_step, review_step, report_step, stepper, workflow_message],
    )
    evidence_continue.click(
        _continue_from_evidence,
        [active_case],
        [case_step, evidence_step, review_step, report_step, stepper, workflow_message],
    )
    evidence_back.click(lambda: (*_workflow_visibility(1)[:5], ""), outputs=[case_step, evidence_step, review_step, report_step, stepper, workflow_message])
    review_back.click(lambda: (*_workflow_visibility(2)[:5], ""), outputs=[case_step, evidence_step, review_step, report_step, stepper, workflow_message])
    review_continue.click(
        _continue_from_review,
        [audit_output],
        [case_step, evidence_step, review_step, report_step, stepper, workflow_message],
    )
    report_back.click(lambda: (*_workflow_visibility(3)[:5], ""), outputs=[case_step, evidence_step, review_step, report_step, stepper, workflow_message])
    nav_case.click(lambda: (*_workflow_visibility(1)[:5], ""), outputs=[case_step, evidence_step, review_step, report_step, stepper, workflow_message])
    nav_evidence.click(lambda: (*_workflow_visibility(2)[:5], ""), outputs=[case_step, evidence_step, review_step, report_step, stepper, workflow_message])
    nav_findings.click(lambda: (*_workflow_visibility(3)[:5], ""), outputs=[case_step, evidence_step, review_step, report_step, stepper, workflow_message])
    nav_report.click(lambda: (*_workflow_visibility(4)[:5], ""), outputs=[case_step, evidence_step, review_step, report_step, stepper, workflow_message])
    mobile_nav.change(
        _navigate_to_step,
        [mobile_nav],
        [case_step, evidence_step, review_step, report_step, stepper, workflow_message],
    )

    create_btn.click(
        create_case, [case_name, deposit, claimed, proposed, move_in, move_out],
        [active_case, case_message, case_dropdown],
    )
    load_btn.click(
        load_selected_case, [case_dropdown],
        [active_case, deductions_view, invoice_dropdown, photo_dropdown,
         link_deduction_dropdown, load_result],
    )
    upload_btn.click(
        upload_documents, [active_case, document_files, upload_category],
        [upload_output, invoice_dropdown], show_progress="full",
    )
    photo_btn.click(
        upload_photo_pair, [active_case, photo_item, photo_in, photo_out],
        [photo_output, photo_dropdown], show_progress="full",
    )
    photo_in.change(
        lambda path: gr.update(value=path, visible=bool(path)),
        [photo_in], [photo_in_preview], show_progress="hidden",
    )
    photo_out.change(
        lambda path: gr.update(value=path, visible=bool(path)),
        [photo_out], [photo_out_preview], show_progress="hidden",
    )
    audit_btn.click(
        _run_audit_for_workflow, [active_case],
        [audit_output, report_output, report_file, deductions_view],
        show_progress="full",
    ).then(
        lambda: (*_workflow_visibility(4)[:5], ""),
        outputs=[case_step, evidence_step, review_step, report_step, stepper, workflow_message],
    )
    gr.HTML(
        '<noscript><p>JavaScript is required to navigate this interactive audit application.</p></noscript>'
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=DESIGN_CSS,
    )