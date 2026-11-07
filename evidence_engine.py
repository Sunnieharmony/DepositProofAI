"""
Evidence engine for DepositProof AI.

Everything in this file is deterministic Python (plus an optional small local embedding
model for clause retrieval). Language/vision models live in llm_integration.py and can
only *add explanation* on top of what is decided here; they never override a confirmed
contradiction found by these rules.

Public functions kept from the original starter:
    terms_for, find_relevant_clauses, verify_quote, extract_invoice_fields,
    classify_deduction, compare_images
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from datetime import date
from typing import Any

SUPPORTED = "Supported"
INCONSISTENT = "Inconsistent"
INSUFFICIENT = "Insufficiently Documented"
CLASSIFICATIONS = (SUPPORTED, INCONSISTENT, INSUFFICIENT)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine-similarity cut-offs. These are heuristics for a hackathon demo, NOT calibrated
# probabilities. Tune them with the environment variables if retrieval is too strict/loose.
RELEVANT_MIN = float(os.getenv("CLAUSE_RELEVANT_MIN", "0.40"))
POTENTIAL_MIN = float(os.getenv("CLAUSE_POTENTIAL_MIN", "0.28"))

# --------------------------------------------------------------------------
# Text normalisation and quote verification
# --------------------------------------------------------------------------
_PUNCT_MAP = {ord(c): "'" for c in "‘’‚‛′`´"}
_PUNCT_MAP.update({ord(c): '"' for c in "“”„‟″"})
_PUNCT_MAP.update({ord(c): "-" for c in "‐‑‒–—―−"})


def normalize_text(s: str) -> str:
    """Harmless-format normalisation only: unicode form, smart quotes/dashes, whitespace, case."""
    s = unicodedata.normalize("NFKC", s or "").translate(_PUNCT_MAP)
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_quote(quote: str, source_text: str) -> bool:
    """
    True only if `quote` appears verbatim in `source_text` after normalising whitespace,
    case, smart quotes and dashes. A quote written with "..." must have every fragment
    present. Very short quotes (< 8 characters) are never accepted.
    """
    if not quote or not source_text:
        return False
    src = normalize_text(source_text)
    q = normalize_text(quote).strip(" \"'")
    fragments = [f.strip(" \"'") for f in re.split(r"\.\.\.|…", q) if f.strip(" \"'")]
    if not fragments:
        return False
    return all(len(f) >= 8 and f in src for f in fragments)


# --------------------------------------------------------------------------
# Vocabulary: service groups used for matching claims to invoices / inspections
# --------------------------------------------------------------------------
SERVICE_GROUPS: dict[str, list[str]] = {
    "painting": ["paint", "repaint", "whitewash", "distemper", "emulsion", "primer", "putty"],
    "cleaning": ["clean", "sanit", "scrub", "mop", "housekeep", "vacuum", "dusting", "disinfect"],
    "furniture": ["table", "chair", "sofa", "desk", "bed", "wardrobe", "cupboard", "shelf", "shelves",
                  "furniture", "cabinet", "couch", "mattress", "carpent", "stool", "drawer"],
    "appliance": ["refrigerator", "fridge", "washing", "oven", "microwave", "geyser", "conditioner",
                  "dishwasher", "appliance", "chimney", "stove", "cooktop", "purifier", "heater"],
    "plumbing": ["plumb", "pipe", "tap", "faucet", "leak", "drain", "flush", "sink", "toilet",
                 "cistern", "commode", "seepage"],
    "electrical": ["electric", "wiring", "switch", "socket", "bulb", "mcb", "circuit", "lighting"],
    "structural": ["ceiling", "crack", "structural", "plaster", "masonry", "tile", "floor", "flooring",
                   "door", "window", "glass", "lock"],
}
CATEGORY_TO_GROUP = {
    "painting": "painting", "repainting": "painting", "cleaning": "cleaning",
    "furniture": "furniture", "appliance": "appliance", "plumbing": "plumbing",
    "electrical": "electrical", "structural": "structural",
}
# Kept for backwards compatibility with the original starter.
KEYWORDS = {
    "painting": ["paint", "painting", "repaint", "wall", "whitewash"],
    "cleaning": ["clean", "cleaning", "sanit", "deep clean"],
    "furniture": ["table", "chair", "sofa", "furniture", "scratch", "wood"],
    "appliance": ["appliance", "refrigerator", "fridge", "washing machine", "oven", "repair"],
    "plumbing": ["plumb", "pipe", "tap", "leak", "drain"],
    "electrical": ["electric", "wiring", "switch", "socket", "light"],
    "structural": ["structural", "wall", "ceiling", "crack", "damage"],
    "deposit": ["deposit", "refund", "deduct", "wear and tear", "maintenance", "tenant"],
}
_STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was were
with will shall may not any all per into than then them they their there these those which who whom
tenant landlord lessee lessor claim claimed charge charges deduction deductions amount rent rental""".split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", normalize_text(text))


def _token_matches(token: str, stem: str) -> bool:
    if len(stem) >= 5:
        return token.startswith(stem)
    return token in (stem, stem + "s", stem + "es", stem + "ed", stem + "ing")


def groups_for(text: str) -> set[str]:
    """Which service groups (painting, cleaning, furniture...) does this text talk about?"""
    toks = _tokens(text)
    return {g for g, stems in SERVICE_GROUPS.items()
            if any(_token_matches(t, s) for t in toks for s in stems)}


def deduction_groups(description: str, category: str | None = None) -> set[str]:
    groups = groups_for(description or "")
    cat = (category or "").lower()
    for key, group in CATEGORY_TO_GROUP.items():
        if key in cat:
            groups.add(group)
    return groups


def terms_for(description: str) -> set[str]:
    """Original helper, kept for compatibility (used by the keyword fallback)."""
    words = {w for w in re.findall(r"[a-zA-Z]{3,}", (description or "").lower()) if w not in _STOPWORDS}
    expanded = set(words)
    for _category, keywords in KEYWORDS.items():
        if any(k in (description or "").lower() for k in keywords):
            expanded.update(keywords)
    return expanded


# --------------------------------------------------------------------------
# Passage splitting
# --------------------------------------------------------------------------
def split_passages(text: str, pages: dict | None = None, max_chars: int = 700, min_chars: int = 40) -> list[dict]:
    """
    Split an agreement into clause-sized passages. Passage text is always copied from the
    source (never rewritten), so verify_quote() will confirm it. Page numbers are only
    reported when the passage can be found on a page; otherwise "unknown".
    """
    if not text or not text.strip():
        return []
    # Split at blank lines and at numbered-clause starts such as "4. Painting".
    blocks = [b.strip() for b in re.split(r"\n\s*\n|\n(?=\s*\d+(?:\.\d+)*[.)]\s+\S)", text) if b.strip()]
    pieces: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            pieces.append(block)
            continue
        sentences = re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9(])", block)
        current = ""
        for sent in sentences:
            if current and len(current) + len(sent) + 1 > max_chars:
                pieces.append(current)
                current = sent
            else:
                current = f"{current} {sent}".strip()
        if current:
            pieces.append(current)
    # Merge headings / very short fragments into the following piece.
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        piece = f"{carry}\n{piece}".strip() if carry else piece
        if len(piece) < min_chars:
            carry = piece
            continue
        merged.append(piece)
        carry = ""
    if carry:
        merged.append(carry)

    norm_pages = {k: normalize_text(v) for k, v in (pages or {}).items() if k != "unknown"}
    passages = []
    for idx, piece in enumerate(merged):
        norm = normalize_text(piece)
        page = "unknown"
        for probe_len in (80, 40):
            probe = norm[:probe_len]
            hit = next((k for k, v in norm_pages.items() if probe and probe in v), None)
            if hit:
                page = hit
                break
        passages.append({"idx": idx, "text": piece, "page": page})
    return passages


# --------------------------------------------------------------------------
# Embedding model (lazy, CPU, loaded once)
# --------------------------------------------------------------------------
_MODEL: Any = None
_MODEL_ERR: str | None = None
_PASSAGE_CACHE: dict[str, tuple[list[dict], list[list[float]]]] = {}


def _embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _load_embedder():
    """Return the SentenceTransformer, or None (reason saved in _MODEL_ERR)."""
    global _MODEL, _MODEL_ERR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERR:
        return None
    if os.getenv("DISABLE_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes"):
        _MODEL_ERR = "disabled by DISABLE_EMBEDDINGS"
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_embedding_model_name(), device="cpu")
    except Exception as exc:  # ImportError, offline download failure, etc.
        _MODEL_ERR = f"{type(exc).__name__}: {str(exc)[:160]}"
        _MODEL = None
    return _MODEL


def embedding_status() -> dict:
    """Cheap status check (does not load the model)."""
    return {"model": _embedding_model_name(), "loaded": _MODEL is not None, "error": _MODEL_ERR}


def _encode(model, texts: list[str]) -> list[list[float]]:
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [[float(x) for x in row] for row in vecs]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Clause retrieval
# --------------------------------------------------------------------------
def _keyword_scores(query: str, passages: list[dict], category: str | None = None) -> list[float]:
    """Fallback scoring: number of distinct query concepts found in a passage."""
    q_tokens = [t for t in _tokens(query) if len(t) >= 4 and t not in _STOPWORDS]
    q_groups = deduction_groups(query, category)
    scores = []
    for p in passages:
        toks = _tokens(p["text"])
        hit_terms = {q for q in q_tokens if any(_token_matches(t, q[:6] if len(q) > 6 else q) for t in toks)}
        hit_groups = {g for g in q_groups if any(_token_matches(t, s) for t in toks for s in SERVICE_GROUPS[g])}
        scores.append(float(len(hit_terms) + len(hit_groups)))
    return scores


def find_relevant_clauses(description: str, text: str, filename: str, pages: dict,
                          top_k: int = 5, category: str | None = None) -> list[dict]:
    """
    Rank agreement passages for a deduction.

    Uses sentence-transformers (all-MiniLM-L6-v2 by default) when it can be loaded.
    Otherwise falls back to keyword matching, and every returned item says so in
    `retrieval_method`, so the app never claims semantic search it did not do.

    Each item: filename, page, quote (exact source text), relevance_score,
    relevance ("relevant" | "potentially_relevant"), retrieval_method, quote_verified, passage_id.
    An empty list means "no relevant clause was identified in the extracted text".
    """
    passages = split_passages(text, pages)
    if not passages:
        return []
    query = (description or "").strip()
    if not query:
        return []

    model = _load_embedder()
    scored: list[tuple[float, dict, str]] = []
    method = "keyword_fallback"
    if model is not None:
        try:
            key = hashlib.sha1(f"{filename}|{_embedding_model_name()}|{text}".encode("utf-8", "ignore")).hexdigest()
            if key not in _PASSAGE_CACHE:
                _PASSAGE_CACHE[key] = (passages, _encode(model, [p["text"] for p in passages[:500]]))
            cached_passages, vectors = _PASSAGE_CACHE[key]
            qv = _encode(model, [f"{query} (rental agreement, security deposit)"])[0]
            sims = [_dot(qv, v) for v in vectors]
            method = f"semantic:{_embedding_model_name()}"
            for p, s in zip(cached_passages, sims):
                if s >= POTENTIAL_MIN:
                    scored.append((s, p, "relevant" if s >= RELEVANT_MIN else "potentially_relevant"))
        except Exception as exc:  # fall back, but remember why
            global _MODEL_ERR
            _MODEL_ERR = f"encode failed: {type(exc).__name__}: {str(exc)[:120]}"
            scored, method = [], "keyword_fallback"
    if method == "keyword_fallback":
        for p, s in zip(passages, _keyword_scores(query, passages, category)):
            if s >= 1:
                scored.append((s, p, "relevant" if s >= 2 else "potentially_relevant"))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, p, label in scored[:top_k]:
        quote = p["text"][:1200]
        out.append({
            "filename": filename,
            "page": p["page"],
            "quote": quote,
            "relevance_score": round(score, 3),
            "relevance": label,
            "retrieval_method": method,
            "quote_verified": verify_quote(quote, text),
            "passage_id": f"{filename}#{p['idx']}",
        })
    return out


AGREEMENT_TOPICS = [
    ("deposit_refund", "security deposit amount and refund of the deposit after move-out"),
    ("deduction_conditions", "conditions under which the landlord may deduct amounts from the security deposit"),
    ("cleaning", "tenant cleaning obligations at move-out"),
    ("painting", "painting or repainting of walls at the end of the tenancy"),
    ("repairs_and_damage", "repairs, damage, breakage and who bears the repair cost"),
    ("wear_and_tear", "normal wear and tear exceptions"),
    ("notice_and_inspection", "notice, inspection or documentation required before deductions"),
]


def retrieve_topic_passages(text: str, pages: dict, filename: str, per_topic: int = 2) -> dict:
    """Non-LLM retrieval of the top passages for each standard agreement topic."""
    result = {}
    for key, query in AGREEMENT_TOPICS:
        result[key] = find_relevant_clauses(query, text, filename, pages, top_k=per_topic)
    return result


# --------------------------------------------------------------------------
# Invoice / inspection / photo checks (deterministic)
# --------------------------------------------------------------------------
def _num(x) -> float | None:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def money_text(value: Any, currency: str = "INR") -> str:
    n = _num(value)
    if n is None:
        return "unknown"
    return f"{currency} {n:,.2f}".replace(".00", "")


def extract_invoice_fields(text: str) -> dict:
    from document_processing import extract_invoice_fields as _extract
    return _extract(text)


def _iso(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def check_invoice(payload: dict) -> dict:
    """
    Compare the associated invoice with the claim. Returns:
      status: none | unreadable | related | unrelated | inconclusive
      amount_status: match | mismatch | unknown | not_checked
      plus human-readable `notes`, `contradictions` and `warnings`.
    Semantic similarity is never treated as proof: matching uses normalised service groups.
    """
    currency = payload.get("currency") or "INR"
    claimed = _num(payload.get("amount")) or 0.0
    invoice = payload.get("invoice") or {}
    out = {"status": "none", "amount_status": "not_checked", "notes": [], "contradictions": [], "warnings": []}
    if not invoice:
        return out

    text = (invoice.get("extracted_text") or "")
    name = invoice.get("filename", "the selected invoice")
    if not text.strip():
        out["status"] = "unreadable"
        out["notes"].append(f"{name} has no extractable text (scanned or unreadable), so it could not be checked.")
        return out

    # --- relation to the claimed work ---
    ded_groups = deduction_groups(payload.get("description", ""), payload.get("category"))
    inv_groups = groups_for(invoice.get("service_description") or "") or groups_for(text)
    if ded_groups and inv_groups:
        if ded_groups & inv_groups:
            out["status"] = "related"
            out["notes"].append(f"Invoice work ({', '.join(sorted(inv_groups))}) overlaps the claimed work ({', '.join(sorted(ded_groups))}).")
        else:
            out["status"] = "unrelated"
            out["contradictions"].append(
                f"Invoice {name} describes {', '.join(sorted(inv_groups))} work, but the claim is for "
                f"{', '.join(sorted(ded_groups))}; the invoice appears to cover unrelated work.")
    else:
        out["status"] = "inconclusive"
        out["warnings"].append(f"Could not tell from the text whether {name} relates to the claimed work; manual check needed.")

    # --- amount (pure Python) ---
    total = _num(invoice.get("total_amount"))
    candidates = [c for c in (_num(x) for x in (invoice.get("amount_candidates") or [])) if c is not None]
    if total is not None:
        if abs(total - claimed) <= 0.5:
            out["amount_status"] = "match"
            out["notes"].append(f"Invoice total {money_text(total, currency)} equals the claimed {money_text(claimed, currency)}.")
        else:
            out["amount_status"] = "mismatch"
            out["contradictions"].append(
                f"Invoice total {money_text(total, currency)} differs from the claimed amount {money_text(claimed, currency)} "
                f"(difference {money_text(abs(total - claimed), currency)}).")
    elif candidates:
        if any(abs(c - claimed) <= 0.5 for c in candidates):
            out["amount_status"] = "match"
            out["notes"].append(f"An amount on the invoice equals the claimed {money_text(claimed, currency)} (the invoice total itself was not identified).")
        else:
            out["amount_status"] = "unknown"
            out["warnings"].append("No invoice total could be identified and none of the extracted amounts equals the claimed amount; check the invoice manually.")
    else:
        out["amount_status"] = "unknown"
        out["warnings"].append("No amount could be extracted from the invoice.")

    # --- timeline ---
    inv_date = _iso(invoice.get("invoice_date"))
    move_in = _iso((payload.get("case_dates") or {}).get("move_in"))
    if inv_date and move_in and inv_date < move_in:
        out["warnings"].append(f"Invoice date {inv_date.isoformat()} is before the move-in date {move_in.isoformat()}; check that it concerns this tenancy.")
    return out


_MOVE_OUT_RE = re.compile(r"move[\s_-]?out|\bexit\b|\bfinal\b|vacat|check[\s_-]?out|departure", re.I)
_MOVE_IN_RE = re.compile(r"move[\s_-]?in|\bentry\b|check[\s_-]?in|hand[\s_-]?over|\bcommencement\b", re.I)
_NOT_CLEAN_RE = re.compile(r"\bnot\s+(?:been\s+)?clean|\bunclean|needs?\s+cleaning|requires?\s+cleaning|\bdirty|\bgreas", re.I)
_CONTRA_RE = re.compile(
    r"good\s+condition|excellent\s+condition|well[\s-]maintained|\bno\s+(?:visible\s+)?(?:\w+\s+){0,2}"
    r"(?:damage|stain|scratch|mark|crack|dent|defect|issue)|undamaged|\bintact\b|\bspotless|\bsatisfactory|"
    r"\bas\s+new\b|\bclean\b|properly\s+cleaned|in\s+order", re.I)
_SUPPORT_RE = re.compile(
    r"damag|stain|scratch|broken|crack|peel|chipped|dent|burn|torn|missing|mou?ld|discolou?r|"
    r"\bmarks?\b|\bholes?\b|smell|odou?r|worn\s+out|not\s+working", re.I)


def inspection_context(filename: str, text: str) -> str:
    probe = f"{filename} {(text or '')[:800]}"
    out_hit, in_hit = bool(_MOVE_OUT_RE.search(probe)), bool(_MOVE_IN_RE.search(probe))
    if out_hit and not in_hit:
        return "move_out"
    if in_hit and not out_hit:
        return "move_in"
    return "unknown"


def analyze_inspections(payload: dict) -> list[dict]:
    """Find sentences in inspection reports that talk about the claimed item and label their polarity."""
    ded_groups = deduction_groups(payload.get("description", ""), payload.get("category"))
    desc_terms = [t for t in _tokens(payload.get("description", "")) if len(t) >= 4 and t not in _STOPWORDS
                  and t not in ("repair", "repairs", "damage", "damaged", "charges", "work")]
    findings = []
    for doc in payload.get("inspection_documents") or []:
        text = doc.get("text") or ""
        context = inspection_context(doc.get("filename", ""), text)
        for sent in [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]:
            toks = _tokens(sent)
            relevant = (ded_groups & groups_for(sent)) or any(_token_matches(t, d) for t in toks for d in desc_terms)
            if not relevant:
                continue
            if _NOT_CLEAN_RE.search(sent):
                polarity = "damage_or_issue_recorded"
            elif _CONTRA_RE.search(sent):
                polarity = "good_condition_recorded"
            elif _SUPPORT_RE.search(sent):
                polarity = "damage_or_issue_recorded"
            else:
                polarity = "neutral"
            findings.append({"filename": doc.get("filename", "inspection report"), "context": context,
                             "sentence": sent[:300], "polarity": polarity})
            if len(findings) >= 12:
                return findings
    return findings


def photo_status(obs: dict | None) -> str:
    """none | not_analyzed | inconclusive | difference_observed | no_difference_observed"""
    if not obs:
        return "none"
    if obs.get("mode") != "llm_analyzed":
        return "not_analyzed"
    if obs.get("comparable") == "no":
        return "inconclusive"
    return "difference_observed" if obs.get("visible_differences") else "no_difference_observed"


def clause_status(payload: dict, clauses: list[dict]) -> str:
    """found | potential | none_identified | undeterminable (the four states in the spec)."""
    if payload.get("agreement_available") is False or payload.get("agreement_problem"):
        return "undeterminable"
    verified = [c for c in clauses if c.get("quote_verified")]
    if any(c.get("relevance", "relevant") == "relevant" for c in verified):
        return "found"
    if verified:
        return "potential"
    return "none_identified"


# --------------------------------------------------------------------------
# Classification rules
# --------------------------------------------------------------------------
def evaluate_locality_benchmark(payload: dict) -> dict | None:
    """Compare a deduction amount with a matching selected-locality sample range."""
    locality = payload.get("locality_benchmarks")
    if not locality:
        return None

    common = {
        "city": locality.get("city"),
        "locality": locality.get("locality"),
        "source_dataset": locality.get("source_dataset"),
        "sample_data_warning": locality.get("sample_data_warning"),
    }
    deduction_service_groups = deduction_groups(
        payload.get("description", ""), payload.get("category")
    )
    matching_rates = [
        rate for rate in locality.get("rates", [])
        if deduction_service_groups & groups_for(str(rate.get("category") or ""))
    ]
    matched_groups = {
        group
        for rate in matching_rates
        for group in deduction_service_groups & groups_for(str(rate.get("category") or ""))
    }
    if not matching_rates or len(matched_groups) != 1:
        return {**common, "status": "no_match"}

    lower = min(float(rate["minimum"]) for rate in matching_rates)
    upper = max(float(rate["maximum"]) for rate in matching_rates)
    amount = _num(payload.get("amount"))
    if amount is None:
        return {**common, "status": "no_amount"}

    if amount < lower:
        position = "below"
        difference = lower - amount
        difference_percent = difference / lower * 100 if lower else None
    elif amount > upper:
        position = "above"
        difference = amount - upper
        difference_percent = difference / upper * 100 if upper else None
    else:
        position, difference, difference_percent = "within", 0.0, 0.0

    category = " / ".join(sorted({str(rate["category"]) for rate in matching_rates}))
    return {
        **common,
        "status": "matched",
        "category": category,
        "match_description": "Selected locality category sample range",
        "record_count": len(matching_rates),
        "lower": lower,
        "upper": upper,
        "claimed_amount": amount,
        "position": position,
        "difference": difference,
        "difference_percent": difference_percent,
        "matched_records": [
            {
                "city": locality.get("city"),
                "locality": locality.get("locality"),
                "category": str(rate["category"]),
                "minimum": float(rate["minimum"]),
                "maximum": float(rate["maximum"]),
            }
            for rate in matching_rates
        ],
    }


def classify_deduction(payload: dict) -> dict:
    """
    Rules-based classification (no LLM). Transparent and conservative.

    INCONSISTENT  only on a concrete contradiction: invoice total != claim, invoice for unrelated
                  work, or an inspection record that says the opposite. Missing evidence never
                  makes a claim Inconsistent.
    SUPPORTED     only if ALL hold: a verified relevant agreement clause, a readable related
                  invoice whose amount equals the claim, condition evidence (a comparable photo
                  pair showing a difference, or a move-out inspection note recording the issue),
                  and no contradictions or open warnings.
    INSUFFICIENT  everything else; the missing items are listed.
    """
    currency = payload.get("currency") or "INR"
    clauses = payload.get("clauses") or []
    verified = [c for c in clauses if c.get("quote_verified")]
    c_status = clause_status(payload, clauses)
    inv = check_invoice(payload)
    inspections = analyze_inspections(payload)
    p_status = photo_status(payload.get("photo_observation"))
    photo = payload.get("photo_observation") or {}

    supporting: list[str] = []
    missing: list[str] = []
    contradictions: list[str] = list(inv["contradictions"])
    warnings: list[str] = list(inv["warnings"])

    # Agreement
    if c_status == "found":
        best = next(c for c in verified if c.get("relevance", "relevant") == "relevant")
        supporting.append(f"Agreement: relevant clause found in {best['filename']} (page {best.get('page', 'unknown')}).")
    elif c_status == "potential":
        missing.append("Only potentially relevant agreement passages were found; none clearly covers this charge.")
    elif c_status == "undeterminable":
        missing.append(payload.get("agreement_problem") or "No readable rental agreement was uploaded, so contract terms could not be checked.")
    else:
        missing.append("No relevant clause was identified in the available extracted agreement text (this does not prove the agreement has none).")

    # Invoice
    if inv["status"] == "none":
        missing.append("No invoice or receipt was associated with this deduction.")
    elif inv["status"] == "unreadable":
        missing.append(inv["notes"][0])
    elif inv["status"] == "related" and inv["amount_status"] == "match":
        supporting.append("Invoice: related work and the amount equals the claim.")
    elif inv["status"] == "related" and inv["amount_status"] == "unknown":
        missing.append("The invoice appears related but its amount could not be confirmed against the claim.")

    # Inspection records
    move_out_issue = [f for f in inspections if f["context"] == "move_out" and f["polarity"] == "damage_or_issue_recorded"]
    move_out_good = [f for f in inspections if f["context"] == "move_out" and f["polarity"] == "good_condition_recorded"]
    move_in_issue = [f for f in inspections if f["context"] == "move_in" and f["polarity"] == "damage_or_issue_recorded"]
    unknown_timing = [f for f in inspections if f["context"] == "unknown" and f["polarity"] != "neutral"]
    if move_out_good and not move_out_issue:
        f = move_out_good[0]
        contradictions.append(f"Move-out inspection ({f['filename']}) records good condition for this item: \"{f['sentence']}\"")
    if move_in_issue:
        f = move_in_issue[0]
        contradictions.append(f"Move-in inspection ({f['filename']}) already records a similar issue: \"{f['sentence']}\"")
    if move_out_issue and not move_out_good:
        supporting.append(f"Inspection: move-out report ({move_out_issue[0]['filename']}) records an issue for this item.")
    if unknown_timing:
        warnings.append(f"An inspection note ({unknown_timing[0]['filename']}) mentions this item but it is unclear whether it is a move-in or move-out record; review it manually.")

    # Photos
    photo_supports = False
    if p_status == "none":
        missing.append("No move-in/move-out photo pair is linked to this deduction.")
    elif p_status == "not_analyzed":
        missing.append("A photo pair is linked but was not analysed (vision model unavailable or failed).")
    elif p_status == "inconclusive":
        missing.append("The photo pair could not be reliably compared (" + str(photo.get("limitations", "images not comparable"))[:160] + ").")
    elif p_status == "no_difference_observed":
        warnings.append("The comparable photo pair shows no visible difference; this does not settle the claim but leaves it unsupported by photos.")
    else:  # difference_observed
        photo_supports = True
        supporting.append("Photos: a visible difference between move-in and move-out images was reported (a visible change is not proof of damage or of who caused it).")
        if photo.get("comparable") == "partially":
            warnings.append("The photos are only partially comparable.")
    condition_evidence = photo_supports or bool(move_out_issue and not move_out_good)

    # Decision
    if contradictions:
        classification = INCONSISTENT
        explanation = ("A material mismatch was found: " + " ".join(contradictions) +
                       " This flags a discrepancy to clarify with the landlord; it is not a finding that the charge is invalid.")
    elif (c_status == "found" and inv["status"] == "related" and inv["amount_status"] == "match"
          and condition_evidence and not warnings):
        classification = SUPPORTED
        explanation = ("The documents provided are consistent with this claim: " + " ".join(supporting) +
                       " This is an evidence classification only; it does not establish that the tenant is responsible, "
                       "that the charge is legally enforceable, or what amount may lawfully be deducted.")
    else:
        classification = INSUFFICIENT
        parts = []
        if missing:
            parts.append("Missing or inconclusive: " + " ".join(missing))
        if warnings:
            parts.append("Open points: " + " ".join(warnings))
        explanation = (" ".join(parts) + " Insufficient documentation does not mean the claim is false; "
                       "it means the available evidence cannot confirm it.")

    return {
        "classification": classification,
        "explanation": explanation,
        "evidence_reviewed": (
            f"{len(verified)} verified clause(s) [{c_status}]; invoice: {inv['status']} (amount: {inv['amount_status']}); "
            f"photos: {p_status}; {len(inspections)} relevant inspection sentence(s)."),
        "missing_or_conflicting_evidence": "; ".join(
            ([f"CONFLICT: {c}" for c in contradictions]) + missing + [f"REVIEW: {w}" for w in warnings]
        ) or "No gaps detected by the rule checks.",
        "limitations": ("The rules engine determines the evidence classification by checking evidence of damage, repair, "
                        "cost and contract terms. In LLM-assisted findings, the model may add an explanation but does not "
                        "replace the classification rules or decide responsibility or legal liability."),
        "mode": "rules_fallback_not_llm",
        "supporting_evidence": supporting,
        "missing_evidence": missing,
        "contradictions": contradictions,
        "warnings": warnings,
        "clause_status": c_status,
        "invoice_status": inv["status"],
        "amount_status": inv["amount_status"],
        "photo_status": p_status,
        "inspection_findings": inspections,
        "locality_benchmark": evaluate_locality_benchmark(payload),
    }


def compare_images(image_a: str, image_b: str) -> dict:
    """Compatibility wrapper: runs the hosted vision model if configured, never fakes a result."""
    from llm_integration import analyze_image_pair_with_llm, get_last_error, llm_is_configured
    if llm_is_configured():
        result = analyze_image_pair_with_llm(image_a, image_b, "item")
        if result:
            return result
        reason = get_last_error() or "the vision model did not return a valid result"
    else:
        reason = "no Hugging Face token is configured"
    return {
        "observation": f"No image comparison was performed: {reason}.",
        "limitations": "No automated visual comparison took place.",
        "mode": "not_analyzed",
    }