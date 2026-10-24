from __future__ import annotations

import re
from datetime import date
from pathlib import Path


# --------------------------------------------------------------------------
# PDF text extraction (PyMuPDF preferred, with pypdf fallback; no OCR)
# --------------------------------------------------------------------------
def extract_pdf(path: str) -> tuple[str, dict[str, str], str]:
    """
    Extract PDF text with PyMuPDF, falling back to pypdf if its backend cannot load.
    Returns (full_text, {page_number_string: page_text}, status).

    The status string is shown to the user, so it says plainly why a file
    could not be read instead of pretending it was processed.
    """
    path_obj = Path(path)
    if path_obj.suffix.lower() != ".pdf":
        return "", {}, "unsupported_for_text_extraction"

    pymupdf_error = None
    try:
        import fitz  # PyMuPDF
        with fitz.open(path) as doc:
            if doc.needs_pass:
                return "", {}, "encrypted_pdf; remove the password and upload again"
            page_text: dict[str, str] = {}
            for i, page in enumerate(doc, start=1):
                page_text[str(i)] = page.get_text("text") or ""
    except Exception as exc:  # corrupt / not really a PDF
        pymupdf_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    else:
        return _pdf_text_result(page_text, "pymupdf")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        return "", {}, (
            "pdf_parser_unavailable: PyMuPDF could not load "
            f"({pymupdf_error}); install pypdf with 'pip install pypdf'"
        )

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            return "", {}, "encrypted_pdf; remove the password and upload again"
        page_text = {
            str(i): page.extract_text() or ""
            for i, page in enumerate(reader.pages, start=1)
        }
    except Exception as exc:
        return "", {}, (
            f"unreadable_pdf:{type(exc).__name__}; "
            f"PyMuPDF failed ({pymupdf_error}) and pypdf could not read the PDF"
        )

    return _pdf_text_result(page_text, "pypdf")


def _pdf_text_result(page_text: dict[str, str], parser: str) -> tuple[str, dict[str, str], str]:
    full_text = "\n\n".join(page_text.values()).strip()
    if not full_text:
        return "", page_text, (
            f"no_text_found; {parser} found no extractable text. "
            "This may be a scanned PDF; OCR is not implemented."
        )
    empty_pages = [p for p, t in page_text.items() if not t.strip()]
    if empty_pages:
        shown = ", ".join(empty_pages[:8])
        return full_text, page_text, f"extracted_with_{parser}; partial: no text on page(s) {shown}"
    return full_text, page_text, f"extracted_with_{parser}"


# --------------------------------------------------------------------------
# Invoice extraction (rule-based, clearly labelled as such)
# --------------------------------------------------------------------------
_NUM = r"([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)"
_NUM_RE = re.compile(_NUM)
_EXCLUDE_LABELS = re.compile(
    r"sub\s*-?\s*total|\btax\b|\bgst\b|cgst|sgst|igst|\bvat\b|discount|advance|\bpaid\b|round(?:ing)?[\s-]*off|\bqty\b|quantity|\brate\b",
    re.I,
)
_TOTAL_PRIORITY = [
    re.compile(r"grand\s*total|total\s*payable|amount\s*(?:due|payable)|net\s*(?:amount|payable)|balance\s*due", re.I),
    re.compile(r"total\s*amount|invoice\s*total|bill\s*total", re.I),
    re.compile(r"\btotal\b", re.I),
]
_DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{2,4}\b",
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.I)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_date(value: str) -> str | None:
    """Parse a date to ISO (YYYY-MM-DD). Numeric dates are read DAY-FIRST (Indian format)."""
    s = (value or "").strip().replace(",", "")
    try:
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s)
        if m:
            y = int(m[3])
            y = y + 2000 if y < 100 else y
            return date(y, int(m[2]), int(m[1])).isoformat()
        m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{2,4})", s)
        if m and m[2][:3].lower() in _MONTHS:
            y = int(m[3])
            y = y + 2000 if y < 100 else y
            return date(y, _MONTHS[m[2][:3].lower()], int(m[1])).isoformat()
    except ValueError:
        return None
    return None


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def extract_invoice_fields(text: str) -> dict:
    """
    Rule-based invoice extraction. NOT an AI model; every value is a candidate that
    the user should confirm against the original document.

    Keeps the original keys (amount_candidates, date_candidates, note) and adds:
    total_amount, vendor, invoice_date, service_description, property_reference.
    """
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # ---- amounts (original behaviour preserved) ----
    amount_candidates = [
        x.replace(",", "") for x in re.findall(
            r"(?:₹|INR|Rs\.?|Total|Amount|Grand Total|Net Amount)\s*[:\-]?\s*([0-9,]+(?:\.[0-9]{1,2})?)",
            text, flags=re.IGNORECASE)
    ]

    # ---- the invoice total: only lines that are labelled as a total, never subtotal/tax ----
    total_amount, total_source = None, None
    for pattern in _TOTAL_PRIORITY:
        found = None
        for ln in lines:
            if _EXCLUDE_LABELS.search(ln) or not pattern.search(ln):
                continue
            nums = [_to_float(n) for n in _NUM_RE.findall(ln)]
            nums = [n for n in nums if n is not None]
            if nums:
                found = (nums[-1], ln)  # last labelled line of this priority wins
        if found:
            total_amount, total_source = found
            break

    # ---- dates ----
    date_candidates = [m.group(0) for m in _DATE_RE.finditer(text)]
    invoice_date = None
    for ln in lines:
        if re.search(r"invoice\s*date|date\s*of\s*invoice|bill\s*date|^date\b|\bdate\s*[:\-]", ln, re.I):
            m = _DATE_RE.search(ln)
            if m:
                invoice_date = parse_date(m.group(0))
                if invoice_date:
                    break
    if not invoice_date:
        for cand in date_candidates:
            invoice_date = parse_date(cand)
            if invoice_date:
                break

    # ---- vendor: first plain line that is not a title/date/number line ----
    vendor = None
    skip = re.compile(r"^(tax\s*)?(invoice|receipt|bill|cash\s*memo|estimate|quotation)\b|^date\b|^no\.?\b|^#|^\d", re.I)
    for ln in lines[:8]:
        m = re.match(r"(?:from|vendor|billed\s*by|issued\s*by|sold\s*by)\s*[:\-]\s*(.+)", ln, re.I)
        if m:
            vendor = m.group(1).strip()[:80]
            break
    if not vendor:
        for ln in lines[:6]:
            if len(ln) <= 80 and re.search(r"[A-Za-z]{3}", ln) and not skip.search(ln):
                vendor = ln
                break

    # ---- service description: lines under a Description/Particulars header, until totals ----
    description_lines: list[str] = []
    capture = False
    for ln in lines:
        if re.search(r"^(description|particulars|item|work\s*done|service|details)\b", ln, re.I):
            capture = True
            rest = re.sub(r"^(description|particulars|item|work\s*done|service|details)\s*[:\-]?\s*", "", ln, flags=re.I)
            if rest and not re.fullmatch(r"(amount|qty|rate|price|total)[\s/]*", rest, re.I):
                description_lines.append(rest)
            continue
        if capture:
            if any(p.search(ln) for p in _TOTAL_PRIORITY) or _EXCLUDE_LABELS.search(ln):
                break
            description_lines.append(ln)
            if len(description_lines) >= 5:
                break
    service_description = " | ".join(description_lines)[:400] or None

    # ---- property / item reference ----
    property_reference = None
    for ln in lines:
        m = re.match(r"(?:property|premises|flat|apartment|unit|site|address)\s*(?:no\.?|ref\.?)?\s*[:\-]\s*(.+)", ln, re.I)
        if m:
            property_reference = m.group(1).strip()[:120]
            break

    return {
        "amount_candidates": amount_candidates,
        "date_candidates": date_candidates,
        "total_amount": total_amount,
        "total_amount_source": total_source,
        "vendor": vendor,
        "invoice_date": invoice_date,
        "service_description": service_description,
        "property_reference": property_reference,
        "extraction_method": "rule_based_not_ai",
        "note": "Rule-based candidates only; verify against the original invoice.",
    }


# --------------------------------------------------------------------------
# Inspection report extraction (rule-based)
# --------------------------------------------------------------------------
_ROOMS = ["living room", "bedroom", "master bedroom", "kitchen", "bathroom", "toilet", "balcony",
          "hall", "dining", "study", "terrace", "entrance", "wall", "ceiling", "floor"]
_CONDITION_WORDS = re.compile(
    r"damag|stain|scratch|dirty|clean|broken|crack|peel|chip|dent|burn|torn|missing|mould|mold|"
    r"good condition|intact|fine|satisfactory|paint|worn|working|not working", re.I)


def extract_inspection_fields(text: str) -> dict:
    """Rule-based: inspection date candidates, rooms mentioned, and condition sentences."""
    text = text or ""
    dates = [m.group(0) for m in _DATE_RE.finditer(text)]
    iso = next((d for d in (parse_date(x) for x in dates) if d), None)
    lower = text.lower()
    rooms = [r for r in _ROOMS if r in lower]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    observations = [s[:300] for s in sentences if _CONDITION_WORDS.search(s)][:20]
    return {
        "date_candidates": dates,
        "inspection_date": iso,
        "rooms_or_items": rooms,
        "recorded_conditions": observations,
        "extraction_method": "rule_based_not_ai",
        "note": "Rule-based extraction; verify against the original inspection report.",
    }
