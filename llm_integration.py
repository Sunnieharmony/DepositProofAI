"""
Hosted-model integration for DepositProof AI.

Hosted-model calls handle vision, language tasks, and an optional explanation
of deterministic repair-rate calculations (plus a small local embedding model
in evidence_engine.py):

    analyze_image_pair_with_llm()  -> vision-language request, compares move-in / move-out photos
    assess_deduction_with_llm()    -> chat request, EXPLAINS a deduction; Python rules decide the class
    analyze_agreement_with_llm()   -> chat request, summarises agreement terms with verified quotes

Design rules (see README):
  * Hosted inference only. No model weights are downloaded here.
  * Every function returns None (and records a readable reason, see get_last_error()) when the
    model is unavailable or its output is invalid. Nothing is ever fabricated.
  * The model can only add explanation. It cannot upgrade a classification beyond what the
    rules allow, override a confirmed contradiction, change an amount, or invent a quotation.
  * Provider credentials are read from the environment and are never logged or returned.

Setup: see .env.example. Google Gemini and Hugging Face use their OpenAI-compatible
chat-completion APIs through plain urllib, so no extra dependency is needed.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable

import evidence_engine as ee

DEFAULT_HF_ROUTER_URL = "https://router.huggingface.co/v1"
DEFAULT_GEMINI_ROUTER_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_LLM = "google/gemma-2-2b-it"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

MAX_IMAGE_BYTES = int(float(os.getenv("MAX_IMAGE_MB", "12")) * 1024 * 1024)
MAX_IMAGE_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "768"))
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}  # MPO = many phone JPEGs

INTERPRETATION_NOTE = (
    "A visible difference between two photographs is an observation only. It does not prove damage, "
    "when a change happened, who caused it, whether it is tenant responsibility, or what a repair should cost."
)


class LLMError(Exception):
    """A model call failed or returned unusable output. Message is safe to show to the user."""


class ImageError(ValueError):
    """An image could not be used. Message is safe to show to the user."""


# --------------------------------------------------------------------------
# Configuration and error tracking
# --------------------------------------------------------------------------
_LAST_ERROR: str | None = None


def get_last_error() -> str | None:
    """Human-readable reason for the most recent failed model call (token removed)."""
    return _LAST_ERROR


def _set_error(msg: str | None) -> None:
    global _LAST_ERROR
    _LAST_ERROR = _sanitize(msg) if msg else None


def _sanitize(msg: str) -> str:
    msg = re.sub(r"hf_[A-Za-z0-9]{6,}", "hf_***", str(msg))
    for variable in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN", "LLM_API_KEY"):
        token = os.getenv(variable) or ""
        if token:
            msg = msg.replace(token, "***")
    return msg


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "huggingface").strip().lower()


def _api_key() -> str:
    if _provider() == "google":
        return (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        ).strip()
    return (os.getenv("LLM_API_KEY") or os.getenv("HF_TOKEN") or "").strip()


def _base_url() -> str:
    default = DEFAULT_GEMINI_ROUTER_URL if _provider() == "google" else DEFAULT_HF_ROUTER_URL
    return (os.getenv("LLM_BASE_URL") or default).rstrip("/")


def vision_model() -> str:
    if _provider() == "google":
        return (os.getenv("GEMINI_VISION_MODEL", "").strip() or os.getenv("GEMINI_MODEL", "").strip()
                or DEFAULT_GEMINI_MODEL)
    return os.getenv("HF_VLM_MODEL", "").strip() or DEFAULT_VLM


def text_model() -> str:
    if _provider() == "google":
        return (os.getenv("GEMINI_TEXT_MODEL", "").strip() or os.getenv("GEMINI_MODEL", "").strip()
                or DEFAULT_GEMINI_MODEL)
    return os.getenv("HF_LLM_MODEL", "").strip() or os.getenv("HF_MODEL", "").strip() or DEFAULT_LLM


def llm_is_configured() -> bool:
    """
    True only when the selected provider has an API key. This does NOT prove the
    models are reachable; a failed call reports why.
    """
    provider = _provider()
    if provider not in ("", "google", "huggingface", "openai_compatible", "ollama"):
        return False
    local_endpoint = bool(os.getenv("LLM_BASE_URL")) and ("localhost" in _base_url() or "127.0.0.1" in _base_url())
    return bool(_api_key()) or local_endpoint


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def prepare_image(path: str) -> tuple[str, dict]:
    """
    Validate an image and return (data_url, info) for the API.
    The image is rotated per EXIF, converted to RGB, shrunk to IMAGE_MAX_SIDE (default 768 px)
    and re-encoded as JPEG, which keeps cost and latency low while preserving enough detail.
    Raises ImageError with a friendly message on any problem.
    """
    if not path or not os.path.isfile(path):
        raise ImageError("The image file is missing.")
    size = os.path.getsize(path)
    if size == 0:
        raise ImageError("The image file is empty.")
    if size > MAX_IMAGE_BYTES:
        raise ImageError(f"The image is {size / 1048576:.1f} MB; the limit is {MAX_IMAGE_BYTES / 1048576:.0f} MB. Please upload a smaller photo.")
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImageError("Pillow is not installed (pip install Pillow).") from exc
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
            if fmt not in ALLOWED_IMAGE_FORMATS:
                raise ImageError(f"Unsupported image format '{fmt or 'unknown'}'. Use JPEG, PNG or WebP.")
            original_size = img.size
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "P"):
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                img = background
            else:
                img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
    except ImageError:
        raise
    except Exception as exc:  # corrupt files, decompression bombs, truncated data...
        raise ImageError(f"This file is not a readable image ({type(exc).__name__}).") from exc
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    info = {"original_size": original_size, "sent_size": img.size, "sent_kb": round(len(buf.getvalue()) / 1024, 1), "format": fmt}
    return f"data:image/jpeg;base64,{data}", info


def validate_image_file(path: str) -> tuple[bool, str]:
    """Convenience for the UI: (ok, message)."""
    try:
        _, info = prepare_image(path)
        return True, f"OK ({info['original_size'][0]}x{info['original_size'][1]} {info['format']})"
    except ImageError as exc:
        return False, str(exc)


# --------------------------------------------------------------------------
# HTTP + JSON helpers
# --------------------------------------------------------------------------
def _post_chat(payload: dict, timeout: float) -> dict:
    """POST to {base}/chat/completions. Tests replace this function; it is the only network call."""
    key = _api_key()
    if not key and "localhost" not in _base_url() and "127.0.0.1" not in _base_url():
        variable = "GEMINI_API_KEY" if _provider() == "google" else "HF_TOKEN"
        raise LLMError(f"No API key configured for {_provider()}. Set {variable} in your .env file.")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{_base_url()}/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")[:240]
        except (OSError, ValueError):
            body = ""
        finally:
            exc.close()
        if _provider() == "google":
            hints = {
                400: "The request or model is unsupported; check the Gemini model ID and request format.",
                401: "Google rejected the API key. Check that GEMINI_API_KEY is correct.",
                403: "Google denied access. Check that the Gemini API is enabled for the key's project and that the key is allowed to use it.",
                404: "The requested Gemini model or endpoint was not found. Check the model ID.",
                429: "Gemini quota or rate limit reached. Check the project's AI Studio rate limits and try again later.",
                500: "Gemini had a server error. Try again later.",
                503: "Gemini is temporarily unavailable. Try again later.",
            }
        else:
            hints = {
                400: "The provider rejected the request (this model may not accept this input type).",
                401: "The token was rejected (check HF_TOKEN).",
                402: "Inference credits are exhausted (see your Hugging Face billing page).",
                403: "Access denied: the token may lack 'Inference Providers' permission, or the model is gated (accept its licence on the Hub).",
                404: "The model is not available through any provider for your account. Choose another model or append a provider, e.g. ':featherless-ai'.",
                429: "Rate limited by the provider. Wait a minute and retry.",
                503: "The model is loading or temporarily unavailable. Retry shortly.",
            }
        try:
            parsed = json.loads(body)
            error = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            detail = error.get("message", "") if isinstance(error, dict) else ""
        except json.JSONDecodeError:
            detail = body
        detail = detail or body
        hint = hints.get(exc.code, "")
        if (_provider() == "google" and "api key" in detail.lower()
                and ("invalid" in detail.lower() or "not valid" in detail.lower())):
            hint = "Google says the Gemini API key is invalid. Check GEMINI_API_KEY in your .env file."
        raise LLMError(
            f"Model request failed (HTTP {exc.code}). {hint} {detail}".strip()
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"Could not reach the model service ({type(exc).__name__}). Check your internet connection.") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("The model service returned a non-JSON response.") from exc


def _message_text(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("The model response had an unexpected structure.") from exc
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise LLMError("The model returned an empty reply.")
    return content


def _extract_json(text: str) -> dict:
    """Pull one JSON object out of a model reply (handles ``` fences and surrounding prose)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("The model did not return a JSON object.")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("The model returned malformed JSON.") from exc
    if not isinstance(obj, dict):
        raise LLMError("The model returned JSON that is not an object.")
    return obj


def _chat_json(model: str, messages: list[dict], validator: Callable[[dict], dict],
               max_tokens: int, timeout: float) -> dict:
    """Call the model; validate its JSON; retry ONCE with a reminder if the JSON was unusable."""
    last: LLMError | None = None
    convo = list(messages)
    for attempt in range(2):
        response = _post_chat({"model": model, "messages": convo, "max_tokens": max_tokens, "temperature": 0.1}, timeout)
        text = _message_text(response)
        try:
            return validator(_extract_json(text))
        except LLMError as exc:
            last = exc
            convo = list(messages) + [
                {"role": "assistant", "content": text[:1500]},
                {"role": "user", "content": f"That reply was not usable ({exc}). Reply again with ONLY the JSON object in the requested format."},
            ]
    raise last or LLMError("The model output could not be validated.")


def _validate_benchmark_explanation(data: dict) -> dict:
    explanation = data.get("explanation")
    if not isinstance(explanation, str) or not 20 <= len(explanation.strip()) <= 700:
        raise LLMError("Expected a brief repair-rate explanation.")
    explanation = explanation.strip()
    if re.search(r"\d|[₹$€£]", explanation):
        raise LLMError("The explanation must not introduce or restate numeric price estimates.")
    return {"explanation": explanation}


def explain_repair_benchmark_with_llm(benchmark: dict[str, Any]) -> str | None:
    """Explain an existing deterministic benchmark without changing its figures."""
    if benchmark.get("status") != "matched":
        _set_error("No comparable repair-rate result is available to explain.")
        return None
    if not llm_is_configured():
        _set_error("No API key is configured for the selected model provider; hosted-model explanation was not used.")
        return None
    facts = {
        key: benchmark.get(key)
        for key in (
            "city", "locality", "category",
            "match_level", "match_description", "position", "record_count",
        )
    }
    facts.update({
        "sample_data_warning": benchmark.get("sample_data_warning"),
        "sample_range": {"lower": benchmark.get("lower"), "upper": benchmark.get("upper")},
        "claimed_amount": benchmark.get("claimed_amount"),
        "matched_sources": [
            {
                key: row.get(key)
                for key in ("city", "locality", "category", "minimum", "maximum")
            }
            for row in (benchmark.get("matched_records") or [])[:3]
        ],
    })
    prompt = (
        "Explain the meaning and limitations of this deterministic repair-rate comparison in plain "
        "language. It uses sample data, not verified quotations. Do not determine whether the landlord "
        "is legally entitled to deduct the amount, do not change the calculation, and do not claim the "
        "sample proves the actual repair cost. Do not include any digits, currency symbols, percentages, "
        "or new price estimates in your explanation. Return only JSON with one string property named "
        "'explanation'. Facts:\n" + json.dumps(facts, ensure_ascii=False)
    )
    try:
        result = _chat_json(
            text_model(),
            [{"role": "user", "content": prompt}],
            _validate_benchmark_explanation,
            max_tokens=250,
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )
    except LLMError as exc:
        _set_error(f"Repair-rate explanation failed: {exc}")
        return None
    _set_error(None)
    return result["explanation"]


def _str_list(value: Any, max_items: int = 8, max_len: int = 240) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LLMError("Expected a list of strings.")
    return [item.strip()[:max_len] for item in value if item.strip()][:max_items]


# --------------------------------------------------------------------------
# A. Image pair comparison (vision-language model)
# --------------------------------------------------------------------------
def _vision_prompt(item_name: str) -> str:
    return (
        f"You are reviewing two photographs of the same room or item: \"{item_name}\". "
        "Image 1 was taken at MOVE-IN and image 2 at MOVE-OUT.\n"
        "Describe only what is visible. Do NOT assume both photos show the same spot: if the angle, lighting, "
        "distance or area differs so that a fair comparison is not possible, say so. Never say who caused "
        "anything, when it happened, whether the tenant is responsible, whether a change requires the claimed repair, "
        "or what a repair would cost. State a location only when visually apparent; otherwise use an empty string. "
        "A visible difference is not proof of damage.\n"
        "comparable must be yes, partially, or no. uncertainty must be low, medium, or high. "
        "Reply with ONE valid JSON object and nothing else, using exactly these keys. Limit observed_changes to five entries:\n"
        '{"move_in_condition": "concise description of the relevant area in image 1",'
        ' "move_out_condition": "concise description of the relevant area in image 2",'
        ' "observed_changes": [{"feature": "visible change", "move_in_condition": "what is visible in image 1",'
        ' "move_out_condition": "what is visible in image 2", "location": "visible relative location or empty string",'
        ' "interpretation": "cautious comparison, not a damage/responsibility finding"}],'
        ' "summary": "2-3 sentences of what is visibly different, or that nothing is",'
        ' "comparable": "yes",'
        ' "same_area_visible_in_both": true,'
        ' "visible_differences": ["short factual items"],'
        ' "possible_damage_or_deterioration": ["short items, phrased as possibilities"],'
        ' "not_reliably_comparable": ["areas or objects that cannot be compared"],'
        ' "uncertainty": "medium",'
        ' "limitations": "one sentence"}'
    )


def _validate_vision(obj: dict) -> dict:
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise LLMError("missing 'summary'")
    comparable = str(obj.get("comparable", "")).strip().lower()
    if comparable not in ("yes", "partially", "no"):
        raise LLMError("'comparable' must be yes, partially or no")
    same = obj.get("same_area_visible_in_both")
    if not isinstance(same, bool):
        raise LLMError("'same_area_visible_in_both' must be true or false")
    unc = str(obj.get("uncertainty", "")).strip().lower()
    if unc not in ("low", "medium", "high"):
        raise LLMError("'uncertainty' must be low, medium or high")
    conditions = {}
    for key in ("move_in_condition", "move_out_condition"):
        value = obj.get(key, "")
        if not isinstance(value, str):
            raise LLMError(f"'{key}' must be a string")
        conditions[key] = value.strip()[:400]
    raw_changes = obj.get("observed_changes", [])
    if not isinstance(raw_changes, list):
        raise LLMError("'observed_changes' must be a list")
    changes = []
    for change in raw_changes[:5]:
        if not isinstance(change, dict):
            raise LLMError("Each observed change must be an object")
        parsed = {}
        for key in ("feature", "move_in_condition", "move_out_condition", "location", "interpretation"):
            value = change.get(key, "")
            if not isinstance(value, str):
                raise LLMError(f"Observed change '{key}' must be a string")
            parsed[key] = value.strip()[:240]
        if not parsed["feature"]:
            raise LLMError("Each observed change must include a feature")
        changes.append(parsed)
    return {
        "summary": summary.strip()[:600],
        **conditions,
        "observed_changes": changes,
        "comparable": comparable,
        "same_area_visible_in_both": same,
        "visible_differences": _str_list(obj.get("visible_differences")),
        "possible_damage_or_deterioration": _str_list(obj.get("possible_damage_or_deterioration")),
        "not_reliably_comparable": _str_list(obj.get("not_reliably_comparable")),
        "uncertainty": unc,
        "limitations": str(obj.get("limitations", "")).strip()[:400],
    }


def analyze_image_pair_with_llm(move_in_path: str, move_out_path: str, item_name: str) -> dict[str, Any] | None:
    """
    Send BOTH images to the hosted vision model and return a validated observation dict, or None
    (reason available from get_last_error()). Signature unchanged from the starter.

    Returned keys: observation, visible_differences, possible_damage_or_deterioration,
    not_reliably_comparable, comparable, same_area_visible_in_both, model_uncertainty (qualitative,
    NOT a probability), limitations, interpretation_note, mode ("llm_analyzed"), model, images_sent.
    """
    _set_error(None)
    if not llm_is_configured():
        _set_error("No API key is configured for the selected model provider, so photos were not analysed.")
        return None
    try:
        in_url, in_info = prepare_image(move_in_path)
        out_url, out_info = prepare_image(move_out_path)
        content = [
            {"type": "text", "text": "Image 1 - MOVE-IN photograph:"},
            {"type": "image_url", "image_url": {"url": in_url}},
            {"type": "text", "text": "Image 2 - MOVE-OUT photograph:"},
            {"type": "image_url", "image_url": {"url": out_url}},
            {"type": "text", "text": _vision_prompt(item_name or "item")},
        ]
        model = vision_model()
        result = _chat_json(model, [{"role": "user", "content": content}], _validate_vision,
                            max_tokens=700, timeout=float(os.getenv("VLM_TIMEOUT", "90")))
    except (LLMError, ImageError) as exc:
        _set_error(f"Photo analysis failed: {exc}")
        return None

    limitations = result["limitations"]
    differences = result["visible_differences"]
    damage = result["possible_damage_or_deterioration"]
    comparable = result["comparable"]
    # Rule: never present a change as real if the same area is not clearly visible in both photos.
    if result["same_area_visible_in_both"] is False or comparable == "no":
        if differences or damage:
            limitations = (limitations + " The model listed differences, but they were discarded because the same area "
                           "is not clearly visible in both photographs.").strip()
        differences, damage, comparable = [], [], "no"
        result["observed_changes"] = []
    return {
        "observation": result["summary"],
        "move_in_condition": result["move_in_condition"],
        "move_out_condition": result["move_out_condition"],
        "observed_changes": result["observed_changes"],
        "visible_differences": differences,
        "possible_damage_or_deterioration": damage,
        "not_reliably_comparable": result["not_reliably_comparable"],
        "comparable": comparable,
        "same_area_visible_in_both": result["same_area_visible_in_both"],
        "model_uncertainty": result["uncertainty"],
        "limitations": limitations or "No limitations stated by the model.",
        "interpretation_note": INTERPRETATION_NOTE,
        "mode": "llm_analyzed",
        "model": model,
        "images_sent": 2,
        "image_info": {"move_in": in_info, "move_out": out_info},
    }


# --------------------------------------------------------------------------
# B. Deduction assessment (small chat model explains; rules decide)
# --------------------------------------------------------------------------
def _build_facts(payload: dict, base: dict) -> tuple[dict, dict[str, dict]]:
    currency = payload.get("currency") or "INR"
    id_map: dict[str, dict] = {}
    passages = []
    for i, c in enumerate([c for c in payload.get("clauses") or [] if c.get("quote_verified")][:5], 1):
        cid = f"C{i}"
        id_map[cid] = c
        passages.append({"id": cid, "file": c.get("filename"), "page": c.get("page"),
                         "relevance": c.get("relevance"), "text": str(c.get("quote", ""))[:500]})
    inv = payload.get("invoice") or {}
    invoice = None
    if inv:
        invoice = {"file": inv.get("filename"), "vendor": inv.get("vendor"), "date": inv.get("invoice_date"),
                   "description": inv.get("service_description"), "total": inv.get("total_amount"),
                   "other_amounts": (inv.get("amount_candidates") or [])[:6],
                   "readable": bool((inv.get("extracted_text") or "").strip())}
    photo = payload.get("photo_observation") or {}
    facts = {
        "claim": {"category": payload.get("category"), "description": payload.get("description"),
                  "claimed_amount": payload.get("amount"), "currency": currency},
        "rule_checks": {
            "agreement_clause_status": base.get("clause_status"), "invoice_status": base.get("invoice_status"),
            "amount_check": base.get("amount_status"), "photo_status": base.get("photo_status"),
            "contradictions": base.get("contradictions"), "missing": base.get("missing_evidence"),
            "open_warnings": base.get("warnings"),
        },
        "agreement_passages": passages,
        "invoice": invoice,
        "photo_observation": ({"summary": photo.get("observation"), "visible_differences": photo.get("visible_differences"),
                               "comparable": photo.get("comparable"), "limitations": photo.get("limitations")} if photo else None),
        "inspection_notes": [{"file": f["filename"], "timing": f["context"], "note": f["sentence"], "type": f["polarity"]}
                             for f in (base.get("inspection_findings") or [])[:6]],
        "rules_classification": base.get("classification"),
    }
    return facts, id_map


_ASSESS_PROMPT = (
    "You help a tenant review one security-deposit deduction. Use ONLY the FACTS below. Do not invent documents, "
    "clauses, amounts, dates or quotations. Do not decide legal liability or responsibility. Keep evidence of damage, "
    "of a repair, of cost, and of contract terms separate. Missing evidence is not proof that a claim is false.\n"
    "FACTS:\n{facts}\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    '{{"classification_suggestion": "Supported" | "Inconsistent" | "Insufficiently Documented",'
    ' "explanation": "3-5 plain sentences explaining what the evidence shows and what is missing; refer to clauses by id (C1...) and do not copy quotations",'
    ' "cited_clause_ids": ["C1"],'
    ' "missing_evidence": ["short items"],'
    ' "contradictions": ["short items, only if the FACTS show a conflict"],'
    ' "limitations": "one sentence"}}'
)


def _validate_assessment(obj: dict) -> dict:
    raw = str(obj.get("classification_suggestion", "")).strip().lower()
    mapping = {c.lower(): c for c in ee.CLASSIFICATIONS}
    mapping.update({"insufficient": ee.INSUFFICIENT, "insufficiently_documented": ee.INSUFFICIENT})
    if raw not in mapping:
        raise LLMError("'classification_suggestion' must be Supported, Inconsistent or Insufficiently Documented")
    explanation = obj.get("explanation")
    if not isinstance(explanation, str) or len(explanation.strip()) < 20:
        raise LLMError("missing 'explanation'")
    return {
        "classification_suggestion": mapping[raw],
        "explanation": explanation.strip()[:1200],
        "cited_clause_ids": _str_list(obj.get("cited_clause_ids"), 6, 8),
        "missing_evidence": _str_list(obj.get("missing_evidence"), 6, 200),
        "contradictions": _str_list(obj.get("contradictions"), 6, 200),
        "limitations": str(obj.get("limitations", "")).strip()[:300],
    }


_QUOTE_RE = re.compile(r'"([^"]{20,})"|“([^”]{20,})”')
_NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUMBER_RE.findall(text or ""):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _explanation_problem(explanation: str, facts_json: str, corpus: str) -> str | None:
    """Return a reason to discard the model's explanation, or None if it is acceptable."""
    for m in _QUOTE_RE.finditer(explanation):
        fragment = m.group(1) or m.group(2)
        if not ee.verify_quote(fragment, corpus):
            return "it contained a quotation that could not be verified against the source documents"
    allowed = _numbers(facts_json) | _numbers(corpus)
    for n in _numbers(explanation):
        if n >= 100 and not (1900 <= n <= 2100) and n not in allowed:
            return f"it mentioned an amount ({n:g}) that does not appear in the evidence"
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = re.sub(r"\W+", " ", item.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def merge_llm_assessment(base: dict, llm: dict, id_map: dict[str, dict], corpus: str, facts_json: str, model: str) -> dict:
    """
    Combine the rules result with the model's explanation.
      * The model may only keep or DOWNGRADE a Supported result. It cannot create Supported,
        and it cannot override a rules-confirmed Inconsistent result.
      * Issues the model raises that the rules did not confirm are listed separately as unverified.
    """
    final = dict(base)
    rules_cls, suggested = base["classification"], llm["classification_suggestion"]
    notes: list[str] = []
    classification = rules_cls
    if rules_cls == ee.SUPPORTED and suggested != ee.SUPPORTED:
        classification = ee.INSUFFICIENT
        notes.append(f"The model suggested '{suggested}', so the result was downgraded from Supported to be safe.")
    elif suggested != rules_cls:
        notes.append(f"The model suggested '{suggested}', but the rule checks decided '{rules_cls}'; the rule result was kept.")

    problem = _explanation_problem(llm["explanation"], facts_json, corpus)
    if problem:
        explanation = base["explanation"]
        notes.append(f"The model's explanation was discarded because {problem}.")
        final["explanation_source"] = "rules (model explanation rejected)"
    else:
        explanation = llm["explanation"]
        final["explanation_source"] = "model"
    if classification == ee.INSUFFICIENT and rules_cls == ee.SUPPORTED:
        explanation = f"{explanation} (Classification limited to 'Insufficiently Documented' after the model reported gaps.)"

    cited = [{"filename": id_map[i].get("filename"), "page": id_map[i].get("page"),
              "quote": id_map[i].get("quote"), "passage_id": id_map[i].get("passage_id")}
             for i in llm["cited_clause_ids"] if i in id_map]
    known_missing = " ".join(base.get("missing_evidence") or []).lower()
    model_flagged = [c for c in llm["contradictions"] if c.lower() not in " ".join(base.get("contradictions") or []).lower()]

    final.update({
        "classification": classification,
        "rules_classification": rules_cls,
        "explanation": explanation,
        "rules_explanation": base["explanation"],
        "cited_clauses": cited,
        "missing_evidence": _dedupe(list(base.get("missing_evidence") or []) + [m for m in llm["missing_evidence"] if m.lower() not in known_missing]),
        "model_flagged_issues_unverified": model_flagged,
        "llm_classification_suggestion": suggested,
        "reconciliation_notes": notes,
        "limitations": (llm["limitations"] + " " + base.get("limitations", "")).strip(),
        "mode": "llm",
        "model": model,
    })
    final["missing_or_conflicting_evidence"] = "; ".join(
        [f"CONFLICT: {c}" for c in base.get("contradictions") or []] + final["missing_evidence"] +
        [f"REVIEW: {w}" for w in base.get("warnings") or []] +
        [f"MODEL-FLAGGED (unverified): {m}" for m in model_flagged]) or "No gaps reported."
    return final


def assess_deduction_with_llm(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Assess one deduction. Steps: (1) deterministic rules run first, (2) the chat model is given the
    verified facts and asked for an explanation as JSON, (3) the reply is validated, quotations and
    amounts are checked against the sources, and (4) the result is merged under the rules above.

    Returns the full assessment dict (classification in {Supported, Inconsistent, Insufficiently
    Documented}) or None when the model call fails (reason in get_last_error()); the caller then
    uses the rules-only result and labels it as a fallback.
    """
    _set_error(None)
    if not llm_is_configured():
        _set_error("No API key is configured for the selected model provider; the language model was not used.")
        return None
    base = ee.classify_deduction(payload)
    facts, id_map = _build_facts(payload, base)
    facts_json = json.dumps(facts, ensure_ascii=False)
    corpus = "\n".join(
        [c.get("quote", "") for c in id_map.values()] +
        [(payload.get("invoice") or {}).get("extracted_text", "")[:6000]] +
        [f["sentence"] for f in base.get("inspection_findings") or []] +
        [str((payload.get("photo_observation") or {}).get("observation", ""))])
    model = text_model()
    try:
        llm = _chat_json(model, [{"role": "user", "content": _ASSESS_PROMPT.format(facts=facts_json)}],
                         _validate_assessment, max_tokens=600, timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    except LLMError as exc:
        _set_error(f"Deduction assessment failed: {exc}")
        return None
    return merge_llm_assessment(base, llm, id_map, corpus, facts_json, model)


# --------------------------------------------------------------------------
# C. Agreement analysis
# --------------------------------------------------------------------------
_AMOUNT_RE = re.compile(r"(?:₹|Rs\.?|INR)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?", re.I)

_AGREEMENT_PROMPT = (
    "Below are numbered passages extracted from a rental agreement. Summarise what the agreement says about each topic, "
    "using ONLY these passages. If no passage addresses a topic, set status to \"not_found\". Do not guess and do not "
    "describe terms that are not in the passages. Write summaries in your own words (do not copy quotations).\n\n"
    "TOPICS: {topics}\n\nPASSAGES:\n{passages}\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    '{{"topics": {{"<topic_key>": {{"status": "found" | "not_found", "summary": "1-2 sentences", "passage_ids": ["P1"]}}}},'
    ' "ambiguities": ["short items about unclear or conflicting wording"]}}'
)


def _validate_agreement(obj: dict) -> dict:
    topics = obj.get("topics")
    if not isinstance(topics, dict) or not topics:
        raise LLMError("missing 'topics'")
    return {"topics": topics, "ambiguities": _str_list(obj.get("ambiguities"), 6, 240)}


def analyze_agreement_with_llm(document_text: str, pages: dict | None = None, filename: str = "agreement") -> dict[str, Any] | None:
    """
    Summarise key agreement terms. Only the most relevant passages (not the whole document) are
    sent to the model. Quotations in the output are copied from the document by code and verified,
    never produced by the model. Returns None on failure (see get_last_error()).
    (`pages` and `filename` are optional additions to the starter signature.)
    """
    _set_error(None)
    if not document_text or not document_text.strip():
        _set_error("The agreement has no extracted text.")
        return None
    if not llm_is_configured():
        _set_error("No API key is configured for the selected model provider; the language model was not used.")
        return None

    retrieved = ee.retrieve_topic_passages(document_text, pages or {}, filename, per_topic=2)
    seen: dict[str, str] = {}
    registry: dict[str, dict] = {}
    for topic_passages in retrieved.values():
        for c in topic_passages:
            if c["passage_id"] not in seen and len(registry) < 12:
                pid = f"P{len(registry) + 1}"
                seen[c["passage_id"]] = pid
                registry[pid] = c
    if not registry:
        _set_error("No passages could be retrieved from the agreement text.")
        return None
    passages_text = "\n".join(f"[{pid}] (page {c['page']}) {c['quote'][:600]}" for pid, c in registry.items())
    topic_text = "; ".join(f"{k} = {q}" for k, q in ee.AGREEMENT_TOPICS)
    try:
        result = _chat_json(text_model(), [{"role": "user", "content": _AGREEMENT_PROMPT.format(topics=topic_text, passages=passages_text)}],
                            _validate_agreement, max_tokens=900, timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    except LLMError as exc:
        _set_error(f"Agreement analysis failed: {exc}")
        return None

    out_topics, not_found = [], []
    for key, _query in ee.AGREEMENT_TOPICS:
        entry = result["topics"].get(key)
        ids = [i for i in (entry.get("passage_ids") if isinstance(entry, dict) else []) or [] if i in registry]
        found = isinstance(entry, dict) and entry.get("status") == "found" and ids and isinstance(entry.get("summary"), str)
        if not found:
            not_found.append(key)
            continue
        out_topics.append({
            "topic": key,
            "summary": entry["summary"].strip()[:400],
            "summary_is_model_generated": True,
            "quotes": [{"quote": registry[i]["quote"], "page": registry[i]["page"], "passage_id": registry[i]["passage_id"],
                        "verified": ee.verify_quote(registry[i]["quote"], document_text)} for i in ids],
        })

    deposit_mentions = []
    for p in ee.split_passages(document_text, pages):
        if re.search(r"deposit", p["text"], re.I) and _AMOUNT_RE.search(p["text"]):
            deposit_mentions.append({"amount_text": _AMOUNT_RE.search(p["text"]).group(0), "quote": p["text"][:500],
                                     "page": p["page"], "verified": ee.verify_quote(p["text"][:500], document_text)})
        if len(deposit_mentions) >= 3:
            break

    method = next((c["retrieval_method"] for c in registry.values()), "unknown")
    return {
        "filename": filename,
        "topics": out_topics,
        "topics_not_found_in_retrieved_passages": not_found,
        "security_deposit_amount_mentions": deposit_mentions,
        "ambiguities": result["ambiguities"],
        "limitations": [
            "Only the top passages per topic were analysed, not every sentence. 'Not found' means no matching passage was retrieved, "
            "not that the agreement is silent.",
            f"Retrieval method: {method}.",
            "Summaries are model-generated; quotations are copied verbatim from the document and verified.",
        ],
        "mode": "llm",
        "model": text_model(),
        "retrieval_method": method,
    }