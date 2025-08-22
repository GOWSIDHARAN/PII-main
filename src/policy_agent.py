import json
import os
from typing import Any, Dict, List, Tuple

import google.generativeai as genai

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")


def _ensure_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)


def classify_and_redact(text: str, detections: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    """Use Gemini to decide action and propose redactions.

    Returns (action, redacted_text, justification)
    action in {ALLOW, REDACT, BLOCK}
    """
    # If no LLM key, default to REDACT via placeholders
    if not os.getenv("GEMINI_API_KEY"):
        return _fallback(text, detections)

    _ensure_genai()
    # Compose prompt
    prompt = (
        "You are a security policy agent. Given input text and detected PII, decide: ALLOW, REDACT, or BLOCK.\n"
        "- If PII is present, prefer REDACT by masking detected spans with tokens like [EMAIL], [PHONE NUMBER], etc.\n"
        "- Provide a brief justification.\n"
        "Return ONLY JSON with keys: 'action' (string), 'justification' (string), 'redactions' (list of {start,end,mask}).\n"
        f"Text: {text}\n"
        f"Detections: {json.dumps(detections, ensure_ascii=False)}\n"
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        resp = model.generate_content(prompt)
        raw = (resp.text or "{}").strip()
        if raw.startswith("```"):
            raw = raw.strip('`')
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        action = str(data.get("action", "REDACT")).upper()
        justification = str(data.get("justification", ""))
        redactions = data.get("redactions", []) or []
        # Apply redactions
        redacted = _apply_redactions(text, redactions)
        return action, redacted, justification
    except Exception:
        return _fallback(text, detections)


def _apply_redactions(text: str, redactions: List[Dict[str, Any]]) -> str:
    s = text
    offset = 0
    # sort by start ascending
    for r in sorted(redactions, key=lambda x: int(x.get("start", 0))):
        start = int(r.get("start", -1))
        end = int(r.get("end", -1))
        mask = str(r.get("mask", "[REDACTED]"))
        if start >= 0 and end >= start:
            s = s[: start + offset] + mask + s[end + offset :]
            offset += len(mask) - (end - start)
    return s


def _fallback(text: str, detections: List[Dict[str, Any]]):
    # default simple placeholder redaction
    s = text
    offset = 0
    for d in sorted(detections, key=lambda x: x.get("start", 0)):
        start = int(d.get("start", -1))
        end = int(d.get("end", -1))
        mask = f"[{str(d.get('type','PII')).upper()}]"
        if start >= 0 and end >= start:
            s = s[: start + offset] + mask + s[end + offset :]
            offset += len(mask) - (end - start)
    justification = "Fallback redaction applied without LLM."
    return "REDACT", s, justification
