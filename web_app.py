def _normalize_text(s: str) -> str:
    """Normalize Unicode artifacts and common PDF glyph issues before detection.
    - NFKC normalization
    - Remove zero-width and control chars
    - Replace fancy quotes/dashes with plain ASCII
    """
    if not s:
        return s
    s = unicodedata.normalize('NFKC', s)
    # Remove zero-width and control characters except \n, \t
    s = re.sub(r"[\u200B\u200C\u200D\uFEFF\u2060]", "", s)
    s = ''.join(ch for ch in s if (ch in '\n\t' or (ord(ch) >= 32)))
    # Normalize punctuation
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u00A0': ' '
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Replace standalone symbol glyphs (icons, dingbats) with space
    try:
        buf = []
        for ch in s:
            cat = unicodedata.category(ch)
            if ch in '\n\t':
                buf.append(ch)
            elif cat.startswith(('S',)) or cat in ('Cf', 'Co'):
                buf.append(' ')
            else:
                buf.append(ch)
        s = ''.join(buf)
    except Exception:
        pass
    # Collapse weird separator artifacts
    s = re.sub(r"[|·•▪◦●○●•◼◾▫▸►▶➤➔➣➢⦁·▹▻➤⌢¶♂♀✓✔✗✘★☆✦✧✪❖❯❮❱❰❯•■□▲△▶▷▸►◀◁◂◄◆◇❖❑❒❙❚❘]+", " ", s)
    # Collapse excessive spaces
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

def _text_quality_score(s: str) -> float:
    """Heuristic quality score: ratio of [A-Za-z0-9@._:/+-] and spaces vs total.
    Returns 0..1. Lower => more garbled.
    """
    if not s:
        return 0.0
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@._:/+- ,\n\t()[]{}#")
    good = sum(1 for ch in s if ch in allowed)
    return good / max(1, len(s))

from flask import Flask, render_template_string, request, jsonify
import re
from datetime import datetime
import os
import sys
import json
from typing import List, Dict, Any
import io

from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import unicodedata

"""IMPORTANT: load .env before importing local modules that read env (e.g., Pinecone)."""
# Load environment from this project folder explicitly (override to ensure .env values take precedence)
_BASE_DIR = os.path.dirname(__file__)
_DOTENV_PATH = os.path.join(_BASE_DIR, ".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
print(f"[Env] Loaded .env from {_DOTENV_PATH}")
print(f"[Env] VECTOR_BACKEND={(os.getenv('VECTOR_BACKEND') or 'chroma').lower()} CHROMA_CLOUD={os.getenv('CHROMA_CLOUD')} CHROMA_TENANT={os.getenv('CHROMA_TENANT')} CHROMA_DATABASE={os.getenv('CHROMA_DATABASE')}")
ENABLE_BENCH = (os.getenv("ENABLE_BENCH", "false").strip().lower() in {"1","true","yes","on"})
if not ENABLE_BENCH:
    print("[Feature] Retrieval benchmark UI is disabled (set ENABLE_BENCH=true to enable /bench)")

# Local modules
SRC_PATH = os.path.join(os.path.dirname(__file__), "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from embeddings import embed_text, embed_texts  # type: ignore
from chroma_store import ChromaStore  # type: ignore
from policy_agent import classify_and_redact  # type: ignore
from audit_logger import log_decision, fetch_recent_benchmarks  # type: ignore

app = Flask(__name__)

# Gemini config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
USE_GEMINI = os.getenv("USE_GEMINI", "true").strip().lower() in {"1","true","yes","on"}
if GEMINI_API_KEY and USE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)

# Optional external tool paths for OCR
TESSERACT_CMD = os.getenv("TESSERACT_CMD")  # e.g., C:\\Program Files\\Tesseract-OCR\\tesseract.exe
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
POPPLER_PATH = os.getenv("POPPLER_PATH")  # e.g., C:\\poppler-xx\\Library\\bin

# Vector store: ChromaDB only
vector_store = None
LAST_VECTOR_ERROR = None
try:
    vector_store = ChromaStore()
    print("[Vector] Using ChromaDB backend")
except Exception as e:
    LAST_VECTOR_ERROR = f"Chroma: {e}"
    print(f"[ERROR] Chroma initialization failed: {e}")
    vector_store = None

# Domain removal strictness toggle: off | normal | strict
DOMAIN_REMOVAL_STRICTNESS = os.getenv("DOMAIN_REMOVAL_STRICTNESS", "normal").strip().lower()

# Reusable regexes for cleanup/counting
BARE_DOMAIN_RE = r'(?i)(?<![@\w])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,})(?:/[\w\-\./%?#=&+]*)?(?![@\w])'
OFFICIAL_HANDLE_RE = r'(?i)(?:(?<=\s)|^|/)[A-Za-z0-9][A-Za-z0-9\-]{1,62}-official\b'

def _count_by_type(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in detections:
        t = d.get('type', 'Unknown')
        counts[t] = counts.get(t, 0) + 1
    return counts

def _count_pattern(text: str, pattern: str) -> int:
    try:
        return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
    except Exception:
        return 0

# --- Retrieval benchmark helpers (UI) ---
def _load_corpus(dir_path: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    try:
        for fn in os.listdir(dir_path):
            if not fn.lower().endswith('.txt'):
                continue
            fid = os.path.splitext(fn)[0]
            fp = os.path.join(dir_path, fn)
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                items.append((fid, f.read()))
    except Exception:
        pass
    return items

def _recall_at_k(expected_ids: list[str], got_ids: list[str]) -> float:
    if not expected_ids:
        return 0.0
    return 1.0 if any(g in expected_ids for g in got_ids) else 0.0


# -----------------------------
# Retrieval Benchmark Web Route (disabled by request)
# -----------------------------
@app.route('/bench', methods=['GET', 'POST'])
def _bench_disabled():  # type: ignore
    return ("Not Found", 404)

# Simple PII detection patterns (Resume-focused)
PII_PATTERNS = {
    'Name': [
        # Personal name line, optionally preceded by 'Name:' label; 2-4 tokens, title-cased
        r'(?im)^\s*(?:Name\s*[:\-]\s*)?(?:[A-Z][a-z]{1,29})(?:\s+[A-Z](?:\.)?)?(?:\s+[A-Z][a-z]{1,29}){1,3}\s*$',
        # ALL-CAPS name lines (2-4 tokens), allow single-letter initials
        r'(?im)^\s*(?:Name\s*[:\-]\s*)?(?:[A-Z]{2,}|[A-Z]\.?)\s+(?:[A-Z]{2,}|[A-Z]\.?(?:\s+[A-Z]{2,}|\s+[A-Z]\.?){0,2})\s*$'
    ],
    'Phone Number': [
        # Generic North America and simple 10-digit (e.g., India) numbers
        r'\b(?:\+?\d{1,3}[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
        r'\b\d{10}\b',
        # India-style with spaces: (+91 )? 5+5 split (e.g., 84386 46468)
        r'\b(?:\+?\d{1,3}\s*)?\d{5}\s*\d{5}\b'
    ],
    'Email': [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ],
    'Address': [
        # Entire line starting with Address:
        r'(?im)^\s*(?:Address|Location)\s*[:\-]\s*.+$'
    ],
    'Date of Birth': [
        r'(?im)\b(?:DOB|D\.O\.B\.|Date of Birth)\s*[:\-]?\s*(?:[0-3]?\d[\/\-.][01]?\d[\/\-.](?:\d{2}|\d{4})|\b\w+\s+\d{1,2},\s+\d{4}\b)'
    ],
    'PAN': [
        r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
    ],
    'Aadhaar': [
        # 12 digits continuous, or grouped 4-4-4 with spaces or hyphens
        r'\b\d{12}\b',
        r'\b\d{4}[\s\-]\d{4}[\s\-]\d{4}\b',
        # Label-based (Aadhaar/Aadhar) with optional No/Number, then digits/spaces/hyphens
        r'(?im)^\s*(?:Aadhaar|Aadhar)\s*(?:No\.?|Number)?\s*[:\-]\s*[0-9\s\-]{12,}\b.*$'
    ],
    'LinkedIn': [
        r'\bhttps?://(www\.)?linkedin\.com/(?:in|pub)/[A-Za-z0-9-_/]+\b',
        r'\blinkedin\.com/(?:in|pub)/[A-Za-z0-9-_/]+\b',
        # Obfuscated or slug-only cases: consume optional leading '/' to avoid leftover slash
        r'(?i)(?:(?<=\s)|^)/?linkedin\s*[\s:/-]*[A-Za-z0-9][A-Za-z0-9\-_/]{2,}\b',
        # Concatenated 'linkedin' + slug with no delimiter, e.g., '/linkedingowsidharan-s-t'
        r'(?i)(?:(?<=\s)|^)/?linkedin[A-Za-z0-9\-_/]{3,}\b'
    ],
    'GitHub': [
        r'\bhttps?://(www\.)?github\.com/[A-Za-z0-9-_/\.]+\b',
        r'\bgithub\.com/[A-Za-z0-9-_/\.]+\b',
        # OCR/slug variants like "Github / Sangeethraj" or "Git hub: Sangeethraj"
        r'(?i)(?:(?<=\s)|^)(?:git\s*hub|github)\s*[:|/\\-]*\s*[@/]?[A-Za-z0-9][A-Za-z0-9\-]{1,39}\b'
    ],
    'LeetCode': [
        r'\bhttps?://(www\.)?leetcode\.com/(?:u/)?[A-Za-z0-9_\-/.]+\b',
        r'\bleetcode\.com/(?:u/)?[A-Za-z0-9_\-/.]+\b',
        # OCR/slug variants like "LeetCode / Sangeethraj" or "Leet Code: Sangeethraj"
        r'(?i)(?:(?<=\s)|^)(?:leet\s*code|leetcode)\s*[:|/\\-]*\s*[@/]?[A-Za-z0-9][A-Za-z0-9_\-]{1,39}\b'
    ],
    'GeeksForGeeks': [
        r'\bhttps?://(www\.)?geeksforgeeks\.org/user/[A-Za-z0-9_\-/.]+\b',
        r'\bgeeksforgeeks\.org/user/[A-Za-z0-9_\-/.]+\b',
        # OCR/slug variants including misspelling 'GeekForGeeks' (first word singular)
        r'(?i)(?:(?<=\s)|^)(?:geek[s]?\s*for\s*geeks|geeksforgeeks|gfg)\s*[:|/\\-]*\s*[@/]?[A-Za-z0-9][A-Za-z0-9_\-]{1,39}\b'
    ],
    'Gender': [
        r'(?im)^\s*(?:Gender|Sex)\s*[:\-]\s*(Male|Female|Other|Non\-binary|Prefer not to say)\b.*$'
    ],
    'Nationality': [
        r'(?im)^\s*(?:Nationality|Citizenship)\s*[:\-]\s*.+$'
    ],
    'Marital Status': [
        r'(?im)^\s*(?:Marital\s*Status)\s*[:\-]\s*(Single|Married|Divorced|Widowed)\b.*$'
    ],
    'Driver License': [
        # India: e.g., KA01 20201234567 (state+RTO+13 digits), conservative
        r'\b[A-Z]{2}\d{2}\s?\d{11}\b',
        # Label-based generic (reduces false positives)
        r'(?im)^\s*(?:Driver\'?s\s*License|DL\s*No\.|License\s*No\.)\s*[:\-]\s*[A-Z0-9\-]{5,20}\b.*$'
    ],
    'Voter ID': [
        # India EPIC: 3 letters + 7 digits
        r'\b[A-Z]{3}\d{7}\b',
        r'(?im)^\s*(?:Voter\s*ID|EPIC)\s*[:\-]\s*[A-Z0-9]{5,20}\b.*$'
    ],
    'Student ID': [
        r'(?im)^\s*(?:Student\s*ID|Roll\s*(?:No\.|Number)|Registration\s*No\.)\s*[:\-]\s*[A-Z0-9\-/]{3,20}\b.*$'
    ],
    'Employee ID': [
        r'(?im)^\s*(?:Employee\s*ID|Emp\s*ID)\s*[:\-]\s*[A-Z0-9\-/]{3,20}\b.*$'
    ],
    'Bank Account': [
        r'(?im)^\s*(?:Bank\s*(?:A/c|Account)\s*(?:No\.|Number)?)\s*[:\-]\s*\d{8,18}\b.*$'
    ],
    'IFSC': [
        r'\b[A-Z]{4}0[0-9A-Z]{6}\b'
    ],
    'SWIFT': [
        # Label-based only to avoid mass false positives on normal words
        r'(?im)^\s*(?:SWIFT|BIC|SWIFT\s*Code)\s*[:\-]\s*[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b.*$'
    ],
    'Salary': [
        r'(?im)^\s*(?:Salary|CTC|Compensation|Expected\s*Salary|Current\s*Salary)\s*[:\-]\s*.*(?:₹|INR|USD|EUR|LPA|pa)\b.*$'
    ],
    'IP Address': [
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b'
    ],
    'GPS Coordinates': [
        r'\b-?\d{1,2}\.\d{3,},\s*-?\d{1,3}\.\d{3,}\b'
    ],
    'Social Handle': [
        r'\bhttps?://(www\.)?(twitter|x|instagram|facebook|fb|threads)\.com/[A-Za-z0-9_\.\-/]+\b',
        r'(?<!\S)@([A-Za-z0-9_]{3,30})(?!\S)'
    ],
    'URL': [
        r'\bhttps?://[\w\-\.]+(?:\.[a-z]{2,})(?:/[\w\-\./%?#=&+]*)?\b'
    ],
    'Institution': [
        r'(?im)^\s*(?:University|College|Institute|School)\s*[:\-]\s*.+$'
    ],
    'SSN': [
        r'\b\d{3}-\d{2}-\d{4}\b',
        r'\b\d{9}\b'
    ],
    'Credit Card': [
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'
    ],
    'Passport Number': [
        # Indian passport: 1 letter followed by 7 digits
        r'\b[A-PR-WYa-pr-wy][0-9]{7}\b',
        # Generic: a letter followed by 7 or 8 digits (very conservative)
        r'\b[A-Z][0-9]{7,8}\b',
        r'\b\w[0-9]{7,8}\b'
    ],
    # (Deduplicated LinkedIn/GitHub entries moved above)
    
}

def detect_pii(text):
    """Simple PII detection using regex with conservative Name heuristics."""
    detections = []
    
    for pii_type, patterns in PII_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                item = {
                    'text': match.group(),
                    'type': pii_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 90,
                    'method': 'Regex Pattern'
                }
                detections.append(item)

    # Post-filter: reduce Name false positives (headings, degrees, sections)
    NAME_STOPWORDS = {
        'objective','career','summary','education','projects','experience','skills','technical',
        'certification','certifications','languages','tools','web','development','machine','learning',
        'bachelor','master','secondary','higher','diploma','associate','ai','artificial','intelligence',
        'data','science','engineering','engineer'
    }
    filtered: List[Dict[str, Any]] = []
    for d in detections:
        if d['type'] != 'Name':
            filtered.append(d)
            continue
        t = d['text'].strip()
        # Basic shape checks
        words = [w for w in re.split(r"\s+", t) if w]
        if len(words) < 2 or len(words) > 4:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        _name_label_re = re.compile(r'^name\s*[:\-]\s*', flags=re.IGNORECASE)
        core = _name_label_re.sub('', t).strip()
        lw = [w.strip('.').lower() for w in core.split()]
        # If majority are stopwords (sections/degrees), drop
        stop_count = sum(1 for w in lw if w in NAME_STOPWORDS)
        if stop_count >= max(1, len(lw) - 1):
            continue
        # If tokens are not Titlecase words, ALL-CAPS words, or initials (A or A.), drop
        if not all(re.match(r'^(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Z]\.?)$', w) for w in words):
            continue
        filtered.append(d)

    # Heuristic: If no Name remained, try to infer a Name line near contact details
    has_name = any(d['type'] == 'Name' for d in filtered)
    if not has_name:
        contact_spans = []
        for d in filtered:
            if d['type'] in ('Email', 'Phone Number'):
                contact_spans.append((d['start'], d['end']))
        if contact_spans:
            # Define a window around first contact
            start_win = max(0, min(s for s, _ in contact_spans) - 200)
            end_win = min(len(text), max(e for _, e in contact_spans) + 200)
            window = text[start_win:end_win]
            # Look for a title-cased 2-4 token line (allow initials) in the window's first 5 lines
            lines = window.splitlines()
            candidates = []
            for i, line in enumerate(lines[:5]):
                l = line.strip()
                if not l or any(ch.isdigit() for ch in l):
                    continue
                words = [w for w in re.split(r"\s+", l) if w]
                if not (2 <= len(words) <= 4):
                    continue
                # Title/Name check: allow Titlecase, ALL-CAPS words, or initials like 'S'/'S.'
                if not all(re.match(r'^(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Z]\.?)$', w) for w in words):
                    continue
                lw = [w.lower().strip('.') for w in words]
                if sum(1 for w in lw if w in NAME_STOPWORDS) > 0:
                    continue
                # Map back to absolute offsets
                abs_start = text.find(line, start_win, end_win)
                if abs_start != -1:
                    abs_end = abs_start + len(line)
                    candidates.append((l, abs_start, abs_end))
            if candidates:
                # Prefer the first candidate
                cand_text, cs, ce = candidates[0]
                filtered.append({
                    'text': cand_text,
                    'type': 'Name',
                    'start': cs,
                    'end': ce,
                    'confidence': 80,
                    'method': 'Heuristic (near contact)'
                })

    # Fallback: If still no Name, attempt top-of-resume title-cased line (first 12 lines)
    has_name = any(d['type'] == 'Name' for d in filtered)
    if not has_name:
        head = text.splitlines()[:12]
        for line in head:
            l = line.strip()
            if not l or any(ch.isdigit() for ch in l):
                continue
            words = [w for w in re.split(r"\s+", l) if w]
            if not (2 <= len(words) <= 4):
                continue
            if not all(re.match(r'^(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Z]\.?)$', w) for w in words):
                continue
            lw = [w.lower().strip('.') for w in words]
            if sum(1 for w in lw if w in NAME_STOPWORDS) > 0:
                continue
            abs_start = text.find(line)
            if abs_start != -1:
                filtered.append({
                    'text': l,
                    'type': 'Name',
                    'start': abs_start,
                    'end': abs_start + len(line),
                    'confidence': 75,
                    'method': 'Heuristic (header)'
                })
                break

    return filtered


def detect_pii_llm(text: str) -> List[Dict[str, Any]]:
    """Detect PII using Gemini structured JSON output. Falls back to []."""
    if not (GEMINI_API_KEY and USE_GEMINI):
        return []
    prompt = (
        "You are extracting personally identifiable information (PII) from raw text.\n"
        "Return ONLY a JSON object with key 'pii_entities' as a list of objects with fields: "
        "type (Email|Phone Number|SSN|Credit Card|Address|Name|Passport Number|LinkedIn|GitHub), text, start (int), end (int), confidence (0-100).\n"
        "Rules to minimize false positives across arbitrary templates:\n"
        "- Do NOT mark headings, section titles, bullet headers, degree names, certifications, skills, technologies, job titles, company names, or institution names as Name.\n"
        "- Only mark Name for the actual personal name(s) of the document subject or a specific person's name in narrative context.\n"
        "- Prefer names that appear near contact details (email/phone) or clearly in running sentences (e.g., 'My name is ...').\n"
        "- Do NOT output organization names, degree programs, course titles, section labels (e.g., 'Education', 'Projects', 'Technical Skills'), or generic phrases as Name.\n"
        "- Ensure 'start' and 'end' are character offsets referencing the provided text, with 0 <= start < end, and the slice text[start:end] exactly equals 'text'.\n"
        "- Avoid overlapping or duplicate spans of the same content.\n"
        "- If uncertain, omit the entity.\n\n"
        f"Text:\n{text}\n"
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        resp = model.generate_content(prompt)
        raw = (resp.text or "{}").strip()
        # Strip common markdown code fences
        if raw.startswith("```"):
            raw = raw.strip('`')
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        # Attempt load
        data = json.loads(raw)
        items = data.get("pii_entities", []) if isinstance(data, dict) else []
        detections: List[Dict[str, Any]] = []
        for ent in items:
            try:
                detections.append({
                    'text': ent.get('text', ''),
                    'type': ent.get('type', 'Unknown'),
                    'start': int(ent.get('start', -1)),
                    'end': int(ent.get('end', -1)),
                    'confidence': int(ent.get('confidence', 75)),
                    'method': 'Gemini'
                })
            except Exception:
                continue
        return [d for d in detections if d.get('start', -1) >= 0 and d.get('end', -1) >= 0]
    except Exception as e:
        print(f"[WARN] Gemini detection failed, falling back to regex: {e}")
        return []

def _merge_spans(detections: List[Dict[str, Any]]) -> List[tuple]:
    """Merge overlapping/adjacent detection spans. Returns list of (start, end)."""
    spans = sorted([(int(d['start']), int(d['end'])) for d in detections if d.get('start') is not None and d.get('end') is not None])
    merged: List[tuple] = []
    for s, e in spans:
        if not merged:
            merged.append((s, e))
        else:
            ls, le = merged[-1]
            if s <= le:  # overlap or touching
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
    return merged


def _strip_empty_label_lines(text: str) -> str:
    """Remove lines that only contain a label with no value or only punctuation after deletions."""
    label_prefix = re.compile(r"^\s*(Email|Phone|Address|Passport\s*No|LinkedIn|GitHub)\s*:\s*[\W_]*$", re.IGNORECASE)
    lines = text.splitlines()
    kept = []
    for line in lines:
        if label_prefix.match(line.strip()):
            # drop the line entirely
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    # Normalize excessive blank lines and trailing spaces
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def redact_pii(text, detections):
    """Deterministically remove PII spans from text (delete, no masking)."""
    # Normalize hidden/unicode spaces that break regex matches (e.g., ZWJ/ZWNJ/NBSP/thin spaces)
    try:
        hidden_space_chars = [
            '\u200b',  # zero-width space
            '\u200c',  # zero-width non-joiner
            '\u200d',  # zero-width joiner
            '\ufeff',  # BOM
            '\u00a0',  # non-breaking space
            '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a'  # en/em/thin spaces
        ]
        for ch in hidden_space_chars:
            text = text.replace(ch, ' ')
        # Collapse repeated spaces
        text = re.sub(r'[ \t]{2,}', ' ', text)
    except Exception:
        pass

    # Capture common section headers and their first following content line as anchors.
    # We'll use these anchors to reinsert headers at the correct location if removed.
    try:
        header_names = (
            r'(Education|Experience|Projects|Work\s*Experience|Professional\s*Experience|'
            r'Activities?|C[O0][-\s]*Curricular(?:\s*Activities)?|Extra[-\s]*Curricular|'
            r'Tech(?:nical|inical)(?:\s+Skills)?|Skills|Certifications?|Achievements?|'
            r'Objective|Summary|Academics?|Area\s*of\s*(?:Interest|Intrest)|Interests?)'
        )
        header_pat = re.compile(rf'(?im)^\s*(?:{header_names})\s*:?\s*$', re.IGNORECASE)
        lines = text.splitlines()
        section_markers: list[dict] = []
        seen = set()
        def _normalize_hdr(s: str) -> str:
            s_low = s.lower().strip()
            if 'work' in s_low and 'experience' in s_low:
                return 'Experience:'
            if 'professional' in s_low and 'experience' in s_low:
                return 'Experience:'
            if (s_low.startswith('co') or s_low.startswith('c0')) and 'curricular' in s_low:
                return 'Co-Curricular:'
            if 'extra' in s_low and 'curricular' in s_low:
                return 'Extra Curricular:'
            if s_low.startswith('area of interest') or s_low.startswith('area of intrest') or ('area' in s_low and ('interest' in s_low or 'intrest' in s_low)):
                return 'Area of Interest:'
            # Default: Title-case and append colon
            return f"{s.title()}:"
        for idx, line in enumerate(lines):
            m = header_pat.match(line.strip())
            if not m:
                continue
            raw_hdr = m.group(0).strip().rstrip(':')
            norm_hdr = _normalize_hdr(raw_hdr)
            # Find the first non-empty content line after the header to anchor insertion
            anchor = ""
            for j in range(idx + 1, min(idx + 8, len(lines))):  # look ahead a few lines only
                nxt = lines[j].strip()
                if nxt:
                    anchor = nxt[:120]
                    break
            key = (norm_hdr.lower(), anchor.lower())
            if key in seen:
                continue
            seen.add(key)
            section_markers.append({"header": norm_hdr, "anchor": anchor})
    except Exception:
        section_markers = []

    # Merge spans first to avoid partial artifacts
    merged_spans = _merge_spans(detections)
    redacted = text
    # Delete from end to start to avoid offset math
    for start, end in reversed(merged_spans):
        redacted = redacted[:start] + redacted[end:]
    # Clean up label-only lines
    redacted = _strip_empty_label_lines(redacted)
    # Post-pass cleanup: catch obfuscated social/profile slugs that might remain
    try:
        cleanup_specs = [
            # LinkedIn
            (r'(?i)(?:(?<=\s)|^)/?linkedin\s*[\s:/-]*[A-Za-z0-9][A-Za-z0-9\-_/]{2,}\b', ''),
            (r'(?i)(?:(?<=\s)|^)/?linkedin[A-Za-z0-9\-_/]{3,}\b', ''),
            # GitHub (OCR variants too)
            (r'(?i)(?:(?<=\s)|^)(?:git\s*hub|github)\s*[:|/\\-]*\s*[@/]?[A-Za-z0-9][A-Za-z0-9\-]{1,39}\b', ''),
            (r'(?i)\b(?:https?://(?:www\.)?github\.com|github\.com)/[A-Za-z0-9\-_/\.]+\b', ''),
            # LeetCode
            (r'(?i)(?:(?<=\s)|^)(?:leet\s*code|leetcode)\s*[:|/\\-]*\s*[@/]?[A-Za-z0-9][A-Za-z0-9_\-]{1,39}\b', ''),
            (r'(?i)\b(?:https?://(?:www\.)?leetcode\.com|leetcode\.com)/(?:u/)?[A-Za-z0-9_\-/.]+\b', ''),
            # GeeksForGeeks / GFG
            (r'(?i)(?:(?<=\s)|^)(?:geeks\s*for\s*geeks|geeksforgeeks|gfg)\s*[:|/\\-]*\s*[@/]?[A-Za-z0-9][A-Za-z0-9_\-]{1,39}\b', ''),
            (r'(?i)\b(?:https?://(?:www\.)?geeksforgeeks\.org|geeksforgeeks\.org)/user/[A-Za-z0-9_\-/.]+\b', ''),
            # Inline platform/handle pairs anywhere in a line (not only at start)
            (r'(?i)\b(?:git\s*hub|github)\s*/\s*[@/]?[A-Za-z0-9][A-Za-z0-9\-]{1,39}\b', ''),
            (r'(?i)\b(?:leet\s*code|leetcode)\s*/\s*[@/]?[A-Za-z0-9][A-Za-z0-9_\-]{1,39}\b', ''),
            (r'(?i)\b(?:geek[s]?\s*for\s*geeks|geeksforgeeks|gfg)\s*/\s*[@/]?[A-Za-z0-9][A-Za-z0-9_\-]{1,39}\b', ''),
            # Platform followed by handle without slash (within 0-20 chars)
            (r'(?i)(?:github|git\s*hub)\s*[:\- ]{0,3}[\s\S]{0,20}?\b[@/]?[A-Za-z][A-Za-z0-9_\-]{2,39}\b', ''),
            (r'(?i)(?:leetcode|leet\s*code)\s*[:\- ]{0,3}[\s\S]{0,20}?\b[@/]?[A-Za-z][A-Za-z0-9_\-]{2,39}\b', ''),
            (r'(?i)(?:geek[s]?\s*for\s*geeks|geeksforgeeks|gfg)\s*[:\- ]{0,3}[\s\S]{0,20}?\b[@/]?[A-Za-z][A-Za-z0-9_\-]{2,39}\b', ''),
            # Bare domains (no scheme) — appended conditionally based on DOMAIN_REMOVAL_STRICTNESS
            # Handle-like tokens ending with '-official' (allow optional preceding slash)
            (r'(?i)(?:(?<=\s)|^|/)[A-Za-z0-9][A-Za-z0-9\-]{1,62}-official\b', ''),
            # Parenthesized ALL-CAPS name tokens like '(HARIKISHORE S)'; keep case-sensitive via (?-i)
            (r'(?-i)\((?:[A-Z]{2,}(?:\s+[A-Z]{1,2}\.?))(?:\s+[A-Z]{2,}){0,2}\)', ''),
            # Slug immediately followed by the words 'Linked In' (e.g., 'balakumar-m-d-0451b023aLinked In')
            (r'(?i)\b[A-Za-z][A-Za-z0-9\-]{8,}\s*Linked\s*In\b', ''),
        ]
        # Conditionally apply bare domain removal
        if DOMAIN_REMOVAL_STRICTNESS in ("normal", "strict"):
            cleanup_specs.append((BARE_DOMAIN_RE, ''))
        for pat, repl in cleanup_specs:
            redacted = re.sub(pat, repl, redacted, flags=re.IGNORECASE)
    except Exception:
        pass

    # Scrub occurrences of detected Name tokens/phrases across the text (inline),
    # including spaced-letter variants. This aggressive pass is disabled by default
    # to avoid corrupting content like email local parts. Enable via AGGRESSIVE_NAME_SCRUB=true.
    try:
        import os as _os
        if _os.getenv('AGGRESSIVE_NAME_SCRUB', 'false').lower() == 'true':
            name_texts = [d['text'] for d in detections if d.get('type') == 'Name' and d.get('text')]
            tokens: set[str] = set()
            phrases: set[str] = set()
            for nt in name_texts:
                # normalize whitespace
                phrase = re.sub(r"\s+", " ", nt.strip())
                if phrase:
                    phrases.add(phrase)
                for t in re.split(r"[^A-Za-z]+", phrase):
                    if len(t) >= 3:  # ignore 1-2 letter tokens
                        tokens.add(t)
            # Full phrase removal
            for ph in sorted(phrases, key=len, reverse=True):
                pat = rf"(?i)(?<![\w@./-]){re.escape(ph)}(?![\w@./-])"
                redacted = re.sub(pat, "", redacted)
            # Token removal and spaced-letter variants
            for tk in tokens:
                pat_tok = rf"(?i)(?<![\w@./-]){re.escape(tk)}(?![\w@./-])"
                redacted = re.sub(pat_tok, "", redacted)
                letters = "\s*".join(list(tk))
                pat_spaced = rf"(?i)(?<![\w@./-]){letters}(?![\w@./-])"
                redacted = re.sub(pat_spaced, "", redacted)
    except Exception:
        pass

    # Remove orphaned label-only lines/fragments like "obile /envel" (from Mobile/Envelope icons)
    try:
        # 1) Drop entire lines that contain only labels and separators
        label_only_line = r'(?im)^\s*(?:m?obile|phone|contact|email|envel(?:ope)?|linkedin|linked\s*in|github|address|passport(?:\s*no)?|aadhaar|aadhar|links?|portfolio|national\s*insurance\s*(?:number|no)?|father\'?s\s*name)\s*(?:[:|/,-]\s*(?:m?obile|phone|contact|email|envel(?:ope)?|linkedin|linked\s*in|github|address|passport(?:\s*no)?|aadhaar|aadhar|links?|portfolio|national\s*insurance\s*(?:number|no)?|father\'?s\s*name)\s*)*$\n?'
        redacted = re.sub(label_only_line, '', redacted)
        # 2) Remove near-empty label lines like 'Email:' or with <=3 non-space chars after label
        near_empty_label = r'(?im)^\s*(?:email|m?obile|phone|contact|linkedin|linked\s*in|github|address|passport(?:\s*no)?|aadhaar|aadhar|links?|portfolio|national\s*insurance\s*(?:number|no)?|father\'?s\s*name)\s*[:\-]\s*(?:\S{0,3})?\s*$\n?'
        redacted = re.sub(near_empty_label, '', redacted)
        # Also remove dangling label fragments at end of lines (no value after)
        redacted = re.sub(r'(?im)(?:m?obile|email|envel(?:ope)?|linkedin|linked\s*in|github|address|passport|aadhaar|aadhar|links?|portfolio|national\s*insurance\s*(?:number|no)?|father\'?s\s*name)\s*(?:[:|/,-]\s*)*(?=\n|$)', '', redacted)
        # 3) Collapse leftover separators at line ends
        redacted = re.sub(r'(?m)[\s|/,:;-]+(?=\n|$)', '', redacted)
        # 4) Clean multiple blank lines
        redacted = re.sub(r'\n{3,}', '\n\n', redacted)
    except Exception:
        pass

    # Drop entire lines that still contain PII values with explicit labels after spans were removed
    try:
        # Lines with explicit labels (more strict delete)
        pii_label_line = r'(?im)^\s*(?:email|e-mail|m?obile|phone|contact|linkedin|linked\s*in|github|address|passport(?:\s*no\.? )?|aadhaar|aadhar|links?|portfolio|national\s*insurance\s*(?:number|no\.? )|father\'?s\s*name)\s*[:\-].*$\n?'
        redacted = re.sub(pii_label_line, '', redacted)
        # Heuristic: Indian address-like lines containing a house number (e.g., 15/94) and a locality keyword
        indian_addr = r'(?im)^.*\b\d{1,4}(?:[\/-]\d{1,4})\b.*\b(?:nagar|road|rd\.?|street|st\.?|colony|layout|lane|cross|phase|sector|block|main|avenue|av\.)\b.*$\n?'
        redacted = re.sub(indian_addr, '', redacted)
    except Exception:
        pass

    # Remove header lines that are just ALL-CAPS name tokens (2–4 tokens),
    # e.g., 'SHESHANTH R S' or 'SHESHANTH R S Design Portfolio'
    try:
        caps_name_line = r'(?im)^\s*(?:[A-Z]{2,}|[A-Z])(?:\s+(?:[A-Z]{2,}|[A-Z])){1,3}\s*(?:Design\s+Portfolio)?\s*$\n?'
        redacted = re.sub(caps_name_line, '', redacted)
    except Exception:
        pass

    # Remove header lines that are just spaced uppercase letters composing a name, e.g., 'H A R I K I S H O R E S'
    try:
        spaced_caps_line = r'(?im)^\s*\(?(?:[A-Z]\s+){4,}[A-Z]\)?\s*$\n?'
        redacted = re.sub(spaced_caps_line, '', redacted)
    except Exception:
        pass

    # Remove inline parenthetical all-caps names like '(HARIKISHORE S)' or '( H A R I K I S H O R E S )'
    try:
        parenthetical_caps = r'\(\s*(?:[A-Z]{2,}(?:\s+[A-Z]{2,}){0,3}|(?:[A-Z]\s+){2,}[A-Z])\s*\)'
        redacted = re.sub(parenthetical_caps, '', redacted)
    except Exception:
        pass

    # Remove orphan username line like '/Sangeethraj' left by OCR line breaks
    try:
        orphan_handle = r'(?im)^\s*/[A-Za-z0-9][A-Za-z0-9_\-]{1,39}\s*$\n?'
        redacted = re.sub(orphan_handle, '', redacted)
    except Exception:
        pass

    # Aggressive catch-all: lines that are just <Platform(optional)> '/' <handle>
    # Examples:
    #   'GeekForGeeks/Sangeethraj'
    #   'Git hub / Sangeethraj'
    #   'Leet Code: /Sangeethraj'
    try:
        # Require an explicit platform prefix or an http(s) URL; do NOT match generic single-word headings.
        platform_handle_line = (
            r'(?im)^\s*(?:'
            r'https?://\S+'
            r'|(?:linkedin|git\s*hub|github|leet\s*code|leetcode|geeks\s*for\s*geeks|geeksforgeeks|gfg)'
            r'\s*[:|/\\-]*\s*/?[@]?[A-Za-z0-9][A-Za-z0-9_\-]{1,39}'
            r')\s*$\n?'
        )
        redacted = re.sub(platform_handle_line, '', redacted)
    except Exception:
        pass

    # Remove lines that are just 1-4 handle-like tokens (e.g., 'arun-kumar-s Arun-Kumar2003')
    try:
        # A handle-like token: contains a hyphen (slug) or ends with digits (username pattern)
        handle_like = r'(?:[A-Za-z][A-Za-z0-9_]*-[A-Za-z0-9_\-]+|[A-Za-z][A-Za-z0-9_\-]*\d{1,4})'
        multi_handles_line = rf'(?im)^\s*(?:{handle_like})(?:\s+{handle_like}){{0,3}}\s*$\n?'
        redacted = re.sub(multi_handles_line, '', redacted)
    except Exception:
        pass

    # If a line contains a phone/email and an adjacent username-like token (e.g., 'arun-kumar-s' or 'Arun-Kumar2003'), redact the token
    try:
        def _redact_adjacent_handles(m):
            line = m.group(0)
            handle_pat = re.compile(r'(?i)(?<![A-Za-z0-9_])[@/]?[A-Za-z][A-Za-z0-9_\-]{2,39}(?:\d{1,4})?(?![A-Za-z0-9_])')
            # Remove up to 3 handle-like tokens on that line
            cnt = 0
            def repl(hm):
                nonlocal cnt
                cnt += 1
                return '' if cnt <= 3 else hm.group(0)
            return handle_pat.sub(repl, line)
        contact_context = re.compile(r'(?im)^.*?(?:\b\d{5}\s*\d{5}\b|\b\d{10}\b|@[A-Za-z0-9_.-]+).*$', re.MULTILINE)
        redacted = contact_context.sub(_redact_adjacent_handles, redacted)
    except Exception:
        pass

    # Strict mode: aggressively scrub handle-like tokens globally (protect emails/URLs)
    try:
        import os as _os2
        if _os2.getenv('REDACT_STRICT', 'false').strip().lower() in {"1","true","yes","on"}:
            strict_patterns = [
                # Hyphenated slugs like 'arun-kumar-s', allow up to 3 hyphen groups
                r'(?i)(?<![\w@./-])[A-Za-z][A-Za-z0-9_]*-(?:[A-Za-z0-9_\-]+)(?:-[A-Za-z0-9_\-]+){0,2}(?![\w@./-])',
                # Usernames ending with digits like 'Arun-Kumar2003' or 'ArunKumar2003'
                r'(?i)(?<![\w@./-])[A-Za-z][A-Za-z0-9_\-]{1,20}\d{2,4}(?![\w@./-])',
            ]
            for pat in strict_patterns:
                redacted = re.sub(pat, '', redacted)
            # Tidy spaces after strict removals
            redacted = re.sub(r'[ \t]{2,}', ' ', redacted)
            redacted = re.sub(r'\n{3,}', '\n\n', redacted)
    except Exception:
        pass

    # Note: We no longer perform a global sweep of hyphenated/digit username-like tokens
    # to avoid deleting legitimate words like 'AI-based'. Handle social handles via targeted rules above.
    
    # Reinsert missing headers at the right place using the anchor line
    try:
        for sm in section_markers:
            hdr = sm.get("header", "").strip()
            anc = sm.get("anchor", "").strip()
            if not hdr:
                continue
            # If header already present, skip
            if re.search(rf'(?im)^\s*{re.escape(hdr)}\s*$', redacted):
                continue
            # If we have an anchor, try to insert header immediately before the line containing it
            if anc:
                # Build a relaxed regex for the anchor: collapse spaces
                prefix = re.escape(anc[:40])  # use a prefix to be robust
                prefix = prefix.replace(r'\ ', r'\s+')
                m = re.search(prefix, redacted, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    # Find start of the matched line
                    line_start = redacted.rfind('\n', 0, m.start())
                    insert_pos = 0 if line_start == -1 else line_start + 1
                    redacted = redacted[:insert_pos] + f"{hdr}\n" + redacted[insert_pos:]
                    continue
            # Fallback: if no anchor found, append header at end (less intrusive than top)
            redacted = redacted.rstrip() + f"\n\n{hdr}\n"
    except Exception:
        pass
    return redacted


def extract_text_from_pdf(file_storage) -> str:
    """Extract text from an uploaded PDF (Werkzeug FileStorage).
    1) Try text layer via PyPDF2.
    2) If empty and Gemini configured, render pages and use Gemini Vision to extract text.
    3) If Gemini not available or fails, fall back to OCR with pdf2image + Tesseract.
    Returns the extracted text. Sets a local variable 'method' to indicate extraction method.
    """
    # Ensure we have bytes
    try:
        stream = file_storage.stream
        try:
            stream.seek(0)
        except Exception:
            pass
        pdf_bytes = stream.read()
    except Exception as e:
        print(f"[ERROR] Reading PDF bytes failed: {e}")
        return ""

    # Attempt text extraction via PyPDF2
    text_out = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        text_out = "\n".join(t for t in texts if t)
    except Exception as e:
        print(f"[WARN] PyPDF2 failed, will try OCR: {e}")

    # Evaluate quality and possibly try alternatives
    if text_out.strip():
        norm_text = _normalize_text(text_out)
        score = _text_quality_score(norm_text)
        if score >= 0.95:  # stricter: require very clean text-layer
            extract_text_from_pdf._last_method = 'text-layer'
            return norm_text
        else:
            print(f"[INFO] PyPDF2 text quality low ({score:.2f}); trying pdfminer/Tesseract")

    # Try pdfminer.six for more robust text-layer extraction
    try:
        pm_text = pdfminer_extract_text(io.BytesIO(pdf_bytes)) or ""
        if pm_text.strip():
            pm_norm = _normalize_text(pm_text)
            pm_score = _text_quality_score(pm_norm)
            if pm_score >= 0.92:  # prefer OCR unless pdfminer is quite clean
                extract_text_from_pdf._last_method = 'pdfminer'
                return pm_norm
            else:
                print(f"[INFO] pdfminer text quality low ({pm_score:.2f}); will try OCR")
    except Exception as e:
        print(f"[WARN] pdfminer.six extraction failed: {e}")

    # Gemini Vision OCR fallback (preferred over Tesseract if available)
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
        if GEMINI_API_KEY and USE_GEMINI:
            gem_text = _gemini_ocr_images(images)
            if gem_text.strip():
                gem_norm = _normalize_text(gem_text)
                extract_text_from_pdf._last_method = 'gemini-vision'
                return gem_norm
    except Exception as e:
        print(f"[WARN] Gemini Vision OCR stage failed: {e}")

    # Tesseract OCR final fallback
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
        ocr_texts = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = img.convert('RGB')
            try:
                ocr_texts.append(pytesseract.image_to_string(img) or "")
            except Exception as e:
                print(f"[WARN] Tesseract OCR failed on a page: {e}")
                continue
        ocr_join = "\n".join(t for t in ocr_texts if t)
        ocr_norm = _normalize_text(ocr_join)
        extract_text_from_pdf._last_method = 'tesseract-ocr'
        return ocr_norm
    except Exception as e:
        print(f"[ERROR] OCR fallback failed: {e}")
        return ""

def _gemini_ocr_images(images: List[Image.Image]) -> str:
    """Use Gemini Vision to extract plain text from a list of PIL Images.
    Returns concatenated text for all pages. Requires GEMINI_API_KEY.
    """
    if not (GEMINI_API_KEY and USE_GEMINI):
        return ""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        page_texts: List[str] = []
        for idx, img in enumerate(images, start=1):
            if not isinstance(img, Image.Image):
                img = img.convert('RGB')
            # Encode image to JPEG bytes
            buf = io.BytesIO()
            img.convert('RGB').save(buf, format='JPEG', quality=90)
            img_bytes = buf.getvalue()
            parts = [
                {"text": "Extract all readable text from this page image. Return plain text only, no markdown or explanations."},
                {"mime_type": "image/jpeg", "data": img_bytes},
            ]
            try:
                resp = model.generate_content(parts)
                txt = (resp.text or "").strip()
                page_texts.append(txt)
            except Exception as e:
                print(f"[WARN] Gemini image OCR failed on page {idx}: {e}")
                page_texts.append("")
        return "\n".join(t for t in page_texts if t)
    except Exception as e:
        print(f"[ERROR] Gemini OCR pipeline failed: {e}")
        return ""

def _spans_overlap(a: tuple, b: tuple) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _consolidate_detections(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping detections of the same type by keeping the longer span.
    If lengths tie, prefer the one whose text contains a label marker like 'Address:' or 'Email:'.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        t = d.get('type', 'Unknown')
        by_type.setdefault(t, []).append(d)
    consolidated: List[Dict[str, Any]] = []
    for t, items in by_type.items():
        items = sorted(items, key=lambda x: (int(x['start']), int(x['end'])))
        kept: List[Dict[str, Any]] = []
        for cur in items:
            cur_span = (int(cur['start']), int(cur['end']))
            replaced = False
            for i, prev in enumerate(list(kept)):
                prev_span = (int(prev['start']), int(prev['end']))
                if _spans_overlap(cur_span, prev_span):
                    cur_len = cur_span[1] - cur_span[0]
                    prev_len = prev_span[1] - prev_span[0]
                    # Decide which to keep
                    def has_label(txt: str) -> bool:
                        return any(lbl in txt for lbl in [":", "Email", "Phone", "Address", "Passport", "LinkedIn", "GitHub"])  # simple heuristic
                    cur_score = (cur_len, 1 if has_label(cur.get('text', '')) else 0)
                    prev_score = (prev_len, 1 if has_label(prev.get('text', '')) else 0)
                    if cur_score >= prev_score:
                        kept[i] = cur
                    replaced = True
                    break
            if not replaced:
                kept.append(cur)
        consolidated.extend(kept)
    # Sort consolidated by start position for stable output
    return sorted(consolidated, key=lambda x: (int(x['start']), int(x['end'])))

# HTML template with inline CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔒 PII Sentinel - Personal Information Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-section h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        textarea {
            width: 100%;
            min-height: 150px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #4facfe;
        }
        
        .button-group {
            margin: 20px 0;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .results-section {
            margin-top: 30px;
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .results-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        .results-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .results-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .pii-type {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            color: white;
            background: #007bff !important; /* Force all pills to the Email blue */
        }
        
        /* Specific classes retained but overridden by base .pii-type */
        .pii-phone { background: #28a745; }
        .pii-email { background: #007bff; }
        .pii-ssn { background: #dc3545; }
        .pii-credit { background: #fd7e14; }
        
        .confidence {
            font-weight: 600;
            color: #28a745;
        }
        
        .redacted-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #4facfe;
        }
        
        .redacted-text {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-family: monospace;
            white-space: pre-wrap;
            border: 1px solid #e0e0e0;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .stat-card {
            flex: 1;
            min-width: 150px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4facfe;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .main-content {
                padding: 20px;
            }
            
            .button-group {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
                text-align: center;
            }
            
            .stats {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 PII Sentinel</h1>
            <p>Advanced Personal Information Detection & Redaction System</p>
            <!-- Benchmark button removed -->
        </div>
        
        <div class="main-content">
            <div class="input-section">
                <h3>📝 Text or PDF to Analyze</h3>
                <div style="display:flex; gap:12px; align-items:center; margin-bottom:10px; flex-wrap:wrap;">
                    <input type="file" id="pdfFile" accept="application/pdf" />
                    <small style="color:#555">Optional: choose a PDF (if selected, PDF will be analyzed instead of the text box)</small>
                </div>
                <textarea id="inputText" placeholder="Paste your text here to detect personally identifiable information (PII)...

Try this sample text:
Hello, my name is John Smith and I work at Acme Corp. You can reach me at john.smith@email.com or call me at (555) 123-4567. My SSN is 123-45-6789 and my credit card number is 4532-1234-5678-9012."></textarea>
                
                <div class="button-group">
                    <button class="btn btn-secondary" onclick="redactPII()">🚫 Redact PII</button>
                    <button class="btn" onclick="clearAll()" style="background: #6c757d; color: white;">🗑️ Clear All</button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing text for PII...</p>
            </div>
            
            <!-- stats removed -->
            
            <!-- detection results removed -->
            
            <div class="redacted-section" id="redactedSection" style="display: none;">
                <h3>🚫 Redacted Text <span id="redactMethodBadge" style="margin-left:8px; font-size:12px; background:#e9f2ff; color:#0b5ed7; padding:3px 8px; border-radius:12px; display:none;"></span></h3>
                <p>Personal information has been replaced with placeholders:</p>
                <div class="redacted-text" id="redactedText"></div>
            </div>
        </div>
    </div>

    <script>
        // detection feature removed
        
        // detectPII removed
        
        // displayResults removed
        
        async function redactPII() {
            const text = document.getElementById('inputText').value.trim();
            const fileInput = document.getElementById('pdfFile');
            const pdfFile = fileInput && fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
            if (!text && !pdfFile) {
                alert('Please enter text or choose a PDF to redact');
                return;
            }
            try {
                let response;
                if (pdfFile) {
                    const form = new FormData();
                    form.append('file', pdfFile);
                    response = await fetch('/api/redact_pdf', { method: 'POST', body: form });
                } else {
                    response = await fetch('/api/redact', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                }
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('redactedText').textContent = result.redacted_text;
                    document.getElementById('redactedSection').style.display = 'block';
                    const m = result.extraction_method || (pdfFile ? 'unknown' : 'text-input');
                    const badge = document.getElementById('redactMethodBadge');
                    if (badge) { badge.textContent = `source: ${m}`; badge.style.display = 'inline-block'; }
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error redacting PII: ' + error.message);
            }
        }
        
        function clearAll() {
            document.getElementById('inputText').value = '';
            document.getElementById('redactedSection').style.display = 'none';
            // stats and detection removed
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page with PII detection interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Feature removed: Detect PII endpoint disabled."""
    return jsonify({'success': False, 'error': 'Detect PII feature has been removed'}), 410


@app.route('/api/audit/export', methods=['GET'])
def api_audit_export():
    """Export audit log to CSV in data/ and return path."""
    try:
        from audit_logger import export_csv  # local import to avoid cycles
        ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        path = os.path.join(os.path.dirname(__file__), 'data', f'audit-{ts}.csv')
        out = export_csv(path)
        return jsonify({'success': True, 'path': out})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/drift/eval', methods=['POST'])
def api_drift_eval():
    """Run a simple drift evaluation on provided labeled dataset.
    Body: { samples: [{text, label_entities:[{start,end,type}], predicted_entities?}], threshold_fp: 0.10 }
    """
    try:
        from drift_monitor import evaluate_dataset, propose_regex_improvements, write_mock_pr
        data = request.get_json() or {}
        samples = data.get('samples', [])
        threshold_fp = float(data.get('threshold_fp', 0.10))

        metrics, fp_samples = evaluate_dataset(samples, detector=lambda t: detect_pii_llm(t) or detect_pii(t))
        suggestions = []
        pr_path = None
        if metrics.get('fp_rate', 0.0) > threshold_fp and fp_samples:
            suggestions = propose_regex_improvements(fp_samples)
            pr_path = write_mock_pr(suggestions)

        return jsonify({'success': True, 'metrics': metrics, 'suggestions': suggestions, 'pr_path': pr_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/redact', methods=['POST'])
def api_redact():
    """API endpoint for PII redaction"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        _client_dets = data.get('detections', [])  # ignored for robustness
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        # Recompute detections server-side to ensure correct spans
        server_llm = detect_pii_llm(text)
        server_rgx = detect_pii(text)
        # Merge unique by span
        key = lambda d: (d['start'], d['end'])
        merged_map = {key(d): d for d in server_rgx}
        for d in server_llm:
            merged_map[key(d)] = d
        server_detections = _consolidate_detections(list(merged_map.values()))
        # Deterministic deletion-based redaction (no placeholders)
        action = "REDACT"
        redacted_text = redact_pii(text, server_detections)
        justification = "Removed PII spans by type per policy (no placeholders)."
        # Build audit extras
        try:
            summary = {
                "counts_by_type": _count_by_type(server_detections),
                "redactions_made": len(_merge_spans(server_detections)),
                "domain_removal_strictness": DOMAIN_REMOVAL_STRICTNESS,
                "bare_domain_hits_in_input": _count_pattern(text, BARE_DOMAIN_RE),
                "official_handle_hits_in_input": _count_pattern(text, OFFICIAL_HANDLE_RE),
            }
            extra_json = json.dumps(summary, ensure_ascii=False)
        except Exception:
            extra_json = None
        try:
            log_decision(
                action,
                snippet=text[:500],
                detections_json=json.dumps(server_detections, ensure_ascii=False),
                redacted_text=redacted_text,
                justification=justification,
                source="api/redact",
                model=GEMINI_MODEL if GEMINI_API_KEY else "regex-fallback",
                extra=extra_json,
            )
        except Exception as e:
            print(f"[WARN] audit log write failed: {e}")

        return jsonify({
            'success': True,
            'original_text': text,
            'redacted_text': redacted_text,
            'action': action,
            'justification': justification,
            'redactions_made': len(_merge_spans(server_detections)),
            'extraction_method': 'text-input'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/index', methods=['POST'])
def api_index():
    """Index documents in Pinecone: expects { items: [{id, text, metadata}] }"""
    if pinecone_store is None:
        return jsonify({'success': False, 'error': 'Pinecone not configured'}), 400
    try:
        data = request.get_json() or {}
        items = data.get('items', [])
        ids = [it['id'] for it in items]
        texts = [it['text'] for it in items]
        metas = [it.get('metadata', {}) for it in items]
        vectors = embed_texts(texts)
        pinecone_store.upsert_texts(ids, vectors, metas)
        return jsonify({'success': True, 'upserted': len(ids)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def api_search():
    """Semantic search via Pinecone: expects { query: str, top_k?: int }"""
    if pinecone_store is None:
        return jsonify({'success': False, 'error': 'Pinecone not configured'}), 400
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        top_k = int(data.get('top_k', 5))
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        vec = embed_text(query)
        res = pinecone_store.query(vec, top_k=top_k)
        return jsonify({'success': True, 'matches': res.get('matches', []), 'namespace': res.get('namespace')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect_pdf', methods=['POST'])
def api_detect_pdf():
    """Feature removed: Detect PII endpoint for PDFs disabled."""
    return jsonify({'success': False, 'error': 'Detect PII feature has been removed'}), 410


@app.route('/api/redact_pdf', methods=['POST'])
def api_redact_pdf():
    """Redact PII from uploaded PDF and return redacted text."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        file = request.files['file']
        text = extract_text_from_pdf(file)
        if not text:
            return jsonify({'success': False, 'error': 'Could not extract text from PDF'}), 400
        # Recompute detections server-side
        server_llm = detect_pii_llm(text)
        server_rgx = detect_pii(text)
        key = lambda d: (d['start'], d['end'])
        merged_map = {key(d): d for d in server_rgx}
        for d in server_llm:
            merged_map[key(d)] = d
        server_detections = _consolidate_detections(list(merged_map.values()))
        redacted_text = redact_pii(text, server_detections)
        # Build audit extras
        try:
            summary = {
                "counts_by_type": _count_by_type(server_detections),
                "redactions_made": len(_merge_spans(server_detections)),
                "domain_removal_strictness": DOMAIN_REMOVAL_STRICTNESS,
                "bare_domain_hits_in_input": _count_pattern(text, BARE_DOMAIN_RE),
                "official_handle_hits_in_input": _count_pattern(text, OFFICIAL_HANDLE_RE),
            }
            extra_json = json.dumps(summary, ensure_ascii=False)
        except Exception:
            extra_json = None
        try:
            log_decision(
                "REDACT",
                snippet=text[:500],
                detections_json=json.dumps(server_detections, ensure_ascii=False),
                redacted_text=redacted_text,
                justification="Removed PII spans by type per policy (no placeholders).",
                source="api/redact_pdf",
                model=GEMINI_MODEL if GEMINI_API_KEY else "regex-fallback",
                extra=extra_json,
            )
        except Exception as e:
            print(f"[WARN] audit log write failed: {e}")
        return jsonify({'success': True, 'original_text': text, 'redacted_text': redacted_text, 'action': 'REDACT', 'redactions_made': len(_merge_spans(server_detections)), 'extraction_method': getattr(extract_text_from_pdf, '_last_method', 'pdf')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 PII Sentinel Web Interface Starting...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("✅ Server ready!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
