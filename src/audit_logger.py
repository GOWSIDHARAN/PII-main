import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional, Iterable, List
import json

DB_PATH = os.getenv("AUDIT_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "audit.db"))
DB_PATH = os.path.abspath(DB_PATH)


def _ensure_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT,
                snippet TEXT,
                decision TEXT NOT NULL,
                justification TEXT,
                detections_json TEXT,
                redacted_text TEXT,
                model TEXT,
                extra JSON,
                avg_recall REAL,
                avg_latency_ms REAL
            )
            """
        )
        # --- Lightweight migration for existing DBs: add columns if missing ---
        cur = conn.execute("PRAGMA table_info(decisions)")
        cols = {row[1] for row in cur.fetchall()}
        if "avg_recall" not in cols:
            conn.execute("ALTER TABLE decisions ADD COLUMN avg_recall REAL")
        if "avg_latency_ms" not in cols:
            conn.execute("ALTER TABLE decisions ADD COLUMN avg_latency_ms REAL")
        conn.commit()
    finally:
        conn.close()


def log_decision(
    decision: str,
    *,
    snippet: str,
    detections_json: str,
    redacted_text: Optional[str] = None,
    justification: Optional[str] = None,
    source: Optional[str] = None,
    model: Optional[str] = None,
    extra: Optional[str] = None,
) -> None:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        # Attempt to parse avg_recall / avg_latency_ms from extra JSON if provided
        avg_recall_val: Optional[float] = None
        avg_latency_val: Optional[float] = None
        if extra:
            try:
                ej = json.loads(extra)
                if isinstance(ej, dict):
                    ar = ej.get("avg_recall")
                    al = ej.get("avg_latency_ms")
                    avg_recall_val = float(ar) if ar is not None else None
                    avg_latency_val = float(al) if al is not None else None
            except Exception:
                pass

        conn.execute(
            "INSERT INTO decisions (ts, source, snippet, decision, justification, detections_json, redacted_text, model, extra, avg_recall, avg_latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(),
                source,
                snippet,
                decision,
                justification,
                detections_json,
                redacted_text,
                model,
                extra,
                avg_recall_val,
                avg_latency_val,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def export_csv(path: str) -> str:
    _ensure_db()
    rows: Iterable[tuple]
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute("SELECT ts, source, decision, model, snippet, detections_json, redacted_text, justification, extra, avg_recall, avg_latency_ms FROM decisions ORDER BY id DESC")
        rows = cur.fetchall()
    finally:
        conn.close()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "source", "decision", "model", "snippet", "detections_json", "redacted_text", "justification", "extra", "avg_recall", "avg_latency_ms"])
        for r in rows:
            writer.writerow(r)
    return os.path.abspath(path)


def fetch_recent_benchmarks(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent retrieval benchmark runs stored via log_decision.

    We treat rows with decision = 'RETRIEVAL_BENCH' (preferred) or
    source like '/bench' as benchmark runs. The 'extra' JSON is parsed
    and included as a dict under key 'extra'.
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows: List[Dict[str, Any]] = []
    try:
        cur = conn.execute(
            """
            SELECT id, ts, source, decision, model, snippet, detections_json, redacted_text, justification, extra
            FROM decisions
            WHERE decision = 'RETRIEVAL_BENCH' OR source = '/bench'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        for r in cur.fetchall():
            item = {k: r[k] for k in r.keys()}
            try:
                import json as _json
                item["extra"] = _json.loads(item.get("extra") or "null")
            except Exception:
                pass
            rows.append(item)
    finally:
        conn.close()
    return rows
