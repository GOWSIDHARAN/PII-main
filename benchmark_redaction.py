import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional

# Import detection/redaction from web_app without starting Flask app
import importlib.util
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app import detect_pii, detect_pii_llm, redact_pii  # type: ignore


def consolidate_detections(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Local minimal consolidation (unique by (start,end,type,text))
    seen = set()
    out: List[Dict[str, Any]] = []
    for d in dets:
        key = (int(d.get("start", -1)), int(d.get("end", -1)), d.get("type", ""), d.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def load_labels(label_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            return data.get("labels", []) or data.get("pii_entities", []) or []
        if isinstance(data, list):
            return data
        return []


def span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return (inter / union) if union > 0 else 0.0


def eval_prf(
    gold: List[Dict[str, Any]],
    pred: List[Dict[str, Any]],
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    # Match by type and IoU overlap
    gold_used = [False] * len(gold)
    tp = 0
    fp = 0
    fn = 0
    for p in pred:
        p_type = p.get("type")
        p_span = (int(p.get("start", -1)), int(p.get("end", -1)))
        matched = False
        for i, g in enumerate(gold):
            if gold_used[i]:
                continue
            if g.get("type") != p_type:
                continue
            g_span = (int(g.get("start", -1)), int(g.get("end", -1)))
            if span_iou(p_span, g_span) >= iou_thr:
                gold_used[i] = True
                matched = True
                tp += 1
                break
        if not matched:
            fp += 1
    fn = gold_used.count(False)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def process_file(path: str, use_llm: bool) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    t0 = time.perf_counter()
    det_rgx = detect_pii(text)
    det_llm = detect_pii_llm(text) if use_llm else []
    # Merge
    merged_map = {(int(d['start']), int(d['end'])): d for d in det_rgx}
    for d in det_llm:
        merged_map[(int(d['start']), int(d['end']))] = d
    detections = consolidate_detections(list(merged_map.values()))
    t1 = time.perf_counter()
    redacted = redact_pii(text, detections)
    t2 = time.perf_counter()
    return {
        "text": text,
        "detections": detections,
        "redacted": redacted,
        "detect_time_ms": (t1 - t0) * 1000,
        "redact_time_ms": (t2 - t1) * 1000,
        "total_time_ms": (t2 - t0) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PII redaction speed and accuracy")
    parser.add_argument("--input", required=True, help="Input directory containing .txt files")
    parser.add_argument("--ext", default=".txt", help="File extension to scan (default: .txt)")
    parser.add_argument("--use-llm", action="store_true", help="Include LLM detections if configured")
    parser.add_argument("--metrics-out", default=None, help="Path to write JSON summary metrics")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for matching spans")
    args = parser.parse_args()

    files = [
        os.path.join(args.input, fn)
        for fn in os.listdir(args.input)
        if fn.lower().endswith(args.ext.lower())
    ]
    files.sort()

    results = []
    total_detect_ms = 0.0
    total_redact_ms = 0.0
    total_total_ms = 0.0

    prf_agg = {"tp": 0, "fp": 0, "fn": 0}

    for fp in files:
        res = process_file(fp, use_llm=args.use_llm)
        results.append({
            "file": os.path.basename(fp),
            "detect_time_ms": res["detect_time_ms"],
            "redact_time_ms": res["redact_time_ms"],
            "total_time_ms": res["total_time_ms"],
            "detections": len(res["detections"]),
        })
        total_detect_ms += res["detect_time_ms"]
        total_redact_ms += res["redact_time_ms"]
        total_total_ms += res["total_time_ms"]

        # Optional accuracy if sidecar labels exist: <file>.labels.json
        label_path = fp + ".labels.json"
        gold = load_labels(label_path)
        if gold:
            prf = eval_prf(gold, res["detections"], iou_thr=args.iou_thr)
            prf_agg["tp"] += prf["tp"]
            prf_agg["fp"] += prf["fp"]
            prf_agg["fn"] += prf["fn"]
            print(f"[ACC] {os.path.basename(fp)}: P={prf['precision']:.3f} R={prf['recall']:.3f} F1={prf['f1']:.3f}")

        print(f"[SPEED] {os.path.basename(fp)}: detect={res['detect_time_ms']:.2f}ms redact={res['redact_time_ms']:.2f}ms total={res['total_time_ms']:.2f}ms detections={len(res['detections'])}")

    n = max(1, len(files))
    avg_detect = total_detect_ms / n
    avg_redact = total_redact_ms / n
    avg_total = total_total_ms / n

    summary: Dict[str, Any] = {
        "files": len(files),
        "avg_detect_time_ms": avg_detect,
        "avg_redact_time_ms": avg_redact,
        "avg_total_time_ms": avg_total,
        "results": results,
    }

    if prf_agg["tp"] + prf_agg["fp"] + prf_agg["fn"] > 0:
        prec = prf_agg["tp"] / max(1, prf_agg["tp"] + prf_agg["fp"])
        rec = prf_agg["tp"] / max(1, prf_agg["tp"] + prf_agg["fn"])
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        summary["precision"] = prec
        summary["recall"] = rec
        summary["f1"] = f1

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote metrics to {args.metrics_out}")

    print("=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
