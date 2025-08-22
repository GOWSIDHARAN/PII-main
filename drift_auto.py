import os
import json
import time
import argparse
from typing import List, Dict, Any, Callable

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in (ROOT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from web_app import detect_pii, detect_pii_llm  # type: ignore
from drift_monitor import evaluate_dataset, propose_regex_improvements, write_mock_pr  # type: ignore


def load_samples(path: str) -> List[Dict[str, Any]]:
    """
    Expects JSON list of objects: {"text": str, "label_entities": [ {start,end,type}, ... ]}
    Or a dict with key "samples".
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("samples", [])
    return data


def make_detector(use_llm: bool) -> Callable[[str], List[Dict[str, Any]]]:
    def _detector(text: str) -> List[Dict[str, Any]]:
        rgx = detect_pii(text)
        if use_llm:
            llm = detect_pii_llm(text)
            # Unique by (start,end)
            mp = {(int(d['start']), int(d['end'])): d for d in rgx}
            for d in llm:
                mp[(int(d['start']), int(d['end']))] = d
            return list(mp.values())
        return rgx
    return _detector


def run_once(dataset_path: str, use_llm: bool, threshold_fp: float) -> Dict[str, Any]:
    samples = load_samples(dataset_path)
    det = make_detector(use_llm)
    metrics, fp_samples = evaluate_dataset(samples, det)
    result: Dict[str, Any] = {
        "metrics": metrics,
        "fp_samples": fp_samples[:10],  # cap in output
        "pr_path": None,
        "improvements": [],
    }
    if metrics.get("fp_rate", 0.0) > threshold_fp:
        suggestions = propose_regex_improvements(fp_samples)
        pr_path = write_mock_pr(suggestions)
        result["improvements"] = suggestions
        result["pr_path"] = pr_path
    return result


def main():
    ap = argparse.ArgumentParser(description="Drift automation: daily PRF eval + mock PR on FP drift")
    ap.add_argument("--dataset", required=True, help="Path to labeled dataset JSON")
    ap.add_argument("--use-llm", action="store_true", help="Include LLM detections if Gemini is configured")
    ap.add_argument("--threshold-fp", type=float, default=0.10, help="Trigger threshold for FP rate")
    ap.add_argument("--loop", action="store_true", help="Run continuously at intervals")
    ap.add_argument("--interval-min", type=float, default=1440.0, help="Loop interval in minutes (default: daily)")
    ap.add_argument("--out", default=None, help="Optional path to write latest JSON result")
    args = ap.parse_args()

    def emit(res: Dict[str, Any]):
        print(json.dumps(res, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

    if not args.loop:
        res = run_once(args.dataset, args.use_llm, args.threshold_fp)
        emit(res)
        return

    try:
        while True:
            res = run_once(args.dataset, args.use_llm, args.threshold_fp)
            emit(res)
            time.sleep(max(1.0, args.interval_min * 60.0))
    except KeyboardInterrupt:
        print("Drift automation stopped.")


if __name__ == "__main__":
    main()
