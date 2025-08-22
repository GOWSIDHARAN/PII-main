import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple

# Make local imports work
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in (ROOT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from embeddings import embed_text, embed_texts  # type: ignore
from pinecone_store import PineconeStore  # type: ignore


def load_corpus(corpus_dir: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for fn in os.listdir(corpus_dir):
        if not fn.lower().endswith(".txt"):
            continue
        fid = os.path.splitext(fn)[0]
        with open(os.path.join(corpus_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
            items.append((fid, f.read()))
    return items


def load_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect a list of {query: str, expected_ids: [str]}
    if isinstance(data, dict):
        data = data.get("queries", [])
    return data


def recall_at_k(expected: List[str], got_ids: List[str]) -> float:
    if not expected:
        return 0.0
    hit = any(g in expected for g in got_ids)
    return 1.0 if hit else 0.0


def main():
    parser = argparse.ArgumentParser(description="Retrieval benchmark for Pinecone")
    parser.add_argument("--corpus-dir", required=True, help="Directory of .txt documents to index")
    parser.add_argument("--queries", required=True, help="JSON file with queries [{query, expected_ids:[...]}, ...]")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--namespace", default="bench")
    parser.add_argument("--results-out", default=None, help="Write JSON results to path")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_dir)
    ids = [doc_id for doc_id, _ in corpus]
    texts = [txt for _, txt in corpus]

    print(f"Indexing {len(ids)} docs to Pinecone namespace '{args.namespace}'...")
    store = PineconeStore()
    vecs = embed_texts(texts)
    metas = [{"source": doc_id, "namespace": args.namespace} for doc_id in ids]
    # For namespace separation in one index, prefix ids
    prefixed_ids = [f"{args.namespace}:{i}" for i in ids]
    store.upsert_texts(prefixed_ids, vecs, metas)
    print("Indexing done.")

    queries = load_queries(args.queries)
    results: List[Dict[str, Any]] = []
    total_recall = 0.0
    total_ms = 0.0

    for q in queries:
        qtext = q.get("query", "")
        expected_ids = q.get("expected_ids", [])
        t0 = time.perf_counter()
        qvec = embed_text(qtext)
        res = store.query(qvec, top_k=args.top_k)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        total_ms += ms
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        got_ids = []
        for m in matches:
            mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
            if isinstance(mid, str) and mid.startswith(f"{args.namespace}:"):
                mid = mid.split(":", 1)[1]
            if mid:
                got_ids.append(mid)
        r = recall_at_k(expected_ids, got_ids)
        total_recall += r
        results.append({
            "query": qtext,
            "latency_ms": ms,
            "top_k": args.top_k,
            "expected_ids": expected_ids,
            "got_ids": got_ids,
            "recall_at_k": r,
        })
        print(f"Q: {qtext[:60]}... | recall@{args.top_k}={r:.0f} | {ms:.2f} ms | got={got_ids}")

    n = max(1, len(queries))
    summary = {
        "docs_indexed": len(ids),
        "queries": len(queries),
        "avg_latency_ms": total_ms / n,
        "avg_recall_at_k": total_recall / n,
        "results": results,
    }

    if args.results_out:
        with open(args.results_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote retrieval benchmark to {args.results_out}")

    print("=== Retrieval Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
