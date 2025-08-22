import os
import json
import random
import argparse
from typing import Optional, Dict, Any, List

from pinecone import Pinecone


def get_index() -> tuple[Pinecone, Any, int, str]:
    index_name = os.getenv("PINECONE_INDEX", "pii-sentinel")
    dim = int(os.getenv("EMBED_DIM", "768"))

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in environment")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return pc, index, dim, index_name


def describe(index) -> None:
    stats = index.describe_index_stats()
    print("=== Index Stats ===")
    print(json.dumps(stats, indent=2))


def sample_query(index, dim: int, top_k: int, filter_obj: Optional[Dict[str, Any]] = None) -> None:
    vec = [random.random() for _ in range(dim)]
    res = index.query(vector=vec, top_k=top_k, include_metadata=True, filter=filter_obj)
    print("=== Sample Query Results ===")
    matches: List[Dict[str, Any]] = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    for i, m in enumerate(matches, 1):
        mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
        metadata = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)
        print(f"[{i}] id={mid} score={score}\nmetadata={json.dumps(metadata, indent=2)}\n---")


def fetch_ids(index, ids: List[str]) -> None:
    if not ids:
        print("No IDs provided to fetch.")
        return
    res = index.fetch(ids=ids)
    print("=== Fetch Results ===")
    print(json.dumps(res, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inspect Pinecone index contents")
    parser.add_argument("--top-k", type=int, default=5, help="Top K matches for sample query")
    parser.add_argument("--filter-json", type=str, default=None, help="JSON string filter (Pinecone metadata filter)")
    parser.add_argument("--fetch-ids", type=str, nargs="*", help="IDs to fetch explicitly")

    args = parser.parse_args()

    _, index, dim, index_name = get_index()
    print(f"Using index: {index_name}; dim={dim}")

    describe(index)

    filter_obj = None
    if args.filter_json:
        try:
            filter_obj = json.loads(args.filter_json)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --filter-json: {e}")

    sample_query(index, dim=dim, top_k=args.top_k, filter_obj=filter_obj)

    if args.fetch_ids:
        fetch_ids(index, ids=args.fetch_ids)


if __name__ == "__main__":
    main()
