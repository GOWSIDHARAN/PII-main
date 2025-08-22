import os
import time
import argparse
import json
from typing import Dict, Any, List, Tuple

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in (ROOT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from embeddings import embed_texts  # type: ignore
from pinecone_store import PineconeStore  # type: ignore


def chunk_text(text: str, max_tokens: int = 256) -> List[str]:
    # Simple whitespace chunker by ~tokens
    words = text.split()
    chunks: List[str] = []
    cur: List[str] = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


essential_meta_keys = ["source", "mode", "namespace"]


def upsert_chunks(store: PineconeStore, namespace: str, source_id: str, chunks: List[str], meta_extra: Dict[str, Any]) -> int:
    if not chunks:
        return 0
    ids = [f"{namespace}:{source_id}:{i}" for i in range(len(chunks))]
    vecs = embed_texts(chunks)
    metas = [{"source": source_id, "namespace": namespace, **meta_extra} for _ in chunks]
    store.upsert_texts(ids, vecs, metas)
    return len(chunks)


def run_tail_file(path: str, namespace: str, poll_sec: float) -> None:
    store = PineconeStore()
    print(f"Tailing file: {path}")
    last_size = 0
    buf = ""
    try:
        while True:
            try:
                cur_size = os.path.getsize(path)
                if cur_size < last_size:
                    # rotated or truncated
                    last_size = 0
                if cur_size > last_size:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(last_size)
                        data = f.read()
                        buf += data
                        last_size = cur_size
                        # chunk per ~256 tokens when buffer exceeds threshold
                        if len(buf.split()) >= 256:
                            chunks = chunk_text(buf, max_tokens=256)
                            # keep last partial chunk in buffer
                            carry = chunks.pop() if chunks else ""
                            n = upsert_chunks(store, namespace, os.path.basename(path), chunks, {"mode": "tail_file"})
                            print(f"Upserted {n} chunks from tail")
                            buf = carry
                time.sleep(poll_sec)
            except FileNotFoundError:
                print("Waiting for file...")
                time.sleep(poll_sec)
    except KeyboardInterrupt:
        print("Stopping tail.")


def run_watch_dir(dir_path: str, namespace: str, poll_sec: float) -> None:
    store = PineconeStore()
    print(f"Watching directory: {dir_path}")
    mtimes: Dict[str, float] = {}
    try:
        while True:
            for fn in os.listdir(dir_path):
                if not fn.lower().endswith(".txt"):
                    continue
                fp = os.path.join(dir_path, fn)
                try:
                    mtime = os.path.getmtime(fp)
                except FileNotFoundError:
                    continue
                prev = mtimes.get(fp)
                if prev is None or mtime > prev:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    chunks = chunk_text(text, max_tokens=256)
                    n = upsert_chunks(store, namespace, os.path.splitext(fn)[0], chunks, {"mode": "watch_dir"})
                    mtimes[fp] = mtime
                    print(f"Indexed {n} chunks from {fn}")
            time.sleep(poll_sec)
    except KeyboardInterrupt:
        print("Stopping watch.")


def main():
    ap = argparse.ArgumentParser(description="Real-time ingestion simulator (file tail or directory watch) for Pinecone")
    ap.add_argument("--mode", choices=["file", "dir"], required=True)
    ap.add_argument("--path", required=True, help="File to tail or directory to watch")
    ap.add_argument("--namespace", default="stream")
    ap.add_argument("--poll-sec", type=float, default=2.0)
    args = ap.parse_args()

    if args.mode == "file":
        run_tail_file(args.path, args.namespace, args.poll_sec)
    else:
        run_watch_dir(args.path, args.namespace, args.poll_sec)


if __name__ == "__main__":
    main()
