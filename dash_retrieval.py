import os
import time
from typing import List, Dict, Any, Tuple

import streamlit as st

# Local imports
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
    try:
        for fn in os.listdir(corpus_dir):
            if not fn.lower().endswith(".txt"):
                continue
            fid = os.path.splitext(fn)[0]
            with open(os.path.join(corpus_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
                items.append((fid, f.read()))
    except Exception:
        pass
    return items


def recall_at_k(expected: List[str], got_ids: List[str]) -> float:
    if not expected:
        return 0.0
    hit = any(g in expected for g in got_ids)
    return 1.0 if hit else 0.0


@st.cache_resource(show_spinner=False)
def get_store() -> PineconeStore:
    return PineconeStore()


def run_once(store: PineconeStore, namespace: str, corpus: List[Tuple[str, str]], queries: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    ids = [doc_id for doc_id, _ in corpus]
    texts = [txt for _, txt in corpus]

    if ids:
        vecs = embed_texts(texts)
        metas = [{"source": doc_id, "namespace": namespace} for doc_id in ids]
        prefixed_ids = [f"{namespace}:{i}" for i in ids]
        store.upsert_texts(prefixed_ids, vecs, metas)

    results: List[Dict[str, Any]] = []
    total_recall = 0.0
    total_ms = 0.0

    for q in queries:
        qtext = q.get("query", "")
        expected_ids = q.get("expected_ids", [])
        t0 = time.perf_counter()
        qvec = embed_text(qtext)
        res = store.query(qvec, top_k=top_k)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        total_ms += ms
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        got_ids = []
        for m in matches:
            mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
            if isinstance(mid, str) and mid.startswith(f"{namespace}:"):
                mid = mid.split(":", 1)[1]
            if mid:
                got_ids.append(mid)
        r = recall_at_k(expected_ids, got_ids)
        total_recall += r
        results.append({
            "query": qtext,
            "latency_ms": ms,
            "top_k": top_k,
            "expected_ids": ", ".join(expected_ids),
            "got_ids": ", ".join(got_ids),
            "recall_at_k": r,
        })

    n = max(1, len(queries))
    summary = {
        "docs_indexed": len(ids),
        "queries": len(queries),
        "avg_latency_ms": total_ms / n,
        "avg_recall_at_k": total_recall / n,
        "results": results,
    }
    return summary


def main():
    st.set_page_config(page_title="Retrieval Live Benchmark", layout="wide")
    st.title("Retrieval Live Benchmark (Pinecone + Gemini)")

    with st.sidebar:
        st.header("Configuration")
        corpus_dir = st.text_input("Corpus directory (txt files)", value=os.path.join(ROOT_DIR, "data", "corpus"))
        queries_text = st.text_area(
            "Queries JSON (list of {query, expected_ids})",
            value='[{"query":"contact details","expected_ids":["sample"]}]',
            height=150,
        )
        namespace = st.text_input("Namespace", value="livebench")
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1)
        refresh_sec = st.number_input("Refresh (seconds)", min_value=2, max_value=3600, value=10, step=1)
        do_run = st.button("Run Now")
        st.markdown("Env: uses GEMINI_API_KEY + Pinecone creds from .env")

    try:
        queries = []
        if queries_text.strip():
            import json as _json
            data = _json.loads(queries_text)
            if isinstance(data, dict):
                data = data.get("queries", [])
            queries = data
    except Exception as e:
        st.error(f"Invalid queries JSON: {e}")
        queries = []

    store = get_store()

    placeholder = st.empty()

    def render_once():
        corpus = load_corpus(corpus_dir)
        if not corpus:
            st.warning("No .txt documents found in corpus directory.")
        summary = run_once(store, namespace, corpus, queries, int(top_k))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Docs Indexed", summary["docs_indexed"]) 
        with col2:
            st.metric("Queries", summary["queries"]) 
        with col3:
            st.metric("Avg Latency (ms)", f"{summary['avg_latency_ms']:.2f}")
        with col4:
            st.metric(f"Avg Recall@{top_k}", f"{summary['avg_recall_at_k']:.2f}")
        st.dataframe(summary["results"], use_container_width=True)

    if do_run:
        with placeholder.container():
            render_once()

    # Auto refresh
    st_autorefresh = st.session_state.get("_autorefresh", True)
    st.session_state["_autorefresh"] = st.checkbox("Auto refresh", value=True)
    if st.session_state["_autorefresh"]:
        import streamlit.runtime.scriptrunner.script_run_context as src
        # Simple timer loop using experimental_rerun-like behavior:
        st.experimental_singleton.clear()  # no-op safety
        # Use st.experimental_rerun via a timer
        st.markdown("<script>setTimeout(function(){window.location.reload();}, %d);</script>" % int(refresh_sec*1000), unsafe_allow_html=True)
        with placeholder.container():
            render_once()


if __name__ == "__main__":
    main()
