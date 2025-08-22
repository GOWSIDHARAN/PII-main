import os
from typing import List, Dict, Any, Optional

# NOTE: Avoid importing chromadb at module import time because some versions
# attempt to initialize a default ONNX embedding function during import,
# which requires onnxruntime and can crash the app if not present.
# We'll import chromadb lazily inside ChromaStore.__init__ instead.

DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "pii_sentinel")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma"))
CHROMA_CLOUD = os.getenv("CHROMA_CLOUD", "false").strip().lower() in {"1","true","yes","on"}
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")


class ChromaStore:
    """Local vector store using ChromaDB with persistent storage.

    Methods mirror the PineconeStore API used by the app:
    - upsert_texts(ids, vectors, metadatas)
    - query(vector, top_k)
    """

    def __init__(self, collection_name: str = DEFAULT_COLLECTION) -> None:
        # Lazy import to prevent module-level side effects if onnxruntime is missing
        import importlib
        chromadb = importlib.import_module("chromadb")
        if CHROMA_CLOUD:
            if not (CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE):
                raise ValueError("CHROMA_CLOUD is enabled but CHROMA_API_KEY, CHROMA_TENANT, or CHROMA_DATABASE not set")
            # Chroma Cloud
            self.client = chromadb.CloudClient(
                api_key=CHROMA_API_KEY,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
            )
        else:
            # Local persistent Chroma (new client API)
            os.makedirs(CHROMA_DIR, exist_ok=True)
            # Use PersistentClient per Chroma migration guide
            self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        # Use our own embeddings; disable Chroma's default ONNX embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

    def upsert_texts(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in ids]
        # Chroma add() will upsert on duplicate ids
        self.collection.add(
            ids=[str(i) for i in ids],
            embeddings=vectors,
            metadatas=metadatas,
            documents=[m.get("text", "") for m in metadatas],
        )
        # Persist to disk
        self.client.persist()

    def query(self, vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter or {},
        )
        # Normalize to Pinecone-like response
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        matches = []
        for _id, _dist, _meta in zip(ids, distances, metas):
            matches.append({
                "id": _id,
                "score": _dist,
                "metadata": _meta or {},
            })
        return {"matches": matches}
