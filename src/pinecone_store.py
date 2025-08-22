import os
from typing import List, Dict, Any, Optional

from pinecone import Pinecone, ServerlessSpec


DEFAULT_INDEX = os.getenv("PINECONE_INDEX", "pii-sentinel")
DEFAULT_DIM = int(os.getenv("EMBED_DIM", "768"))
DEFAULT_METRIC = os.getenv("PINECONE_METRIC", "cosine")


class PineconeStore:
    def __init__(self, index_name: str = DEFAULT_INDEX, dimension: int = DEFAULT_DIM) -> None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")

        self.pc = Pinecone(api_key=api_key)

        # Determine cloud/region
        env = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
        # Allow explicit overrides (strictly preferred)
        cloud_override = (os.getenv("PINECONE_CLOUD") or "").strip().lower()
        region_override = (os.getenv("PINECONE_REGION") or "").strip().lower()
        cloud_name = cloud_override or None
        region_name = region_override or None
        if not (cloud_name and region_name):
            parts = [p for p in env.split("-") if p]
            known_clouds = {"gcp", "aws", "azure"}
            cloud_parsed = None
            region_parsed = None
            if len(parts) >= 2:
                # If last token matches known cloud, use it; region is the rest joined
                if parts[-1].lower() in known_clouds:
                    cloud_parsed = parts[-1].lower()
                    region_parsed = "-".join(parts[:-1])
                else:
                    # Fallbacks for 2-3 tokens like us-east1-gcp or us-east-1
                    if len(parts) >= 3:
                        region_parsed = f"{parts[0]}-{parts[1]}"
                        cloud_parsed = parts[2].lower()
                    else:
                        region_parsed = parts[0]
                        cloud_parsed = parts[1].lower()
            else:
                region_parsed, cloud_parsed = "us-east1", "gcp"
            cloud_name = (cloud_name or (cloud_parsed.lower() if cloud_parsed else None))
            region_name = (region_name or (region_parsed.lower() if region_parsed else None))

        # Basic validation
        if not cloud_name or not region_name:
            raise RuntimeError(f"Unable to resolve Pinecone cloud/region from env='{env}', overrides cloud='{cloud_override}', region='{region_override}'")

        # Log resolved config for troubleshooting
        print(f"[Pinecone] Using cloud='{cloud_name}', region='{region_name}', index='{index_name}', dim={dimension}")

        spec = ServerlessSpec(cloud=cloud_name, region=region_name)

        # Ensure index exists
        try:
            existing = {idx.name: idx for idx in self.pc.list_indexes()}
            if index_name not in existing:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=DEFAULT_METRIC,
                    spec=spec,
                )
            self.index = self.pc.Index(index_name)
        except Exception as e:
            # Enrich error with resolved settings
            raise RuntimeError(f"Pinecone init failed with cloud='{cloud_name}', region='{region_name}', index='{index_name}': {e}") from e

    def upsert_texts(self, ids: List[str], vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in ids]
        items = [
            {
                "id": id_,
                "values": vec,
                "metadata": meta,
            }
            for id_, vec, meta in zip(ids, vectors, metadatas)
        ]
        self.index.upsert(vectors=items)

    def query(self, vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.index.query(vector=vector, top_k=top_k, include_metadata=True, filter=filter)
