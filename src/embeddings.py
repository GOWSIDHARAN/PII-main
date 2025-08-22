import os
from typing import List

import google.generativeai as genai


EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")


def _ensure_genai_config() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment")
    genai.configure(api_key=api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings using Gemini text-embedding-004.

    Args:
        texts: list of strings
    Returns:
        list of float vectors
    """
    if not texts:
        return []
    _ensure_genai_config()
    model = genai.GenerativeModel(EMBED_MODEL)
    # Batch embed using client method
    # google-generativeai exposes genai.embed_content for single item; for batch, loop for simplicity
    vectors: List[List[float]] = []
    for t in texts:
        resp = genai.embed_content(model=EMBED_MODEL, content=t)
        vectors.append(resp["embedding"])  # type: ignore[index]
    return vectors


def embed_text(text: str) -> List[float]:
    vecs = embed_texts([text])
    return vecs[0] if vecs else []
