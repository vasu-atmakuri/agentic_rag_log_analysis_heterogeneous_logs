"""Dense semantic index over chunks: all-MiniLM-L6-v2 embeddings, L2-normalized, exact
inner-product search with faiss.IndexFlatIP (equivalent to cosine similarity)."""
from __future__ import annotations

import hashlib
import json
import os
import re

# On macOS, torch and faiss each ship an OpenMP runtime. Loading faiss first and then running torch
# (sentence-transformers) crashes with a segmentation fault, so torch is loaded first and faiss is
# kept single-threaded (exact search over a few thousand vectors does not need threads).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None

import faiss  # noqa: E402
import numpy as np  # noqa: E402

faiss.omp_set_num_threads(1)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=32, show_progress_bar=len(texts) > 500,
                                    convert_to_numpy=True, normalize_embeddings=True)
        return vectors.astype("float32")


class HashingEmbedder:
    """Offline stand-in for unit tests only (token hashing). Never used for reported results."""

    name = "hashing-test-embedder"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


_EMBEDDER_CACHE = {}


def get_embedder(kind: str = "sentence_transformer"):
    if kind not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[kind] = HashingEmbedder() if kind == "hashing_test" else SentenceTransformerEmbedder()
    return _EMBEDDER_CACHE[kind]


class SemanticChunkIndex:
    def __init__(self, chunks: list[dict], vectors: np.ndarray, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @classmethod
    def build(cls, chunks: list[dict], embedder) -> "SemanticChunkIndex":
        vectors = embedder.encode([c["text"] for c in chunks]) if chunks else np.zeros((0, 384), dtype="float32")
        return cls(chunks, vectors, embedder)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "chunk_vectors.faiss"))
        with open(os.path.join(directory, "chunks.jsonl"), "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk) + "\n")

    @classmethod
    def load(cls, directory: str, embedder) -> "SemanticChunkIndex":
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(os.path.join(directory, "chunk_vectors.faiss"))
        with open(os.path.join(directory, "chunks.jsonl"), "r", encoding="utf-8") as handle:
            obj.chunks = [json.loads(line) for line in handle]
        obj.embedder = embedder
        return obj

    def search(self, query: str, k: int, excluded_types=frozenset(), only_type: str | None = None) -> list[tuple[dict, float]]:
        if not self.chunks:
            return []
        filtered = bool(excluded_types) or only_type is not None
        fetch = min(len(self.chunks), max(k * 40, 200) if filtered else k)
        vector = self.embedder.encode([query])
        scores, ids = self.index.search(vector, fetch)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if chunk["chunk_type"] in excluded_types or (only_type and chunk["chunk_type"] != only_type):
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results
