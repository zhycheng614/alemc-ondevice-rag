"""Query / passage embedder with graceful backend fallback.

Preference order:
  1. fastembed (ONNX bge-small-en-v1.5) — no PyTorch, works on ARM/Termux.
  2. sentence-transformers — if a full torch stack is present (hub only).
  3. deterministic mock — hashed bag-of-words, so the harness and all
     downstream code can be validated without any model. Mock vectors are
     NOT semantically meaningful and must never back a reported result.

All backends L2-normalize output so a FAISS inner-product index behaves as
cosine similarity.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMBED_MODEL_NAME, EMBED_DIM


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class Embedder:
    """Unified embedding interface. `backend` attribute records which path ran."""

    def __init__(self, prefer: str = "auto"):
        self.backend = None
        self._impl = None
        self.dim = EMBED_DIM

        if prefer in ("auto", "fastembed"):
            if self._try_fastembed():
                return
        if prefer in ("auto", "sentence-transformers"):
            if self._try_sentence_transformers():
                return
        self._init_mock()

    # -- backends -----------------------------------------------------------
    def _try_fastembed(self) -> bool:
        try:
            from fastembed import TextEmbedding
        except Exception:
            return False
        try:
            self._impl = TextEmbedding(model_name=EMBED_MODEL_NAME)
            self.backend = "fastembed"
            return True
        except Exception:
            return False

    def _try_sentence_transformers(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return False
        try:
            self._impl = SentenceTransformer(EMBED_MODEL_NAME)
            self.backend = "sentence-transformers"
            return True
        except Exception:
            return False

    def _init_mock(self):
        self.backend = "mock"
        self._impl = None

    # -- encoding -----------------------------------------------------------
    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "fastembed":
            vecs = np.array(list(self._impl.embed(texts)), dtype=np.float32)
        elif self.backend == "sentence-transformers":
            vecs = np.asarray(self._impl.encode(texts), dtype=np.float32)
        else:
            vecs = self._mock_encode(texts)
        return _l2_normalize(vecs)

    def _mock_encode(self, texts: List[str]) -> np.ndarray:
        """Deterministic hashed bag-of-words -> EMBED_DIM vector."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return out

    @property
    def is_mock(self) -> bool:
        return self.backend == "mock"


if __name__ == "__main__":
    emb = Embedder()
    v = emb.encode(["what is the capital of france", "who wrote hamlet"])
    print(f"backend={emb.backend} shape={v.shape} "
          f"norm0={np.linalg.norm(v[0]):.3f} mock={emb.is_mock}")
