"""Sentence embeddings for metric 7, behind an interface.

The interface exists so the VERDICT LOGIC can be tested on exact arithmetic. If
correctness.py imported a real model, its tests would assert on whatever MiniLM
happens to output for a given sentence -- which is neither stable across versions
nor a statement about the code under test.

Lexical overlap is not an option here. The architecture document's own worked
example is two sentences meant to score 0.88 that share almost no vocabulary
("not wearing a hard hat" vs "has no helmet on"). TF-IDF scores that near zero and
would mark a correct explanation incorrect.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# Pinned in requirements/models.lock, like every other weight in this study. An
# unpinned embedding model silently changes metric 7 between runs.
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """One row of floats per input string."""
        ...


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, with an empty vector scoring 0 rather than nan.

    A missing or empty rationale is a real case in this data. Returning nan would
    propagate into every aggregate; returning 0 says "no similarity", which is the
    truthful reading.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FakeEmbedder:
    """Scripted embeddings for tests.

    Unknown text gets a unique basis vector, so anything unscripted is orthogonal
    to everything else. A fake that returned zeros would make unscripted text
    score 0 against a real vector too, hiding the difference between "dissimilar"
    and "not configured".
    """

    def __init__(self, table: dict[str, list[float]], dim: int = 8):
        self._table = {k: np.asarray(v, dtype=float) for k, v in table.items()}
        self._dim = max(dim, max((len(v) for v in self._table.values()), default=dim))
        self._unknown: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t in self._table:
                v = np.zeros(self._dim)
                src = self._table[t]
                v[:len(src)] = src
            else:
                if t not in self._unknown:
                    idx = self._dim - 1 - (len(self._unknown) % self._dim)
                    u = np.zeros(self._dim)
                    u[idx] = 1.0
                    self._unknown[t] = u
                v = self._unknown[t]
            out.append(v)
        return np.vstack(out)


class MiniLMEmbedder:
    """The real embedder. Loaded lazily so importing this module costs nothing."""

    def __init__(self, model_id: str = EMBED_MODEL_ID, revision: str | None = None):
        self._model_id = model_id
        self._revision = revision
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            kw = {"revision": self._revision} if self._revision else {}
            self._model = SentenceTransformer(self._model_id, device="cpu", **kw)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._load().encode(texts, show_progress_bar=False))
