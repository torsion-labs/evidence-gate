"""A deterministic, offline embedder.

This is not a toy. It is a hashed bag of words: tokens are hashed into a fixed
number of buckets, weighted sub-linearly and L2-normalised, which gives real
lexical similarity without a model, a download or a network call.

It exists for two reasons, and the second matters more than the first.

The obvious one is that the test suite must run anywhere, instantly, with no
API key. The less obvious one is that it makes every test in this repository
verifiable by a stranger. A retrieval test that depends on a hosted model is
not a test of your code, it is a test of someone else's service on the day you
ran it.

A hosted embedder is a drop-in replacement through the Embedder protocol. What
does not change is that the tests keep passing without it.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def normalise(text: str) -> str:
    """Casefold and strip accents so 'Pagina' and 'página' tokenise alike."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.casefold()


def tokenize(text: str) -> list[str]:
    """Split into tokens, keeping things like 'contract-v3.pdf' and '48h' whole."""
    return _TOKEN.findall(normalise(text))


class HashingEmbedder:
    """Deterministic lexical embedder. Same text in, same vector out, forever."""

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self._dimensions
        # A sign derived from the same hash keeps unrelated tokens from piling
        # up in the same direction when they collide in a bucket.
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            index, sign = self._bucket(token)
            # Sub-linear weighting: the tenth occurrence of a word says much
            # less than the first.
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors, assuming neither is pre-normalised."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
