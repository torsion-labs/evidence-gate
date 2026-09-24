"""An in-memory vector store that can be written to disk and read back.

Small on purpose. The point of this repository is the gate and the evidence it
produces, not a database. Swapping in a real vector store means implementing
the Store protocol; the tests above it do not change, which is the property
worth having.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from .embedding import HashingEmbedder, cosine
from .ports import Embedder
from .types import Chunk, Passage, Source

SCHEMA_VERSION = 1


class InMemoryStore:
    """Holds chunks and vectors, and ranks them by cosine similarity."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder: Embedder = embedder or HashingEmbedder()
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._ids: set[str] = set()

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks. Re-adding the same chunk id is a no-op, not a duplicate.

        Idempotence matters here for the same reason it matters in any pipeline
        that can be re-run: an ingestion that ran twice must not make a passage
        twice as likely to be retrieved.
        """
        fresh = [c for c in chunks if c.id not in self._ids]
        if not fresh:
            return
        vectors = self._embedder.embed([c.text for c in fresh])
        for chunk, vector in zip(fresh, vectors):
            self._chunks.append(chunk)
            self._vectors.append(list(vector))
            self._ids.add(chunk.id)

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[Passage]:
        """Return the k best passages, highest score first.

        Ties are broken by chunk ordinal and then id, so the ranking is total
        and identical between runs. Non-determinism in a retrieval test is not
        a flake to retry; it is a result that cannot be trusted.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._chunks:
            return []

        query_vector = self._embedder.embed([query])[0]
        scored: list[tuple[float, int, str, Passage]] = []
        for chunk, vector in zip(self._chunks, self._vectors):
            if where is not None and not where(chunk):
                continue
            score = cosine(query_vector, vector)
            scored.append((score, chunk.ordinal, chunk.id, Passage(chunk, score)))

        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [row[3] for row in scored[:k]]

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "dimensions": self._embedder.dimensions,
            "chunks": [
                {
                    "id": c.id,
                    "text": c.text,
                    "ordinal": c.ordinal,
                    "metadata": dict(c.metadata),
                    "source": {
                        "file": c.source.file,
                        "page": c.source.page,
                        "section": c.source.section,
                    },
                    "vector": v,
                }
                for c, v in zip(self._chunks, self._vectors)
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "InMemoryStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"index schema {data.get('schema')!r} was written by another "
                f"version; expected {SCHEMA_VERSION}. Re-index rather than guess."
            )

        store = cls(embedder)
        if store._embedder.dimensions != data["dimensions"]:
            raise ValueError(
                f"index has {data['dimensions']} dimensions but the embedder "
                f"produces {store._embedder.dimensions}. These vectors are not "
                f"comparable; re-index with the embedder you intend to query with."
            )

        for row in data["chunks"]:
            src = row["source"]
            chunk = Chunk(
                id=row["id"],
                text=row["text"],
                source=Source(
                    file=src["file"], page=src["page"], section=src["section"]
                ),
                ordinal=row["ordinal"],
                metadata=row.get("metadata", {}),
            )
            store._chunks.append(chunk)
            store._vectors.append(list(row["vector"]))
            store._ids.add(chunk.id)
        return store
