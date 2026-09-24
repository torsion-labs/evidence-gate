"""The seams.

Everything that could reach the network, cost money or behave differently
between two runs sits behind one of these protocols. The default
implementations shipped with the library are deterministic and offline, which
is why the test suite needs no API key and no connection.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .types import Chunk, Passage


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Implementations must be deterministic for a fixed input: the same string
    embeds to the same vector, always. A retrieval system whose index shifts
    under it cannot be tested, and cannot be audited either.
    """

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class Store(Protocol):
    """Holds chunks and their vectors, and answers nearest-neighbour queries."""

    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int = 5) -> list[Passage]: ...

    def __len__(self) -> int: ...


@runtime_checkable
class Writer(Protocol):
    """Turns passages into prose.

    A writer receives the passages the gate approved and returns text. It never
    returns citations, because it is not allowed to decide what the answer
    rests on. That decision was already made and is not up for revision by a
    language model.
    """

    def write(self, question: str, passages: Sequence[Passage]) -> str: ...


@runtime_checkable
class Reader(Protocol):
    """Turns a file into text, page by page.

    Page numbers are 1-based and must be real. A reader that cannot tell which
    page a piece of text came from should return None rather than guess, and
    the chunker will record the absence instead of inventing a number.
    """

    def read(self, path: str) -> list[tuple[int | None, str]]: ...
