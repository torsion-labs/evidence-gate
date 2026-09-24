"""Core data types.

Everything here is frozen. A chunk that has been indexed cannot be edited in
place, and neither can a verdict: if something needs to change, a new value is
built and the old one stays available for the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


@dataclass(frozen=True)
class Source:
    """Where a piece of text came from.

    This is the reason the whole library exists. A chunk without a source is
    not indexable, because an answer built from it could not be checked.
    """

    file: str
    page: int | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        if not self.file or not self.file.strip():
            raise ValueError("Source.file must be a non-empty path")
        if self.page is not None and self.page < 1:
            raise ValueError(f"Source.page must be 1-based, got {self.page}")

    def cite(self) -> str:
        """A short human-readable citation, e.g. 'contract-v3.pdf p.12'."""
        parts = [self.file]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        return " ".join(parts)


@dataclass(frozen=True)
class Chunk:
    """A unit of text with its provenance attached."""

    id: str
    text: str
    source: Source
    ordinal: int = 0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Chunk.id must not be empty")
        if not self.text.strip():
            raise ValueError(f"Chunk {self.id!r} has no text")


@dataclass(frozen=True)
class Passage:
    """A chunk retrieved for a query, with the score that retrieved it."""

    chunk: Chunk
    score: float

    @property
    def source(self) -> Source:
        return self.chunk.source


@dataclass(frozen=True)
class Verdict:
    """The evidence gate's decision.

    `answerable` is the whole point. When it is False the caller must not
    answer, and `missing` says what was not found so the refusal can be
    explained rather than merely delivered.
    """

    answerable: bool
    passages: tuple[Passage, ...]
    reason: str
    missing: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.answerable and not self.passages:
            raise ValueError("an answerable verdict must carry its passages")
        if not self.reason:
            raise ValueError("every verdict must state its reason")

    @property
    def needs_review(self) -> bool:
        """True when the gate let it through but wants a human to look.

        Answerable and clean are not the same thing. A verdict can pass every
        threshold and still sit on top of evidence that was nearly ambiguous,
        and saying so costs nothing next to the cost of not saying it.
        """
        return bool(self.flags)


@dataclass(frozen=True)
class Citation:
    """A claim tied to the chunk it came from."""

    chunk_id: str
    source: Source
    quote: str


@dataclass(frozen=True)
class Answer:
    """What the caller gets back.

    Either `answered` is True and there is at least one citation, or it is
    False and the text explains what was missing. There is no third state, and
    no answer without citations.
    """

    answered: bool
    text: str
    citations: tuple[Citation, ...] = ()
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.answered and not self.citations:
            raise ValueError("an answer without citations is not an answer")
        if not self.answered and self.citations:
            raise ValueError("a refusal must not carry citations")

    @property
    def cited_files(self) -> tuple[str, ...]:
        seen: list[str] = []
        for c in self.citations:
            if c.source.file not in seen:
                seen.append(c.source.file)
        return tuple(seen)

    def with_text(self, text: str) -> "Answer":
        """Replace the prose, keeping the citations untouched.

        This is the only way a prose writer is allowed to contribute: it may
        rewrite how the answer reads, never which sources it rests on.
        """
        return replace(self, text=text)
