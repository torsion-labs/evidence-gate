"""Building the answer, and the rule a language model does not get to break.

The model writes prose. It never decides what the answer rests on.

Citations are derived here, from the passages the gate approved, before any
writer is called. A writer is then handed those passages and returns text, and
that text replaces the prose and nothing else. If it invents a source, the
invented source is not in the citation list, because the citation list was
never its to build.

This is not a guard against a badly behaved model. It is an ordering. There is
no point in the pipeline where a model is asked which document an answer came
from, so there is no point at which it can be wrong about it.
"""

from __future__ import annotations

from typing import Sequence

from .ports import Writer
from .types import Answer, Citation, Passage, Verdict

MAX_QUOTE_CHARS = 320


def _quote(passage: Passage) -> str:
    text = passage.chunk.text.strip()
    if len(text) <= MAX_QUOTE_CHARS:
        return text
    cut = text.rfind(" ", 0, MAX_QUOTE_CHARS)
    return text[: cut if cut > 0 else MAX_QUOTE_CHARS].rstrip() + "..."


def citations_for(passages: Sequence[Passage]) -> tuple[Citation, ...]:
    """Derive citations from passages. The only place citations are created."""
    return tuple(
        Citation(chunk_id=p.chunk.id, source=p.source, quote=_quote(p))
        for p in passages
    )


class QuotingWriter:
    """The default writer: quotes the passages and says where each came from.

    It composes nothing, so it can invent nothing. Boring, and correct, and it
    means the pipeline is useful before any model is wired in at all.
    """

    def write(self, question: str, passages: Sequence[Passage]) -> str:
        lines = []
        for p in passages:
            lines.append(f"[{p.source.cite()}] {_quote(p)}")
        return "\n\n".join(lines)


def refusal_text(verdict: Verdict) -> str:
    """What the caller says when the gate says no.

    It names what was missing. "I could not find that" is a dead end for the
    person asking; "I could not find anything mentioning 48" tells them their
    document may not cover it, or that they should ask differently.
    """
    if verdict.missing:
        return (
            "I could not answer that from your documents. The passages I found "
            "are about the right topic but none of them mentions: "
            + ", ".join(verdict.missing)
            + "."
        )
    return f"I could not answer that from your documents: {verdict.reason}."


def build(
    question: str,
    verdict: Verdict,
    *,
    writer: Writer | None = None,
) -> Answer:
    """Turn a verdict into an answer, or into a refusal that explains itself."""
    if not verdict.answerable:
        return Answer(
            answered=False,
            text=refusal_text(verdict),
            missing=verdict.missing,
        )

    citations = citations_for(verdict.passages)
    writer = writer or QuotingWriter()
    prose = writer.write(question, verdict.passages)

    if not prose.strip():
        # A writer that returns nothing does not get to turn a good verdict
        # into an empty answer. Fall back to the quoting writer, which cannot.
        prose = QuotingWriter().write(question, verdict.passages)

    return Answer(answered=True, text=prose, citations=citations)
