"""Splitting text into chunks without losing where it came from.

Two rules govern this module.

The first is that a chunk keeps its source. File, page and section travel with
the text through the whole pipeline, because a citation produced at the end is
only as trustworthy as the provenance recorded at the start.

The second is subtler and comes from a real failure. When two sections of a
document say almost the same thing and differ in one qualifier -- "after 24
hours" against "after 48 hours" -- retrieval scores them nearly alike and
picks the wrong one, with prose that reads perfectly. Carrying the section
heading into the chunk text makes that qualifier part of what is matched, and
turns an invisible error into a visible one.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .types import Chunk, Source

_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}\s+(?P<hash>.+)|(?P<upper>[A-Z][A-Z0-9 ,.'/&()-]{6,})\s*)$")
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class ChunkingPolicy:
    """How text is split.

    `prepend_section` is on by default and is the fix described above. Turning
    it off is supported so the cost of the fix can be measured rather than
    asserted: see tests/test_chunking.py.
    """

    max_chars: int = 900
    overlap_chars: int = 120
    prepend_section: bool = True
    min_chars: int = 40

    def __post_init__(self) -> None:
        if self.max_chars <= self.min_chars:
            raise ValueError("max_chars must exceed min_chars")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("overlap_chars must be within [0, max_chars)")


def detect_heading(line: str) -> str | None:
    """Return the heading text if this line looks like a heading, else None.

    Deliberately conservative. Two shapes are recognised: Markdown hashes, and
    a short all-capitals line. Anything else is treated as prose, because a
    false heading silently mislabels every chunk beneath it.
    """
    match = _HEADING.match(line.rstrip())
    if not match:
        return None
    text = match.group("hash") or match.group("upper") or ""
    text = text.strip()
    return text or None


def _chunk_id(source: Source, ordinal: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{source.file}|{source.page}|{ordinal}|{text}".encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _split_long(block: str, policy: ChunkingPolicy) -> list[str]:
    """Split an oversized block on sentence ends, with overlap."""
    if len(block) <= policy.max_chars:
        return [block]

    pieces: list[str] = []
    start = 0
    while start < len(block):
        end = min(start + policy.max_chars, len(block))
        if end < len(block):
            window = block[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("; "))
            if cut > policy.min_chars:
                end = start + cut + 1
        piece = block[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(block):
            break
        start = max(end - policy.overlap_chars, start + 1)
    return pieces


def chunk_page(
    text: str,
    *,
    file: str,
    page: int | None = None,
    policy: ChunkingPolicy | None = None,
    start_ordinal: int = 0,
) -> list[Chunk]:
    """Split one page of text into chunks that remember where they are."""
    policy = policy or ChunkingPolicy()
    chunks: list[Chunk] = []
    ordinal = start_ordinal
    section: str | None = None

    for block in _PARAGRAPH.split(text):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        heading = detect_heading(lines[0])
        if heading is not None:
            section = heading
            body_lines = lines[1:]
        else:
            body_lines = lines

        body = " ".join(ln.strip() for ln in body_lines).strip()
        if len(body) < policy.min_chars:
            # Too short to stand alone. A lone heading is not evidence, and a
            # stray line is not worth a citation.
            continue

        source = Source(file=file, page=page, section=section)
        for piece in _split_long(body, policy):
            text_for_index = (
                f"{section}\n{piece}" if policy.prepend_section and section else piece
            )
            chunks.append(
                Chunk(
                    id=_chunk_id(source, ordinal, piece),
                    text=text_for_index,
                    source=source,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    return chunks


def chunk_document(
    pages: list[tuple[int | None, str]],
    *,
    file: str,
    policy: ChunkingPolicy | None = None,
) -> list[Chunk]:
    """Chunk a whole document, preserving page numbers across the sequence."""
    policy = policy or ChunkingPolicy()
    out: list[Chunk] = []
    for page, text in pages:
        out.extend(
            chunk_page(
                text, file=file, page=page, policy=policy, start_ordinal=len(out)
            )
        )
    return out
