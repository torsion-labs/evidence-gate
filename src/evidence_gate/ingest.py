"""Reading a set of files into chunks, and saying what was left out.

Ingestion is where a retrieval system most often goes wrong without anyone
noticing. A scanned page has no text, so it is not indexed, so no question
ever retrieves it, so nobody finds out it was missing -- the assistant simply
behaves as if that page did not exist. The same happens to a file in a format
nobody wrote a reader for, and to a PDF read on a machine without pypdf.

So `ingest` never drops anything quietly. It returns the chunks together with
a report of every page that yielded no text and every file it could not use,
with the reason. `report.summary()` is written to be read by the person who
owns the documents, not by the engineer.

`report.complete` is False when a page or a file was left out. Fragments too
short to index are listed too, but they do not make the ingestion incomplete:
almost every PDF has page numbers and footers, and an alarm that rings on every
document is one nobody reads. They are listed so that the rare short fragment
that mattered -- a two-line table, a one-line rule -- can still be found.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .chunking import ChunkingPolicy, _chunk_document
from .ports import Reader
from .readers import DocxReader, MissingDependency, PdfReader, TextReader
from .types import Chunk

DEFAULT_READERS: Mapping[str, Reader] = {
    ".pdf": PdfReader(),
    ".docx": DocxReader(),
    ".txt": TextReader(),
    ".md": TextReader(),
    ".markdown": TextReader(),
}


@dataclass(frozen=True)
class Skipped:
    """A file that contributed nothing to the index, and why."""

    file: str
    reason: str


@dataclass(frozen=True)
class IngestReport:
    """The chunks, and an honest account of what did not make it in."""

    chunks: tuple[Chunk, ...]
    read: tuple[str, ...]
    empty_pages: tuple[tuple[str, int], ...] = ()
    skipped: tuple[Skipped, ...] = ()
    too_short: tuple[tuple[str, int | None, str], ...] = ()
    min_chars: int = ChunkingPolicy().min_chars

    @property
    def complete(self) -> bool:
        """True only when every page of every file was indexed."""
        return not self.empty_pages and not self.skipped

    def summary(self) -> str:
        lines = [f"{len(self.read)} file(s) read, {len(self.chunks)} chunk(s) indexed."]

        by_file: dict[str, list[int]] = {}
        for file, page in self.empty_pages:
            by_file.setdefault(file, []).append(page)
        for file, pages in by_file.items():
            listed = ", ".join(str(p) for p in pages)
            lines.append(
                f"- {file}: no text on page(s) {listed}. Usually a scanned "
                f"image; it needs OCR before it can be searched."
            )
        for s in self.skipped:
            lines.append(f"- {s.file}: {s.reason}")

        if self.complete:
            lines.append("Nothing was left out.")

        if self.too_short:
            examples = ", ".join(repr(text) for _, _, text in self.too_short[:3])
            more = " and others" if len(self.too_short) > 3 else ""
            lines.append(
                f"{len(self.too_short)} fragment(s) under {self.min_chars} "
                f"characters were not indexed, usually page numbers and "
                f"footers: {examples}{more}."
            )
        return "\n".join(lines)


def ingest(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
    policy: ChunkingPolicy | None = None,
    readers: Mapping[str, Reader] | None = None,
) -> IngestReport:
    """Read files and folders into chunks, reporting everything left out.

    Folders are walked recursively in a fixed order, skipping hidden files and
    the `~$` lock files Word leaves behind; a file named explicitly is always
    attempted. Each chunk is cited by its path relative to `root` when one is
    given, and by its file name otherwise. Two files that would be cited by
    the same name are an error rather than a warning: a citation that could
    mean either of two documents is not a citation.
    """
    policy = policy or ChunkingPolicy()
    readers = DEFAULT_READERS if readers is None else readers
    root_path = Path(root) if root is not None else None

    files: list[Path] = []
    seen_paths: set[Path] = set()
    for found in _expand(paths):
        resolved = found.resolve()
        if resolved not in seen_paths:  # named twice, indexed once
            seen_paths.add(resolved)
            files.append(found)
    names = [_display_name(f, root_path) for f in files]
    _refuse_duplicates(files, names)

    chunks: list[Chunk] = []
    read: list[str] = []
    empty: list[tuple[str, int]] = []
    skipped: list[Skipped] = []
    short: list[tuple[str, int | None, str]] = []

    for path, name in zip(files, names):
        reader = readers.get(path.suffix.lower())
        if reader is None:
            kind = path.suffix or "files without an extension"
            skipped.append(Skipped(name, f"no reader for {kind}"))
            continue

        try:
            pages = reader.read(path)
        except MissingDependency as exc:
            skipped.append(Skipped(name, str(exc)))
            continue
        except Exception as exc:  # a broken file must not stop the others
            detail = f"{type(exc).__name__}: {exc}"
            skipped.append(Skipped(name, f"could not be read ({detail})"))
            continue

        blank = [p for p, text in pages if p is not None and not text.strip()]
        file_chunks, dropped = _chunk_document(pages, file=name, policy=policy)
        short.extend((name, page, text) for page, text in dropped)

        if not file_chunks:
            if pages and all(not text.strip() for _, text in pages):
                reason = "no extractable text; a scanned document needs OCR first"
            else:
                reason = (
                    f"text found, but no block of at least {policy.min_chars} "
                    f"characters to index"
                )
            skipped.append(Skipped(name, reason))
            continue

        chunks.extend(file_chunks)
        read.append(name)
        empty.extend((name, p) for p in blank)

    return IngestReport(
        chunks=tuple(chunks),
        read=tuple(read),
        empty_pages=tuple(empty),
        skipped=tuple(skipped),
        too_short=tuple(short),
        min_chars=policy.min_chars,
    )


def _expand(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for found in sorted(path.rglob("*")):
                if found.is_file() and not _ignored(found.name):
                    yield found
        else:
            yield path


def _ignored(name: str) -> bool:
    return name.startswith(".") or name.startswith("~$")


def _display_name(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _refuse_duplicates(files: list[Path], names: list[str]) -> None:
    seen: dict[str, Path] = {}
    for path, name in zip(files, names):
        if name in seen and seen[name].resolve() != path.resolve():
            raise ValueError(
                f"two files would both be cited as {name!r} ({seen[name]} and "
                f"{path}); pass root= so citations carry their folder"
            )
        seen.setdefault(name, path)
