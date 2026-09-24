"""Readers: files in, pages of text out, with the page numbers that are real.

Three formats are covered. Plain text and Markdown need nothing. DOCX needs
nothing either: a .docx file is a zip of XML, and the standard library reads
both. PDF needs `pypdf`, installed with `pip install "evidence-gate[pdf]"`,
and only when a PDF is actually read, so the core keeps its promise of no
runtime dependencies.

Every reader follows the contract in ports.py. A page number is returned only
when the format has pages. A PDF does. A DOCX file does not: where its pages
break depends on the fonts and the printer of whoever opens it, so a DOCX
reader returns None and the citation falls back to the section heading, which
is stable. A page number that was guessed would be worse than none, because
it looks exactly like one that was not.

Headings matter as much as pages. The chunker carries the current heading into
every chunk (see chunking.py for the failure that rule comes from), so readers
hand headings over in a shape the chunker recognises: a Markdown heading line
at the start of its own block.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .chunking import detect_heading

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_HEADING_STYLE = re.compile(r"^heading\s+(\d)$")


class MissingDependency(ImportError):
    """A reader needs a library that is not installed.

    Raised when the file is read, not when the module is imported, so a corpus
    with no PDFs never needs pypdf. `ingest` turns this into a line in its
    report instead of stopping, because one unreadable format should not hide
    the files that were fine.
    """

    def __init__(self, extra: str, package: str) -> None:
        super().__init__(
            f"reading these files needs {package}: "
            f'pip install "evidence-gate[{extra}]"'
        )
        self.extra = extra
        self.package = package


class TextReader:
    """Plain text and Markdown.

    A text file has no pages, with one exception: form feeds. Tools that
    convert PDFs to text mark page breaks with them, and when they are there
    they are the only page information that survived the conversion, so they
    are kept.
    """

    def read(self, path: str | Path) -> list[tuple[int | None, str]]:
        text = Path(path).read_text(encoding="utf-8-sig")
        if "\f" in text:
            return [(n, page) for n, page in enumerate(text.split("\f"), start=1)]
        return [(None, text)]


def separate_headings(text: str) -> str:
    """Put each heading line at the start of its own block.

    Text extracted from a PDF comes back as lines with no blank lines between
    them, so a heading in the middle of a page would otherwise be read as one
    more line of prose and never become a section.
    """
    out: list[str] = []
    for line in text.splitlines():
        if detect_heading(line) is not None and out and out[-1].strip():
            out.append("")
        out.append(line)
    return "\n".join(out)


class PdfReader:
    """PDF, one entry per page, numbered from 1.

    A page with no extractable text is returned as an empty string rather than
    dropped. It is usually a scanned image, and whoever reads the ingestion
    report should find out which pages were never indexed instead of assuming
    every page was.
    """

    def read(self, path: str | Path) -> list[tuple[int | None, str]]:
        try:
            from pypdf import PdfReader as _Pdf
        except ImportError as exc:
            raise MissingDependency("pdf", "pypdf") from exc

        pdf = _Pdf(str(path))
        return [
            (number, separate_headings(page.extract_text() or ""))
            for number, page in enumerate(pdf.pages, start=1)
        ]


class DocxReader:
    """Word documents, read from their XML with the standard library.

    Headings are recognised by what the style *is*, not by what it is called
    in the interface language: a Spanish copy of Word stores "Título 1" under
    the internal name "heading 1", and the outline level travels with it.
    Tables are kept, one row per line with cells separated by " | ", because
    in policies and price lists the tables are often where the answer lives.
    """

    def read(self, path: str | Path) -> list[tuple[int | None, str]]:
        with zipfile.ZipFile(path) as package:
            document = ElementTree.fromstring(package.read("word/document.xml"))
            try:
                styles = _read_styles(package.read("word/styles.xml"))
            except KeyError:
                styles = {}

        body = document.find(f"{_W}body")
        if body is None:
            return [(None, "")]

        blocks: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                blocks.append("\n".join(current))
                current.clear()

        for element in _body_elements(body):
            if element.tag == f"{_W}p":
                text = _paragraph_text(element)
                if not text:
                    continue
                level = _heading_level(element, styles)
                if level:
                    flush()
                    blocks.append(f"{'#' * min(level, 6)} {text}")
                else:
                    current.append(text)
            elif element.tag == f"{_W}tbl":
                # A table stays in the block of the section it belongs to. On
                # its own, a three-row price table is shorter than the
                # chunker's minimum and would not be indexed at all.
                current.extend(_table_rows(element))
        flush()

        return [(None, "\n\n".join(blocks))]


def _body_elements(body: ElementTree.Element):
    """Paragraphs and tables in reading order, looking inside content controls."""
    for child in body:
        if child.tag in (f"{_W}p", f"{_W}tbl"):
            yield child
        elif child.tag == f"{_W}sdt":
            content = child.find(f"{_W}sdtContent")
            if content is not None:
                yield from _body_elements(content)


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{_W}t" and node.text:
            parts.append(node.text)
        elif node.tag in (f"{_W}tab", f"{_W}br", f"{_W}cr"):
            parts.append(" ")
    return " ".join("".join(parts).split())


def _table_rows(table: ElementTree.Element) -> list[str]:
    rows: list[str] = []
    for row in table.iter(f"{_W}tr"):
        cells = []
        for cell in row.findall(f"{_W}tc"):
            text = " ".join(
                t for t in (_paragraph_text(p) for p in cell.iter(f"{_W}p")) if t
            )
            cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


def _read_styles(xml: bytes) -> dict[str, tuple[str, int | None, str | None]]:
    """styleId -> (name, outline level, the style it is based on)."""
    root = ElementTree.fromstring(xml)
    styles: dict[str, tuple[str, int | None, str | None]] = {}
    for style in root.iter(f"{_W}style"):
        style_id = style.get(f"{_W}styleId")
        if not style_id:
            continue
        name_el = style.find(f"{_W}name")
        name = (name_el.get(f"{_W}val") if name_el is not None else "") or ""
        outline_el = style.find(f"{_W}pPr/{_W}outlineLvl")
        outline = _outline(outline_el)
        based_el = style.find(f"{_W}basedOn")
        based = based_el.get(f"{_W}val") if based_el is not None else None
        styles[style_id] = (name.strip().lower(), outline, based)
    return styles


def _outline(element: ElementTree.Element | None) -> int | None:
    """Word's outline level is 0-based and 9 means body text."""
    if element is None:
        return None
    try:
        value = int(element.get(f"{_W}val", ""))
    except ValueError:
        return None
    return value + 1 if 0 <= value < 9 else None


def _heading_level(
    paragraph: ElementTree.Element,
    styles: dict[str, tuple[str, int | None, str | None]],
) -> int | None:
    own = _outline(paragraph.find(f"{_W}pPr/{_W}outlineLvl"))
    if own:
        return own

    style_el = paragraph.find(f"{_W}pPr/{_W}pStyle")
    style_id = style_el.get(f"{_W}val") if style_el is not None else None
    for _ in range(10):  # a basedOn chain deeper than this is not a heading
        if not style_id:
            return None
        if style_id not in styles:
            # No styles part to consult: fall back to the English built-in ids.
            match = re.match(r"^Heading(\d)$", style_id)
            return int(match.group(1)) if match else None
        name, outline, based = styles[style_id]
        match = _HEADING_STYLE.match(name)
        if match:
            return int(match.group(1))
        if name == "title":
            return 1
        if outline:
            return outline
        style_id = based
    return None
