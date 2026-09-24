"""Readers and ingestion.

The documents here are built by the tests themselves -- a DOCX is a zip of XML
and a PDF is a short text format -- so there are no binary fixtures to trust
and nothing to download. PDF tests need pypdf and are skipped without it; the
rest need nothing.

Most of what is tested is not the happy path. It is what a reader does with a
page that has no text, a heading that sits at the bottom of a page, a heading
style named in Spanish, a format nobody wrote a reader for. Those are the
cases where an index ends up quietly smaller than the documents it came from.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from evidence_gate import (
    Assistant,
    DocxReader,
    PdfReader,
    Source,
    TextReader,
    chunk_document,
    ingest,
    separate_headings,
)

# --------------------------------------------------------------------------
# Document builders
# --------------------------------------------------------------------------

_WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def para(text: str, *, style: str | None = None, outline: int | None = None) -> str:
    props = ""
    if style or outline is not None:
        inner = f'<w:pStyle w:val="{style}"/>' if style else ""
        inner += f'<w:outlineLvl w:val="{outline}"/>' if outline is not None else ""
        props = f"<w:pPr>{inner}</w:pPr>"
    return f"<w:p>{props}<w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"


def table(rows: list[list[str]]) -> str:
    body = "".join(
        "<w:tr>" + "".join(f"<w:tc>{para(cell)}</w:tc>" for cell in row) + "</w:tr>"
        for row in rows
    )
    return f"<w:tbl>{body}</w:tbl>"


def make_docx(path: Path, body: str, styles: dict[str, str] | None = None) -> Path:
    """A minimal but real .docx: the parts Word needs to open it."""
    styles = {"Heading1": "heading 1", "Heading2": "heading 2"} if styles is None else styles
    style_xml = "".join(
        f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/></w:style>'
        for sid, name in styles.items()
    )
    parts = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>"
        ),
        "word/document.xml": (
            f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{_WNS}">'
            f"<w:body>{body}<w:sectPr/></w:body></w:document>"
        ),
        "word/styles.xml": (
            f'<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="{_WNS}">'
            f"{style_xml}</w:styles>"
        ),
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, xml in parts.items():
            z.writestr(name, xml)
    return path


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    """A real PDF, one list of lines per page. An empty list is a page with no
    text on it, which is what a scanned page looks like to a text extractor."""
    objects: list[bytes] = [b"", b""]  # catalog and page tree, filled in below
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font = len(objects)
    kids: list[int] = []
    for lines in pages:
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 720 Td"]
        ops += [f"({_pdf_escape(line)}) Tj T*" for line in lines]
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1") if lines else b""
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content = len(objects)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"
            ).encode()
        )
        kids.append(len(objects))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    ).encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))
    return path


HANDBOOK_PAGES = [
    [
        "REFUND POLICY",
        "Customers can request a refund within 30 days of purchase by writing",
        "to the support desk. Refunds go back to the original payment method.",
    ],
    [],  # a scanned page: an image, no text layer
    [
        "Standard orders leave the warehouse within two business days, and",
        "tracking details are sent by email once the parcel is collected.",
        "WARRANTY TERMS",
    ],
    [
        "The warranty covers manufacturing defects for twelve months from",
        "the delivery date. It does not cover damage caused by misuse.",
    ],
]

POLICY_BODY = "".join(
    [
        para("Annual leave", style="Heading1"),
        para("Employees accrue annual leave at a rate of two days per month worked."),
        para("Sick leave", style="Heading1"),
        para("Tell your manager before 10:00."),
        para("Bring a doctor's note after three days."),
        para("Allowances", style="Heading1"),
        para("Leave allowance by plan, in working days per calendar year:"),
        table([["Plan", "Days"], ["Standard", "22"], ["Senior", "25"]]),
        para("Carry-over", style="Ttulo2"),
        para("Up to five unused days can be carried into the first quarter of next year."),
        para("Public holidays", outline=1),
        para("Public holidays follow the calendar of the office where you are based."),
    ]
)

# Word in Spanish stores its built-in heading under a localised id; the
# internal name stays "heading 2", and that is what the reader must go by.
POLICY_STYLES = {"Heading1": "heading 1", "Ttulo2": "heading 2"}


# --------------------------------------------------------------------------
# Plain text
# --------------------------------------------------------------------------


class TestTextReader:
    def test_a_text_file_has_no_page_number(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("# Notes\n\nSomething worth keeping.", encoding="utf-8")
        assert TextReader().read(path) == [(None, "# Notes\n\nSomething worth keeping.")]

    def test_form_feeds_are_the_page_breaks_a_conversion_left_behind(self, tmp_path):
        path = tmp_path / "converted.txt"
        path.write_text("page one\fpage two\fpage three", encoding="utf-8")
        assert [p for p, _ in TextReader().read(path)] == [1, 2, 3]

    def test_a_byte_order_mark_does_not_become_text(self, tmp_path):
        path = tmp_path / "bom.txt"
        path.write_bytes("﻿Hello".encode("utf-8"))
        assert TextReader().read(path) == [(None, "Hello")]


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------


class TestDocxReader:
    @pytest.fixture
    def chunks(self, tmp_path):
        path = make_docx(tmp_path / "policy.docx", POLICY_BODY, POLICY_STYLES)
        return chunk_document(DocxReader().read(path), file="policy.docx")

    def _by_section(self, chunks):
        return {c.source.section: c for c in chunks}

    def test_headings_become_sections(self, chunks):
        sections = self._by_section(chunks)
        assert "rate of two days per month" in sections["Annual leave"].text

    def test_a_docx_has_no_page_numbers_and_says_so(self, chunks):
        assert {c.source.page for c in chunks} == {None}

    def test_the_citation_falls_back_to_the_section(self, chunks):
        chunk = self._by_section(chunks)["Annual leave"]
        assert chunk.source.cite() == "policy.docx §Annual leave"

    def test_short_paragraphs_under_one_heading_are_kept_together(self, chunks):
        """Each sentence alone is under the chunker's minimum and would be
        dropped. Read as one block under their heading, they are indexed."""
        text = self._by_section(chunks)["Sick leave"].text
        assert "before 10:00" in text and "after three days" in text

    def test_tables_are_read_row_by_row_with_the_text_that_introduces_them(
        self, chunks
    ):
        text = self._by_section(chunks)["Allowances"].text
        assert "working days per calendar year" in text
        assert "Senior | 25" in text

    def test_a_heading_style_is_recognised_by_what_it_is_not_what_it_is_called(
        self, chunks
    ):
        assert "Carry-over" in self._by_section(chunks)

    def test_an_outline_level_without_a_style_is_a_heading(self, chunks):
        assert "Public holidays" in self._by_section(chunks)

    def test_without_a_styles_part_the_english_ids_still_work(self, tmp_path):
        body = para("Scope", style="Heading1") + para(
            "This policy applies to every employee on a permanent contract."
        )
        path = make_docx(tmp_path / "bare.docx", body, styles={})
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        # rebuild without word/styles.xml at all
        stripped = tmp_path / "stripped.docx"
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(stripped, "w") as dst:
            for name in names:
                if name != "word/styles.xml":
                    dst.writestr(name, src.read(name))
        chunks = chunk_document(DocxReader().read(stripped), file="stripped.docx")
        assert chunks[0].source.section == "Scope"


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


class TestPdfReader:
    @pytest.fixture
    def pages(self, tmp_path):
        pytest.importorskip("pypdf")
        path = make_pdf(tmp_path / "handbook.pdf", HANDBOOK_PAGES)
        return PdfReader().read(path)

    def test_every_page_keeps_its_real_number(self, pages):
        assert [p for p, _ in pages] == [1, 2, 3, 4]

    def test_a_page_without_text_is_returned_empty_not_dropped(self, pages):
        assert pages[1] == (2, "")

    def test_a_heading_in_the_middle_of_a_page_starts_its_own_block(self, pages):
        assert "\n\nWARRANTY TERMS" in pages[2][1]

    def test_a_section_continues_across_a_page_break(self, pages):
        """The heading sits at the bottom of page 3; its text is on page 4."""
        chunks = chunk_document(pages, file="handbook.pdf")
        page_four = [c for c in chunks if c.source.page == 4]
        assert page_four and page_four[0].source.section == "WARRANTY TERMS"
        assert page_four[0].text.startswith("WARRANTY TERMS\n")


def test_separate_headings_leaves_prose_alone():
    text = "An ordinary line.\nAnother ordinary line."
    assert separate_headings(text) == text


def test_a_section_carries_across_pages_without_any_reader():
    pages = [
        (1, "INTRODUCTION\n\nThis guide explains how returns are handled.\n\nRETURNS"),
        (2, "Items can be returned unused within the period stated on the receipt."),
    ]
    chunks = chunk_document(pages, file="guide.pdf")
    assert chunks[-1].source.page == 2
    assert chunks[-1].source.section == "RETURNS"


def test_a_citation_with_a_page_does_not_also_name_the_section():
    assert Source("guide.pdf", 3, "RETURNS").cite() == "guide.pdf p.3"


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


class TestIngest:
    @pytest.fixture
    def folder(self, tmp_path):
        pytest.importorskip("pypdf")
        docs = tmp_path / "docs"
        (docs / "hr").mkdir(parents=True)
        make_pdf(docs / "handbook.pdf", HANDBOOK_PAGES)
        make_pdf(docs / "scanned.pdf", [[], []])
        make_docx(docs / "hr" / "policy.docx", POLICY_BODY, POLICY_STYLES)
        (docs / "prices.xlsx").write_bytes(b"not read by anything here")
        (docs / "broken.docx").write_bytes(b"this is not a zip file")
        (docs / ".hidden.md").write_text("Hidden notes, never meant to be indexed at all.")
        (docs / "~$policy.docx").write_bytes(b"lock file")
        return docs

    @pytest.fixture
    def report(self, folder):
        return ingest([folder], root=folder)

    def test_what_was_read(self, report):
        assert report.read == ("handbook.pdf", "hr/policy.docx")

    def test_every_blank_page_is_reported(self, report):
        assert report.empty_pages == (("handbook.pdf", 2),)

    def test_every_unusable_file_is_reported_with_its_reason(self, report):
        reasons = {s.file: s.reason for s in report.skipped}
        assert set(reasons) == {"broken.docx", "prices.xlsx", "scanned.pdf"}
        assert reasons["prices.xlsx"] == "no reader for .xlsx"
        assert "OCR" in reasons["scanned.pdf"]
        assert reasons["broken.docx"].startswith("could not be read (BadZipFile")

    def test_hidden_and_lock_files_are_not_walked(self, report):
        mentioned = set(report.read) | {s.file for s in report.skipped}
        assert ".hidden.md" not in mentioned and "~$policy.docx" not in mentioned

    def test_an_incomplete_ingestion_says_so(self, report):
        assert not report.complete
        summary = report.summary()
        assert "handbook.pdf: no text on page(s) 2" in summary
        assert "prices.xlsx: no reader for .xlsx" in summary
        assert "Nothing was left out" not in summary

    def test_a_fragment_too_short_to_index_is_listed_not_lost(self, tmp_path):
        """Found by this suite: a three-row table alone under its heading is
        shorter than the chunker's minimum. It is not indexed -- raising the
        minimum for everyone is the wrong fix -- but the report names it, so
        whoever owns the document can see it and decide."""
        body = (
            para("Fees", style="Heading1")
            + table([["Plan", "Fee"], ["Basic", "9"]])
            + para("Payment", style="Heading1")
            + para("Invoices are issued monthly and are due within thirty days.")
        )
        make_docx(tmp_path / "fees.docx", body)
        report = ingest([tmp_path / "fees.docx"])
        assert ("fees.docx", None, "Plan | Fee Basic | 9") in report.too_short
        assert report.complete  # short fragments are listed, not alarmed on
        assert "'Plan | Fee Basic | 9'" in report.summary()

    def test_a_complete_ingestion_says_that_too(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("# Notes\n\nA note long enough to be worth indexing on its own.")
        report = ingest([path])
        assert report.complete
        assert report.summary().endswith("Nothing was left out.")

    def test_a_file_named_twice_is_indexed_once(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("# Notes\n\nA note long enough to be worth indexing on its own.")
        once = ingest([path])
        twice = ingest([path, tmp_path])
        assert len(twice.chunks) == len(once.chunks)

    def test_two_files_that_would_share_a_citation_are_refused(self, tmp_path):
        for folder in ("a", "b"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "notes.md").write_text(
                "# Notes\n\nA note long enough to be worth indexing on its own."
            )
        with pytest.raises(ValueError, match="pass root="):
            ingest([tmp_path / "a", tmp_path / "b"])
        report = ingest([tmp_path], root=tmp_path)
        assert report.read == ("a/notes.md", "b/notes.md")

    def test_a_missing_pdf_library_is_a_line_in_the_report_not_a_crash(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "pypdf", None)
        (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "notes.md").write_text(
            "# Notes\n\nA note long enough to be worth indexing on its own."
        )
        report = ingest([tmp_path])
        assert report.read == ("notes.md",)
        assert 'pip install "evidence-gate[pdf]"' in report.skipped[0].reason


# --------------------------------------------------------------------------
# End to end: real formats through the gate
# --------------------------------------------------------------------------


class TestFromDocumentsToAnswers:
    @pytest.fixture
    def assistant(self, tmp_path):
        pytest.importorskip("pypdf")
        make_pdf(tmp_path / "handbook.pdf", HANDBOOK_PAGES)
        make_docx(tmp_path / "policy.docx", POLICY_BODY, POLICY_STYLES)
        return Assistant.build(ingest([tmp_path]).chunks)

    def test_an_answer_from_a_pdf_cites_its_page(self, assistant):
        answer = assistant.ask("Can customers request a refund within 30 days?")
        assert answer.answered
        assert answer.citations[0].source.cite() == "handbook.pdf p.1"

    def test_an_answer_from_a_section_that_crossed_a_page_break(self, assistant):
        answer = assistant.ask(
            "How long does the warranty cover manufacturing defects?",
            anchors=("twelve months",),
        )
        assert answer.answered
        assert answer.citations[0].source.page == 4
        assert answer.citations[0].source.section == "WARRANTY TERMS"

    def test_an_answer_from_a_docx_cites_its_section(self, assistant):
        answer = assistant.ask(
            "When do I need a doctor's note for sick leave?",
            anchors=("three days",),
        )
        assert answer.answered
        assert answer.citations[0].source.cite() == "policy.docx §Sick leave"

    def test_a_question_the_documents_do_not_cover_is_declined(self, assistant):
        answer = assistant.ask(
            "What is the refund window for gift cards?", anchors=("gift card",)
        )
        assert not answer.answered
        assert answer.missing == ("gift card",)
