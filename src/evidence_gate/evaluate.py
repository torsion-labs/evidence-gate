"""The test set, and the report it produces.

This is the module the whole repository is arranged around.

A case pairs a question with the source that holds its answer. The assistant
passes only when it cites that source. Not when it sounds right, not when a
reviewer nods at the prose: when the citation matches what was agreed before
any code was written.

Cases with `expects_answer=False` are the ones nobody writes. The answer is
deliberately absent from the corpus, and passing means the assistant declined.
They are worth as much as the rest, because a system that never refuses has
not been tested on the only failure that is invisible from the outside.

The report prints failures. A report that only shows passes is not evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .pipeline import Assistant
from .types import Answer


@dataclass(frozen=True)
class Case:
    """One agreed question, and what answering it correctly means."""

    id: str
    question: str
    expected_file: str | None = None
    expected_page: int | None = None
    expects_answer: bool = True
    anchors: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.expects_answer and not self.expected_file:
            raise ValueError(
                f"case {self.id!r} expects an answer but names no expected file; "
                f"a case without an expected source cannot fail honestly"
            )
        if not self.expects_answer and self.expected_file:
            raise ValueError(
                f"case {self.id!r} expects a refusal but names an expected file"
            )


@dataclass(frozen=True)
class Result:
    """What happened when one case ran."""

    case: Case
    answer: Answer
    cited_file: str | None
    cited_page: int | None
    passed: bool
    detail: str
    flags: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "should-answer" if self.case.expects_answer else "should-not-answer"


@dataclass
class Report:
    """The results of a run, and the few numbers worth quoting."""

    results: list[Result] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if not r.passed]

    @property
    def flagged(self) -> list[Result]:
        return [r for r in self.results if r.flags]

    @property
    def refusal_cases(self) -> int:
        return sum(1 for r in self.results if not r.case.expects_answer)

    @property
    def green(self) -> bool:
        return self.total > 0 and not self.failed

    def to_markdown(self) -> str:
        lines = [
            "# Retrieval test report",
            "",
            f"**{self.passed} of {self.total}** citations matched the expected source.",
            f"{len(self.failed)} failed. {len(self.flagged)} flagged for review. "
            f"{self.refusal_cases} of the cases are should-not-answer cases.",
            "",
            "| # | Question | Expected | Cited | Result |",
            "|---|---|---|---|---|",
        ]
        for r in self.results:
            expected = (
                "— not in corpus"
                if not r.case.expects_answer
                else _cite(r.case.expected_file, r.case.expected_page)
            )
            cited = (
                "no answer given"
                if not r.answer.answered
                else _cite(r.cited_file, r.cited_page)
            )
            verdict = "PASS" if r.passed else "FAIL"
            lines.append(
                f"| {r.case.id} | {r.case.question} | {expected} | {cited} | {verdict} |"
            )

        if self.failed:
            lines += ["", "## What failed", ""]
            for r in self.failed:
                lines.append(f"**{r.case.id}** — {r.detail}")
                if r.case.note:
                    lines.append(f"  {r.case.note}")
                lines.append("")

        if self.flagged:
            lines += ["", "## Flagged for review", ""]
            for r in self.flagged:
                for f in r.flags:
                    lines.append(f"**{r.case.id}** — {f}")
            lines.append("")

        lines += [
            "",
            "This report measures citation correctness, not writing quality. "
            "It bounds the questions that were agreed; it says nothing about "
            "questions outside that set.",
        ]
        return "\n".join(lines)


def _cite(file: str | None, page: int | None) -> str:
    if not file:
        return "—"
    return f"{file} p.{page}" if page is not None else file


def run_case(assistant: Assistant, case: Case) -> Result:
    verdict = assistant.consider(
        case.question, anchors=case.anchors if case.anchors else None
    )
    answer = assistant.ask(
        case.question, anchors=case.anchors if case.anchors else None
    )

    cited_file: str | None = None
    cited_page: int | None = None
    if answer.citations:
        cited_file = answer.citations[0].source.file
        cited_page = answer.citations[0].source.page

    if not case.expects_answer:
        passed = not answer.answered
        detail = (
            "declined, as expected"
            if passed
            else f"answered from {_cite(cited_file, cited_page)} when it should have declined"
        )
        return Result(case, answer, cited_file, cited_page, passed, detail, verdict.flags)

    if not answer.answered:
        return Result(
            case,
            answer,
            None,
            None,
            False,
            f"declined, but the answer is in {_cite(case.expected_file, case.expected_page)}"
            + (f" (missing: {', '.join(answer.missing)})" if answer.missing else ""),
            verdict.flags,
        )

    file_ok = cited_file == case.expected_file
    page_ok = case.expected_page is None or cited_page == case.expected_page
    passed = file_ok and page_ok
    if passed:
        detail = "cited the expected source"
    elif not file_ok:
        detail = (
            f"cited {_cite(cited_file, cited_page)}, expected "
            f"{_cite(case.expected_file, case.expected_page)}"
        )
    else:
        detail = (
            f"right file, wrong place: cited p.{cited_page}, "
            f"expected p.{case.expected_page}"
        )
    return Result(case, answer, cited_file, cited_page, passed, detail, verdict.flags)


def run(assistant: Assistant, cases: Iterable[Case]) -> Report:
    return Report(results=[run_case(assistant, c) for c in cases])


def load_cases(path: str | Path) -> list[Case]:
    """Read a test set from JSON.

    Kept to the standard library on purpose: a test set that needs a YAML
    parser installed is one more reason for someone not to run it.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[Case] = []
    for row in data:
        cases.append(
            Case(
                id=str(row["id"]),
                question=row["question"],
                expected_file=row.get("expected_file"),
                expected_page=row.get("expected_page"),
                expects_answer=row.get("expects_answer", True),
                anchors=tuple(row.get("anchors", ())),
                note=row.get("note", ""),
            )
        )
    return cases


def save_cases(cases: Sequence[Case], path: str | Path) -> None:
    rows = []
    for c in cases:
        row: dict = {"id": c.id, "question": c.question}
        if c.expected_file:
            row["expected_file"] = c.expected_file
        if c.expected_page is not None:
            row["expected_page"] = c.expected_page
        if not c.expects_answer:
            row["expects_answer"] = False
        if c.anchors:
            row["anchors"] = list(c.anchors)
        if c.note:
            row["note"] = c.note
        rows.append(row)
    Path(path).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
