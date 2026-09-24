"""The two failures this library exists to catch.

If you read one file in this repository, read this one. Everything else is
plumbing that makes these four tests possible.
"""

from __future__ import annotations

from evidence_gate import Assistant, Case, GatePolicy, run

ESCALATION_48 = "What is the escalation path after 48 hours?"
HEADCOUNT = "How many employees does the company have?"


class TestAdjacentSections:
    """A passage about 24 hours, retrieved for a question about 48.

    The prose is coherent. The citation is real. The document is the right one.
    Only the missing number shows that the answer is wrong, which is why the
    gate looks for the number.
    """

    def test_the_two_sections_really_are_hard_to_tell_apart(self, assistant):
        """Establish the trap before testing the escape from it.

        A test that the gate catches an error is worthless if the error was
        never close to happening. This asserts the passages genuinely compete.
        """
        passages = assistant.store.search(ESCALATION_48, k=4)
        sections = [p.source.section for p in passages[:2]]
        assert "ESCALATION AFTER 48 HOURS" in sections
        assert "ESCALATION AFTER 24 HOURS" in sections

        gap = abs(passages[0].score - passages[1].score)
        assert gap < 0.25, (
            f"the two sections scored {gap:.3f} apart, which is not a near-tie; "
            f"the fixture no longer reproduces the failure it was built for"
        )

    def test_answers_from_the_right_section(self, assistant):
        answer = assistant.ask(ESCALATION_48)

        assert answer.answered
        assert answer.citations[0].source.section == "ESCALATION AFTER 48 HOURS"
        assert "head of operations" in answer.text
        assert "service owner" not in answer.text

    def test_without_the_section_in_the_chunk_it_refuses_rather_than_guesses(
        self, corpus_without_section_prefix
    ):
        """The chunking fix, measured rather than asserted.

        Without the heading carried into the chunk, nothing in the corpus
        contains the number 48, so the anchor is missing everywhere and the
        gate declines. That is the correct outcome for a system that cannot
        tell the two sections apart: it is unhelpful, and it is not wrong.

        The failure being prevented is the third possibility -- answering from
        the 24-hour section with full confidence.
        """
        assistant = Assistant.build(corpus_without_section_prefix)
        answer = assistant.ask(ESCALATION_48)

        assert not answer.answered
        assert "48" in answer.missing

    def test_with_the_gate_relaxed_it_gets_it_wrong(
        self, corpus_without_section_prefix
    ):
        """The control.

        Same corpus, same question, anchor check off. The system answers, from
        the wrong section, and nothing about the answer looks wrong. This is
        what the rest of the tests are protecting against, and it is worth
        seeing it happen once.
        """
        assistant = Assistant.build(
            corpus_without_section_prefix, policy=GatePolicy(require_anchors=False)
        )
        answer = assistant.ask(ESCALATION_48)

        assert answer.answered
        assert answer.citations[0].source.file == "runbook.md"
        assert answer.citations[0].source.section != "ESCALATION AFTER 48 HOURS"

        # It cited a real section of the right file, and got the question
        # wrong. Nothing about the answer is detectably off from reading it,
        # which is exactly the problem.
        #
        # Worth noting what the measurement actually showed here: without the
        # heading carried into the chunk, the two escalation sections stop
        # being findable at all. Their bodies never use the word "escalation"
        # -- it lives only in the heading -- so the best lexical match for an
        # escalation question becomes the general incident section. The fix in
        # chunking.py is doing more work than catching a near-tie: it is what
        # makes these sections retrievable in the first place.


class TestShouldNotAnswer:
    """The answer is not in the corpus, and the assistant has to say so.

    The document about the company is plainly the relevant one, scores well and
    contains nothing about headcount. Retrieval has done its job; the gate has
    to do the rest.
    """

    def test_declines_when_the_answer_is_absent(self, assistant):
        answer = assistant.ask(HEADCOUNT, anchors=("employees",))

        assert not answer.answered
        assert not answer.citations
        assert "employees" in answer.missing

    def test_the_refusal_says_what_was_missing(self, assistant):
        answer = assistant.ask(HEADCOUNT, anchors=("employees",))

        assert "employees" in answer.text
        assert "could not" in answer.text.lower()

    def test_with_the_gate_relaxed_it_answers_from_a_passing_mention(
        self, lenient_assistant
    ):
        """The control again.

        Relax the anchor check and the assistant answers a headcount question
        from a page that never mentions headcount, because that page really is
        the closest thing in the corpus. Retrieval is not wrong here. Answering
        is.
        """
        answer = lenient_assistant.ask(HEADCOUNT, anchors=("employees",))

        assert answer.answered
        assert answer.citations[0].source.file == "handbook.md"


class TestTheSuiteAsDelivered:
    """The test set as a client would receive it: cases in, report out."""

    def test_a_green_run_is_green_and_says_so(self, assistant):
        cases = [
            Case(
                id="01",
                question="What is the notice period for ending the agreement?",
                expected_file="contract.md",
                expected_page=1,
            ),
            Case(
                id="02",
                question=ESCALATION_48,
                expected_file="runbook.md",
                expected_page=1,
            ),
            Case(
                id="03",
                question=HEADCOUNT,
                expects_answer=False,
                anchors=("employees",),
                note="the corpus never states headcount",
            ),
        ]

        report = run(assistant, cases)

        assert report.green
        assert report.total == 3
        assert report.passed == 3
        assert report.refusal_cases == 1

    def test_a_failure_is_reported_with_its_reason(self, assistant):
        """A case pointed at the wrong file must fail, and say what it cited.

        A report that cannot produce a failure has not demonstrated anything.
        """
        cases = [
            Case(
                id="X1",
                question="What is the notice period for ending the agreement?",
                expected_file="handbook.md",
                expected_page=1,
            )
        ]

        report = run(assistant, cases)

        assert not report.green
        assert len(report.failed) == 1
        detail = report.failed[0].detail
        assert "contract.md" in detail
        assert "handbook.md" in detail

        markdown = report.to_markdown()
        assert "## What failed" in markdown
        assert "X1" in markdown
