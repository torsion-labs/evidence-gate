"""Unit tests for the pieces underneath.

Grouped by the property being defended rather than by module, because what
matters about this code is not that each function works but that certain
things cannot happen.
"""

from __future__ import annotations

import pytest

from evidence_gate import (
    Answer,
    Assistant,
    ChunkingPolicy,
    Citation,
    EvidenceGate,
    GatePolicy,
    HashingEmbedder,
    InMemoryStore,
    Passage,
    QuotingWriter,
    Source,
    Verdict,
    chunk_page,
    citations_for,
    detect_heading,
    extract_anchors,
    normalise,
    tokenize,
)
from evidence_gate.types import Chunk


class TestNothingIsWithoutProvenance:
    """The invariants that make a citation mean something."""

    def test_a_source_needs_a_file(self):
        with pytest.raises(ValueError):
            Source(file="   ")

    def test_pages_are_one_based(self):
        with pytest.raises(ValueError):
            Source(file="a.pdf", page=0)

    def test_an_answer_without_citations_is_not_an_answer(self):
        with pytest.raises(ValueError):
            Answer(answered=True, text="Yes, definitely.")

    def test_a_refusal_must_not_smuggle_citations(self):
        cite = Citation(chunk_id="x", source=Source("a.pdf", 1), quote="...")
        with pytest.raises(ValueError):
            Answer(answered=False, text="I could not find it.", citations=(cite,))

    def test_every_verdict_states_its_reason(self):
        with pytest.raises(ValueError):
            Verdict(answerable=False, passages=(), reason="")

    def test_an_empty_chunk_cannot_be_indexed(self):
        with pytest.raises(ValueError):
            Chunk(id="a", text="   \n ", source=Source("a.pdf", 1))


class TestChunkingKeepsWhereItCameFrom:
    def test_page_and_section_survive(self):
        chunks = chunk_page(
            "NOTICE PERIOD\n\nEither party may end this agreement by giving "
            "written notice of 30 days, effective the following month.",
            file="contract.md",
            page=7,
        )
        assert chunks
        assert all(c.source.page == 7 for c in chunks)
        assert chunks[0].source.section == "NOTICE PERIOD"

    def test_the_heading_is_carried_into_the_indexed_text_by_default(self):
        chunks = chunk_page(
            "ESCALATION AFTER 48 HOURS\n\nThe on-call engineer notifies the head "
            "of operations and records the delay in the incident log.",
            file="runbook.md",
            page=1,
        )
        assert "48" in chunks[0].text

    def test_and_is_not_when_the_policy_says_not_to(self):
        chunks = chunk_page(
            "ESCALATION AFTER 48 HOURS\n\nThe on-call engineer notifies the head "
            "of operations and records the delay in the incident log.",
            file="runbook.md",
            page=1,
            policy=ChunkingPolicy(prepend_section=False),
        )
        assert "48" not in chunks[0].text
        assert chunks[0].source.section == "ESCALATION AFTER 48 HOURS"

    def test_a_lone_heading_is_not_evidence(self):
        assert chunk_page("ESCALATION AFTER 48 HOURS\n", file="r.md", page=1) == []

    @pytest.mark.parametrize(
        "line, expected",
        [
            ("## Notice period", "Notice period"),
            ("NOTICE PERIOD", "NOTICE PERIOD"),
            ("Either party may end this agreement.", None),
            ("SHORT", None),
        ],
    )
    def test_heading_detection_is_conservative(self, line, expected):
        assert detect_heading(line) == expected


class TestTheEmbedderIsDeterministic:
    def test_same_text_same_vector_every_time(self):
        e = HashingEmbedder()
        assert e.embed_one("escalation after 48 hours") == e.embed_one(
            "escalation after 48 hours"
        )

    def test_two_instances_agree(self):
        assert HashingEmbedder(256).embed_one("x y z") == HashingEmbedder(
            256
        ).embed_one("x y z")

    def test_accents_and_case_do_not_split_a_token(self):
        assert tokenize("Página") == tokenize("pagina")
        assert normalise("ÁÉÍ") == "aei"

    def test_identifiers_survive_tokenisation(self):
        assert "contract-v3.pdf" in tokenize("see contract-v3.pdf for terms")

    def test_related_text_scores_above_unrelated(self):
        from evidence_gate import cosine

        e = HashingEmbedder()
        q = e.embed_one("notice period to end the agreement")
        near = e.embed_one("either party may end this agreement by giving notice")
        far = e.embed_one("invoices are issued monthly in euros")
        assert cosine(q, near) > cosine(q, far)


class TestTheStoreIsRepeatable:
    def test_indexing_twice_does_not_duplicate(self):
        chunks = chunk_page(
            "NOTICE PERIOD\n\nEither party may end this agreement by giving "
            "written notice of 30 days.",
            file="contract.md",
            page=1,
        )
        store = InMemoryStore()
        store.add(chunks)
        store.add(chunks)
        assert len(store) == len(chunks)

    def test_ranking_is_identical_between_runs(self, corpus):
        a, b = InMemoryStore(), InMemoryStore()
        a.add(corpus)
        b.add(list(reversed(corpus)))
        ids_a = [p.chunk.id for p in a.search("notice period", k=5)]
        ids_b = [p.chunk.id for p in b.search("notice period", k=5)]
        assert ids_a == ids_b

    def test_a_saved_index_comes_back_the_same(self, corpus, tmp_path):
        store = InMemoryStore()
        store.add(corpus)
        path = tmp_path / "index.json"
        store.save(path)

        reloaded = InMemoryStore.load(path)
        assert len(reloaded) == len(store)
        assert [p.chunk.id for p in reloaded.search("invoicing", k=3)] == [
            p.chunk.id for p in store.search("invoicing", k=3)
        ]

    def test_an_index_from_another_embedder_is_refused_not_reused(
        self, corpus, tmp_path
    ):
        """Silently querying a 512-dim index with a 256-dim embedder would
        return nonsense that looks like results. Refusing is the only safe
        behaviour, and the message says what to do about it."""
        store = InMemoryStore()
        store.add(corpus)
        path = tmp_path / "index.json"
        store.save(path)

        with pytest.raises(ValueError, match="not comparable"):
            InMemoryStore.load(path, embedder=HashingEmbedder(256))


class TestTheGate:
    def _passage(self, text, score, *, section=None, file="a.md"):
        return Passage(
            Chunk(
                id=f"{file}:{section}:{score}",
                text=text,
                source=Source(file=file, page=1, section=section),
            ),
            score,
        )

    def test_nothing_retrieved_means_no_answer(self):
        verdict = EvidenceGate().decide("anything?", [])
        assert not verdict.answerable

    def test_everything_below_the_floor_means_no_answer(self):
        gate = EvidenceGate(GatePolicy(min_score=0.5))
        verdict = gate.decide("q", [self._passage("some text", 0.1)])
        assert not verdict.answerable
        assert "floor" in verdict.reason

    def test_a_missing_anchor_declines_even_with_a_good_score(self):
        gate = EvidenceGate(GatePolicy(min_score=0.1))
        verdict = gate.decide(
            "escalation after 48 hours?",
            [self._passage("escalation after 24 hours: notify the owner", 0.9)],
        )
        assert not verdict.answerable
        assert verdict.missing == ("48",)

    def test_the_anchor_present_lets_it_through(self):
        gate = EvidenceGate(GatePolicy(min_score=0.1))
        verdict = gate.decide(
            "escalation after 48 hours?",
            [self._passage("escalation after 48 hours: notify operations", 0.9)],
        )
        assert verdict.answerable
        assert verdict.missing == ()

    def test_declared_anchors_override_the_extractor(self):
        gate = EvidenceGate(GatePolicy(min_score=0.1))
        verdict = gate.decide(
            "how many people work there?",
            [self._passage("the company works with consultancies", 0.9)],
            anchors=("employees",),
        )
        assert not verdict.answerable
        assert verdict.missing == ("employees",)

    def test_a_near_tie_between_sections_is_flagged_not_hidden(self):
        gate = EvidenceGate(GatePolicy(min_score=0.1, ambiguity_margin=0.05))
        verdict = gate.decide(
            "escalation after 48 hours?",
            [
                self._passage("escalation after 48 hours", 0.71, section="S48"),
                self._passage("escalation after 24 hours", 0.69, section="S24"),
            ],
        )
        assert verdict.answerable
        assert verdict.needs_review
        assert "near-tie" in verdict.flags[0]

    def test_a_clear_winner_is_not_flagged(self):
        gate = EvidenceGate(GatePolicy(min_score=0.1, ambiguity_margin=0.05))
        verdict = gate.decide(
            "escalation after 48 hours?",
            [
                self._passage("escalation after 48 hours", 0.90, section="S48"),
                self._passage("escalation after 24 hours", 0.40, section="S24"),
            ],
        )
        assert verdict.answerable
        assert not verdict.needs_review

    @pytest.mark.parametrize(
        "question, expected",
        [
            ("escalation after 48 hours?", ("48",)),
            ('what does "force majeure" cover?', ("force majeure",)),
            ("who approves it?", ()),
            ("the 2026 rate of 21 percent", ("2026", "21")),
        ],
    )
    def test_anchor_extraction(self, question, expected):
        assert extract_anchors(question) == expected

    def test_an_impossible_policy_is_refused_at_construction(self):
        with pytest.raises(ValueError):
            GatePolicy(min_score=1.5)

    def test_the_floor_and_the_anchors_have_different_jobs(self, assistant):
        """Pins the division of labour, because it is easy to blur by accident.

        The headcount question retrieves the company page at a perfectly
        respectable score. It clears the floor. It is declined by the anchor
        check, because the page never mentions employees.

        That distinction is the design. A floor raised high enough to reject
        this question as well would also reject correct answers, and would be
        rejecting for a reason nobody could explain: not "the passage does not
        contain what you asked about" but "the number was small". This test
        fails if someone tunes the floor to cover for a weak anchor.
        """
        question = "How many employees does the company have?"
        best = assistant.store.search(question, k=1)[0]

        assert best.score >= assistant.gate.policy.min_score, (
            "the fixture no longer clears the floor, so this test would pass "
            "for the wrong reason"
        )

        verdict = assistant.consider(question, anchors=("employees",))
        assert not verdict.answerable
        assert verdict.missing == ("employees",)
        assert "floor" not in verdict.reason


class TestAWriterCannotChangeWhatTheAnswerRestsOn:
    """The rule from the brief: the model writes prose, never values.

    It is enforced by ordering rather than by trust. Citations are derived from
    the gated passages before any writer runs, so a writer has nothing to
    invent with.
    """

    def test_a_lying_writer_cannot_add_a_source(self, assistant):
        class LyingWriter:
            def write(self, question, passages):
                return "According to invented-source.pdf p.99, the answer is 42."

        answer = assistant.ask(
            "What is the notice period for ending the agreement?",
        )
        assert answer.answered
        honest = answer.cited_files

        assistant.writer = LyingWriter()
        lied = assistant.ask("What is the notice period for ending the agreement?")

        assert "invented-source.pdf" in lied.text
        assert lied.cited_files == honest
        assert all(c.source.file != "invented-source.pdf" for c in lied.citations)

    def test_an_empty_writer_does_not_produce_an_empty_answer(self, assistant):
        class SilentWriter:
            def write(self, question, passages):
                return "   "

        assistant.writer = SilentWriter()
        answer = assistant.ask("What is the notice period for ending the agreement?")

        assert answer.answered
        assert answer.text.strip()

    def test_citations_point_at_chunks_that_exist(self, assistant):
        answer = assistant.ask("When are invoices issued?")
        indexed = {c.id for c in assistant.store._chunks}  # noqa: SLF001
        assert answer.citations
        assert all(c.chunk_id in indexed for c in answer.citations)

    def test_the_default_writer_quotes_rather_than_composes(self):
        chunk = Chunk(
            id="1",
            text="Invoices are issued monthly in euros.",
            source=Source("contract.md", 1, "INVOICING"),
        )
        text = QuotingWriter().write("q", [Passage(chunk, 0.9)])
        assert "contract.md p.1" in text
        assert "Invoices are issued monthly in euros." in text

    def test_citations_are_derived_only_from_passages(self):
        chunk = Chunk(id="1", text="x" * 500, source=Source("a.md", 2))
        cites = citations_for([Passage(chunk, 0.5)])
        assert len(cites) == 1
        assert cites[0].source.page == 2
        assert cites[0].quote.endswith("...")


class TestTheAssistantCannotSkipTheGate:
    def test_consider_and_ask_agree(self, assistant):
        q = "What is the escalation path after 48 hours?"
        verdict = assistant.consider(q)
        answer = assistant.ask(q)
        assert verdict.answerable == answer.answered

    def test_an_empty_corpus_answers_nothing(self):
        answer = Assistant.build([]).ask("anything at all?")
        assert not answer.answered
