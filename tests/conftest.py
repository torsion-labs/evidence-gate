"""Fixtures.

The corpus below is small and synthetic, and it is built around two specific
failures rather than around being realistic. Both come from the same place: a
retrieval system that is right nearly all the time and wrong in a way nobody
can see.

`runbook.md` has two sections that say almost exactly the same thing. The only
difference between them lives in the heading -- 24 hours against 48 -- and the
bodies are deliberately interchangeable. Retrieval scores them nearly alike. A
system that picks the wrong one produces coherent prose and a real citation,
and a reviewer reading the answer has no way to tell.

`handbook.md` talks about the company at length and never says how many people
work there. A question about headcount retrieves it, because it is obviously
the relevant document. Answering from it is the second failure.
"""

from __future__ import annotations

import pytest

from evidence_gate import Assistant, ChunkingPolicy, GatePolicy, chunk_page

RUNBOOK = """\
INCIDENT RESPONSE

An incident is any interruption a customer can notice. The on-call engineer
acknowledges it, opens an incident record and starts the response clock.

ESCALATION AFTER 24 HOURS

If the incident is still open, the on-call engineer notifies the service owner
and records the delay in the incident log. The service owner decides whether to
widen the response and may bring in a second team.

ESCALATION AFTER 48 HOURS

If the incident is still open, the on-call engineer notifies the head of
operations and records the delay in the incident log. The head of operations
decides whether to widen the response and may bring in a second team.
"""

CONTRACT = """\
NOTICE PERIOD

Either party may end this agreement by giving written notice of 30 days. Notice
takes effect on the first day of the following month and does not affect work
already accepted.

INVOICING

Invoices are issued monthly in euros against accepted milestones. Payment is due
within 21 days of the invoice date.
"""

HANDBOOK = """\
ABOUT THE COMPANY

The company was founded to make technical reporting less manual. It works with
consultancies across several countries and keeps its own documentation in the
same system it sells.

HOW WE WORK

Written first. Decisions are recorded where the work happens rather than in a
meeting, so that someone joining later can reconstruct why a thing was done.
"""


@pytest.fixture
def corpus():
    """The three documents, chunked with the default policy."""
    chunks = []
    chunks += chunk_page(RUNBOOK, file="runbook.md", page=1)
    chunks += chunk_page(CONTRACT, file="contract.md", page=1)
    chunks += chunk_page(HANDBOOK, file="handbook.md", page=1)
    return chunks


@pytest.fixture
def corpus_without_section_prefix():
    """The same documents, chunked without carrying headings into the text.

    This exists so the cost of the chunking fix can be measured instead of
    asserted. See tests/test_acceptance.py.
    """
    policy = ChunkingPolicy(prepend_section=False)
    chunks = []
    chunks += chunk_page(RUNBOOK, file="runbook.md", page=1, policy=policy)
    chunks += chunk_page(CONTRACT, file="contract.md", page=1, policy=policy)
    chunks += chunk_page(HANDBOOK, file="handbook.md", page=1, policy=policy)
    return chunks


@pytest.fixture
def assistant(corpus):
    return Assistant.build(corpus)


@pytest.fixture
def lenient_assistant(corpus):
    """An assistant with the anchor check switched off.

    Not a convenience. It is the control: the tests use it to show what the
    system does when the gate is relaxed, which is the only way to claim the
    gate is doing anything.
    """
    return Assistant.build(corpus, policy=GatePolicy(require_anchors=False))
