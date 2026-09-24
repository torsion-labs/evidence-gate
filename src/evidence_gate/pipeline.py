"""The three pieces wired together.

Retrieve, gate, answer. Nothing clever happens here; the value is that the
order is fixed and the gate cannot be skipped. An assistant that could answer
without consulting the gate would be a different library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .answer import build
from .gate import EvidenceGate, GatePolicy
from .ports import Store, Writer
from .store import InMemoryStore
from .types import Answer, Chunk, Verdict


@dataclass
class Assistant:
    """Answers questions over an indexed corpus, or declines and says why."""

    store: Store
    gate: EvidenceGate
    writer: Writer | None = None
    top_k: int = 8

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk] = (),
        *,
        policy: GatePolicy | None = None,
        writer: Writer | None = None,
    ) -> "Assistant":
        store = InMemoryStore()
        if chunks:
            store.add(chunks)
        return cls(store=store, gate=EvidenceGate(policy), writer=writer)

    def consider(
        self, question: str, *, anchors: Sequence[str] | None = None
    ) -> Verdict:
        """Retrieve and gate, without composing anything.

        Exposed on purpose. The verdict is the auditable part, and a caller
        that only wants to know whether the corpus can answer a question should
        not have to pay for prose to find out.
        """
        passages = self.store.search(question, k=self.top_k)
        return self.gate.decide(question, passages, anchors=anchors)

    def ask(self, question: str, *, anchors: Sequence[str] | None = None) -> Answer:
        return build(question, self.consider(question, anchors=anchors), writer=self.writer)
