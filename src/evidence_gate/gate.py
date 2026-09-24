"""The evidence gate.

One question, asked before anything is written: is what came back enough to
answer with?

Most retrieval systems skip this. They rank passages, hand the best few to a
language model, and the model writes something. It always writes something,
because that is what it is for. The gate is the piece that is allowed to say
no, and it is the reason this repository exists.

Two conditions decline, and a third only flags.

**The floor.** If nothing scored above `min_score`, nothing was close enough.
This is the ordinary case and the easy one.

**The anchors.** If the question pins something down -- a number, a quoted
phrase, a term the caller declared -- then the passage that answers it has to
contain that thing. Not the topic: the thing. This catches the failure that
scores cannot: a passage about escalation after 24 hours, retrieved for a
question about 48, reads perfectly and is wrong. The prose gives no sign. The
citation is real. Only the missing anchor shows it.

**The ambiguity flag.** When the top two passages come from different sections
and their scores are nearly tied, the gate answers but says so. It is the same
failure seen from the other side, and two independent detectors for one
failure is not redundancy, it is how you find out when one of them is broken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .embedding import normalise
from .types import Passage, Verdict

_QUOTED = re.compile(r'"([^"]{2,80})"')
_NUMERIC = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def extract_anchors(question: str) -> tuple[str, ...]:
    """Pull the terms a passage must contain to be answering *this* question.

    Deliberately narrow: quoted phrases and numbers. Both are things the asker
    typed on purpose, and both are exactly what distinguishes two otherwise
    identical sections of a document.

    It does not try to find entities. Guessing which words matter is where a
    heuristic starts being wrong quietly, and quiet wrongness is the thing this
    library is against. When a question needs an anchor that is neither quoted
    nor numeric, declare it: `gate.decide(q, passages, anchors=("escalation",))`.
    That is not a workaround. Explicit beats clever.
    """
    anchors: list[str] = []
    for phrase in _QUOTED.findall(question):
        cleaned = phrase.strip()
        if cleaned:
            anchors.append(cleaned)

    stripped = _QUOTED.sub(" ", question)
    for number in _NUMERIC.findall(stripped):
        anchors.append(number)

    seen: list[str] = []
    for a in anchors:
        if a.casefold() not in {s.casefold() for s in seen}:
            seen.append(a)
    return tuple(seen)


def _contains(haystack: str, needle: str) -> bool:
    return normalise(needle) in normalise(haystack)


@dataclass(frozen=True)
class GatePolicy:
    """Where the gate is set.

    Every number here is a trade, and the trade runs one way: raise the bar and
    the system answers less often and is wrong less often. There is no setting
    that gives you both. Choosing it is the client's call, not the engineer's,
    because only the client knows what a wrong answer costs them.

    **`min_score` must be calibrated against your own corpus.** The default is
    deliberately low, and that is a design choice rather than a loose one: the
    floor is only meant to catch "nothing retrieved is even on topic". Deciding
    whether an on-topic passage actually answers the question is the anchors'
    job, and a floor raised high enough to do it as well would be doing it by
    accident -- declining correct answers and rejecting wrong ones for reasons
    nobody could explain afterwards.

    The defaults here were set from measured scores on the example corpus, not
    chosen. Measure yours before you change them. tests/test_units.py pins the
    division of labour so that raising the floor to paper over a bad anchor
    fails loudly.
    """

    min_score: float = 0.15
    max_passages: int = 4
    require_anchors: bool = True
    ambiguity_margin: float = 0.04

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be within [0, 1]")
        if self.max_passages < 1:
            raise ValueError("max_passages must be at least 1")
        if self.ambiguity_margin < 0.0:
            raise ValueError("ambiguity_margin must not be negative")


class EvidenceGate:
    """Decides whether retrieved passages support an answer."""

    def __init__(self, policy: GatePolicy | None = None) -> None:
        self.policy = policy or GatePolicy()

    def decide(
        self,
        question: str,
        passages: Sequence[Passage],
        *,
        anchors: Sequence[str] | None = None,
    ) -> Verdict:
        policy = self.policy

        if not passages:
            return Verdict(
                answerable=False,
                passages=(),
                reason="nothing was retrieved for this question",
            )

        candidates = tuple(p for p in passages if p.score >= policy.min_score)

        if not candidates:
            best = max(p.score for p in passages)
            return Verdict(
                answerable=False,
                passages=(),
                reason=(
                    f"no passage reached the evidence floor "
                    f"(best {best:.3f} < {policy.min_score:.3f})"
                ),
            )

        required = tuple(anchors) if anchors is not None else extract_anchors(question)
        missing = tuple(
            a
            for a in required
            if not any(_contains(p.chunk.text, a) for p in candidates)
        )

        # Anchors do not only decide, they select. A passage that lacks what
        # the question pinned down is not evidence for that question, and
        # citing it alongside the right one puts the decoy back into the answer
        # after the gate went to the trouble of finding it.
        if required and policy.require_anchors:
            evidence = tuple(
                p
                for p in candidates
                if all(_contains(p.chunk.text, a) for a in required)
            )
            if not evidence:
                if missing:
                    return Verdict(
                        answerable=False,
                        passages=(),
                        reason=(
                            "the retrieved passages are about the right topic "
                            "but do not contain what the question pinned down"
                        ),
                        missing=missing,
                    )
                return Verdict(
                    answerable=False,
                    passages=(),
                    reason=(
                        "no single passage contains everything the question "
                        "pinned down; the terms are spread across different "
                        "passages, which is not the same as being answered"
                    ),
                    missing=required,
                )
        else:
            evidence = candidates

        selected = evidence[: policy.max_passages]

        flags: list[str] = []
        if len(passages) >= 2:
            first, second = passages[0], passages[1]
            same_file = first.source.file == second.source.file
            other_section = first.source.section != second.source.section
            close = abs(first.score - second.score) <= policy.ambiguity_margin
            if same_file and other_section and close:
                flags.append(
                    f"near-tie between two sections of {first.source.file}: "
                    f"{first.source.section!r} ({first.score:.3f}) and "
                    f"{second.source.section!r} ({second.score:.3f})"
                )
        if missing:
            flags.append(
                "answered without these anchors present: " + ", ".join(missing)
            )

        return Verdict(
            answerable=True,
            passages=selected,
            reason=f"{len(selected)} passage(s) above the evidence floor",
            missing=missing,
            flags=tuple(flags),
        )
