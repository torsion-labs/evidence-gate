# evidence-gate

[![tests](https://github.com/torsion-labs/evidence-gate/actions/workflows/tests.yml/badge.svg)](https://github.com/torsion-labs/evidence-gate/actions/workflows/tests.yml)

Retrieval over your own documents, with citations you can check and a gate that is allowed to say no.

Most retrieval systems rank passages, hand the best few to a language model, and the model writes something. It always writes something — that is what it is for. This library adds the piece that is missing: a gate that decides whether what came back is *enough*, and refuses when it is not.

```python
from evidence_gate import Assistant, chunk_page

chunks = chunk_page(open("runbook.md").read(), file="runbook.md", page=1)
assistant = Assistant.build(chunks)

answer = assistant.ask("What is the escalation path after 48 hours?")

answer.answered              # True
answer.citations[0].source.cite()   # 'runbook.md p.1'

answer = assistant.ask("How many employees are there?", anchors=("employees",))

answer.answered              # False
answer.missing               # ('employees',)
answer.text                  # "I could not answer that from your documents.
                             #  The passages I found are about the right topic
                             #  but none of them mentions: employees."
```

## Run the tests before you read the code

```bash
git clone <this repository>
cd evidence-gate
pip install pytest pypdf
python -m pytest
```

No API key. No network. No model download. The suite runs locally; elapsed time depends on the environment. Without `pypdf` the PDF tests are skipped, not failed, and everything else still runs.

The same suite runs on every change, on Python 3.10 to 3.14, once with `pypdf` and once without it. The badge at the top is that run, and nothing is merged into `main` unless it passes.

That is deliberate, and it is the first claim this repository makes about itself. A retrieval test that depends on a hosted model is not a test of your code — it is a test of someone else's service on the day you ran it. Everything that could reach the network sits behind a protocol in [`ports.py`](src/evidence_gate/ports.py), and the defaults that ship are deterministic and offline.

So you can verify what follows yourself, right now, without asking anyone for anything.

## The two failures it exists to catch

Both live in [`tests/test_acceptance.py`](tests/test_acceptance.py). Read that file first; everything else is plumbing that makes those tests possible.

### The adjacent section

A runbook has two sections that say almost exactly the same thing. One is headed *escalation after 24 hours*, the other *after 48*. Their bodies are interchangeable. Retrieval scores them 0.398 and 0.358 — a near-tie.

Pick the wrong one and you get coherent prose, a real citation, the right document, and the wrong answer. **Nothing about it looks wrong.** That is what makes this class of error expensive: it survives review.

Two independent mechanisms catch it. The chunker carries the section heading into the indexed text, so the number becomes part of what is matched. The gate then requires that the passage actually contain what the question pinned down — and if it does not, it declines rather than reaching for the neighbour.

The test suite also runs the control: same corpus, anchor check off, and you can watch it answer confidently from the wrong section.

### The question that should not be answered

A handbook talks about the company at length and never says how many people work there. Ask about headcount and it retrieves — correctly, it is obviously the relevant document. Retrieval has done its job.

Answering is the failure. The gate declines, and says *what* was missing rather than "I could not find that", because the first is actionable and the second is a dead end.

In a delivered test set these cases are typically a fifth of the total. They are the ones nobody writes, and the only failure mode that is invisible from outside the system.

## How the gate decides

| | |
|---|---|
| **The floor** | Nothing scored high enough to be on topic at all. |
| **The anchors** | The question pinned something down — a number, a quoted phrase, a declared term — and no passage contains it. Being about the right topic is not the same as containing the answer. |
| **The ambiguity flag** | Two passages from different sections of one file, nearly tied. It answers, and says so. |

Anchors do not only decide, they select: a passage that lacks what the question pinned down is not cited, because citing it alongside the right one puts the decoy back into the answer after the gate went to the trouble of removing it.

Anchor extraction is deliberately narrow — quoted phrases and numbers, both things the asker typed on purpose. It does not try to guess which words matter, because that is where a heuristic starts being wrong quietly. When a question needs an anchor that is neither, declare it. Explicit beats clever.

## The model writes prose, never values

Citations are derived from the gated passages *before* any writer runs. A writer receives those passages and returns text; that text replaces the prose and nothing else.

This is not a guard against a badly behaved model — it is an ordering. There is no point in the pipeline where a model is asked which document an answer came from, so there is no point at which it can be wrong about it. `test_a_lying_writer_cannot_add_a_source` hands the pipeline a writer that cites an invented file, and shows the citation list unmoved.

## The test set is the acceptance criteria

```python
from evidence_gate import Case, run

cases = [
    Case("01", "What is the notice period?", expected_file="contract.md", expected_page=1),
    Case("02", "How many employees are there?", expects_answer=False, anchors=("employees",)),
]

report = run(assistant, cases)
print(report.to_markdown())
```

A case pairs a question with the source that holds its answer. It passes when the assistant cites *that* source — not when the prose reads well. Agreed before any code is written, run automatically after.

The report prints its failures, with what was cited and what was expected. A report that only shows passes is not evidence.

## Reading your documents

```python
from evidence_gate import Assistant, ingest

report = ingest(["policies/"], root="policies/")
print(report.summary())
# for example:
# 12 file(s) read, 340 chunk(s) indexed.
# - handbook.pdf: no text on page(s) 7, 8. Usually a scanned image; it needs OCR before it can be searched.
# - prices.xlsx: no reader for .xlsx
# 5 fragment(s) under 40 characters were not indexed, usually page numbers and footers: 'Page 1 of 9', 'Page 2 of 9', 'Internal use only' and others.

assistant = Assistant.build(report.chunks)
```

| Format | Needs | A citation points to |
|---|---|---|
| PDF | `pip install "evidence-gate[pdf]"`, which brings `pypdf` | file and page: `handbook.pdf p.12` |
| DOCX | nothing: a .docx is a zip of XML, and the standard library reads both | file and section: `policy.docx §Sick leave` |
| Markdown and text | nothing | file and section; form feeds count as page breaks |

**The report is the point.** Ingestion is where an index most often ends up smaller than the documents it came from, and nobody notices: a scanned page has no text, so it is never retrieved, so the assistant behaves as if it did not exist. The same happens to a format nobody wrote a reader for, and to a PDF read on a machine without `pypdf`. `ingest` never drops anything quietly. `report.complete` is False whenever a page or a file was left out, and `summary()` is written for the person who owns the documents.

Fragments too short to index are listed too, but they do not make an ingestion incomplete: almost every PDF has page numbers and footers, and an alarm that rings on every document is one nobody reads. They are listed because the suite found the case where it matters — a three-row table alone under its heading, shorter than the chunker's minimum.

**A DOCX has no page numbers, and says so.** Where its pages break depends on the fonts and printer of whoever opens it, so the reader returns none and the citation falls back to the section heading. A guessed page number would look exactly like a real one.

**Headings are what the chunker's first fix depends on**, so the readers take care over them. A Spanish copy of Word stores *Título 1* under the internal name *heading 1*, and the reader goes by the internal name. A heading in the middle of a PDF page starts its own block. And a section that crosses a page break keeps its heading on the next page — without that, the adjacent-section failure above comes back at every page break.

Two files that would be cited by the same name are refused rather than merged. Pass `root=` and the citations carry their folder.

## What this is not

- **Not a framework.** Eleven focused modules, no runtime dependencies, standard library only. Reading PDFs is the one optional extra.
- **Not OCR.** A scanned page is reported, not read.
- **Not a benchmark.** It measures citation correctness on the questions you agreed. It says nothing about questions outside that set, and it does not measure writing quality.
- **Not a guarantee.** A test set bounds what it covers. Anything else is unmeasured, and this repository would rather say so than imply otherwise.
- **Not production infrastructure.** No auth, no rate limiting, no hosted index. Those are decisions that belong to your deployment, not to a library.

## Calibration

`GatePolicy.min_score` must be measured against your own corpus, not copied from here. The defaults were set from observed scores on the example corpus and are documented as such in [`gate.py`](src/evidence_gate/gate.py).

The floor and the anchors have different jobs, and blurring them is easy to do by accident: a floor raised high enough to reject a bad question will also reject good ones, and for a reason nobody can explain afterwards. `test_the_floor_and_the_anchors_have_different_jobs` pins that division so it fails loudly if someone tunes one to cover for the other.

## Layout

```
src/evidence_gate/
  types.py       frozen values; an answer without citations cannot be constructed
  ports.py       the seams: Embedder, Store, Writer, Reader
  chunking.py    splitting text without losing where it came from
  embedding.py   deterministic offline embedder
  store.py       in-memory vectors, saved and reloaded
  gate.py        the decision
  answer.py      citations first, prose second
  pipeline.py    the three wired together
  evaluate.py    the test set and its report
  readers.py     PDF, DOCX and text, with the page numbers that are real
  ingest.py      files and folders in, chunks out, and what was left out
```

## Status

Early. The core is complete and tested. Readers for PDF, DOCX and text arrived in 0.2.0, with the ingestion report. Hosted embedder adapters are next and arrive behind the existing protocols.

Built at [Torsion Labs](https://torsion-labs.com).

## License

Apache-2.0.
