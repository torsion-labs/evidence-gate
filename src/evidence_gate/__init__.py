"""evidence-gate — retrieval over your own documents, with citations you can check.

The short version: retrieval that keeps the source file and page with every
chunk, a gate that is allowed to say no, and a test set that decides whether
it works. The gate is the part most systems leave out.
"""

from .answer import QuotingWriter, build, citations_for, refusal_text
from .chunking import ChunkingPolicy, chunk_document, chunk_page, detect_heading
from .embedding import HashingEmbedder, cosine, normalise, tokenize
from .evaluate import Case, Report, Result, load_cases, run, run_case, save_cases
from .gate import EvidenceGate, GatePolicy, extract_anchors
from .pipeline import Assistant
from .ports import Embedder, Reader, Store, Writer
from .store import InMemoryStore
from .types import Answer, Chunk, Citation, Passage, Source, Verdict

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Assistant",
    "Case",
    "Chunk",
    "ChunkingPolicy",
    "Citation",
    "Embedder",
    "EvidenceGate",
    "GatePolicy",
    "HashingEmbedder",
    "InMemoryStore",
    "Passage",
    "QuotingWriter",
    "Reader",
    "Report",
    "Result",
    "Source",
    "Store",
    "Verdict",
    "Writer",
    "build",
    "chunk_document",
    "chunk_page",
    "citations_for",
    "cosine",
    "detect_heading",
    "extract_anchors",
    "load_cases",
    "normalise",
    "refusal_text",
    "run",
    "run_case",
    "save_cases",
    "tokenize",
]
