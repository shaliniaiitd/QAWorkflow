"""File: src/utils/chunking.py

Text chunking for RAG.

Why chunk at all:
- Embedding models and vector search work best on small, semantically
  coherent pieces of text -- a few sentences to a paragraph. Embed a whole
  long document as one vector and you get a blurry average that doesn't
  match specific questions well.
- Chunking also controls what gets stuffed into the LLM's context window
  at generation time: smaller, well-chosen chunks reduce "context
  dilution" (the LLM getting a wall of mostly-irrelevant text and losing
  the useful part in it).

Why overlap:
- Chunk boundaries can fall in the middle of a relevant sentence, so the
  most relevant chunk retrieved is missing its lead-in ("...which reduced
  latency by 30%" without saying *what* reduced latency). A small overlap
  between consecutive chunks means that context usually survives in at
  least one of the neighboring chunks.
- More overlap = more redundancy (safer recall, more storage/compute).
  Less overlap = leaner index, higher risk of losing boundary context.
  A common starting point is overlap ~= 10-20% of chunk_size.

This module uses a simple word-count-based splitter (not sentence- or
token-aware) -- easy to reason about and enough for short/medium text like
a one-page portfolio site. For longer or denser documents (e.g. full
resumes or spec PDFs), a sentence-aware or token-aware splitter is worth
upgrading to later; the chunk() function signature below is written so
swapping the internals later won't change any caller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """One chunk of a larger text, with enough info to trace it back."""

    text: str
    chunk_index: int
    start_word: int
    end_word: int


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 10) -> list[Chunk]:
    """Split `text` into overlapping chunks of ~chunk_size words each.

    Args:
        text: The text to split.
        chunk_size: Target number of words per chunk. Tune this to your
            embedding model and typical query length -- 40-100 words is a
            reasonable range for short factual content like this project.
        overlap: Number of words repeated between consecutive chunks, to
            preserve context that straddles a boundary. Must be smaller
            than chunk_size (otherwise chunks would never advance).

    Returns:
        A list of Chunk objects, in order, covering the whole text.

    Example:
        >>> len(chunk_text("word " * 100, chunk_size=60, overlap=10))
        2
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                text=" ".join(chunk_words),
                chunk_index=index,
                start_word=start,
                end_word=end,
            )
        )
        if end == len(words):
            break
        start += step
        index += 1

    return chunks


if __name__ == "__main__":
    # Quick manual demo: run `python src/utils/chunking.py` to see chunking
    # behavior on a short sample paragraph.
    sample = (
        "Lead SDET at TekSystems architected a Pytest-based data validation "
        "framework with factory fixtures, ClickHouse and SSH connectivity, "
        "credentials management, and Allure/XRAY reporting. This framework "
        "reduced manual data validation effort significantly and became the "
        "standard pattern reused across three later engagements."
    )
    for c in chunk_text(sample, chunk_size=15, overlap=4):
        print(f"[chunk {c.chunk_index}] words {c.start_word}-{c.end_word}: {c.text}")
