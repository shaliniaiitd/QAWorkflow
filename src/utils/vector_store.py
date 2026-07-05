"""File: src/utils/vector_store.py

Vector store for the QA workflow's RAG layer, backed by Chroma
(local, persistent, no external service needed) with embeddings from
Ollama -- reusing the same local Ollama setup workflow.py already
depends on, so there's no second API key or service to run.

Two embedding models running against Ollama at once (the chat model in
workflow.py's MODEL_CONFIG, and the embedding model here) is normal --
they do different jobs. Pull the embedding model once:
    ollama pull nomic-embed-text

Design notes:
- `add_story_to_db` is an *upsert* (Chroma's `collection.upsert`, not
  `add`) keyed on `story_id`, so re-running a seed script is always safe
  -- it updates existing entries instead of duplicating them.
- `retrieve_similar_stories` keeps the original simple signature
  (returns a list[str]) so existing code in workflow.py doesn't need to
  change.
- `retrieve_similar_chunks` is the new, richer function: returns
  metadata and similarity scores alongside the text, for the retrieval
  checkup script and the RAG eval script.
- Metadata (e.g. `section`) is optional on write and filterable on read,
  via Chroma's `where=` clause.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain_ollama import OllamaEmbeddings

PROJECT_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "user_stories"

import os

EMBED_MODEL = "nomic-embed-text"
EMBED_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

_client: chromadb.ClientAPI | None = None
_collection = None
_embedder: OllamaEmbeddings | None = None


def _get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE_URL)
    return _embedder


def _get_collection():
    """Lazily open (or create) the persistent Chroma collection.

    Lazy on purpose: importing this module must not require Chroma/Ollama
    to already be reachable -- only actually calling add/retrieve should.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def add_story_to_db(story_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
    """Upsert one story/chunk into the vector store.

    Args:
        story_id: Unique id. For chunked documents, use something like
            f"{doc_id}::chunk{chunk_index}" so each chunk is addressable
            and re-seeding a doc updates its own chunks, not others'.
        text: The text to embed and store.
        metadata: Optional dict, e.g. {"section": "projects", "chunk_index": 2}.
            Used for filtered retrieval and for the eval/checkup scripts.
    """
    collection = _get_collection()
    embedding = _get_embedder().embed_query(text)
    collection.upsert(
        ids=[story_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata or {}],
    )


def retrieve_similar_stories(query: str, top_k: int = 3) -> list[str]:
    """Return just the text of the top_k most similar stored chunks.

    Kept deliberately simple (list[str]) since this is what workflow.py's
    `retrieve_memory` node already expects. Use retrieve_similar_chunks
    below when you need scores/metadata too.
    """
    results = retrieve_similar_chunks(query, top_k=top_k)
    return [r["text"] for r in results]


def retrieve_similar_chunks(
    query: str,
    top_k: int = 3,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the top_k most similar chunks with metadata and distance.

    Args:
        query: The search text.
        top_k: How many results to return.
        where: Optional Chroma metadata filter, e.g. {"section": "projects"},
            to restrict the search to one part of the source content.

    Returns:
        A list of dicts: {"id", "text", "metadata", "distance"}, ordered
        by similarity (lowest distance = most similar first). Chroma's
        default distance is cosine distance, so 0.0 = identical direction,
        higher = less similar.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = _get_embedder().embed_query(query)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
    )

    results: list[dict[str, Any]] = []
    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for i in range(len(ids)):
        results.append(
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i] or {},
                "distance": distances[i],
            }
        )
    return results


def collection_stats() -> dict[str, Any]:
    """Small helper for the checkup/eval scripts: how many chunks are stored,
    and what sections/metadata keys exist."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return {"count": 0, "sections": {}}
    sample = collection.get(limit=count)
    sections: dict[str, int] = {}
    for meta in sample.get("metadatas", []):
        section = (meta or {}).get("section", "(none)")
        sections[section] = sections.get(section, 0) + 1
    return {"count": count, "sections": sections}