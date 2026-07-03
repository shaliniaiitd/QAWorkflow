"""File: check_retrieval.py

The retrieval checkup script.

The whole point: right now nothing in this project lets you *see* what
gets retrieved or how confident the match is. This script makes that
visible, so "I called retrieve()" becomes "I can show you my retrieval is
actually working, and here's how I know."

What "distance" means here:
- Chroma's default distance metric is cosine distance = 1 - cosine_similarity.
- 0.0 means the query and the chunk point in exactly the same direction
  in embedding space (as similar as it gets).
- Larger values mean less similar. There's no fixed "good" cutoff -- it
  depends on your embedding model and content -- which is exactly why you
  look at real numbers on real queries instead of guessing a threshold.
- A useful habit: run a query you know the right answer to, note the
  distance of the correct result, and use that as your rough calibration
  for "this looks like a real match" vs "this is the store grasping at
  straws because nothing relevant exists."

How to run:
    python check_retrieval.py "your query here"
    python check_retrieval.py "your query here" --top-k 5
    python check_retrieval.py "your query here" --section projects
    python check_retrieval.py   # runs a few built-in demo queries
"""

from __future__ import annotations

import argparse

from src.utils.vector_store import collection_stats, retrieve_similar_chunks

DEMO_QUERIES = [
    "large scale data validation experience",
    "generative AI certifications",
    "academic background and rank",
    "framework design and reusability",
]


def show_query(query: str, top_k: int, section: str | None) -> None:
    where = {"section": section} if section else None
    results = retrieve_similar_chunks(query, top_k=top_k, where=where)

    print(f"\nQuery: {query!r}" + (f"  (filtered to section={section})" if section else ""))
    print("-" * 70)
    if not results:
        print("  (no results -- is the vector store seeded? run seed_vector_db.py)")
        return

    for rank, r in enumerate(results, start=1):
        meta = r["metadata"]
        print(
            f"  #{rank}  distance={r['distance']:.4f}  "
            f"section={meta.get('section', '?')}  "
            f"story_id={meta.get('story_id', '?')}  "
            f"chunk={meta.get('chunk_index', '?')}"
        )
        print(f"       {r['text'][:120]}{'...' if len(r['text']) > 120 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=None, help="Query to test. Omit to run demo queries.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--section", default=None, help="Restrict to one section, e.g. projects")
    args = parser.parse_args()

    stats = collection_stats()
    print(f"Vector store: {stats['count']} chunk(s) across sections: {stats['sections']}")
    if stats["count"] == 0:
        raise SystemExit("Vector store is empty. Run seed_vector_db.py first.")

    if args.query:
        show_query(args.query, args.top_k, args.section)
    else:
        print("\nNo query given -- running built-in demo queries:")
        for q in DEMO_QUERIES:
            show_query(q, args.top_k, args.section)


if __name__ == "__main__":
    main()
