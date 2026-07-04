"""File: rag_eval.py

A small, honest RAG evaluation harness.

Scope, deliberately: this evaluates RETRIEVAL quality (did the vector
search find the right section?), not GENERATION quality (did the LLM use
that context well?). Retrieval eval is simpler, doesn't need an LLM judge,
and is the right first layer -- if retrieval is wrong, nothing downstream
can be right, so it's the first thing worth being confident about.

The metric: Hit@k. For each (query, expected_section) test case, run
retrieval and check whether a chunk from expected_section appears
anywhere in the top_k results. This is a standard, simple retrieval
metric.

How this could grow later (worth knowing the names, not required to build):
- Precision@k / Recall@k if you had multiple relevant chunks per query.
- MRR (Mean Reciprocal Rank) if *ranking* position matters, not just
  presence in top_k.
- A generation-level eval layer (e.g. using RAGAS, or reusing 
  LLM-based evaluation rubric from the EPAM Uber project) that checks
  whether the *final answer* is faithful to the retrieved context, not
  just whether retrieval found the right section.

How to run:
    python rag_eval.py
    python rag_eval.py --top-k 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.utils.vector_store import collection_stats, retrieve_similar_chunks

# Each test case is grounded in a real portfolio section, so a wrong
# result is genuinely wrong, not just "the model output changed" -- the
# same discipline as writing real test assertions instead of snapshot tests.
TEST_CASES = [
    ("large scale data migration and validation", "projects"),
    ("recent generative AI training and certifications", "training_certifications"),
    ("degree, institute, and academic rank", "education"),
    ("reusable schema validation framework design", "frameworks_delivered"),
    ("client recognition and mentoring feedback", "recognition"),
    ("cloud, DevOps, and database tooling breadth", "skills"),
    ("leadership across multiple client engagements", "work_experience"),
]


@dataclass
class EvalResult:
    query: str
    expected_section: str
    hit: bool
    found_sections: list[str]
    best_distance: float | None


def evaluate(top_k: int = 3) -> list[EvalResult]:
    results = []
    for query, expected_section in TEST_CASES:
        retrieved = retrieve_similar_chunks(query, top_k=top_k)
        found_sections = [r["metadata"].get("section", "?") for r in retrieved]
        hit = expected_section in found_sections
        best_distance = retrieved[0]["distance"] if retrieved else None
        results.append(
            EvalResult(
                query=query,
                expected_section=expected_section,
                hit=hit,
                found_sections=found_sections,
                best_distance=best_distance,
            )
        )
    return results


def print_report(results: list[EvalResult]) -> None:
    print(f"{'HIT':<5}{'expected':<24}{'query':<45}top-k sections retrieved")
    print("-" * 110)
    hits = 0
    for r in results:
        mark = "PASS" if r.hit else "FAIL"
        if r.hit:
            hits += 1
        print(f"{mark:<5}{r.expected_section:<24}{r.query[:43]:<45}{r.found_sections}")

    total = len(results)
    rate = hits / total if total else 0.0
    print("-" * 110)
    print(f"Hit@k: {hits}/{total} = {rate:.0%}")
    if rate < 1.0:
        print(
            "\nNot 100%? That's normal and worth investigating, not hiding:\n"
            "  - Check the failing query's best_distance -- if it's much higher\n"
            "    than passing queries, the query phrasing may not overlap enough\n"
            "    with the stored text (a real RAG failure mode: vocabulary mismatch).\n"
            "  - Try --top-k with a larger value -- if it passes at top_k=5 but not\n"
            "    top_k=3, the right chunk is being found, just not ranked highly enough."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    stats = collection_stats()
    print(f"Vector store: {stats['count']} chunk(s) across sections: {stats['sections']}\n")
    if stats["count"] == 0:
        raise SystemExit("Vector store is empty. Run seed_vector_db.py first.")

    results = evaluate(top_k=args.top_k)
    print_report(results)


if __name__ == "__main__":
    main()
