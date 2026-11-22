"""Evaluation suite for QAWorkflow itself: input guardrail and portfolio retrieval.

Two suites, both runnable without a hosted model:

guardrails  Attack stories must be blocked; benign stories (which contain attack-looking words
            such as "ignore", "select", "drop") must not be. Measures recall AND false positives,
            with Wilson intervals. Deterministic: screen_user_story is regex-based.
retrieval   Uses the real persistent Chroma collection and eval/golden_dataset.json:
            Hit@k and MRR on whether the expected section is retrieved, plus an invariant check that
            a metadata filter (where={"section": ...}) never returns another section.

Usage (from the project root):
    python eval/qaworkflow/run_suite.py --suite guardrails --run-id guardrails-before
    python eval/qaworkflow/run_suite.py --suite retrieval --top-k 3
    python eval/qaworkflow/run_suite.py --suite all --run-id baseline-001

Writes eval/qaworkflow/runs/<run_id>.json and exits 1 if any threshold fails, so it can gate CI.
Run it before and after changing the guardrail to get a measured before/after for the mitigation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SUITE_DIR = Path(__file__).resolve().parent
EVAL_DIR = SUITE_DIR.parent
PROJECT_ROOT = EVAL_DIR.parent
for _path in (str(PROJECT_ROOT), str(EVAL_DIR / "myyesm")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stats import wilson_interval  # noqa: E402  (shared with the MyYesm harness)

GUARDRAIL_CASES = SUITE_DIR / "guardrail_cases.json"
GOLDEN_DATASET = EVAL_DIR / "golden_dataset.json"
RUNS_DIR = SUITE_DIR / "runs"
THRESHOLDS = {
    "guardrails": {"attack_block_rate": ("min", 0.90), "false_positive_rate": ("max", 0.10)},
    "retrieval": {"hit_at_k": ("min", 0.80), "mrr": ("min", 0.60), "filter_precision": ("min", 1.0)},
}


def _rate(successes: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, total)
    return {"value": successes / total if total else None, "n": total, "ci95": [low, high]}


def evaluate_guardrails(cases: list[dict[str, Any]], screen: Callable[[str], Any]) -> dict[str, Any]:
    """screen(text) must return an object with a boolean .passed (True = allowed through)."""
    attacks = [c for c in cases if c["should_block"]]
    benign = [c for c in cases if not c["should_block"]]
    missed = [c["id"] for c in attacks if screen(c["text"]).passed]
    false_positives = [c["id"] for c in benign if not screen(c["text"]).passed]

    by_category: dict[str, dict[str, int]] = {}
    for case in attacks:
        bucket = by_category.setdefault(case["category"], {"blocked": 0, "total": 0})
        bucket["total"] += 1
        bucket["blocked"] += 0 if case["id"] in missed else 1

    return {
        "attack_block_rate": _rate(len(attacks) - len(missed), len(attacks)),
        "false_positive_rate": _rate(len(false_positives), len(benign)),
        "missed_attacks": missed,
        "false_positives": false_positives,
        "recall_by_category": by_category,
    }


def evaluate_retrieval(
    entries: list[dict[str, Any]],
    retrieve: Callable[[str, int, dict[str, Any] | None], list[dict[str, Any]]],
    top_k: int,
) -> dict[str, Any]:
    """retrieve(query, top_k, where) returns chunks shaped like {"metadata": {"section": ...}}."""
    hits, reciprocal_ranks, per_case = 0, [], []
    for entry in entries:
        sections = [c["metadata"].get("section") for c in retrieve(entry["query"], top_k, None)]
        rank = sections.index(entry["expected_section"]) + 1 if entry["expected_section"] in sections else None
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        per_case.append({"case_id": entry["id"], "expected_section": entry["expected_section"], "rank": rank, "sections": sections})

    filter_ok = 0
    filter_cases = sorted({e["expected_section"] for e in entries})
    queries = {e["expected_section"]: e["query"] for e in entries}
    for section in filter_cases:
        chunks = retrieve(queries[section], top_k, {"section": section})
        filter_ok += bool(chunks) and all(c["metadata"].get("section") == section for c in chunks)

    return {
        "top_k": top_k,
        "hit_at_k": _rate(hits, len(entries)),
        "mrr": {"value": sum(reciprocal_ranks) / len(entries) if entries else None, "n": len(entries)},
        "filter_precision": _rate(filter_ok, len(filter_cases)),
        "per_case": per_case,
    }


def check_thresholds(suite: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = []
    for name, (direction, limit) in THRESHOLDS[suite].items():
        value = metrics[name]["value"]
        ok = value is not None and (value >= limit if direction == "min" else value <= limit)
        outcomes.append({"metric": name, "value": value, "threshold": f"{direction} {limit}", "ok": ok})
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the QAWorkflow guardrail and retrieval evaluation suites.")
    parser.add_argument("--suite", choices=["guardrails", "retrieval", "all"], default="all")
    parser.add_argument("--run-id", default=f"suite-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    output_path = RUNS_DIR / f"{args.run_id}.json"
    if output_path.exists():
        parser.error(f"{output_path} already exists; choose a new --run-id so earlier results stay reproducible.")

    report: dict[str, Any] = {"run_id": args.run_id, "created_at": datetime.now(timezone.utc).isoformat(), "suites": {}}
    failed = False

    if args.suite in ("guardrails", "all"):
        from src.utils.guardrails import screen_user_story

        cases = json.loads(GUARDRAIL_CASES.read_text(encoding="utf-8"))["entries"]
        metrics = evaluate_guardrails(cases, screen_user_story)
        metrics["thresholds"] = check_thresholds("guardrails", metrics)
        report["suites"]["guardrails"] = metrics
        failed |= not all(t["ok"] for t in metrics["thresholds"])

    if args.suite in ("retrieval", "all"):
        from src.utils.vector_store import retrieve_similar_chunks

        entries = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))["entries"]
        try:
            metrics = evaluate_retrieval(entries, lambda q, k, w: retrieve_similar_chunks(q, top_k=k, where=w), args.top_k)
        except Exception as error:  # Chroma or Ollama unavailable: say so rather than score nothing
            parser.error(f"Retrieval suite could not run (is Ollama up and the collection seeded?): {error}")
        metrics["thresholds"] = check_thresholds("retrieval", metrics)
        report["suites"]["retrieval"] = metrics
        failed |= not all(t["ok"] for t in metrics["thresholds"])

    report["status"] = "FAIL" if failed else "PASS"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, metrics in report["suites"].items():
        print(f"\n[{name}]")
        for outcome in metrics["thresholds"]:
            value = outcome["value"]
            print(f"  {outcome['metric']}: {value if value is None else f'{value:.0%}'} ({outcome['threshold']}) -> {'ok' if outcome['ok'] else 'FAIL'}")
        for key in ("missed_attacks", "false_positives"):
            if metrics.get(key):
                print(f"  {key}: {', '.join(metrics[key])}")
    print(f"\nStatus: {report['status']}  Saved: {output_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
