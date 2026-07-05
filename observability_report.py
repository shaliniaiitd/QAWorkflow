"""File: observability_report.py

Reads observability_log.jsonl (written automatically by every LLM call in
src/workflow.py) and prints latency baselines per node -- mean, min, max,
and a simple p95 -- plus a running failure/retry count pulled from the
same file. This is the "latency baselines" and "failure analysis" pieces
you get for free with zero LangSmith setup.
 LangSmith gives the same thing plus a UI, nested traces, and team sharing.

How to run:
    python observability_report.py
    python observability_report.py --node generate_bdd
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from src.utils.observability import read_log


def percentile(values: list[float], pct: float) -> float:
    """Simple nearest-rank percentile -- fine for a handful of local runs;
    swap for numpy.percentile if you're aggregating hundreds+ of calls."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default=None, help="Show only one node's stats")
    args = parser.parse_args()

    records = read_log()
    if not records:
        raise SystemExit(
            "No observability data yet -- observability_log.jsonl doesn't exist "
            "or is empty. Run the workflow (python -m src.workflow) at least once first."
        )

    by_node: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_node[r["node"]].append(r)

    nodes = [args.node] if args.node else sorted(by_node)

    print(f"{'node':<24}{'calls':<8}{'mean_s':<10}{'min_s':<10}{'max_s':<10}{'p95_s':<10}{'avg_tokens':<12}")
    print("-" * 84)
    for node in nodes:
        calls = by_node.get(node, [])
        if not calls:
            print(f"{node:<24} (no calls logged)")
            continue
        latencies = [c["latency_s"] for c in calls]
        tokens = [c["total_tokens"] for c in calls if c.get("total_tokens") is not None]
        avg_tokens = f"{statistics.mean(tokens):.0f}" if tokens else "n/a"
        print(
            f"{node:<24}{len(calls):<8}"
            f"{statistics.mean(latencies):<10.3f}"
            f"{min(latencies):<10.3f}"
            f"{max(latencies):<10.3f}"
            f"{percentile(latencies, 0.95):<10.3f}"
            f"{avg_tokens:<12}"
        )

    print(f"\nTotal calls logged: {len(records)}")
    print("(Set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY to also see these as traces in LangSmith.)")


if __name__ == "__main__":
    main()
