"""Check whether an evaluation dataset is statistically big and varied enough for the claim you want to make.

A dataset is "ready" for a tier only if it can support that tier's claims with the stated confidence:
- Enough cases per scenario: an all-pass scenario proves a pass rate >= target only above a minimum n
  (Wilson lower bound; see stats.min_cases_for_zero_failures).
- Coverage of the four case families: core intents, edge cases, bias, and adversarial scenarios.
- A corpus with distractors, so retrieval metrics can discriminate.
- A held-out split that prompts were not tuned on.
- Human review on every high-risk case.

Usage (from the project root):
    python eval/myyesm/dataset_requirements.py                       # pilot tier on cases.json
    python eval/myyesm/dataset_requirements.py --tier release --out eval/myyesm/DATASET_READINESS.md
    python eval/myyesm/dataset_requirements.py --dataset eval/myyesm/tool_cases.json --top-k 2

Exit code 1 when the dataset does not meet the chosen tier, so it can gate CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from stats import min_cases_for_zero_failures, required_sample_size  # noqa: E402

FAMILIES = {
    "adversarial": ("injection", "sensitive_data", "misuse", "jailbreak", "poison"),
    "bias": ("fairness", "bias"),
    "edge": ("ambiguous", "unanswerable", "abstention", "restraint"),
}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
TIERS: dict[str, dict[str, Any]] = {
    "pilot": {
        "description": "Directional signal only: finds bugs, cannot certify a rate.",
        "min_cases_per_scenario": 5,
        "min_family_share": {"edge": 0.0, "bias": 0.0, "adversarial": 0.0},
        "min_family_cases": 1,
        "corpus_docs_per_top_k": 3,
        "holdout_share": 0.0,
    },
    "release": {
        "description": "Supports a pass-rate claim with 95% confidence even when every case passes.",
        "target_rate_by_risk": {"low": 0.85, "medium": 0.90, "high": 0.95},
        "min_family_share": {"edge": 0.15, "bias": 0.10, "adversarial": 0.25},
        "min_family_cases": 1,
        "corpus_docs_per_top_k": 10,
        "holdout_share": 0.30,
    },
}


def family_of(scenario: str) -> str:
    for family, keywords in FAMILIES.items():
        if any(keyword in scenario for keyword in keywords):
            return family
    return "core"


def minimum_detectable_drop(cases_per_arm: int, baseline: float = 0.90) -> float | None:
    """Smallest pass-rate drop from `baseline` detectable at 80% power with this many cases."""
    for step in range(1, int(baseline * 100)):
        drop = step / 100
        if required_sample_size(baseline, baseline - drop) <= cases_per_arm:
            return drop
    return None


def check_dataset(
    entries: list[dict[str, Any]], documents: list[dict[str, Any]], tier: str, top_k: int
) -> dict[str, Any]:
    spec = TIERS[tier]
    findings: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        findings.append({"check": name, "ok": ok, "detail": detail})

    scenario_cases: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        scenario_cases.setdefault(entry["scenario"], []).append(entry)

    for scenario, cases in sorted(scenario_cases.items()):
        risk = max((c.get("risk_tier", "low") for c in cases), key=lambda r: RISK_ORDER.get(r, 0))
        needed = spec.get("min_cases_per_scenario") or min_cases_for_zero_failures(spec["target_rate_by_risk"][risk])
        add(f"scenario size: {scenario} ({risk} risk)", len(cases) >= needed, f"{len(cases)} case(s), need {needed}")

    family_counts = Counter(family_of(e["scenario"]) for e in entries)
    total = len(entries)
    for family, share in spec["min_family_share"].items():
        count = family_counts.get(family, 0)
        ok = count >= spec["min_family_cases"] and (count / total if total else 0) >= share
        add(f"family coverage: {family}", ok, f"{count} of {total} ({count / total:.0%}), need >= {share:.0%} and >= {spec['min_family_cases']} case(s)" if total else "empty dataset")

    referenced = {sid for e in entries for sid in e.get("expected_source_ids", [])}
    distractors = [d["source_id"] for d in documents if d["source_id"] not in referenced]
    needed_docs = spec["corpus_docs_per_top_k"] * top_k
    add("corpus size for retrieval metrics", len(documents) >= needed_docs,
        f"{len(documents)} document(s) with top_k={top_k}, need >= {needed_docs}; {len(distractors)} never expected in any case")

    holdout = sum(1 for e in entries if e.get("split") == "holdout")
    add("held-out split", (holdout / total if total else 0) >= spec["holdout_share"],
        f"{holdout} of {total} case(s) tagged split=holdout, need >= {spec['holdout_share']:.0%}")

    unreviewed = [e["id"] for e in entries if e.get("risk_tier") == "high" and not e.get("manual_review")]
    add("human review on high-risk cases", not unreviewed, "all high-risk cases have manual_review" if not unreviewed else f"missing manual_review: {', '.join(unreviewed)}")

    return {
        "tier": tier,
        "description": spec["description"],
        "total_cases": total,
        "findings": findings,
        "ready": all(f["ok"] for f in findings),
        "minimum_detectable_drop_from_90pct": minimum_detectable_drop(total),
    }


def render(result: dict[str, Any]) -> str:
    drop = result["minimum_detectable_drop_from_90pct"]
    lines = [
        f"# Dataset readiness — {result['tier']} tier: {'READY' if result['ready'] else 'NOT READY'}", "",
        f"_{result['description']}_", "",
        f"- Total cases: {result['total_cases']}",
        "- Smallest pass-rate drop from 90% this dataset can detect (80% power, 95% confidence): "
        + (f"{drop:.0%} points" if drop is not None else "none — too few cases to detect any drop"), "",
        "| Check | Status | Detail |", "|---|---|---|",
    ]
    lines += [f"| {f['check']} | {'ok' if f['ok'] else '**gap**'} | {f['detail']} |" for f in result["findings"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dataset size and coverage against a readiness tier.")
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "cases.json")
    parser.add_argument("--documents", type=Path, default=EVAL_DIR / "documents.json")
    parser.add_argument("--tier", choices=sorted(TIERS), default="pilot")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        entries = json.loads(args.dataset.read_text(encoding="utf-8"))["entries"]
        documents = json.loads(args.documents.read_text(encoding="utf-8"))["documents"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        parser.error(f"Cannot read dataset: {error}")
    result = check_dataset(entries, documents, args.tier, args.top_k)
    text = render(result)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
