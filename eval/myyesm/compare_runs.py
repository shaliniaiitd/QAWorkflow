"""Compare two scored runs: did a prompt or model change help, and is the difference real?

Usage (from the project root):
    python eval/myyesm/compare_runs.py baseline-001 hardened-fixed-002
    python eval/myyesm/compare_runs.py baseline-001 hardened-fixed-002 --out eval/myyesm/comparisons/prompt-v2.md

Reads runs/<id>/report.json. Prints metric deltas, the paired per-case outcome table,
Wilson intervals for each run's pass rate, and an exact McNemar p-value, so a 12-case
difference is never presented as stronger evidence than it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from stats import exact_mcnemar_p, wilson_interval  # noqa: E402

RUNS_DIR = EVAL_DIR / "runs"
QUALITY_METRICS = [
    "retrieval_hit_rate", "context_precision", "context_recall", "fact_coverage",
    "required_phrase_coverage", "citation_recall", "json_valid_rate", "json_match_rate",
    "tool_selection_accuracy", "tool_call_accuracy", "forbidden_phrase_case_rate",
    "forbidden_tool_call_case_rate",
]
COST_METRICS = ["latency_p50_ms", "latency_p95_ms", "tokens_total", "cost_usd_total"]
SIGNIFICANCE_LEVEL = 0.05


def load_report(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    path = runs_dir / run_id / "report.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Pure comparison of two report.json payloads."""
    warnings: list[str] = []
    if baseline.get("dataset_sha256") != candidate.get("dataset_sha256"):
        warnings.append("Different datasets: metrics are not comparable.")
    if baseline.get("documents_sha256") != candidate.get("documents_sha256"):
        warnings.append("Different document corpora: retrieval metrics are not comparable.")
    if baseline.get("evaluator_sha256") != candidate.get("evaluator_sha256"):
        warnings.append("Evaluator code changed between runs: re-score the older run before trusting small deltas.")

    metrics = []
    for name in QUALITY_METRICS + COST_METRICS:
        a, b = baseline["aggregate"].get(name), candidate["aggregate"].get(name)
        if a is None and b is None:
            continue
        metrics.append({"name": name, "baseline": a, "candidate": b,
                        "delta": None if a is None or b is None else b - a})

    base_cases = {r["case_id"]: r["automatic_checks_passed"] for r in baseline["results"]}
    cand_cases = {r["case_id"]: r["automatic_checks_passed"] for r in candidate["results"]}
    shared = sorted(set(base_cases) & set(cand_cases))
    only_baseline = [c for c in shared if base_cases[c] and not cand_cases[c]]
    only_candidate = [c for c in shared if cand_cases[c] and not base_cases[c]]
    both_pass = sum(1 for c in shared if base_cases[c] and cand_cases[c])
    both_fail = sum(1 for c in shared if not base_cases[c] and not cand_cases[c])
    base_passes = sum(base_cases[c] for c in shared)
    cand_passes = sum(cand_cases[c] for c in shared)
    p_value = exact_mcnemar_p(len(only_baseline), len(only_candidate))

    return {
        "warnings": warnings,
        "metrics": metrics,
        "paired": {
            "cases": len(shared), "both_pass": both_pass, "both_fail": both_fail,
            "only_baseline_passes": only_baseline, "only_candidate_passes": only_candidate,
            "baseline_pass_rate": base_passes / len(shared) if shared else None,
            "candidate_pass_rate": cand_passes / len(shared) if shared else None,
            "baseline_ci95": wilson_interval(base_passes, len(shared)),
            "candidate_ci95": wilson_interval(cand_passes, len(shared)),
            "mcnemar_p": p_value,
            "significant": p_value < SIGNIFICANCE_LEVEL,
        },
    }


def _fmt(name: str, value: Any) -> str:
    if value is None:
        return "—"
    if name.endswith(("_ms",)):
        return f"{value:.0f} ms"
    if name in ("tokens_total",):
        return f"{value:,}"
    if name == "cost_usd_total":
        return f"${value:.4f}"
    return f"{value:.1%}"


def _fmt_delta(name: str, delta: Any) -> str:
    if delta is None:
        return "—"
    if name.endswith(("_ms",)):
        return f"{delta:+.0f} ms"
    if name == "tokens_total":
        return f"{delta:+,}"
    if name == "cost_usd_total":
        return f"{delta:+.4f}"
    return f"{delta * 100:+.1f} pts"


def render(baseline: dict[str, Any], candidate: dict[str, Any], comparison: dict[str, Any]) -> str:
    paired = comparison["paired"]
    b_variant = json.dumps(baseline["system_variants"], sort_keys=True)
    c_variant = json.dumps(candidate["system_variants"], sort_keys=True)
    lines = [
        f"# Run comparison: `{baseline['run_id']}` vs `{candidate['run_id']}`", "",
        f"- Baseline variant: `{b_variant}`",
        f"- Candidate variant: `{c_variant}`", "",
    ]
    for warning in comparison["warnings"]:
        lines.append(f"> **Warning:** {warning}")
    if comparison["warnings"]:
        lines.append("")
    lines += ["## Metric deltas", "", "| Metric | Baseline | Candidate | Delta |", "|---|---:|---:|---:|"]
    for item in comparison["metrics"]:
        lines.append(f"| {item['name']} | {_fmt(item['name'], item['baseline'])} | "
                     f"{_fmt(item['name'], item['candidate'])} | {_fmt_delta(item['name'], item['delta'])} |")

    b_lo, b_hi = paired["baseline_ci95"]
    c_lo, c_hi = paired["candidate_ci95"]
    lines += [
        "", "## Paired per-case outcome (automatic checks)", "",
        f"- Cases compared: {paired['cases']}",
        f"- Baseline pass rate: {paired['baseline_pass_rate']:.0%} (95% CI {b_lo:.0%}–{b_hi:.0%})"
        if paired["cases"] else "- No shared cases.",
        f"- Candidate pass rate: {paired['candidate_pass_rate']:.0%} (95% CI {c_lo:.0%}–{c_hi:.0%})"
        if paired["cases"] else "",
        f"- Both pass: {paired['both_pass']}, both fail: {paired['both_fail']}",
        f"- Only baseline passes: {', '.join(paired['only_baseline_passes']) or 'none'}",
        f"- Only candidate passes: {', '.join(paired['only_candidate_passes']) or 'none'}",
        f"- Exact McNemar p-value: {paired['mcnemar_p']:.3f}", "",
    ]
    if paired["significant"]:
        lines.append(f"**Verdict:** the difference is statistically significant at the {SIGNIFICANCE_LEVEL:.0%} level.")
    else:
        lines.append(
            f"**Verdict:** not statistically distinguishable at n={paired['cases']}. Treat the deltas above as "
            "directional hints, not proof. Add cases (see DATASET_REQUIREMENTS.md) before claiming an improvement."
        )
    lines.append("")
    return "\n".join(line for line in lines if line is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two scored evaluation runs.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, help="Also write the comparison to this markdown file")
    args = parser.parse_args()
    try:
        baseline = load_report(args.baseline, args.runs_dir)
        candidate = load_report(args.candidate, args.runs_dir)
    except ValueError as error:
        parser.error(str(error))
    text = render(baseline, candidate, compare_reports(baseline, candidate))
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
