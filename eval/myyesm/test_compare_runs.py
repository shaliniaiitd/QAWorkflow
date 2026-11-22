from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_runs  # noqa: E402


def report(run_id: str, passes: dict[str, bool], **aggregate: float) -> dict:
    return {
        "run_id": run_id,
        "dataset_sha256": "d", "documents_sha256": "c", "evaluator_sha256": "e",
        "system_variants": {"prompt_version": [run_id]},
        "aggregate": aggregate,
        "results": [{"case_id": case_id, "automatic_checks_passed": ok} for case_id, ok in passes.items()],
    }


class CompareReportsTests(unittest.TestCase):
    def test_small_difference_is_not_significant(self) -> None:
        cases = [f"c{i}" for i in range(12)]
        baseline = report("a", {c: i < 6 for i, c in enumerate(cases)}, fact_coverage=0.8, tokens_total=1000)
        candidate = report("b", {c: i < 9 for i, c in enumerate(cases)}, fact_coverage=0.9, tokens_total=1500)
        result = compare_runs.compare_reports(baseline, candidate)
        self.assertEqual(result["paired"]["only_candidate_passes"], ["c6", "c7", "c8"])
        self.assertEqual(result["paired"]["only_baseline_passes"], [])
        self.assertFalse(result["paired"]["significant"])
        delta = {m["name"]: m["delta"] for m in result["metrics"]}
        self.assertAlmostEqual(delta["fact_coverage"], 0.1)
        self.assertEqual(delta["tokens_total"], 500)

    def test_large_one_sided_difference_is_significant(self) -> None:
        cases = [f"c{i}" for i in range(12)]
        baseline = report("a", {c: False for c in cases})
        candidate = report("b", {c: True for c in cases})
        self.assertTrue(compare_runs.compare_reports(baseline, candidate)["paired"]["significant"])

    def test_dataset_mismatch_is_flagged(self) -> None:
        baseline = report("a", {"c": True})
        candidate = report("b", {"c": True})
        candidate["dataset_sha256"] = "other"
        warnings = compare_runs.compare_reports(baseline, candidate)["warnings"]
        self.assertTrue(any("datasets" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
