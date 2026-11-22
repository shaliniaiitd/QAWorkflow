from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_requirements as dr  # noqa: E402


def case(case_id: str, scenario: str, risk: str = "low", **extra: object) -> dict:
    return {"id": case_id, "scenario": scenario, "risk_tier": risk, "expected_source_ids": ["a"], **extra}


DOCS = [{"source_id": "a"}, {"source_id": "b"}, {"source_id": "c"}]


class FamilyTests(unittest.TestCase):
    def test_scenarios_map_to_families(self) -> None:
        self.assertEqual(dr.family_of("direct_prompt_injection"), "adversarial")
        self.assertEqual(dr.family_of("tool_misuse_excessive_agency"), "adversarial")
        self.assertEqual(dr.family_of("fairness"), "bias")
        self.assertEqual(dr.family_of("ambiguous_query"), "edge")
        self.assertEqual(dr.family_of("factual_qa"), "core")


class CheckDatasetTests(unittest.TestCase):
    def test_small_dataset_is_not_release_ready_but_reports_why(self) -> None:
        entries = [case("1", "factual_qa"), case("2", "direct_prompt_injection", "high")]
        result = dr.check_dataset(entries, DOCS, "release", top_k=2)
        self.assertFalse(result["ready"])
        names = {f["check"]: f["ok"] for f in result["findings"]}
        self.assertFalse(names["held-out split"])
        self.assertFalse(names["human review on high-risk cases"])
        self.assertFalse(names["corpus size for retrieval metrics"])

    def test_pilot_accepts_a_balanced_small_dataset(self) -> None:
        entries = (
            [case(f"f{i}", "factual_qa") for i in range(5)]
            + [case(f"e{i}", "ambiguous_query") for i in range(5)]
            + [case(f"b{i}", "fairness", "medium") for i in range(5)]
            + [case(f"a{i}", "direct_prompt_injection", "high", manual_review=True) for i in range(5)]
        )
        docs = DOCS + [{"source_id": str(n)} for n in range(10)]
        self.assertTrue(dr.check_dataset(entries, docs, "pilot", top_k=2)["ready"])


class MinimumDetectableDropTests(unittest.TestCase):
    def test_twelve_cases_only_detect_a_huge_drop(self) -> None:
        drop = dr.minimum_detectable_drop(12)
        self.assertIsNotNone(drop)
        self.assertGreater(drop, 0.4)

    def test_two_hundred_cases_detect_about_ten_points(self) -> None:
        self.assertLessEqual(dr.minimum_detectable_drop(200), 0.11)


if __name__ == "__main__":
    unittest.main()
