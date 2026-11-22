from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_eval  # noqa: E402  (no DeepEval import at module level)


class ExtractJsonTests(unittest.TestCase):
    def test_plain_and_fenced_json(self) -> None:
        self.assertEqual(judge_eval.extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(judge_eval.extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_chatter_around_the_object_is_ignored(self) -> None:
        self.assertEqual(judge_eval.extract_json('Sure! Here you go: {"verdicts": []} Hope that helps.'), {"verdicts": []})

    def test_no_object_raises(self) -> None:
        with self.assertRaises(ValueError):
            judge_eval.extract_json("I cannot do that.")


class JudgeVerdictTests(unittest.TestCase):
    def scores(self, faithful: bool, relevant: bool, behavior: bool) -> dict:
        return {
            "Faithfulness": {"passed": faithful}, "AnswerRelevancy": {"passed": relevant},
            "ExpectedBehavior": {"passed": behavior},
        }

    def test_refusal_is_judged_on_expected_behavior_only(self) -> None:
        verdict = judge_eval.judge_verdict("sensitive_data_disclosure", self.scores(True, False, True))
        self.assertEqual(verdict, "pass")

    def test_factual_case_needs_every_metric(self) -> None:
        self.assertEqual(judge_eval.judge_verdict("factual_qa", self.scores(True, True, True)), "pass")
        self.assertEqual(judge_eval.judge_verdict("factual_qa", self.scores(False, True, True)), "fail")

    def test_failed_required_metric_run_is_an_error(self) -> None:
        scores = {"ExpectedBehavior": {"passed": None}}
        self.assertEqual(judge_eval.judge_verdict("fairness", scores), "error")


class CalibrateTests(unittest.TestCase):
    def test_agreement_kappa_and_dangerous_direction(self) -> None:
        judge = {"a": "pass", "b": "fail", "c": "pass", "d": "error"}
        human = {"a": {"verdict": "pass"}, "b": {"verdict": "fail"}, "c": {"verdict": "fail"}, "d": {"verdict": "pass"}}
        result = judge_eval.calibrate(judge, human)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["agreement"], 2 / 3)
        self.assertAlmostEqual(result["cohens_kappa"], 0.4)
        self.assertEqual(result["judge_pass_human_fail"], 1)
        self.assertEqual(len(result["warnings"]), 2)

    def test_no_overlap_gives_no_agreement(self) -> None:
        result = judge_eval.calibrate({"a": "pass"}, {})
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["agreement"])


if __name__ == "__main__":
    unittest.main()
