from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stats  # noqa: E402


class WilsonIntervalTests(unittest.TestCase):
    def test_twelve_of_twelve_cannot_prove_a_high_rate(self) -> None:
        lower, upper = stats.wilson_interval(12, 12)
        self.assertGreater(lower, 0.74)
        self.assertLess(lower, 0.77)
        self.assertEqual(upper, 1.0)

    def test_no_data_gives_the_uninformative_interval(self) -> None:
        self.assertEqual(stats.wilson_interval(0, 0), (0.0, 1.0))

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stats.wilson_interval(5, 3)


class SampleSizeTests(unittest.TestCase):
    def test_all_pass_run_needed_to_certify_a_target_rate(self) -> None:
        self.assertEqual(stats.min_cases_for_zero_failures(0.80), 16)
        self.assertEqual(stats.min_cases_for_zero_failures(0.90), 35)

    def test_detecting_a_ten_point_drop_needs_about_two_hundred_cases_per_arm(self) -> None:
        self.assertTrue(195 <= stats.required_sample_size(0.90, 0.80) <= 205)

    def test_identical_rates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stats.required_sample_size(0.8, 0.8)


class McNemarTests(unittest.TestCase):
    def test_no_discordant_cases_is_not_significant(self) -> None:
        self.assertEqual(stats.exact_mcnemar_p(0, 0), 1.0)

    def test_known_values(self) -> None:
        self.assertAlmostEqual(stats.exact_mcnemar_p(5, 0), 0.0625)
        self.assertAlmostEqual(stats.exact_mcnemar_p(10, 0), 2 / 1024)
        self.assertEqual(stats.exact_mcnemar_p(3, 3), 1.0)


class CohensKappaTests(unittest.TestCase):
    def test_perfect_and_opposite_agreement(self) -> None:
        self.assertAlmostEqual(stats.cohens_kappa([("pass", "pass"), ("fail", "fail")]), 1.0)
        self.assertAlmostEqual(stats.cohens_kappa([("pass", "fail"), ("fail", "pass")]), -1.0)

    def test_undefined_when_every_rating_is_identical(self) -> None:
        self.assertIsNone(stats.cohens_kappa([("pass", "pass"), ("pass", "pass")]))
        self.assertIsNone(stats.cohens_kappa([]))


class SummarizeRepeatsTests(unittest.TestCase):
    def test_mean_and_spread(self) -> None:
        summary = stats.summarize_repeats([0.5, 0.7])
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["mean"], 0.6)
        self.assertAlmostEqual(summary["stdev"], 0.1414, places=3)

    def test_empty_input(self) -> None:
        self.assertEqual(stats.summarize_repeats([])["n"], 0)


if __name__ == "__main__":
    unittest.main()
