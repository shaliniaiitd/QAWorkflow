"""Small statistics toolkit for the evaluation harness (stdlib only).

Why this exists: a pass rate from a dozen cases is a noisy estimate. These helpers
put an honest uncertainty around every rate, say how many cases a claim needs, and
compare two runs without over-reading small differences.
"""

from __future__ import annotations

import math
import statistics
from statistics import NormalDist


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Well behaved at 0% and 100%."""
    if total <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def min_cases_for_zero_failures(target_rate: float, confidence: float = 0.95) -> int:
    """Fewest cases such that an all-pass run proves pass rate >= target_rate.

    With zero failures the Wilson lower bound is n / (n + z^2), so
    n >= target * z^2 / (1 - target).
    """
    if not 0 < target_rate < 1:
        raise ValueError("target_rate must be between 0 and 1")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    return math.ceil(target_rate * z * z / (1 - target_rate))


def required_sample_size(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Cases per arm to detect a pass-rate change p1 -> p2 (unpaired two-proportion test)."""
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_beta = NormalDist().inv_cdf(power)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / (p1 - p2) ** 2)


def exact_mcnemar_p(only_a_passes: int, only_b_passes: int) -> float:
    """Two-sided exact p-value for paired pass/fail results on the same cases.

    Only the discordant cases matter: cases A passed and B failed, and vice versa.
    """
    discordant = only_a_passes + only_b_passes
    if discordant == 0:
        return 1.0
    smaller = min(only_a_passes, only_b_passes)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement between two raters beyond chance. None when undefined."""
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    labels = {label for pair in pairs for label in pair}
    expected = sum(
        (sum(1 for a, _ in pairs if a == label) / total) * (sum(1 for _, b in pairs if b == label) / total)
        for label in labels
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)


def summarize_repeats(values: list[float]) -> dict[str, float | int | None]:
    """Mean and spread across repeated runs of the same configuration."""
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }
