"""V22 binomial minimax estimator for a Bernoulli proportion.

Independent experimental candidate. It uses only current same-subject,
same-benchmark binary labels and has no external dependencies or state.
"""
from __future__ import annotations

import math


def _get_pair(input):
    if not isinstance(input, (list, tuple)) or len(input) != 2:
        return None
    subject, item = input
    if not isinstance(subject, dict) or not isinstance(item, dict):
        return None
    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        return None
    return subject, benchmark


def predict(input: list, labeled: list | None = None) -> float:
    """Predict correctness using the V22 minimax binomial estimator.

    Malformed top-level/evidence records are handled conservatively. For valid
    Starting Kit records, filtering matches V12: exact subject-dict equality
    and exact benchmark_id equality; all revealed binary response records
    (including repeated response rows) count as supplied by the platform.
    """
    pair = _get_pair(input)
    if pair is None:
        return 0.5
    subject, benchmark = pair
    if labeled is None:
        labeled = []
    if not isinstance(labeled, (list, tuple)):
        labeled = []

    n = k = 0
    for entry in labeled:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        observed_pair, label = entry
        if not isinstance(observed_pair, (list, tuple)) or len(observed_pair) != 2:
            continue
        observed_subject, observed_item = observed_pair
        if not isinstance(observed_subject, dict) or not isinstance(observed_item, dict):
            continue
        if observed_subject != subject:
            continue
        if observed_item.get("benchmark_id") != benchmark:
            continue
        # Official labels are integer 0/1. Reject bool, floats, and other
        # malformed values instead of silently coercing them.
        if type(label) is not int or label not in (0, 1):
            continue
        n += 1
        k += label

    if n == 0:
        return 0.5
    root_n = math.sqrt(n)
    p = (k + root_n / 2.0) / (n + root_n)
    if not math.isfinite(p):
        return 0.5
    return float(min(1.0, max(0.0, p)))
