"""V37 fixed, low-capacity task-category local-label predictor.

This module is standard-library-only and follows the official PAIEC predict
contract. It is an experimental offline candidate, not a formal submission.
"""
from __future__ import annotations

import re
from typing import Any

_RULES = {
    "testing": re.compile(r"\b(?:test|tests|testing|pytest|unittest|junit|assert)\b", re.I),
    "dependency_installation": re.compile(
        r"\b(?:dependency|dependencies|installation|install|pip|npm|yarn|pnpm|gradle|maven|cargo)\b|"
        r"\b(?:pom\.xml|go\.mod|requirements\.txt)\b", re.I),
    "api_interface": re.compile(r"\b(?:api|interface|endpoint|signature|protocol|request|response)\b", re.I),
    "bug_exception": re.compile(
        r"\b(?:bug|exception|error|crash|traceback|panic|nullpointer|typeerror|valueerror|failure|failed)\b", re.I),
    "documentation_configuration": re.compile(
        r"\b(?:documentation|docs|readme|configuration|config|yaml|toml|ini|settings)\b|"
        r"\benvironment variable\b", re.I),
}
_SUBJECT_FIELDS = frozenset({
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "reasoning_effort", "harness_version", "subject_features_extra",
})
_ITEM_FIELDS = frozenset({"item_content", "item_features", "interactors", "benchmark_id"})
MIN_GROUP_LABELS = 7
MIN_LOCAL_ITEMS = 3
LOCAL_WEIGHT = 0.25


def task_categories(item_content: Any) -> frozenset[str]:
    """Extract frozen rule categories from prediction-visible text only."""
    if not isinstance(item_content, str) or not item_content:
        return frozenset()
    return frozenset(name for name, pattern in _RULES.items() if pattern.search(item_content))


def _binary_label(value: Any) -> bool:
    return type(value) is int and value in (0, 1)


def _payload_key(item: Any) -> tuple[str, str, str, str] | None:
    if type(item) is not dict or set(item) != _ITEM_FIELDS:
        return None
    values = tuple(item.get(key) for key in ("benchmark_id", "item_content", "item_features", "interactors"))
    if not all(isinstance(value, str) for value in values) or not values[0]:
        return None
    return values  # strict observable payload equality; no fuzzy identity


def _entry(entry: Any):
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        return None
    pair, label = entry
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    subject, item = pair
    if type(subject) is not dict or type(item) is not dict or not _binary_label(label):
        return None
    # Official exact-shape records only; malformed data are excluded defensively.
    if set(subject) != _SUBJECT_FIELDS or not all(isinstance(v, str) for v in subject.values()):
        return None
    if _payload_key(item) is None:
        return None
    return subject, item, label


def _baseline(subject: dict, benchmark: str, labeled: Any) -> tuple[int, int]:
    count = positive = 0
    if not isinstance(labeled, (list, tuple)):
        return 0, 0
    for raw in labeled:
        row = _entry(raw)
        if row is None:
            continue
        source, source_item, label = row
        if source == subject and source_item["benchmark_id"] == benchmark:
            count += 1
            positive += label
    return count, positive


def predict(input: list, labeled: list | None = None) -> float:
    """Predict target correctness from V12 plus a small within-group local term."""
    if not isinstance(input, (list, tuple)) or len(input) != 2:
        return 0.5
    subject, item = input
    if type(subject) is not dict or type(item) is not dict:
        return 0.5
    if set(subject) != _SUBJECT_FIELDS or not all(isinstance(v, str) for v in subject.values()):
        return 0.5
    target_key = _payload_key(item)
    if target_key is None:
        return 0.5
    benchmark = target_key[0]
    n, k = _baseline(subject, benchmark, labeled)
    p12 = (k + 1) / (n + 2)
    if n < MIN_GROUP_LABELS:
        return float(p12)
    target_categories = task_categories(item.get("item_content"))
    if not target_categories or not isinstance(labeled, (list, tuple)):
        return float(p12)

    # Deduplicate by full visible item payload. Repeated equal outcomes are one
    # task for local support; conflicting labels make that payload ambiguous.
    payload_labels: dict[tuple[str, str, str, str], set[int]] = {}
    for raw in labeled:
        row = _entry(raw)
        if row is None:
            continue
        source, source_item, label = row
        if source != subject or source_item["benchmark_id"] != benchmark:
            continue
        key = _payload_key(source_item)
        if key is not None:
            payload_labels.setdefault(key, set()).add(label)

    local_labels: list[int] = []
    for key, outcomes in payload_labels.items():
        if key == target_key or len(outcomes) != 1:
            continue
        categories = task_categories(key[1])
        if categories and categories.intersection(target_categories):
            local_labels.append(next(iter(outcomes)))
    if len(local_labels) < MIN_LOCAL_ITEMS:
        return float(p12)
    local_positive = sum(local_labels)
    p_local = (local_positive + 1) / (len(local_labels) + 2)
    probability = (1.0 - LOCAL_WEIGHT) * p12 + LOCAL_WEIGHT * p_local
    return float(min(1.0, max(0.0, probability)))
