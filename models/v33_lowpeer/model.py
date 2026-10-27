"""V33 live conservative cross-subject transfer candidate.

Uses only the current official ``predict(input, labeled)`` arguments. The
baseline is V12's Beta(1, 1) empirical mean. A bounded, zero-centered residual
from other revealed subject-metadata records on an exact same-benchmark item
payload is considered only after a conservative evidence threshold.
"""

from __future__ import annotations

import json
import math
from typing import Any

MIN_TARGET_LABELS = 7
MIN_OTHER_SUBJECTS = 1
RESIDUAL_PRIOR_STRENGTH = 2.0
SUBJECT_FIELDS = frozenset({
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "harness_version", "reasoning_effort", "subject_features_extra",
})
ITEM_FIELDS = frozenset({"item_content", "item_features", "interactors", "benchmark_id"})


def _valid_binary(value: Any) -> bool:
    return type(value) is int and value in (0, 1)


def _stable_dict_key(value: Any) -> str | None:
    if type(value) is not dict or set(value) != SUBJECT_FIELDS or not all(
        isinstance(key, str) and isinstance(field, str)
        for key, field in value.items()
    ):
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _benchmark(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    value = item.get("benchmark_id")
    return value if isinstance(value, str) and value else None


def _item_key(item: Any, benchmark: str) -> str | None:
    """Return an exact observable payload key, not a claimed platform item ID."""
    if type(item) is not dict or set(item) != ITEM_FIELDS or _benchmark(item) != benchmark:
        return None
    fields = ("item_content", "item_features", "interactors")
    if not all(isinstance(item.get(field), str) for field in fields):
        return None
    if not item["item_content"].strip():
        return None
    return json.dumps(
        [benchmark, *(item[field] for field in fields)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_label(entry: Any) -> tuple[dict, dict, int] | None:
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        return None
    pair, label = entry
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    subject, item = pair
    if not isinstance(subject, dict) or not isinstance(item, dict) or not _valid_binary(label):
        return None
    return subject, item, label


def v12_probability(input: list, labeled: list | None = None) -> float:
    """Reproduce the frozen V12 rule, filtering exact subject and benchmark."""
    if not isinstance(input, (list, tuple)) or len(input) != 2:
        raise ValueError("input must be [subject_dict, item_dict].")
    subject, item = input
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise ValueError("subject and item must be dictionaries.")
    benchmark = _benchmark(item)
    if labeled is None:
        labeled = []
    if not isinstance(labeled, (list, tuple)):
        raise ValueError("labeled must be a list or tuple.")
    n = k = 0
    for entry in labeled:
        parsed = _parse_label(entry)
        if parsed is None:
            continue
        observed_subject, observed_item, label = parsed
        if observed_subject == subject and benchmark is not None and _benchmark(observed_item) == benchmark:
            n += 1
            k += label
    if n == 0:
        return 0.5
    return float((k + 1) / (n + 2))


def _cross_subject_residuals(
    target_subject: dict,
    target_item: dict,
    benchmark: str,
    target_key: str,
    labeled: list | tuple,
) -> list[float]:
    """One item residual per distinct exact subject-attribute dictionary.

    A source subject's baseline is Beta(1,1) smoothed on its *other* distinct
    same-benchmark payloads. Ambiguous duplicate payloads with conflicting
    labels invalidate that source subject's contribution.
    """
    target_subject_key = _stable_dict_key(target_subject)
    per_subject: dict[str, dict[str, set[int]]] = {}
    for entry in labeled:
        parsed = _parse_label(entry)
        if parsed is None:
            continue
        source_subject, source_item, label = parsed
        if _benchmark(source_item) != benchmark:
            continue
        source_subject_key = _stable_dict_key(source_subject)
        source_item_key = _item_key(source_item, benchmark)
        if source_subject_key is None or source_item_key is None:
            continue
        if source_subject_key == target_subject_key or source_subject == target_subject:
            continue
        per_subject.setdefault(source_subject_key, {}).setdefault(source_item_key, set()).add(label)

    residuals: list[float] = []
    for source_key, by_item in per_subject.items():
        target_labels = by_item.get(target_key, set())
        other_items = {key: labels for key, labels in by_item.items() if key != target_key}
        # Require one unambiguous same-benchmark baseline item, and avoid
        # treating repeated/contradictory outcomes as independent evidence.
        if not target_labels or len(target_labels) != 1 or not other_items:
            continue
        if any(len(labels) != 1 for labels in other_items.values()):
            continue
        base_n = len(other_items)
        base_k = sum(next(iter(labels)) for labels in other_items.values())
        source_base = (base_k + 1) / (base_n + 2)
        item_outcome = next(iter(target_labels))
        residuals.append(float(item_outcome - source_base))
    return residuals


def predict(input: list, labeled: list | None = None) -> float:
    """Return V21 probability, strictly reverting to V12 when unsupported.

    An exact match means equality of benchmark_id, item_content,
    item_features, and interactors. The official interface supplies no
    independent item ID, so this is a payload-match heuristic, not an identity
    guarantee. Other-subject revealed outcomes are centered by each source
    subject's own smoothed rate on other revealed items.
    """
    if not isinstance(input, (list, tuple)) or len(input) != 2:
        return 0.5
    subject, item = input
    if not isinstance(subject, dict) or not isinstance(item, dict):
        return 0.5
    if labeled is not None and not isinstance(labeled, (list, tuple)):
        labeled = []
    try:
        p12 = v12_probability(input, labeled)
    except (TypeError, ValueError, OverflowError):
        return 0.5
    if not math.isfinite(p12) or not 0.0 <= p12 <= 1.0:
        return 0.5
    if not isinstance(labeled, (list, tuple)) or len(labeled) == 0:
        return p12
    benchmark = _benchmark(item)
    target_key = _item_key(item, benchmark) if benchmark else None
    if target_key is None:
        return p12

    # A target response should never appear among its own acquisition evidence.
    # In a contaminated/contrafactual input, remove every exact-payload record
    # for this target subject before computing either the V12 fallback or the
    # cross-subject adjustment. Valid PAIEC splits should already ensure this.
    safe_labeled = []
    for entry in labeled:
        parsed = _parse_label(entry)
        if parsed and parsed[0] == subject and _item_key(parsed[1], benchmark) == target_key:
            continue
        safe_labeled.append(entry)
    p12 = v12_probability(input, safe_labeled)

    target_n = 0
    for entry in safe_labeled:
        parsed = _parse_label(entry)
        if parsed and parsed[0] == subject and _benchmark(parsed[1]) == benchmark:
            target_n += 1
    if target_n < MIN_TARGET_LABELS:
        return p12

    residuals = _cross_subject_residuals(subject, item, benchmark, target_key, labeled)
    if len(residuals) < MIN_OTHER_SUBJECTS:
        return p12
    signs = {1 if residual > 0.0 else -1 if residual < 0.0 else 0 for residual in residuals}
    # Require unanimous, non-zero direction as a minimal stability gate. This
    # is not a significance test; it only prevents mixed evidence from moving
    # the baseline in this small-sample prototype.
    if len(signs) != 1 or 0 in signs:
        return p12

    # Do not treat metadata-distinct systems as statistically independent.
    # Aggregate their residuals, but shrink using one effective pooled signal
    # plus two zero-residual pseudo-signals, regardless of source count.
    delta = (sum(residuals) / len(residuals)) * min(1.0 / 3.0, len(residuals) / (len(residuals) + 6.0))
    probability = min(1.0, max(0.0, p12 + delta))
    if not math.isfinite(probability):
        return p12
    return float(probability)


def leave_one_out_diagnostic(records: list[dict]) -> dict:
    """Internal revealed-record LOO diagnostic; never a held-out estimate.

    Each record has ``subject``, ``item``, and binary ``label``. A held-out
    outcome is omitted from the supplied evidence for both predictors.
    """
    v12_losses: list[float] = []
    v20_losses: list[float] = []
    rows: list[dict] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not _valid_binary(record.get("label")):
            continue
        subject, item, outcome = record.get("subject"), record.get("item"), record["label"]
        if not isinstance(subject, dict) or not isinstance(item, dict):
            continue
        target_benchmark = _benchmark(item)
        target_item_key = _item_key(item, target_benchmark) if target_benchmark else None
        evidence = []
        # This function accepts an explicitly ordered synthetic/prequential
        # stream. Only earlier events are visible; never infer temporal order
        # from the official labeled-list order.
        for other_index, other in enumerate(records[:index]):
            if not isinstance(other, dict) or not _valid_binary(other.get("label")):
                continue
            if isinstance(other.get("subject"), dict) and isinstance(other.get("item"), dict):
                # Hold out all repeated rows for the same observable target,
                # so another recorded outcome for identical input cannot leak.
                if (other["subject"] == subject and target_item_key is not None
                        and _item_key(other["item"], target_benchmark) == target_item_key):
                    continue
                evidence.append([[other["subject"], other["item"]], other["label"]])
        pair = [subject, item]
        p_base = v12_probability(pair, evidence)
        p_candidate = predict(pair, evidence)
        loss_base = (p_base - outcome) ** 2
        loss_candidate = (p_candidate - outcome) ** 2
        v12_losses.append(loss_base)
        v20_losses.append(loss_candidate)
        rows.append({"index": index, "n_same_pair_evidence": len([
            row for row in evidence
            if _parse_label(row) is not None
            and _parse_label(row)[0] == subject
            and _benchmark(_parse_label(row)[1]) == target_benchmark
        ]), "p_v12": p_base, "p_v20": p_candidate,
            "outcome": outcome, "brier_v12": loss_base,
            "brier_v20": loss_candidate})
    return {
        "kind": "within-revealed-record LOO; not independent held-out validation",
        "n": len(rows),
        "mean_brier_v12": sum(v12_losses) / len(v12_losses) if rows else None,
        "mean_brier_v20": sum(v20_losses) / len(v20_losses) if rows else None,
        "mean_paired_difference_v20_minus_v12": (
            sum(b - a for a, b in zip(v12_losses, v20_losses)) / len(rows) if rows else None
        ),
        "rows": rows,
    }

