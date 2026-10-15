"""Leakage-guarded paired V12 vs feature-offset Brier ALC evaluation.

This module uses only the Python standard library. A real dataset is rejected
unless its manifest carries explicit PAIEC public-training-pool assignment,
exact source revision, authorization and license-review evidence. Synthetic
unit data require an explicit allow_synthetic=True call.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any

BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
RIDGE_L2 = 1.0  # fixed before outcomes; penalizes item-feature adjustment, not intercept
SPLIT_SEED = 20261014
BOOTSTRAP_SEED = 1701
BOOTSTRAP_REPLICATES = 2000
FEATURE_NAMES = (
    "verification_correct", "verification_incorrect", "verification_unverified",
    "checkable_steps", "observable_obstacle_count_scaled",
    "generation_truncated", "evidence_complete",
)
ALLOWED_VERIFICATION = {"verified_correct", "verified_incorrect", "not_independently_verified"}


class EligibilityError(ValueError):
    pass


def validate_real_eligibility(metadata: dict[str, Any]) -> None:
    """Fail closed unless exact competition-pool and use evidence is recorded."""
    if metadata.get("data_kind") != "real":
        raise EligibilityError("Real-data manifest must declare data_kind='real'.")
    if metadata.get("paiec_pool_status") != "confirmed_public_training_pool":
        raise EligibilityError("No authoritative PAIEC public-training-pool assignment recorded.")
    source = metadata.get("source", {})
    assignment = metadata.get("pool_assignment", {})
    use = metadata.get("use_authorization", {})
    feature_provenance = metadata.get("feature_provenance", {})
    split_provenance = metadata.get("split_provenance", {})
    required = {
        "source.url": source.get("url"),
        "source.revision": source.get("revision"),
        "source.files_sha256": source.get("files_sha256"),
        "pool_assignment.url": assignment.get("url"),
        "pool_assignment.revision_or_date": assignment.get("revision_or_date"),
        "pool_assignment.benchmark_ids": assignment.get("benchmark_ids"),
        "pool_assignment.evidence_field": assignment.get("evidence_field"),
        "use_authorization.status": use.get("status"),
        "use_authorization.evidence_url": use.get("evidence_url"),
        "use_authorization.license_review_reference": use.get("license_review_reference"),
        "feature_provenance.source": feature_provenance.get("source"),
        "feature_provenance.protocol_version": feature_provenance.get("protocol_version"),
        "feature_provenance.outcome_blind": feature_provenance.get("outcome_blind"),
        "split_provenance.definition_url_or_path": split_provenance.get("definition_url_or_path"),
        "split_provenance.definition_sha256": split_provenance.get("definition_sha256"),
        "split_provenance.frozen_before_outcomes": split_provenance.get("frozen_before_outcomes"),
        "label_semantics": metadata.get("label_semantics"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise EligibilityError("Real-data eligibility manifest is incomplete: " + ", ".join(missing))
    if use.get("status") != "confirmed_for_competition_development":
        raise EligibilityError("Use/access authorization is not explicitly confirmed for development.")
    if feature_provenance.get("outcome_blind") is not True:
        raise EligibilityError("Feature provenance must attest features were produced without target outcomes.")
    if split_provenance.get("frozen_before_outcomes") is not True:
        raise EligibilityError("Training/held-out benchmark split must be frozen before outcome inspection.")


def _feature_vector(features: dict[str, Any]) -> tuple[float, ...]:
    expected = {"verification_status", "checkable_steps", "observable_obstacle_count", "generation_truncated", "evidence_complete"}
    if set(features) != expected:
        raise ValueError(f"features must contain exactly {sorted(expected)}")
    status = features["verification_status"]
    if status not in ALLOWED_VERIFICATION:
        raise ValueError("invalid verification_status")
    for name in ("checkable_steps", "generation_truncated", "evidence_complete"):
        if type(features[name]) is not bool:
            raise ValueError(f"{name} must be bool")
    count = features["observable_obstacle_count"]
    if type(count) is not int or count < 0:
        raise ValueError("observable_obstacle_count must be a nonnegative int")
    return (
        float(status == "verified_correct"),
        float(status == "verified_incorrect"),
        float(status == "not_independently_verified"),
        float(features["checkable_steps"]),
        min(count, 10) / 10.0,
        float(features["generation_truncated"]),
        float(features["evidence_complete"]),
    )


def _stable_rank(seed: int, *parts: str) -> str:
    raw = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _prepare_group(group: dict[str, Any], seed: int) -> dict[str, Any] | None:
    subject = group.get("subject_id")
    benchmark = group.get("benchmark_id")
    items = group.get("items")
    if not isinstance(subject, str) or not subject or not isinstance(benchmark, str) or not benchmark:
        raise ValueError("each group needs nonempty subject_id and benchmark_id")
    if not isinstance(items, list):
        raise ValueError("group.items must be a list")
    seen: set[str] = set()
    valid_items = []
    for item in items:
        item_id = item.get("item_id")
        content = item.get("item_content")
        labels = item.get("response_labels")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("item_id must be unique and nonempty within a subject-benchmark group")
        seen.add(item_id)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("each item needs its exact item_content")
        if not isinstance(labels, list) or not labels:
            raise ValueError("response_labels must be a nonempty list of binary outcomes")
        if any(type(y) is not int or y not in (0, 1) for y in labels):
            raise ValueError("only strict binary integer response labels 0/1 are accepted")
        x = _feature_vector(item.get("features", {}))
        valid_items.append({"item_id": item_id, "item_content": content, "response_labels": labels, "x": x})
    # The official split is defined only for groups with at least 80 distinct items.
    if len(valid_items) < 80:
        return None
    ranked = sorted(valid_items, key=lambda it: _stable_rank(seed, subject, benchmark, it["item_id"]))
    cut = len(ranked) // 2
    acquisition = ranked[:cut]
    evaluation = ranked[cut:]
    # Simulate the official random one-response reveal per selected acquisition item
    # with a fixed independent seed; all evaluation response runs remain score rows.
    acquired = []
    for item in sorted(acquisition, key=lambda it: _stable_rank(seed + 1, subject, benchmark, it["item_id"])):
        idx = int(_stable_rank(seed + 2, subject, benchmark, item["item_id"]), 16) % len(item["response_labels"])
        acquired.append({"item_id": item["item_id"], "label": item["response_labels"][idx]})
    return {"subject_id": subject, "benchmark_id": benchmark, "acquisition": acquired, "evaluation": evaluation, "all_normalized_item_content": {" ".join(it["item_content"].casefold().split()) for it in valid_items}}


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-min(z, 700.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def v12_probability(n: int, k: int) -> float:
    if type(n) is not int or type(k) is not int or n < 0 or not 0 <= k <= n:
        raise ValueError("require integers n>=0 and 0<=k<=n")
    return (k + 1.0) / (n + 2.0)


def pair_equal_mean(values: list[float]) -> float:
    """Official macro step: an unweighted mean after one score per pair exists."""
    if not values:
        raise ValueError("cannot average an empty pair collection")
    return sum(values) / len(values)


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise ArithmeticError("singular Newton system")
        m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        m[col] = [v / scale for v in m[col]]
        for r in range(n):
            if r == col:
                continue
            scale = m[r][col]
            if scale:
                m[r] = [m[r][j] - scale * m[col][j] for j in range(n + 1)]
    return [m[i][-1] for i in range(n)]


def _fit_offset_logistic(rows: list[tuple[float, tuple[float, ...], int, float]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no train-only calibration examples")
    d = 1 + len(FEATURE_NAMES)
    beta = [0.0] * d
    total_weight = sum(r[3] for r in rows)
    if total_weight <= 0:
        raise ValueError("training weights must sum to a positive value")
    for iteration in range(100):
        grad = [0.0] * d
        hess = [[0.0] * d for _ in range(d)]
        for base, x, y, weight in rows:
            xv = (1.0,) + x
            z = _logit(base) + sum(beta[j] * xv[j] for j in range(d))
            p = _sigmoid(z)
            w = weight / total_weight
            for j in range(d):
                grad[j] += w * (p - y) * xv[j]
                for q in range(d):
                    hess[j][q] += w * p * (1.0 - p) * xv[j] * xv[q]
        for j in range(1, d):
            grad[j] += RIDGE_L2 * beta[j]
            hess[j][j] += RIDGE_L2
        step = _solve_linear(hess, grad)
        beta = [beta[j] - step[j] for j in range(d)]
        if max(abs(v) for v in step) < 1e-8:
            break
    return {"intercept": beta[0], "feature_coefficients": dict(zip(FEATURE_NAMES, beta[1:])), "l2": RIDGE_L2, "iterations": iteration + 1, "train_examples": len(rows)}


def _predict_candidate(base: float, x: tuple[float, ...], model: dict[str, Any]) -> float:
    adjustment = model["intercept"] + sum(model["feature_coefficients"][name] * value for name, value in zip(FEATURE_NAMES, x))
    p = _sigmoid(_logit(base) + adjustment)
    return min(1.0, max(0.0, p))


def _training_rows(groups: list[dict[str, Any]]) -> list[tuple[float, tuple[float, ...], int, float]]:
    rows = []
    for g in groups:
        evaluation = g["evaluation"]
        eval_n = sum(len(it["response_labels"]) for it in evaluation)
        prefix: list[int] = []
        for budget, alc_w in zip(BUDGETS, ALC_WEIGHTS):
            prefix = [a["label"] for a in g["acquisition"][:budget]]
            n, k = len(prefix), sum(prefix)
            base = v12_probability(n, k)
            # Each train subject-benchmark contributes total weight 1 after its six budget weights.
            row_weight = alc_w / eval_n
            for item in evaluation:
                for y in item["response_labels"]:
                    rows.append((base, item["x"], y, row_weight))
    return rows


def _group_scores(group: dict[str, Any], model: dict[str, Any] | None) -> dict[str, Any]:
    per_budget = {}
    for budget in BUDGETS:
        prefix = [x["label"] for x in group["acquisition"][:budget]]
        n, k = len(prefix), sum(prefix)
        base = v12_probability(n, k)
        v12_losses = []
        new_losses = []
        for item in group["evaluation"]:
            p_new = _predict_candidate(base, item["x"], model) if model is not None else base
            for y in item["response_labels"]:
                v12_losses.append((base - y) ** 2)
                new_losses.append((p_new - y) ** 2)
        per_budget[str(budget)] = {"v12_brier": sum(v12_losses) / len(v12_losses), "candidate_brier": sum(new_losses) / len(new_losses), "n_revealed": n, "evaluation_response_count": len(v12_losses), "acquired_prefix_labels": prefix}
    v12_alc = sum(w * per_budget[str(b)]["v12_brier"] for b, w in zip(BUDGETS, ALC_WEIGHTS))
    cand_alc = sum(w * per_budget[str(b)]["candidate_brier"] for b, w in zip(BUDGETS, ALC_WEIGHTS))
    return {"subject_id": group["subject_id"], "benchmark_id": group["benchmark_id"], "v12_by_budget": {b: per_budget[str(b)]["v12_brier"] for b in BUDGETS}, "candidate_by_budget": {b: per_budget[str(b)]["candidate_brier"] for b in BUDGETS}, "revealed_prefixes": {b: per_budget[str(b)]["acquired_prefix_labels"] for b in BUDGETS}, "evaluation_response_counts": {b: per_budget[str(b)]["evaluation_response_count"] for b in BUDGETS}, "v12_brier_alc": v12_alc, "candidate_brier_alc": cand_alc, "delta_candidate_minus_v12": cand_alc - v12_alc}


def _bootstrap_ci(per_group: list[dict[str, Any]], replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    by_bench: dict[str, list[float]] = {}
    for row in per_group:
        by_bench.setdefault(row["benchmark_id"], []).append(row["delta_candidate_minus_v12"])
    names = sorted(by_bench)
    if len(names) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(replicates):
        chosen = [rng.choice(names) for _ in names]
        sample = [delta for bench in chosen for delta in by_bench[bench]]
        vals.append(sum(sample) / len(sample))
    vals.sort()
    return vals[int(0.025 * (replicates - 1))], vals[int(0.975 * (replicates - 1))]


def evaluate_dataset(dataset: dict[str, Any], *, allow_synthetic: bool = False) -> dict[str, Any]:
    metadata = dataset.get("metadata", {})
    if metadata.get("data_kind") == "synthetic":
        if not allow_synthetic:
            raise EligibilityError("Synthetic data require explicit allow_synthetic=True.")
    else:
        validate_real_eligibility(metadata)
    train_bench = set(metadata.get("train_benchmarks", []))
    heldout_bench = set(metadata.get("heldout_benchmarks", []))
    if not train_bench or not heldout_bench or train_bench & heldout_bench:
        raise ValueError("Declare disjoint nonempty train_benchmarks and heldout_benchmarks before reading outcomes.")
    if metadata.get("data_kind") == "real":
        assigned = set(metadata["pool_assignment"]["benchmark_ids"])
        if not (train_bench | heldout_bench) <= assigned:
            raise EligibilityError("Frozen split contains benchmark IDs absent from the cited official training-pool assignment.")
    groups = dataset.get("groups")
    if not isinstance(groups, list):
        raise ValueError("dataset.groups must be a list")
    seen_group = set()
    prepared = []
    for group in groups:
        key = (group.get("subject_id"), group.get("benchmark_id"))
        if key in seen_group:
            raise ValueError(f"duplicate subject-benchmark group: {key}")
        seen_group.add(key)
        bench = key[1]
        if bench not in train_bench | heldout_bench:
            raise ValueError(f"benchmark {bench!r} has no frozen split assignment")
        row = _prepare_group(group, int(metadata.get("split_seed", SPLIT_SEED)))
        if row is not None:
            row["partition"] = "train" if bench in train_bench else "heldout"
            prepared.append(row)
    train_groups = [g for g in prepared if g["partition"] == "train"]
    test_groups = [g for g in prepared if g["partition"] == "heldout"]
    if not train_groups or not test_groups:
        raise ValueError("Need at least one eligible (>=80-item) train group and held-out group.")
    # Exact normalized duplicate texts may not straddle the benchmark split.
    train_texts = {text for g in train_groups for text in g["all_normalized_item_content"]}
    leaked = sorted({text for g in test_groups for text in g["all_normalized_item_content"]} & train_texts)
    if leaked:
        raise ValueError(f"Exact normalized item-content overlap across train/held-out benchmarks ({len(leaked)} items).")
    train_rows = _training_rows(train_groups)
    model = _fit_offset_logistic(train_rows)
    train_diagnostics = [_group_scores(g, model) for g in train_groups]
    per_group = [_group_scores(g, model) for g in test_groups]
    all_group_rows = per_group
    by_budget = {}
    for budget in BUDGETS:
        b = str(budget)
        # Official pair-equal macro: average within pair first, then unweighted over pairs.
        by_budget[b] = {"v12_brier_pair_macro": pair_equal_mean([r["v12_by_budget"][budget] for r in all_group_rows]), "candidate_brier_pair_macro": pair_equal_mean([r["candidate_by_budget"][budget] for r in all_group_rows])}
    per_benchmark = []
    for bench in sorted({r["benchmark_id"] for r in per_group}):
        rows = [r for r in per_group if r["benchmark_id"] == bench]
        per_benchmark.append({"benchmark_id": bench, "subject_benchmark_pairs": len(rows), "v12_macro_alc": pair_equal_mean([r["v12_brier_alc"] for r in rows]), "candidate_macro_alc": pair_equal_mean([r["candidate_brier_alc"] for r in rows]), "delta_candidate_minus_v12": pair_equal_mean([r["delta_candidate_minus_v12"] for r in rows])})
    v12_macro = pair_equal_mean([r["v12_brier_alc"] for r in per_group])
    cand_macro = pair_equal_mean([r["candidate_brier_alc"] for r in per_group])
    ci = _bootstrap_ci(per_group)
    return {"status": "synthetic_only" if metadata.get("data_kind") == "synthetic" else "heldout_real_evaluation", "data_kind": metadata.get("data_kind"), "split": {"train_benchmarks": sorted(train_bench), "heldout_benchmarks": sorted(heldout_bench), "train_groups_used": len(train_groups), "heldout_groups_used": len(test_groups), "groups_excluded_below_80_items": len(groups) - len(prepared), "split_seed": int(metadata.get("split_seed", SPLIT_SEED))}, "candidate_fit": {**model, "train_benchmarks_only": sorted(train_bench), "ridge_l2_predeclared": RIDGE_L2, "training_in_sample_macro_v12_alc": sum(r["v12_brier_alc"] for r in train_diagnostics) / len(train_diagnostics), "training_in_sample_macro_candidate_alc": sum(r["candidate_brier_alc"] for r in train_diagnostics) / len(train_diagnostics)}, "heldout": {"per_subject_benchmark": per_group, "per_benchmark_macro": per_benchmark, "pair_equal_macro_v12_brier_alc": v12_macro, "pair_equal_macro_candidate_brier_alc": cand_macro, "paired_delta_candidate_minus_v12": cand_macro - v12_macro, "paired_95pct_benchmark_cluster_bootstrap_ci": ci, "brier_macro_by_budget": by_budget}, "scoring": {"budgets": list(BUDGETS), "weights": list(ALC_WEIGHTS), "weight_sum": sum(ALC_WEIGHTS), "within_pair_aggregate": "mean squared error over every recorded evaluation response; same item prediction reused for repeated response runs", "overall_aggregate": "unweighted mean of subject-benchmark pair ALCs", "acquisition": "fixed seeded random 50/50 split by distinct item; at most one recorded binary response sampled per acquired item; same prefix reused for V12 and candidate"}}


def evaluate_files(manifest_path: str, observations_path: str, *, allow_synthetic: bool = False) -> dict[str, Any]:
    """Validate the small manifest before opening any outcome-bearing data file."""
    import json
    from pathlib import Path

    metadata = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if metadata.get("data_kind") == "synthetic":
        if not allow_synthetic:
            raise EligibilityError("Synthetic data require explicit allow_synthetic=True.")
    else:
        validate_real_eligibility(metadata)
    # Only after the eligibility gate succeeds do we open the response-bearing file.
    observations = json.loads(Path(observations_path).read_text(encoding="utf-8"))
    if not isinstance(observations, dict) or not isinstance(observations.get("groups"), list):
        raise ValueError("observations file must contain a groups list")
    return evaluate_dataset({"metadata": metadata, "groups": observations["groups"]}, allow_synthetic=allow_synthetic)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_json", help="Small metadata/provenance manifest checked before outcome data is opened")
    parser.add_argument("observations_json", help="Outcome-bearing groups JSON; opened only after manifest passes")
    parser.add_argument("output_json", help="Where to write paired results")
    parser.add_argument("--allow-synthetic", action="store_true", help="Required to run an explicitly synthetic unit dataset")
    args = parser.parse_args()
    result = evaluate_files(args.manifest_json, args.observations_json, allow_synthetic=args.allow_synthetic)
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "heldout_pairs": result["split"]["heldout_groups_used"], "paired_delta_candidate_minus_v12": result["heldout"]["paired_delta_candidate_minus_v12"]}, ensure_ascii=False))
