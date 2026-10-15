"""Frozen paired real-response comparison: V12 vs the V22 minimax estimator.

Reuses V27/V37 loaders, splits, and nested acquisition prefixes verbatim. No
data are downloaded, no parameters are fitted, and no prior result is changed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results.json"
V37_RUNNER = ROOT / "experiments/v37_task_conditioned/run_trial.py"
V12_ZIP = ROOT / "release_staging/v12_beta11_candidate/v12_beta11_candidate.zip"
V33_ZIP = ROOT / "release_staging/v33_conservative_transfer_candidate/v33_conservative_transfer_candidate.zip"
EXPECTED = {
    "v12": "de12d731b6f1c4b9f6c83226998e17fc930ad55e8c16948c753c6ceaa754fe02",
    "v33": "21e8b461ec483f6a13365e001cadb7f94557ad7a325b5343c941495279477b41",
}
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
REVISION = "54074453da7b9911be0f48e4c560be547bc84d21"
BOOTSTRAP_SEED = 3801
BOOTSTRAP_REPS = 10000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def v12(n: int, k: int) -> float:
    return (k + 1.0) / (n + 2.0)


def v38(n: int, k: int) -> float:
    if n == 0:
        return 0.5
    root_n = math.sqrt(n)
    return (k + root_n / 2.0) / (n + root_n)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    x = q * (len(xs) - 1)
    lo = int(x)
    hi = min(len(xs) - 1, lo + 1)
    return xs[lo] * (hi - x) + xs[hi] * (x - lo)


def bootstrap_mean_ci(values: list[float], seed: int) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values)
             for _ in range(BOOTSTRAP_REPS)]
    return [quantile(means, 0.025), quantile(means, 0.975)]


def score_benchmark(loader: object, name: str) -> dict:
    source = loader.load_source(name)
    frozen = loader.eval_data(source)
    group_results = []
    for group in frozen["groups"]:
        acq = group["acquisition"]
        eval_rows = group["evaluation"]
        budgets = {}
        for budget, weight in zip(BUDGETS, WEIGHTS):
            prefix = acq[:budget]
            n = len(prefix)
            k = sum(row["y"] for row in prefix)
            p12 = v12(n, k)
            p38 = v38(n, k)
            loss12 = sum((p12 - row["y"]) ** 2 for row in eval_rows) / len(eval_rows)
            loss38 = sum((p38 - row["y"]) ** 2 for row in eval_rows) / len(eval_rows)
            budgets[str(budget)] = {
                "n_revealed": n, "k_positive": k,
                "evaluation_response_rows": len(eval_rows),
                "p_v12": p12, "p_v38": p38,
                "v12_brier": loss12, "v38_brier": loss38,
                "delta_v38_minus_v12": loss38 - loss12,
                "alc_weight": weight,
            }
        alc12 = sum(WEIGHTS[i] * budgets[str(b)]["v12_brier"] for i, b in enumerate(BUDGETS))
        alc38 = sum(WEIGHTS[i] * budgets[str(b)]["v38_brier"] for i, b in enumerate(BUDGETS))
        group_results.append({
            "subject_id": str(group["subject_id"]),
            "benchmark_id": name,
            "distinct_source_items": group["n_items"],
            "acquisition_items": len(acq),
            "evaluation_response_rows": len(eval_rows),
            "v12_brier_alc": alc12,
            "v38_brier_alc": alc38,
            "delta_v38_minus_v12": alc38 - alc12,
            "by_budget": budgets,
        })

    per_budget = {}
    for b in BUDGETS:
        key = str(b)
        d = [g["by_budget"][key]["delta_v38_minus_v12"] for g in group_results]
        a = [g["by_budget"][key]["v12_brier"] for g in group_results]
        c = [g["by_budget"][key]["v38_brier"] for g in group_results]
        per_budget[key] = {
            "v12_macro_brier": sum(a) / len(a),
            "v38_macro_brier": sum(c) / len(c),
            "delta_v38_minus_v12": sum(d) / len(d),
            "paired_group_bootstrap_95pct_descriptive_ci": bootstrap_mean_ci(d, BOOTSTRAP_SEED + b),
            "groups_improved": sum(x < -1e-15 for x in d),
            "groups_worsened": sum(x > 1e-15 for x in d),
            "groups_tied": sum(abs(x) <= 1e-15 for x in d),
        }
    d_alc = [g["delta_v38_minus_v12"] for g in group_results]
    v12_alc = sum(g["v12_brier_alc"] for g in group_results) / len(group_results)
    v38_alc = sum(g["v38_brier_alc"] for g in group_results) / len(group_results)
    return {
        "benchmark_id": name,
        "metadata": source["benchmark_meta"],
        "table_counts": source["counts"],
        "data_file_hashes": source["files"],
        "frozen_split": {
            "seed": loader.SEED,
            "group_count": len(group_results),
            "groups": len(group_results),
            "acquisition_item_ids": len(frozen["acq_item_ids"]),
            "evaluation_item_ids": len(frozen["eval_item_ids"]),
            "content_clusters": frozen["content_cluster_count"],
            "excluded_groups": frozen["exclusions"],
            "prefix_order": "reused exactly from V27/V37 eval_data; nested per subject",
            "acquisition_response_rule": "first (trial,response_id) per acquisition item, as frozen in V37",
            "evaluation_rule": "all valid binary evaluation response rows, as frozen in V37",
            "same_acquisition_and_evaluation_for_both_estimators": True,
        },
        "scores": {
            "v12_brier_alc": v12_alc,
            "v38_brier_alc": v38_alc,
            "delta_v38_minus_v12": v38_alc - v12_alc,
            "paired_group_bootstrap_95pct_descriptive_ci": bootstrap_mean_ci(d_alc, BOOTSTRAP_SEED),
            "bootstrap_unit": "subject-benchmark group; descriptive, not item-level inference",
            "bootstrap_groups": len(group_results),
            "bootstrap_replicates": BOOTSTRAP_REPS if len(group_results) >= 2 else 0,
            "groups_improved": sum(x < -1e-15 for x in d_alc),
            "groups_worsened": sum(x > 1e-15 for x in d_alc),
            "groups_tied": sum(abs(x) <= 1e-15 for x in d_alc),
            "per_budget_macro": per_budget,
            "groups": group_results,
        },
    }


def binomial_risk(theta: float, n: int, estimator) -> float:
    """Exact iid Bernoulli expected single-outcome Brier via finite sum."""
    total = 0.0
    for k in range(n + 1):
        prob = math.comb(n, k) * theta ** k * (1.0 - theta) ** (n - k)
        q = estimator(n, k)
        # Integrate the independent future Bernoulli outcome analytically.
        total += prob * (theta * (1.0 - q) ** 2 + (1.0 - theta) * q ** 2)
    return total


def math_diagnostics() -> dict:
    thetas = (0.05, 0.25, 0.5, 0.75, 0.95)
    by_theta = {}
    for theta in thetas:
        by_budget = {}
        for b, w in zip(BUDGETS, WEIGHTS):
            r12 = binomial_risk(theta, b, v12)
            r38 = binomial_risk(theta, b, v38)
            by_budget[str(b)] = {"v12_expected_brier": r12,
                                 "v38_expected_brier": r38,
                                 "delta_v38_minus_v12": r38-r12}
        by_theta[str(theta)] = {
            "per_budget": by_budget,
            "v12_expected_brier_alc": sum(w * by_budget[str(b)]["v12_expected_brier"] for b,w in zip(BUDGETS,WEIGHTS)),
            "v38_expected_brier_alc": sum(w * by_budget[str(b)]["v38_expected_brier"] for b,w in zip(BUDGETS,WEIGHTS)),
            "delta_v38_minus_v12": sum(w * by_budget[str(b)]["delta_v38_minus_v12"] for b,w in zip(BUDGETS,WEIGHTS)),
        }
    return {
        "assumptions": "iid Bernoulli revealed labels and future evaluation outcomes with one shared theta; this is a mathematical reference, not asserted for PAIEC.",
        "budget_1_predictions": {"k=0": {"v12": v12(1,0), "v38": v38(1,0)},
                                 "k=1": {"v12": v12(1,1), "v38": v38(1,1)}},
        "theta_risks": by_theta,
        "interpretation": "Exact binomial summation of future-outcome Brier. Minimax squared-error risk for estimating theta is a different decision problem from minimizing PAIEC's realized macro Brier ALC.",
    }


def main() -> None:
    start = time.perf_counter()
    before = {"v12": sha256(V12_ZIP), "v33": sha256(V33_ZIP)}
    if before != EXPECTED:
        raise RuntimeError(f"Frozen ZIP hash mismatch before run: {before}")
    spec = importlib.util.spec_from_file_location("v37_frozen_runner", V37_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen V37 evaluator")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    studies = [score_benchmark(loader, name) for name in ("algotune", "multi_swebench")]
    after = {"v12": sha256(V12_ZIP), "v33": sha256(V33_ZIP)}
    if after != EXPECTED or after != before:
        raise RuntimeError(f"Frozen ZIP changed during run: {after}")
    result = {
        "status": "completed_exploratory_retrospective_paired_real_response_evaluation",
        "data_revision": REVISION,
        "sources": ["inventory-listed AlgoTune", "inventory-listed Multi-SWE-bench"],
        "excluded_sources_outcomes_read": [],
        "protocol": {"budgets": list(BUDGETS), "weights": list(WEIGHTS),
                     "split_and_prefix": "reuse V27 AlgoTune and V37 Multi-SWE-bench frozen split/order exactly; no re-sampling",
                     "v12": "(k+1)/(n+2)",
                     "v38": "n=0: 0.5; n>0: (k+sqrt(n)/2)/(n+sqrt(n))",
                     "valid_n": "actual valid revealed acquisition prefix for the current subject-benchmark group, not all labeled rows"},
        "math_diagnostics": math_diagnostics(),
        "studies": studies,
        "frozen_zip_hashes": {"before": before, "after": after},
        "runtime_seconds": time.perf_counter() - start,
        "limits": [
            "Retrospective analysis on previously used public, inventory-listed sources; not an independent unseen dataset evaluation.",
            "Real response labels and identical frozen acquisition/evaluation splits are used, but benchmark outcomes may be dependent and non-iid.",
            "Bootstrap resamples subject-benchmark groups only and is descriptive; AlgoTune has only six held-out subjects.",
            "The binomial minimax result concerns squared error in estimating a common Bernoulli parameter under iid assumptions; it does not guarantee lower PAIEC Brier ALC.",
            "No hidden evaluation outcomes, excluded-source outcomes, or network data were accessed; no ZIP was changed or uploaded."
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"],
                      "studies": [{"benchmark": s["benchmark_id"], "groups": s["scores"]["bootstrap_groups"],
                                   "v12": s["scores"]["v12_brier_alc"], "v38": s["scores"]["v38_brier_alc"],
                                   "delta": s["scores"]["delta_v38_minus_v12"],
                                   "group_wins": s["scores"]["groups_improved"],
                                   "group_losses": s["scores"]["groups_worsened"]} for s in studies]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
