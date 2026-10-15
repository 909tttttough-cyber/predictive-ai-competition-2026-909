"""Frozen, item-disjoint paired V12 vs train-only Beta-binomial prior trial.

Reads only the pinned AlgoTune Measurement DB formatted tables. It keeps all
response rows for scoring and uses one acquisition label per distinct item
(max_trial=1 in this collection). No network access is performed here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
import sys
import importlib.util

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "raw" / "algotune" / "formatted_tables"
PAIR_EVAL_PATH = ROOT.parents[1] / "experiments" / "solve_assess_validation" / "pair_eval.py"
_spec = importlib.util.spec_from_file_location("v27_pair_eval_core", PAIR_EVAL_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load existing paired evaluator: {PAIR_EVAL_PATH}")
_pair_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pair_eval)
ALC_WEIGHTS, BUDGETS = _pair_eval.ALC_WEIGHTS, _pair_eval.BUDGETS
v12_probability, pair_equal_mean = _pair_eval.v12_probability, _pair_eval.pair_equal_mean
SEED = 20261015
TRAIN_SUBJECTS = 12
BOUNDS = (0.05, 100.0)
MAX_ITER = 4000
TOL = 1e-8
BOOTSTRAP_REPS = 10000


def rank(*parts: str) -> str:
    return hashlib.sha256(("\x1f".join(map(str, parts))).encode("utf-8")).hexdigest()


def normalize_content(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("item content must be nonempty text")
    return " ".join(value.casefold().split())


def load_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = pq.read_table(DATA / "items.parquet", columns=["item_id", "benchmark_id", "content"]).to_pylist()
    responses = pq.read_table(DATA / "responses.parquet", columns=["subject_id", "item_id", "benchmark_id", "trial", "response"]).to_pylist()
    benchmarks = pq.read_table(DATA / "benchmarks.parquet", columns=["benchmark_id", "release", "response_type", "license", "n_items", "n_subjects", "n_responses", "max_trial"]).to_pylist()
    return items, responses, benchmarks


def _simplex_minimize(fn, x0: tuple[float, float]) -> tuple[tuple[float, float], float, int, bool]:
    """Bounded deterministic 2-D Nelder-Mead; no held-out data involved."""
    lo, hi = math.log(BOUNDS[0]), math.log(BOUNDS[1])
    clamp = lambda x: tuple(min(hi, max(lo, z)) for z in x)
    simplex = [clamp(x0), clamp((x0[0] + 0.1, x0[1])), clamp((x0[0], x0[1] + 0.1))]
    vals = [fn(x) for x in simplex]
    for it in range(1, MAX_ITER + 1):
        order = sorted(range(3), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]; vals = [vals[i] for i in order]
        spread = max(abs(vals[i] - vals[0]) for i in (1, 2))
        diameter = max(max(abs(simplex[i][j] - simplex[0][j]) for j in (0, 1)) for i in (1, 2))
        if spread < TOL and diameter < TOL:
            return simplex[0], vals[0], it, True
        best, second, worst = simplex
        centroid = ((best[0] + second[0]) / 2, (best[1] + second[1]) / 2)
        reflected = clamp((centroid[0] + (centroid[0] - worst[0]), centroid[1] + (centroid[1] - worst[1])))
        fr = fn(reflected)
        if fr < vals[0]:
            expanded = clamp((centroid[0] + 2 * (reflected[0] - centroid[0]), centroid[1] + 2 * (reflected[1] - centroid[1])))
            fe = fn(expanded)
            simplex[2], vals[2] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < vals[1]:
            simplex[2], vals[2] = reflected, fr
        else:
            outside = fr < vals[2]
            point = clamp((centroid[0] + 0.5 * ((reflected if outside else worst)[0] - centroid[0]), centroid[1] + 0.5 * ((reflected if outside else worst)[1] - centroid[1])))
            fp = fn(point)
            if fp < (fr if outside else vals[2]):
                simplex[2], vals[2] = point, fp
            else:
                simplex[1] = clamp((best[0] + 0.5 * (second[0] - best[0]), best[1] + 0.5 * (second[1] - best[1])))
                simplex[2] = clamp((best[0] + 0.5 * (worst[0] - best[0]), best[1] + 0.5 * (worst[1] - best[1])))
                vals[1], vals[2] = fn(simplex[1]), fn(simplex[2])
    i = min(range(3), key=lambda j: vals[j])
    return simplex[i], vals[i], MAX_ITER, False


def fit_beta_binomial(groups: list[tuple[int, int]]) -> dict[str, Any]:
    if len(groups) < 3 or any(n <= 0 or k < 0 or k > n for k, n in groups):
        raise ValueError("need >=3 nonempty train groups with valid binary counts")
    def nll(log_ab: tuple[float, float]) -> float:
        a, b = math.exp(log_ab[0]), math.exp(log_ab[1])
        total = 0.0
        for k, n in groups:
            total += (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                      + math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b)
                      + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
        return -total
    point, obj, iterations, converged = _simplex_minimize(nll, (0.0, 0.0))
    alpha, beta = math.exp(point[0]), math.exp(point[1])
    return {"alpha": alpha, "beta": beta, "mean": alpha/(alpha+beta), "concentration": alpha+beta,
            "train_groups": len(groups), "train_counts": [{"k": k, "n": n} for k, n in groups],
            "fit": "bounded_beta_binomial_marginal_MLE_Nelder_Mead", "bounds_each": list(BOUNDS),
            "initial_alpha_beta": [1.0, 1.0], "negative_log_likelihood": obj,
            "iterations": iterations, "converged": converged}


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if len(s) == 1: return s[0]
    x = q * (len(s)-1); lo = int(x); hi = min(len(s)-1, lo+1)
    return s[lo] * (hi-x) + s[hi] * (x-lo)


def run() -> dict[str, Any]:
    items, responses, bench_rows = load_tables()
    if len(bench_rows) != 1 or bench_rows[0]["benchmark_id"] != "algotune":
        raise ValueError("expected only the pinned public AlgoTune benchmark")
    bm = bench_rows[0]
    if bm["release"] != "public" or bm["response_type"] != "binary" or bm["license"] != "MIT":
        raise ValueError("benchmark release/type/license checks failed")
    by_item = {r["item_id"]: r for r in items}
    if len(by_item) != len(items): raise ValueError("duplicate item rows")
    content = {iid: normalize_content(row["content"]) for iid, row in by_item.items()}
    clusters: dict[str, list[str]] = {}
    for iid, text in content.items(): clusters.setdefault(text, []).append(iid)
    # Split by exact normalized content cluster so repeated prompts cannot cross sides.
    units = sorted(clusters, key=lambda text: rank("item", SEED, text))
    target = len(items) // 2
    acq_text: set[str] = set(); count = 0
    for text in units:
        if count < target:
            acq_text.add(text); count += len(clusters[text])
    acq_ids = {iid for text in acq_text for iid in clusters[text]}
    eval_ids = set(by_item) - acq_ids
    if acq_ids & eval_ids or len(acq_ids) < 31 or len(eval_ids) < 31:
        raise ValueError("invalid item-disjoint split / insufficient budgets")
    subjects = sorted({r["subject_id"] for r in responses})
    if len(subjects) != 18: raise ValueError("unexpected subject count")
    subjects = sorted(subjects, key=lambda s: rank("subject", SEED, s))
    train_subjects, test_subjects = subjects[:TRAIN_SUBJECTS], subjects[TRAIN_SUBJECTS:]
    groups: dict[tuple[str, str], dict[str, list[Any]]] = {}
    seen_response_keys = set()
    for r in responses:
        if r["benchmark_id"] != "algotune" or r["item_id"] not in by_item:
            raise ValueError("response-to-item benchmark/key mismatch")
        response_key=(r["subject_id"],r["benchmark_id"],r["item_id"],r["trial"])
        if response_key in seen_response_keys: raise ValueError("duplicate subject-benchmark-item-trial response")
        seen_response_keys.add(response_key)
        y = r["response"]
        if type(y) not in (int, float) or y not in (0, 1): raise ValueError("non-binary outcome encountered")
        group = groups.setdefault((r["subject_id"], r["benchmark_id"]), {"acq": [], "eval": []})
        side = "acq" if r["item_id"] in acq_ids else "eval"
        group[side].append((r["item_id"], int(y)))
    if len(groups) != 18: raise ValueError("expected 18 subject-benchmark pairs")
    train_counts = []
    for subject in train_subjects:
        rows = groups[(subject, "algotune")]["acq"]
        if len({i for i,_ in rows}) < 31: raise ValueError("train pair has too few acquisition items")
        train_counts.append((sum(y for _,y in rows), len(rows)))
    prior = fit_beta_binomial(train_counts)
    train_successes=sum(k for k,_ in train_counts); train_trials=sum(n for _,n in train_counts)
    pair_rows=[]
    heldout_eval_successes=0; heldout_eval_trials=0
    for subject in test_subjects:
        g=groups[(subject,"algotune")]
        acq=sorted(g["acq"],key=lambda z:rank("prefix",SEED,subject,z[0]))
        ev=g["eval"]
        heldout_eval_successes += sum(y for _,y in ev); heldout_eval_trials += len(ev)
        if len({i for i,_ in acq})<31 or len({i for i,_ in ev})<31: raise ValueError("heldout pair too small")
        by_budget={}
        for budget,weight in zip(BUDGETS,ALC_WEIGHTS):
            prefix=[y for _,y in acq[:budget]]
            n,k=len(prefix),sum(prefix)
            p12=v12_probability(n,k)
            pc=(k+prior["alpha"])/(n+prior["alpha"]+prior["beta"])
            loss12=sum((p12-y)**2 for _,y in ev)/len(ev)
            lossc=sum((pc-y)**2 for _,y in ev)/len(ev)
            by_budget[str(budget)]={"v12_brier":loss12,"candidate_brier":lossc,"delta_candidate_minus_v12":lossc-loss12,"n_revealed":n,"k_correct":k,"evaluation_response_count":len(ev)}
        v12_alc=sum(w*by_budget[str(b)]["v12_brier"] for b,w in zip(BUDGETS,ALC_WEIGHTS))
        can_alc=sum(w*by_budget[str(b)]["candidate_brier"] for b,w in zip(BUDGETS,ALC_WEIGHTS))
        pair_rows.append({"subject_id":subject,"benchmark_id":"algotune","acquisition_unique_items":len(g["acq"]),"evaluation_response_rows":len(ev),"v12_brier_alc":v12_alc,"candidate_brier_alc":can_alc,"delta_candidate_minus_v12":can_alc-v12_alc,"by_budget":by_budget})
    deltas=[r["delta_candidate_minus_v12"] for r in pair_rows]
    v12_macro=pair_equal_mean([r["v12_brier_alc"] for r in pair_rows])
    cand_macro=pair_equal_mean([r["candidate_brier_alc"] for r in pair_rows])
    import random
    rng=random.Random(1701); boots=[]
    for _ in range(BOOTSTRAP_REPS):
        boots.append(sum(rng.choice(deltas) for _ in deltas)/len(deltas))
    budget_macro={}
    for b in BUDGETS:
        budget_macro[str(b)]={key:pair_equal_mean([r["by_budget"][str(b)][key] for r in pair_rows])
                              for key in ("v12_brier","candidate_brier","delta_candidate_minus_v12")}
    provenance_files=[]
    for f in [ROOT/"raw"/"algotune"/"metadata.yaml", ROOT/"raw"/"algotune"/"characterization.yaml", ROOT/"raw"/"algotune"/"curation_record.md", *sorted(DATA.glob("*.parquet"))]:
        raw=f.read_bytes(); provenance_files.append({"path":str(f.relative_to(ROOT)),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
    return {"status":"real_public_training_channel_offline_experiment","data":{"repo":"https://huggingface.co/datasets/aims-foundations/measurement-db","revision":"54074453da7b9911be0f48e4c560be547bc84d21","collection":"algotune","benchmark_id":"algotune","release":"public","license":"MIT per pinned metadata","response_type":"binary","pool_caveat":"Based on current official public training release channel; item-level PAIEC pool assignment not separately confirmed.","table_rows":{"items":len(items),"responses":len(responses),"subjects":len(subjects)},"unique_items":len(by_item),"distinct_content_clusters":len(clusters),"acquisition_items":len(acq_ids),"evaluation_items":len(eval_ids),"response_rows_used_for_training_prior":train_trials,"training_prior_empirical_rate":train_successes/train_trials,"heldout_evaluation_outcome_rate":heldout_eval_successes/heldout_eval_trials,"heldout_evaluation_response_rows":heldout_eval_trials,"source_row_counts_per_pair":{"min":min(len(groups[k]["acq"])+len(groups[k]["eval"]) for k in groups),"max":max(len(groups[k]["acq"])+len(groups[k]["eval"]) for k in groups)},"downloaded_file_sha256":provenance_files},"protocol":{"split_seed":SEED,"item_split":"global deterministic exact-normalized-content cluster split, outcome-blind, near 50/50","train_subject_count":len(train_subjects),"heldout_subject_count":len(test_subjects),"train_subject_ids":train_subjects,"heldout_subject_ids":test_subjects,"budget_order":"nested deterministic SHA-256 order over acquisition items","budgets":list(BUDGETS),"weights":list(ALC_WEIGHTS),"response_unit":"all evaluation response rows retained; one row per task/model according metadata"},"candidate_fit":prior,"heldout":{"pair_count":len(pair_rows),"v12_pair_macro_brier_alc":v12_macro,"candidate_pair_macro_brier_alc":cand_macro,"delta_candidate_minus_v12":cand_macro-v12_macro,"paired_group_bootstrap_95pct_descriptive_ci":[pct(boots,.025),pct(boots,.975)],"bootstrap_replicates":BOOTSTRAP_REPS,"per_budget_pair_macro":budget_macro,"per_group":pair_rows,"interpretation":"negative candidate-minus-V12 favors candidate; bootstrap has only six heldout groups and is descriptive"},"limitations":["One official benchmark only; this is not benchmark-held-out validation.","Current data are public-release research data; per-benchmark competition pool assignment is not separately mapped in the repository.","AlgoTune binary labels encode reported speedup >=1.0 / explicit N/A threshold, not a fresh PAIEC evaluation or necessarily semantic correctness.","Only six heldout anonymous subjects; the group bootstrap is exploratory.","Does not establish transfer to PAIEC private benchmarks or Codabench hidden scoring."]}


if __name__ == "__main__":
    result=run()
    out=ROOT/"real_trial_results.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"pairs":result["heldout"]["pair_count"],"v12_alc":result["heldout"]["v12_pair_macro_brier_alc"],"candidate_alc":result["heldout"]["candidate_pair_macro_brier_alc"],"delta":result["heldout"]["delta_candidate_minus_v12"]},ensure_ascii=False))
