"""Run frozen paired V12/V37 studies on AlgoTune and Multi-SWE-bench.

Requires the project's PyArrow environment. All split/order rules are frozen in
protocol_frozen.md. No model fitting or external network access is performed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REVISION = "54074453da7b9911be0f48e4c560be547bc84d21"
SEED = 20261015
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
ALGO_DIR = ROOT / "experiments/v27_official_data/raw/algotune/formatted_tables"
MULTI_DIR = HERE / "raw/multi_swebench/formatted_tables"
V12_ZIP = ROOT / "release_staging/v12_beta11_candidate/v12_beta11_candidate.zip"
V33_ZIP = ROOT / "release_staging/v33_conservative_transfer_candidate/v33_conservative_transfer_candidate.zip"
V12_SHA = "de12d731b6f1c4b9f6c83226998e17fc930ad55e8c16948c753c6ceaa754fe02"
V33_SHA = "21e8b461ec483f6a13365e001cadb7f94557ad7a325b5343c941495279477b41"

spec = importlib.util.spec_from_file_location("v37_model", HERE / "model.py")
model = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(model)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def norm_content(value: Any) -> str:
    return " ".join(value.casefold().split()) if isinstance(value, str) else ""


def text_clusters(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in items:
        text = norm_content(row.get("content"))
        key = text if text else "\x00empty-content:" + row["item_id"]
        clusters[key].append(row["item_id"])
    return dict(clusters)


def split_ids(items: list[dict[str, Any]], split_style: str) -> tuple[set[str], set[str], int]:
    clusters = text_clusters(items)
    units = sorted(clusters, key=lambda text: rank("item", SEED, text))
    target = len(items) // 2
    acq_text: set[str] = set()
    count = 0
    for text in units:
        if count < target:
            acq_text.add(text)
            count += len(clusters[text])
    acq = {item_id for text in acq_text for item_id in clusters[text]}
    all_ids = {row["item_id"] for row in items}
    ev = all_ids - acq
    if acq & ev or len(acq) < 31 or len(ev) < 1:
        raise RuntimeError(f"split cannot support six budgets: acq={len(acq)} eval={len(ev)}")
    return acq, ev, len(clusters)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Expected text field in official tables, got {type(value).__name__}")
    return value


def load_source(name: str) -> dict[str, Any]:
    table_dir = ALGO_DIR if name == "algotune" else MULTI_DIR
    item_file = table_dir / "items.parquet"
    response_file = table_dir / "responses.parquet"
    benchmark_file = table_dir / "benchmarks.parquet"
    subjects_file = (ROOT / "experiments/v27_official_data/raw/algotune/formatted_tables/subjects.parquet"
                     if name == "algotune" else MULTI_DIR / "subjects.parquet")
    if name == "algotune":
        item_table = pq.read_table(item_file, columns=["item_id", "benchmark_id", "content", "item_features"])
    else:
        item_table = pq.read_table(item_file, columns=["item_id", "benchmark_id", "content", "item_features"])
    response_table = pq.read_table(response_file, columns=["subject_id", "item_id", "benchmark_id", "trial", "response_id", "response", "interactors"])
    benchmark_table = pq.read_table(benchmark_file, columns=["benchmark_id", "release", "license", "response_type", "n_items", "n_subjects", "n_responses", "max_trial"])
    subject_table = pq.read_table(subjects_file, columns=["subject_id", "normalized_name", "provider", "release_date", "access_date", "harness", "reasoning_effort", "harness_version", "subject_features_extra"])
    items = item_table.to_pylist()
    responses = response_table.to_pylist()
    benchmark_rows = benchmark_table.to_pylist()
    subjects = subject_table.to_pylist()
    if len(benchmark_rows) != 1:
        raise ValueError(f"{name}: expected one benchmark metadata row")
    bm = benchmark_rows[0]
    if bm["release"] != "public" or bm["response_type"] != "binary":
        raise ValueError(f"{name}: metadata is not public/binary: {bm}")
    id_to_item: dict[str, dict[str, Any]] = {}
    for row in items:
        if row["benchmark_id"] != name or row["item_id"] in id_to_item:
            raise ValueError(f"{name}: duplicate item ID or benchmark mismatch")
        id_to_item[row["item_id"]] = {
            "item_id": row["item_id"], "benchmark_id": name,
            "content": _as_text(row.get("content")),
            "item_features": _as_text(row.get("item_features")),
        }
    subject_fields = ("normalized_name", "provider", "release_date", "access_date", "harness",
                      "reasoning_effort", "harness_version", "subject_features_extra")
    subject_map = {}
    for row in subjects:
        subject_map[row["subject_id"]] = {field: _as_text(row.get(field)) for field in subject_fields}
    if len(subject_map) != len(subjects):
        raise ValueError(f"{name}: duplicate subject ID")
    valid_rows = []
    nonbinary = 0
    unknown_item = 0
    for row in responses:
        if row["benchmark_id"] != name or row["item_id"] not in id_to_item or row["subject_id"] not in subject_map:
            unknown_item += 1
            continue
        y = row["response"]
        if isinstance(y, bool) or type(y) not in (int, float) or y not in (0, 1, 0.0, 1.0):
            nonbinary += 1
            continue
        interactor = _as_text(row.get("interactors"))
        valid_rows.append({
            "subject_id": row["subject_id"], "item_id": row["item_id"],
            "benchmark_id": name, "trial": int(row["trial"]),
            "response_id": _as_text(row.get("response_id")), "y": int(y),
            "interactors": interactor,
        })
    return {
        "name": name, "items": list(id_to_item.values()), "item_map": id_to_item,
        "responses": valid_rows, "subject_map": subject_map, "benchmark_meta": bm,
        "counts": {"item_rows": len(items), "response_rows": len(responses),
                   "valid_binary_response_rows": len(valid_rows), "nonbinary_response_rows": nonbinary,
                   "unmatched_response_rows": unknown_item, "subject_rows": len(subjects)},
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha_file(p)}
                  for p in (item_file, response_file, benchmark_file, subjects_file)},
    }


def runtime_item(source: dict[str, Any], item_id: str, interactors: str) -> dict[str, str]:
    row = source["item_map"][item_id]
    return {"item_content": row["content"], "item_features": row["item_features"],
            "interactors": interactors, "benchmark_id": source["name"]}


def eval_data(source: dict[str, Any]) -> dict[str, Any]:
    acq_ids, eval_ids, cluster_count = split_ids(source["items"], source["name"])
    by_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"acq": [], "eval": []})
    for row in source["responses"]:
        side = "acq" if row["item_id"] in acq_ids else "eval"
        by_group[row["subject_id"]][side].append(row)
    groups = []
    exclusions = Counter()
    allowed_subjects = None
    if source["name"] == "algotune":
        # Reuse the exact six subject-held-out identities frozen in V27-B.
        prior_path = ROOT / "experiments/v27_official_data/real_trial_results.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        allowed_subjects = set(prior["protocol"]["heldout_subject_ids"])
    for sid, parts in by_group.items():
        if allowed_subjects is not None and sid not in allowed_subjects:
            exclusions["not_in_frozen_v27_heldout_subjects"] += 1
            continue
        # One revealed label per distinct acquisition item, deterministically the
        # first recorded trial. Evaluation keeps every valid response row.
        by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in parts["acq"]:
            by_item[row["item_id"]].append(row)
        acquisition = []
        for iid, rows in by_item.items():
            rows.sort(key=lambda x: (x["trial"], x["response_id"]))
            chosen = rows[0]
            acquisition.append(chosen)
        acquisition.sort(key=lambda row: rank("prefix", SEED, sid, row["item_id"]))
        evaluation = parts["eval"]
        total_distinct = len({r["item_id"] for r in parts["acq"] + parts["eval"]})
        if total_distinct < 80:
            exclusions["below_80_distinct_items"] += 1
            continue
        if len(acquisition) < 31:
            exclusions["fewer_than_31_acquisition_items"] += 1
            continue
        if not evaluation:
            exclusions["no_evaluation_rows"] += 1
            continue
        groups.append({"subject_id": sid, "subject": source["subject_map"][sid],
                       "acquisition": acquisition, "evaluation": evaluation,
                       "n_items": total_distinct, "acq_ids": acq_ids, "eval_ids": eval_ids})
    if not groups:
        raise RuntimeError(f"{source['name']}: no groups satisfy complete six-budget protocol")
    return {"groups": groups, "acq_item_ids": acq_ids, "eval_item_ids": eval_ids,
            "content_cluster_count": cluster_count, "exclusions": dict(exclusions)}


def score_source(source: dict[str, Any]) -> dict[str, Any]:
    split = eval_data(source)
    pair_results = []
    per_budget = {str(b): {"v12": [], "v37": [], "delta": [], "trigger_rate": [], "categorized_rate": []}
                  for b in BUDGETS}
    trigger_all = categorized_all = eval_unique_all = 0
    for group in split["groups"]:
        subj = group["subject"]
        acq = group["acquisition"]
        eval_rows = group["evaluation"]
        by_budget = {}
        for budget, weight in zip(BUDGETS, WEIGHTS):
            prefix = acq[:budget]
            labeled = [[[subj, runtime_item(source, r["item_id"], r["interactors"])], r["y"]] for r in prefix]
            # Repeated outcomes with identical observable evaluation input reuse
            # one prediction, matching the platform's prediction reuse behavior.
            input_predictions: dict[tuple[str, str], tuple[float, float, bool, bool]] = {}
            category_prediction_cache: dict[tuple[str, ...], float] = {}
            p12 = (sum(r["y"] for r in prefix) + 1)/(len(prefix)+2)
            for r in eval_rows:
                key = (r["item_id"], r["interactors"])
                if key in input_predictions:
                    continue
                target_item = runtime_item(source, r["item_id"], r["interactors"])
                cats = model.task_categories(target_item["item_content"])
                category_key = tuple(sorted(cats))
                # Given the frozen content-cluster split, no evaluation content
                # cluster can occur in the revealed prefix. Thus the predictor's
                # only target-dependent statistic is this fixed category set;
                # memoization reuses mathematically identical outputs.
                if category_key not in category_prediction_cache:
                    category_prediction_cache[category_key] = model.predict([subj, target_item], labeled)
                p37 = category_prediction_cache[category_key]
                input_predictions[key] = (p12, p37, p12 != p37, bool(cats))
            loss12 = loss37 = 0.0
            for r in eval_rows:
                p12, p37, _, _ = input_predictions[(r["item_id"], r["interactors"])]
                loss12 += (p12-r["y"])**2
                loss37 += (p37-r["y"])**2
            loss12 /= len(eval_rows); loss37 /= len(eval_rows)
            changed = sum(1 for row in input_predictions.values() if row[2])
            categorized = sum(1 for row in input_predictions.values() if row[3])
            unique_count = len(input_predictions)
            rate = changed/unique_count if unique_count else 0.0
            cat_rate = categorized/unique_count if unique_count else 0.0
            by_budget[str(budget)] = {
                "n_revealed": len(prefix), "k_positive": sum(r["y"] for r in prefix),
                "evaluation_response_rows": len(eval_rows), "evaluation_unique_inputs": unique_count,
                "v12_brier": loss12, "v37_brier": loss37,
                "delta_v37_minus_v12": loss37-loss12,
                "changed_prediction_inputs": changed, "trigger_rate": rate,
                "categorized_input_count": categorized, "categorized_input_rate": cat_rate,
            }
            per_budget[str(budget)]["v12"].append(loss12)
            per_budget[str(budget)]["v37"].append(loss37)
            per_budget[str(budget)]["delta"].append(loss37-loss12)
            per_budget[str(budget)]["trigger_rate"].append(rate)
            per_budget[str(budget)]["categorized_rate"].append(cat_rate)
            if budget == 31:
                trigger_all += changed
                categorized_all += categorized
                eval_unique_all += unique_count
        alc12 = sum(w*by_budget[str(b)]["v12_brier"] for b,w in zip(BUDGETS,WEIGHTS))
        alc37 = sum(w*by_budget[str(b)]["v37_brier"] for b,w in zip(BUDGETS,WEIGHTS))
        pair_results.append({"subject_id": group["subject_id"], "benchmark_id": source["name"],
                             "distinct_source_items": group["n_items"],
                             "acquisition_distinct_items": len(acq),
                             "evaluation_response_rows": len(eval_rows),
                             "evaluation_unique_inputs": len({(r["item_id"], r["interactors"]) for r in eval_rows}),
                             "v12_brier_alc": alc12, "v37_brier_alc": alc37,
                             "delta_v37_minus_v12": alc37-alc12,
                             "by_budget": by_budget})
    # per_budget values are lists; macro-average across eligible groups.
    macro_budget = {b: {key: sum(metrics[key])/len(metrics[key]) for key in metrics}
                    for b, metrics in per_budget.items()}
    trigger_counts = {}
    for b in map(str, BUDGETS):
        changed = sum(g["by_budget"][b]["changed_prediction_inputs"] for g in pair_results)
        categorized = sum(g["by_budget"][b]["categorized_input_count"] for g in pair_results)
        total_inputs = sum(g["by_budget"][b]["evaluation_unique_inputs"] for g in pair_results)
        trigger_counts[b] = {"changed_unique_inputs": changed,
                             "categorized_unique_inputs": categorized,
                             "total_unique_inputs": total_inputs,
                             "overall_trigger_rate": changed/total_inputs if total_inputs else 0.0,
                             "overall_category_coverage": categorized/total_inputs if total_inputs else 0.0}
    v12_macro = sum(r["v12_brier_alc"] for r in pair_results)/len(pair_results)
    v37_macro = sum(r["v37_brier_alc"] for r in pair_results)/len(pair_results)
    deltas = [r["delta_v37_minus_v12"] for r in pair_results]
    rng = random.Random(3701)
    bootstrap = []
    if len(deltas) >= 2:
        for _ in range(10000):
            bootstrap.append(sum(rng.choice(deltas) for _ in deltas)/len(deltas))
    def quantile(values, q):
        values = sorted(values)
        if not values: return None
        x=q*(len(values)-1); lo=int(x); hi=min(len(values)-1,lo+1)
        return values[lo]*(hi-x)+values[hi]*(x-lo)
    tags_all = Counter()
    item_rows = source["items"]
    for it in item_rows:
        for tag in model.task_categories(it["content"]): tags_all[tag] += 1
    tagged_items = sum(1 for it in item_rows if model.task_categories(it["content"]))
    return {
        "benchmark_id": source["name"], "metadata": source["benchmark_meta"], "table_counts": source["counts"],
        "files": source["files"],
        "split": {"seed": SEED, "rule": "global exact normalized content clusters; stable SHA256 half split",
                  "content_clusters": split["content_cluster_count"],
                  "all_item_rows": len(source["items"]), "acquisition_item_ids": len(split["acq_item_ids"]),
                  "evaluation_item_ids": len(split["eval_item_ids"]),
                  "acquisition_evaluation_overlap": len(split["acq_item_ids"] & split["eval_item_ids"]),
                  "eligible_group_count": len(pair_results), "excluded_groups": split["exclusions"],
                  "algotune_subject_partition": "V27-B frozen six held-out subjects only; non-heldout subjects excluded",
                  "budgets": list(BUDGETS), "weights": list(WEIGHTS),
                  "prefix_order": "SHA256(prefix, seed, subject_id, item_id), nested per subject",
                  "response_rule": "acquisition: first trial,response_id per item; evaluation: all exact-binary rows retained"},
        "feature_coverage": {"items_with_any_category": tagged_items, "all_items": len(item_rows),
                             "item_coverage_rate": tagged_items/len(item_rows), "per_category_item_count": dict(tags_all)},
        "scores": {"v12_brier_alc": v12_macro, "v37_brier_alc": v37_macro,
                   "delta_v37_minus_v12": v37_macro-v12_macro,
                   "paired_group_bootstrap_95pct_descriptive": [quantile(bootstrap,.025),quantile(bootstrap,.975)] if bootstrap else None,
                   "bootstrap_groups": len(pair_results), "bootstrap_replicates": len(bootstrap),
                   "per_budget_macro": macro_budget,
                   "groups": pair_results},
        "trigger_summary": {"budget_31_changed_unique_inputs": trigger_all,
                            "budget_31_categorized_unique_inputs": categorized_all,
                            "budget_31_total_unique_eval_inputs": eval_unique_all,
                            "budget_31_trigger_rate": trigger_all/eval_unique_all if eval_unique_all else 0,
                            "budget_31_category_coverage": categorized_all/eval_unique_all if eval_unique_all else 0,
                            "trigger_rate_by_budget_group_macro": {b: macro_budget[b]["trigger_rate"] for b in macro_budget},
                            "trigger_counts_by_budget": trigger_counts},
        "category_accuracy_diagnostic": category_diagnostic(source, split, pair_results),
    }


def category_diagnostic(source: dict[str, Any], split: dict[str, Any], pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive category/outcome association; no model parameters are fit."""
    evaluation_ids = split["eval_item_ids"]
    by_group_item = defaultdict(list)
    for row in source["responses"]:
        if row["item_id"] in evaluation_ids:
            by_group_item[(row["subject_id"], row["item_id"])].append(row["y"])
    stats = defaultdict(lambda: {"items": 0, "response_rows": 0, "correct": 0, "subject_rates": []})
    subject_category = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (sid, iid), ys in by_group_item.items():
        cats = model.task_categories(source["item_map"][iid]["content"])
        for cat in cats:
            d = stats[cat]; d["items"] += 1; d["response_rows"] += len(ys); d["correct"] += sum(ys)
            subject_category[cat][sid][0] += sum(ys); subject_category[cat][sid][1] += len(ys)
    for cat, subjects in subject_category.items():
        stats[cat]["subject_positive_rate_range"] = [
            min(k/n for k,n in subjects.values() if n), max(k/n for k,n in subjects.values() if n)
        ] if subjects else None
        stats[cat]["subject_count"] = len(subjects)
        stats[cat]["mean_subject_positive_rate"] = sum(k/n for k,n in subjects.values() if n)/len(subjects) if subjects else None
    for d in stats.values():
        d["positive_rate"] = d["correct"]/d["response_rows"] if d["response_rows"] else None
    return {"scope": "evaluation outcome descriptive only; not used for feature/rule selection",
            "category_counts_and_rates": dict(stats),
            "misclassification_rate": None,
            "misclassification_note": "No independently annotated gold task taxonomy exists; false-positive/false-negative category error cannot be estimated from outcomes without circularly defining categories."}


def main():
    for path, expected in ((V12_ZIP,V12_SHA),(V33_ZIP,V33_SHA)):
        got=sha_file(path)
        if got != expected:
            raise RuntimeError(f"Frozen candidate changed: {path} {got}")
    sources = [load_source("algotune"), load_source("multi_swebench")]
    result = {
        "status": "completed_exploratory_retrospective_paired_real_response_evaluation",
        "data_revision": REVISION,
        "source_boundary": "inventory-listed AlgoTune and Multi-SWE-bench only; no excluded collection outcomes read",
        "category_rule_protocol": str((HERE / "protocol_frozen.md").relative_to(ROOT)),
        "v12_v33_frozen_zip_hashes_verified": {"v12": sha_file(V12_ZIP), "v33": sha_file(V33_ZIP)},
        "studies": [score_source(s) for s in sources],
        "interpretation_limits": [
            "Both sources/results have been used previously in project research; this is retrospective exploratory comparison, not independent confirmation.",
            "Task categories are fixed keyword proxies, not a gold-labeled taxonomy; category error rate is not identifiable without independent annotation.",
            "AlgoTune labels are the published final_speedup threshold; Multi-SWE labels represent issue resolution. Label semantics differ.",
            "No hidden PAIEC evaluation outcomes are accessed. Results are not Codabench scores and do not establish transfer to private benchmarks.",
            "If categories overlap, a revealed item is counted once in local support. Local support is deduplicated by exact observable item payload and conflicting repeated outcomes are excluded.",
        ],
    }
    out = HERE / "real_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "results": [
        {"benchmark": s["benchmark_id"], "items": s["table_counts"]["item_rows"],
         "responses": s["table_counts"]["valid_binary_response_rows"],
         "groups": s["scores"]["bootstrap_groups"], "v12": s["scores"]["v12_brier_alc"],
         "v37": s["scores"]["v37_brier_alc"], "delta": s["scores"]["delta_v37_minus_v12"],
         "trigger31": s["trigger_summary"]["budget_31_trigger_rate"]}
        for s in result["studies"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
