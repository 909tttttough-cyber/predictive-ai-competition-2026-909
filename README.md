# PAIEC 2026: reproducible predictors and offline studies

This repository contains the team's selected **V33 Low-peer** predictor, the earlier V12 and V33 predictors, and code and aggregate outputs for the documented AlgoTune and Multi-SWE-bench studies. The selected model is `models/v33_lowpeer/model.py`. A user-provided Codabench page screenshot records submission **#1002801** as Finished with formative Brier ALC **0.164578** and upload filename `v33_lowpeer_formative_candidate.zip`. The screenshot does not establish that the uploaded ZIP is byte-identical to the local ZIP used for this source release.

## Task and metric

PAIEC asks a predictor to estimate the probability that a configured AI subject answers an item correctly, using the current prediction input and revealed binary labels. For each subject–benchmark pair, Brier score is averaged over evaluation responses at budgets 0, 1, 3, 7, 15, and 31. Brier ALC is the weighted sum with weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1; the competition aggregate gives subject–benchmark groups equal weight.

## Predictors

- **V12** (`models/v12/model.py`): exact target subject and benchmark filtering; `p=(k+1)/(n+2)` with a Beta(1,1) prior. Stateless; Python standard library only.
- **Historical V33** (`models/v33/model.py`): V12 fallback plus a conservative residual correction from at least three distinct peer configurations with exact same-benchmark observable payload matches. This older candidate is retained unchanged.
- **Selected V33 Low-peer** (`models/v33_lowpeer/model.py`): the same Beta(1,1) subject–benchmark fallback, strict binary filtering and target-payload exclusion, with a correction from one or more eligible peers after at least seven valid target labels. Exact matching uses `benchmark_id`, `item_content`, `item_features`, and `interactors`. Peer outcomes are centered against each peer's other distinct, unambiguous same-benchmark payloads. All eligible residuals must have the same nonzero sign. Their mean receives coefficient `min(1/3, J/(J+6))`: **1/7 for one peer, 1/4 for two, and 1/3 for three or more**. Missing or ambiguous evidence returns the subject baseline. Payload equality is not an independent platform item ID, and the official correction trigger rate is unknown. See [the Low-peer notes](models/v33_lowpeer/README.md).

The team's final #1002801 report describes these Low-peer gates and coefficients. Earlier archived reports describing a three-peer minimum refer to the historical V33 candidate.

The V12 and historical three-peer V33 archives remain under `release_staging/`. The selected Low-peer source was extracted without modification from a separately verified local ZIP; that ZIP is not redistributed here. Run candidates separately; do not combine their files.

## Starting Kit checker and smoke test

The official Starting Kit is third-party material and is deliberately not redistributed here. Obtain the current kit from the official PAIEC/Codabench source and follow its license. Set `$kit` to the extracted kit directory, then from this repository root run:

```powershell
python "$kit/tools/check_submission_zip.py" "release_staging/v12_beta11_candidate/v12_beta11_candidate.zip"
python "$kit/tools/run_smoke_test.py" "models/v12"
python "$kit/tools/check_submission_zip.py" "release_staging/v33_conservative_transfer_candidate/v33_conservative_transfer_candidate.zip"
python "$kit/tools/run_smoke_test.py" "models/v33"
python -B "models/v33_lowpeer/test_model.py"
python "$kit/tools/run_smoke_test.py" "models/v33_lowpeer"
```

To construct a fresh ZIP from the selected source and its repository notes, run from the repository root:

```powershell
python -c "import zipfile; z=zipfile.ZipFile('v33_lowpeer_rebuilt.zip','w',zipfile.ZIP_DEFLATED); z.write('models/v33_lowpeer/model.py','model.py'); z.write('models/v33_lowpeer/README.md','README.md'); z.close()"
python "$kit/tools/check_submission_zip.py" "v33_lowpeer_rebuilt.zip"
```

The rebuilt archive contains the same model source but has different packaging metadata and README, so its ZIP hash need not match the original. If the verified original ZIP is available locally, run the checker directly on that file to validate the original archive. The smoke test checks an unpacked predictor directory. These engineering checks do not establish prediction gains or the official trigger frequency.

## Permitted offline-study reproduction

The included real-data studies use only the inventory-listed AlgoTune and Multi-SWE-bench formatted collections at Measurement DB revision `54074453da7b9911be0f48e4c560be547bc84d21`. Response tables are **not included**. Obtain the tables yourself only through the authorized Measurement DB access path and subject to its terms. Use the same pinned revision for every table, verify SHA-256 against `manifest.json`, and place files exactly as follows:

```text
experiments/v27_official_data/raw/algotune/metadata.yaml
experiments/v27_official_data/raw/algotune/characterization.yaml
experiments/v27_official_data/raw/algotune/curation_record.md
experiments/v27_official_data/raw/algotune/formatted_tables/{benchmarks,items,responses,subjects}.parquet
experiments/v37_task_conditioned/raw/multi_swebench/formatted_tables/{benchmarks,items,responses,subjects}.parquet
```

Use Python 3.12.3. Install the research-only dependency in an isolated environment:

```powershell
python -m pip install -r requirements-research.txt
```

From the repository root, run the included tests:

```powershell
python -S experiments/v33_live_transfer/tests.py
python -S experiments/v36_prediction_semantics/tests.py
python -S experiments/v38_minimax_real/tests.py
```

To reproduce the paired studies, execute in order from a disposable working copy (the runners overwrite result JSON files):

```powershell
python experiments/v27_official_data/real_trial.py
python experiments/v37_task_conditioned/run_trial.py
python experiments/v38_minimax_real/run_trial.py
```

V27 freezes the AlgoTune subject split and fits its comparison prior from training subjects only. V37 reuses that split for AlgoTune and the frozen content-cluster split for Multi-SWE-bench. V38 reuses the same splits and prefixes. These scripts write result JSON files in their experiment folders; use a disposable working copy if you need to preserve the checked-in reference outputs. Full protocols and aggregate reference results are next to each runner. Do not use AppWorld, MathArena, ResearchCodeBench, CMMLU, hidden/private outcomes, or any source outside the PAIEC benchmark inventory for competition-specific development.

## Validation and references

The Low-peer regression tests and the historical V33 and V36 tests use synthetic official-shape inputs; V38 includes mathematical/property tests. V27/V37/V38 aggregate JSON files contain study summaries, not response-table payloads. `experiments/solve_assess_validation/pair_eval.py` is the shared evaluator module. Validation records are descriptive; different formative submissions may resample hidden evaluation groups. Aggregate score differences therefore are not paired causal comparisons. No real-platform diagnostic exposes how often Low-peer correction triggered.

## Provenance, tooling, and licensing

The predictor and research code in this repository was developed by the team with assistance from OpenAI Codex. Chen Linhe and Liu Yanbo authorized MIT licensing for code the team owns and confirmed that they did not directly copy code from other GitHub projects or the official Starting Kit. Codex assisted with implementation, code review, experiment packaging, and report drafting. Local Qwen models used in separate exploratory work are not loaded here and are not included.

The root `LICENSE` applies to team-owned project code and documentation in this release. It does not relicense third-party materials or dependencies. The Measurement DB response tables, upstream benchmark datasets, model weights, credentials, and official Starting Kit source are not bundled; their own terms remain applicable. See `docs/ATTRIBUTION_AND_LICENSES.md` for the scope boundary.

For questions about this release, contact Chen Linhe and Liu Yanbo at [909tttttough@gmail.com](mailto:909tttttough@gmail.com). The author names, affiliation (Saint Petersburg State University), and contact address are included with the authors’ approval.

## Integrity

- V12 archive SHA-256: `de12d731b6f1c4b9f6c83226998e17fc930ad55e8c16948c753c6ceaa754fe02`
- Historical three-peer V33 archive SHA-256: `21e8b461ec483f6a13365e001cadb7f94557ad7a325b5343c941495279477b41`
- Selected Low-peer `model.py` SHA-256: `53646d9b3d38ae2330a4fb5943428024b2f9d8bec588806fa56fb16d73c3b7ac`
- Locally verified Low-peer ZIP SHA-256: `16ac64c6c361c7948df656e1db5381f193f8628ca652e13b74ca8847f86b22c6`; the screenshot does not verify the uploaded ZIP's digest.

The predictor and research code in this repository was developed by the team with assistance from OpenAI Codex. Chen Linhe and Liu Yanbo have authorized MIT licensing for code the team owns. The authors confirm that they did not directly copy code from other GitHub projects or the official Starting Kit. Third-party materials remain subject to their own terms; the Starting Kit, Measurement DB response tables, upstream benchmark datasets, model weights, and credentials are not included or relicensed here. Codex assisted with implementation, code review, experiment packaging, and report drafting. Local Qwen models used in separate exploratory work are not loaded or included.

The root `LICENSE` applies to the team-owned project code and documentation in this release. It does not grant rights to external dependencies, datasets, or other third-party materials; their original terms continue to apply.
