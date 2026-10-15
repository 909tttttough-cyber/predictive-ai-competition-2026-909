# PAIEC 2026: reproducible predictors and offline studies

This repository contains the frozen V12 and V33 PAIEC probability predictors and a minimal set of code and aggregate outputs for reproducing the permitted AlgoTune and Multi-SWE-bench studies. It is a source/research release, not a Codabench submission; it documents both frozen candidates without designating the official selection.

## Task and metric

PAIEC asks a predictor to estimate the probability that a configured AI subject answers an item correctly, using the current prediction input and revealed binary labels. For each subject–benchmark pair, Brier score is averaged over evaluation responses at budgets 0, 1, 3, 7, 15, and 31. Brier ALC is the weighted sum with weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1; the competition aggregate gives subject–benchmark groups equal weight.

## Predictors

- **V12** (`models/v12/model.py`): exact target subject and benchmark filtering; `p=(k+1)/(n+2)` with a Beta(1,1) prior. Stateless; Python standard library only.
- **V33** (`models/v33/model.py`): V12 fallback plus a conservative residual correction from at least three distinct peer configurations with exact same-benchmark observable payload matches, after excluding the matched payload from each peer baseline. Any ambiguous or insufficient evidence returns V12. There is no independent item ID in this matching rule, and actual hidden-platform trigger frequency is unknown.

The corresponding frozen upload archives are preserved byte-for-byte under `release_staging/`. Their SHA-256 digests are recorded in `manifest.json` and below. Run candidates separately; do not combine their files.

## Starting Kit checker and smoke test

The official Starting Kit is third-party material and is deliberately not redistributed here. Obtain the current kit from the official PAIEC/Codabench source and follow its license. Set `$kit` to the extracted kit directory, then from this repository root run:

```powershell
python "$kit/tools/check_submission_zip.py" "release_staging/v12_beta11_candidate/v12_beta11_candidate.zip"
python "$kit/tools/run_smoke_test.py" "models/v12"
python "$kit/tools/check_submission_zip.py" "release_staging/v33_conservative_transfer_candidate/v33_conservative_transfer_candidate.zip"
python "$kit/tools/run_smoke_test.py" "models/v33"
```

The smoke test checks the unpacked predictor directory; the checker validates each unchanged archive. These engineering checks do not establish prediction gains or V33 trigger frequency.

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

The V33 test suite and V36 contract tests use synthetic official-shape inputs; V38 includes mathematical/property tests. V27/V37/V38 aggregate JSON files contain study summaries, not response-table payloads. `experiments/solve_assess_validation/pair_eval.py` is the shared evaluator module. Validation records are descriptive; formative Codabench submissions resample hidden evaluation subsets. V12 and V33 formative totals therefore are not paired causal comparisons. No real-platform diagnostic exposes how often V33's correction triggered.

## Provenance, tooling, and licensing

The predictor and research code in this repository was developed by the team with assistance from OpenAI Codex. Chen Linhe and Liu Yanbo authorized MIT licensing for code the team owns and confirmed that they did not directly copy code from other GitHub projects or the official Starting Kit. Codex assisted with implementation, code review, experiment packaging, and report drafting. Local Qwen models used in separate exploratory work are not loaded here and are not included.

The root `LICENSE` applies to team-owned project code and documentation in this release. It does not relicense third-party materials or dependencies. The Measurement DB response tables, upstream benchmark datasets, model weights, credentials, and official Starting Kit source are not bundled; their own terms remain applicable. See `docs/ATTRIBUTION_AND_LICENSES.md` for the scope boundary.

For questions about this release, contact Chen Linhe and Liu Yanbo at [909tttttough@gmail.com](mailto:909tttttough@gmail.com). The author names, affiliation (Saint Petersburg State University), and contact address are included with the authors’ approval.

## Integrity

- V12 archive SHA-256: `de12d731b6f1c4b9f6c83226998e17fc930ad55e8c16948c753c6ceaa754fe02`
- V33 archive SHA-256: `21e8b461ec483f6a13365e001cadb7f94557ad7a325b5343c941495279477b41`

No repository creation, Git push, Codabench/OpenReview upload, or email is performed by this preparation.

The predictor and research code in this repository was developed by the team with assistance from OpenAI Codex. Chen Linhe and Liu Yanbo have authorized MIT licensing for code the team owns. The authors confirm that they did not directly copy code from other GitHub projects or the official Starting Kit. Third-party materials remain subject to their own terms; the Starting Kit, Measurement DB response tables, upstream benchmark datasets, model weights, and credentials are not included or relicensed here. Codex assisted with implementation, code review, experiment packaging, and report drafting. Local Qwen models used in separate exploratory work are not loaded or included.

The root `LICENSE` applies to the team-owned project code and documentation in this release. It does not grant rights to external dependencies, datasets, or other third-party materials; their original terms continue to apply.
