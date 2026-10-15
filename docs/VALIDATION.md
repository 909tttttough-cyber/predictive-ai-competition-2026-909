# Validation record

Validation ran from the prepared public repository tree on 2026-10-16 using the local official Starting Kit (not redistributed here).

- V33 unit tests: 11/11 passed with `python -S`.
- V36 frozen-ZIP prediction-semantics tests: 18/18 passed with `python -S`.
- V38 estimator/math tests: 6/6 passed with `python -S`.
- Official Starting Kit ZIP checker: V12 and V33 archives both passed.
- Official Starting Kit smoke test: V12 passed (2 predictions); V33 passed (2 predictions).
- Frozen archive SHA-256 checks: recorded in `manifest.json`.

Real-data runner scripts were not rerun during release preparation. The required authorized Parquet files are not included. Their locations, pinned revision, and expected digests are documented in `README.md` and `manifest.json`; these studies require users to obtain data under the applicable access terms.
