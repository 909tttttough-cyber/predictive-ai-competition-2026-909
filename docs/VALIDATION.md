# Validation record

Validation ran from the prepared public repository tree on 2026-10-16 using the local official Starting Kit (not redistributed here).

- V33 unit tests: 11/11 passed with `python -S`.
- V36 frozen-ZIP prediction-semantics tests: 18/18 passed with `python -S`.
- V38 estimator/math tests: 6/6 passed with `python -S`.
- Official Starting Kit ZIP checker: V12 and V33 archives both passed.
- Official Starting Kit smoke test: V12 passed (2 predictions); V33 passed (2 predictions).
- Frozen archive SHA-256 checks: recorded in `manifest.json`.

Real-data runner scripts were not rerun during release preparation. The required authorized Parquet files are not included. Their locations, pinned revision, and expected digests are documented in `README.md` and `manifest.json`; these studies require users to obtain data under the applicable access terms.

## Selected Low-peer source update

On 2026-10-28, `models/v33_lowpeer/model.py` was verified byte-for-byte against the `model.py` member of the locally held Low-peer ZIP with SHA-256 `16ac64c6c361c7948df656e1db5381f193f8628ca652e13b74ca8847f86b22c6`. Five synthetic Low-peer regression tests, the official Starting Kit ZIP checker on that local ZIP, and the official smoke test on the extracted source directory passed. The historical V33 synthetic suite also passed. The smoke fixture is an interface check, not a benchmark score.

The displayed Codabench #1002801 filename, Finished status, and formative Brier ALC 0.164578 are recorded from a user-provided official-page screenshot. Uploaded archive byte identity and official peer-correction trigger frequency remain unknown. The team's final #1002801 PDF describes the same Low-peer gates and peer-count scaling; an earlier archived PDF describes the historical three-peer candidate.
