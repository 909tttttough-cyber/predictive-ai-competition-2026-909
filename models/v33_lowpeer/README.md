# Final selected V33 Low-peer predictor

`model.py` is the byte-for-byte `model.py` member of the locally verified
`v33_lowpeer_formative_candidate.zip`. Its SHA-256 is
`53646d9b3d38ae2330a4fb5943428024b2f9d8bec588806fa56fb16d73c3b7ac`.
The reference local ZIP has SHA-256
`16ac64c6c361c7948df656e1db5381f193f8628ca652e13b74ca8847f86b22c6`.
The Codabench screenshot associates submission #1002801 and formative Brier ALC
0.164578 with that ZIP filename; the uploaded ZIP's bytes have not been
independently matched to the local ZIP.

The predictor accepts the official `predict(input, labeled=None)` interface and
uses only the Python standard library. Its subject-benchmark fallback is the
Beta(1,1) posterior mean `(1+k)/(2+n)` over strictly binary revealed labels.
When a valid target payload is available, the target subject's records on that
exact payload are excluded before the fallback and the target-label gate.

Peer matching requires exact equality of the benchmark and the observable
`item_content`, `item_features`, and `interactors`. At least seven valid target
labels and one eligible peer are required. Each peer's target outcome is
centered by its Beta(1,1) baseline on other distinct, unambiguous payloads in
the same benchmark. A peer is excluded if its target or other payload labels
conflict, or if it lacks another baseline item. All retained residuals must
have the same nonzero sign. Their mean is multiplied by `min(1/3, J/(J+6))`:
the factors are 1/7 for one peer, 1/4 for two, and 1/3 for three or more.
The result is clipped to [0,1]. If a gate fails, the predictor returns the
subject baseline; malformed input can return 0.5. Exact payload equality is an
observable heuristic, not an official item-ID guarantee.

No `labeling.py`, model weights, external data, network access, or persistent
cross-call state is required. The official runtime's peer-correction trigger
rate is unknown. Formative submissions can use resampled evaluation groups;
their aggregate scores are not paired causal comparisons.

Run the synthetic regression checks from the repository root:

```powershell
python -B models/v33_lowpeer/test_model.py
```

See the repository README for Starting Kit checker and smoke-test commands.
