# V12 Beta(1,1) Smoothed Empirical Mean

Independent experimental candidate for the PAIEC Starting Kit. It uses only
binary responses revealed to the current `predict(input, labeled=None)` call
that match the complete subject dictionary and target `benchmark_id`.

For `n` matching labels, of which `k` are correct, it returns
`(k + 1) / (n + 2)`. With no matching evidence, including budget 0, this is
`0.5`. The only algorithmic difference from frozen V11 is the fixed symmetric
Beta prior: V11 uses Beta(0.5, 0.5) and `(k + 0.5) / (n + 1)`.

The stronger shrinkage toward 0.5 may help for some true rates nearer 0.5 and
may hurt for rates near 0 or 1. No real-data advantage is established. This
candidate has no historical training data, model artifact, text features,
acquisition hook, external dependency, network access, or persistent state.
