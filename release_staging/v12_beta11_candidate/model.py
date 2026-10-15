"""Beta-Binomial smoothed empirical mean over currently revealed labels."""


def predict(input: list, labeled: list | None = None) -> float:
    """Predict correctness from matching revealed labels with fixed Beta(1,1).

    Matching follows the official empirical-mean baseline: subject dictionaries
    must compare equal, and labels are pooled only within the target
    benchmark_id. Entries from other subjects or benchmarks are ignored.
    """
    if not isinstance(input, (list, tuple)) or len(input) != 2:
        raise ValueError("input must be [subject_dict, item_dict].")
    subject, item = input
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise ValueError("subject and item inputs must be dictionaries.")
    if labeled is None:
        labeled = []
    if not isinstance(labeled, (list, tuple)):
        raise ValueError("labeled must be a list of official-style entries.")
    # Match the upstream baseline's no-evidence fast path, including its
    # compatibility with checker inputs that omit benchmark_id at budget 0.
    if not labeled:
        return 0.5

    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("The smoothed mean predictor requires an anonymous item benchmark_id.")

    successes = 0
    count = 0
    for entry in labeled:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("Each labeled entry must be [[subject_dict, item_dict], label].")
        pair, label = entry
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Labeled input pair must contain subject_dict and item_dict.")
        observed_subject, observed_item = pair
        if not isinstance(observed_subject, dict) or not isinstance(observed_item, dict):
            raise ValueError("Labeled subject and item inputs must be dictionaries.")
        if observed_subject != subject:
            continue
        observed_benchmark = observed_item.get("benchmark_id")
        if not isinstance(observed_benchmark, str) or not observed_benchmark:
            raise ValueError("Acquired items must include an anonymous item benchmark_id.")
        if observed_benchmark != benchmark:
            continue
        if label not in (0, 1):
            raise ValueError("Acquired responses must be binary (0 or 1).")
        successes += int(label)
        count += 1

    if count == 0:
        return 0.5
    return float((successes + 1) / (count + 2))
