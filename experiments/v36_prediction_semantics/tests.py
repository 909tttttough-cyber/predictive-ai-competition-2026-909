"""Contract-focused tests against the model.py bytes inside frozen ZIPs.

Run with: python experiments/v36_prediction_semantics/tests.py
Only the standard library is required. Tests deliberately distinguish valid
Starting Kit inputs from malformed cases that the contract does not define.
"""
from __future__ import annotations

import hashlib
import io
import math
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V12_ZIP = ROOT / "release_staging/v12_beta11_candidate/v12_beta11_candidate.zip"
V33_ZIP = ROOT / "release_staging/v33_conservative_transfer_candidate/v33_conservative_transfer_candidate.zip"
EXPECTED = {
    "V12": "de12d731b6f1c4b9f6c83226998e17fc930ad55e8c16948c753c6ceaa754fe02",
    "V33": "21e8b461ec483f6a13365e001cadb7f94557ad7a325b5343c941495279477b41",
}
SUBJECT_FIELDS = (
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "reasoning_effort", "harness_version", "subject_features_extra",
)


def subject(tag: str = "target") -> dict[str, str]:
    return {
        "normalized_name": tag,
        "provider": "provider",
        "release_date": "2026-01-01",
        "access_date": "2026-02-01",
        "harness": "harness",
        "reasoning_effort": "medium",
        "harness_version": "1.0",
        "subject_features_extra": "",
    }


def item(content: str = "target task", benchmark: str = "bench-A", *, features: str = "", interactors: str = "") -> dict[str, str]:
    return {"item_content": content, "item_features": features,
            "interactors": interactors, "benchmark_id": benchmark}


def load_model(path: Path, name: str):
    with zipfile.ZipFile(path) as zf:
        assert "model.py" in zf.namelist(), f"{name} frozen ZIP has no root model.py"
        source = zf.read("model.py")
    ns: dict[str, object] = {"__name__": f"frozen_{name.lower()}"}
    exec(compile(source, f"{path}!/model.py", "exec"), ns)
    return ns["predict"]


def run() -> tuple[int, list[str]]:
    predictors = {"V12": load_model(V12_ZIP, "V12"), "V33": load_model(V33_ZIP, "V33")}
    for key, path in (("V12", V12_ZIP), ("V33", V33_ZIP)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == EXPECTED[key], f"{key} ZIP hash changed: {digest}"
    p12, p33 = predictors["V12"], predictors["V33"]
    passed: list[str] = []

    def check(name, fn):
        fn()
        passed.append(name)

    s, i = subject(), item()
    target = [s, i]

    # Valid official-contract cases.
    check("official empty/None evidence: budget 0 is 0.5", lambda: (
        (_ for _ in ()).throw(AssertionError()) if p12(target, []) != 0.5 or p12(target, None) != 0.5 else None))

    def target_one_label(label: int) -> float:
        return p12(target, [[[s, item("acq-1")], label]])
    check("official budget 1: labels 0/1 produce 1/3 and 2/3", lambda: (
        None if target_one_label(0) == 1 / 3 and target_one_label(1) == 2 / 3 else (_ for _ in ()).throw(AssertionError())))

    check("official exact eight-field subject matching", lambda: (
        None if math.isclose(p12(target, [[[s, item("a")], 1], [[subject("other"), item("b")], 0]]), 2 / 3)
        else (_ for _ in ()).throw(AssertionError())))

    check("official benchmark isolation", lambda: (
        None if math.isclose(p12(target, [[[s, item("x", "bench-B")], 0], [[s, item("y", "bench-A")], 1]]), 2 / 3)
        else (_ for _ in ()).throw(AssertionError())))

    check("official duplicate item outcomes count as independent labels, including conflict", lambda: (
        None if math.isclose(p12(target, [[[s, i], 1], [[s, i], 0]]), 1 / 2)
        else (_ for _ in ()).throw(AssertionError())))

    def budgets():
        # The protocol exposes caps; the predictor sees the actual revealed list,
        # so test actual n rather than passing a budget field that is not in input.
        expected = {0: .5, 1: 2/3, 3: 4/5, 7: 8/9, 15: 16/17, 31: 32/33}
        for n, value in expected.items():
            labels = [[[s, item(f"acq-{j}")], 1] for j in range(n)]
            assert math.isclose(p12(target, labels), value), (n, p12(target, labels), value)
    check("official six budgets use received valid labels (all-positive prefixes)", budgets)

    def v33_budget_behavior():
        for n in (0, 1, 3, 7, 15, 31):
            evidence = [[[s, item(f"target-acq-{j}")], j % 2] for j in range(n)]
            for j in range(3):
                peer = subject(f"budget-peer-{j}")
                evidence += [[[peer, item("target task")], 1],
                             [[peer, item(f"budget-base-{j}")], 0]]
            got = p33(target, evidence)
            assert math.isfinite(got) and 0 <= got <= 1
            if n < 7:
                assert got == p12(target, evidence), (n, got, p12(target, evidence))
    check("V33 all six budgets: finite output and mandatory fallback below n=7", v33_budget_behavior)

    def budget_is_not_inferred_from_list_length_as_protocol_field():
        # There is no separate budget argument in predict(); actual valid
        # received labels determine n, even if an external caller informally
        # describes a different cap.
        evidence = [[[s, item("only-revealed")], 0]]
        assert p12(target, evidence) == 1 / 3
        assert p33(target, evidence) == p12(target, evidence)
    check("prediction uses actual received labels; no independent budget field exists", budget_is_not_inferred_from_list_length_as_protocol_field)

    def v12_unrelated_records():
        base = [[[s, item("acq")], 1]]
        extra = base + [[[subject(f"unrelated-{j}"), item(f"u{j}")], j % 2] for j in range(100)]
        assert p12(target, extra) == p12(target, base)
    check("official many unrelated subjects do not change V12 target count", v12_unrelated_records)

    check("official legal empty strings are matched as values, not missing fields", lambda: (
        None if p12(target, [[[s, item("acq", features="", interactors="")], 1]]) == 2 / 3
        else (_ for _ in ()).throw(AssertionError())))

    def v33_no_peers_exact_fallback():
        evidence = [[[s, item(f"target-acq-{j}")], j % 2] for j in range(7)]
        assert p33(target, evidence) == p12(target, evidence)
    check("V33 no legal peer evidence strictly equals V12", v33_no_peers_exact_fallback)

    def v33_peer_gate_and_trigger():
        evidence = [[[s, item(f"target-acq-{j}")], j % 2] for j in range(7)]
        # A source's other distinct same-benchmark item provides its baseline;
        # same full payload and benchmark supplies its already revealed target outcome.
        for peer_idx in range(3):
            peer = subject(f"peer-{peer_idx}")
            evidence.append([[peer, item("target task")], 1])
            evidence.append([[peer, item(f"peer-baseline-{peer_idx}")], 0])
        assert p33(target, evidence) != p12(target, evidence)
        one_two = evidence[:7] + evidence[7:11]
        assert p33(target, one_two) == p12(target, one_two)
    check("V33 three distinct peers with own baselines trigger; one/two do not", v33_peer_gate_and_trigger)

    def v33_interactor_benchmark_identity_gates():
        target_with_interactor = [s, item("target task", interactors="tool-A")]
        evidence = [[[s, item(f"a{j}")], 1] for j in range(7)]
        for j in range(3):
            peer = subject(f"peer-{j}")
            evidence += [[[peer, item("target task", interactors="tool-B")], 1],
                         [[peer, item(f"base{j}"),], 0]]
        assert p33(target_with_interactor, evidence) == p12(target_with_interactor, evidence)
        wrong_bench = [[[s, item(f"a{j}")], 1] for j in range(7)]
        for j in range(3):
            peer = subject(f"peer-{j}")
            wrong_bench += [[[peer, item("target task", "bench-B"),], 1],
                            [[peer, item(f"base{j}", "bench-B")], 0]]
        assert p33(target, wrong_bench) == p12(target, wrong_bench)
    check("V33 requires full item payload and same benchmark", v33_interactor_benchmark_identity_gates)

    def v33_same_subject_not_counted_and_peer_baselines_isolated():
        evidence = [[[s, item(f"target-acq-{j}")], 1] for j in range(7)]
        # Target subject may have other revealed items, but its identity cannot
        # count as an independent peer for the queried item.
        evidence.append([[s, item("target-subject-extra")], 0])
        for j in range(2):
            peer = subject(f"peer-{j}")
            evidence += [[[peer, item("target task")], 1], [[peer, item(f"base{j}")], 0]]
        assert p33(target, evidence) == p12(target, evidence)
        # Peer baseline from another benchmark cannot establish a baseline here.
        evidence = [[[s, item(f"target-acq-{j}")], 1] for j in range(7)]
        for j in range(3):
            peer = subject(f"peer-{j}")
            evidence += [[[peer, item("target task")], 1], [[peer, item(f"base{j}", "bench-B")], 0]]
        assert p33(target, evidence) == p12(target, evidence)
    check("V33 excludes target identity and cross-benchmark peer baselines", v33_same_subject_not_counted_and_peer_baselines_isolated)

    def v33_target_outcome_not_used():
        evidence = [[[s, item(f"target-acq-{j}")], j % 2] for j in range(7)]
        for j in range(3):
            peer = subject(f"peer-{j}")
            evidence += [[[peer, item("target task")], 1], [[peer, item(f"base{j}")], 0]]
        # No target-subject record for the queried exact item is part of valid
        # evaluation input. A separate contamination check is outside contract.
        before = p33(target, evidence)
        contaminated = evidence + [[[s, i], 0]]
        assert p33(target, contaminated) == before
    check("V33 target outcome cannot enter its own prediction; contaminated duplicate stripped", v33_target_outcome_not_used)

    # Contract-violating or unspecified cases are recorded separately by report;
    # these checks characterize behavior but are not counted as platform bugs.
    check("out-of-contract bool labels differ by implementation (V12 accepts bool; V33 ignores)", lambda: (
        None if p12(target, [[[s, item("bool")], True]]) == 2/3 and p33(target, [[[s, item("bool")], True]]) == .5
        else (_ for _ in ()).throw(AssertionError())))

    def string_label_out_of_contract():
        try:
            p12(target, [[[s, item("string-label")], "1"]])
        except ValueError:
            pass
        else:
            raise AssertionError("V12 should reject non-binary string label")
        assert p33(target, [[[s, item("string-label")], "1"]]) == .5
    check("out-of-contract string labels: V12 rejects, V33 ignores", string_label_out_of_contract)

    def malformed_fields():
        missing_subject = dict(s); missing_subject.pop("provider")
        missing_item = dict(i); missing_item.pop("interactors")
        # These are not official-shape inputs; ensure behavior is recorded and finite.
        for predictor in (p12, p33):
            value = predictor([missing_subject, missing_item], [[[missing_subject, missing_item], 1]])
            assert math.isfinite(value) and 0 <= value <= 1
    check("out-of-contract missing fields return finite probabilities", malformed_fields)

    return len(passed), passed


if __name__ == "__main__":
    count, names = run()
    print(f"PASS {count}/{count}")
    for name in names:
        print(f"- {name}")
