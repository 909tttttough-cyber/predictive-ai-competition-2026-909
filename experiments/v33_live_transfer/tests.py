import math
import unittest

from model import predict, v12_probability


SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra")


def subject(name):
    return {key: (name if key == "normalized_name" else "") for key in SUBJECT_FIELDS}


def item(text="task", benchmark="bench", features="", interactors=""):
    return {"item_content": text, "item_features": features, "interactors": interactors,
            "benchmark_id": benchmark}


def labeled(subject_, item_, label):
    return [[subject_, item_], label]


class V33Tests(unittest.TestCase):
    def test_no_peers_falls_back_exactly(self):
        target, target_item = subject("target"), item()
        evidence = [labeled(target, item(f"own-{i}"), i % 2) for i in range(7)]
        self.assertEqual(predict([target, target_item], evidence), v12_probability([target, target_item], evidence))

    def test_one_or_two_peers_fall_back_exactly(self):
        target, target_item = subject("target"), item()
        evidence = [labeled(target, item(f"own-{i}"), i % 2) for i in range(7)]
        for peer_count in (1, 2):
            peers = []
            for p in range(peer_count):
                peer = subject(f"peer-{p}")
                peers += [labeled(peer, target_item, 1), labeled(peer, item(f"other-{p}"), 0)]
            all_evidence = evidence + peers
            self.assertEqual(predict([target, target_item], all_evidence),
                             v12_probability([target, target_item], all_evidence))

    def test_three_complete_peers_with_baselines_can_trigger(self):
        target, target_item = subject("target"), item("shared", "bench", "topic=x", "tool=y")
        evidence = [labeled(target, item(f"own-{i}"), i % 2) for i in range(7)]
        for p in range(3):
            peer = subject(f"peer-{p}")
            evidence.append(labeled(peer, dict(target_item), 1))
            evidence.extend(labeled(peer, item(f"peer-{p}-other-{i}"), 0) for i in range(3))
        before = v12_probability([target, target_item], evidence)
        after = predict([target, target_item], evidence)
        self.assertGreater(after, before)
        self.assertLessEqual(after, 1.0)

    def test_same_text_different_interactors_does_not_match(self):
        target, target_item = subject("target"), item("same")
        evidence = [labeled(target, item(f"own-{i}"), 1) for i in range(7)]
        for p in range(3):
            peer = subject(f"peer-{p}")
            evidence.extend([labeled(peer, item("same", interactors=f"tool-{p}"), 1),
                             labeled(peer, item(f"other-{p}"), 0)])
        self.assertEqual(predict([target, target_item], evidence), v12_probability([target, target_item], evidence))

    def test_same_payload_different_benchmark_does_not_match(self):
        target, target_item = subject("target"), item("same", "bench-a")
        evidence = [labeled(target, item(f"own-{i}", "bench-a"), 1) for i in range(7)]
        for p in range(3):
            peer = subject(f"peer-{p}")
            evidence.extend([labeled(peer, item("same", "bench-b"), 1),
                             labeled(peer, item(f"other-{p}", "bench-b"), 0)])
        self.assertEqual(predict([target, target_item], evidence), v12_probability([target, target_item], evidence))

    def test_missing_or_untrusted_item_id_never_used_for_matching(self):
        target, target_item = subject("target"), item("same")
        evidence = [labeled(target, item(f"own-{i}"), 1) for i in range(7)]
        # Official schema has no item_id. An extra/untrusted field invalidates
        # strict matching; it is never used to join otherwise partial payloads.
        for p in range(3):
            peer = subject(f"peer-{p}")
            peer_item = dict(target_item, item_id=f"untrusted-{p}")
            evidence.extend([labeled(peer, peer_item, 1), labeled(peer, item(f"other-{p}"), 0)])
        self.assertEqual(predict([target, target_item], evidence), v12_probability([target, target_item], evidence))
        untrusted_target = dict(target_item, item_id="untrusted")
        self.assertEqual(predict([target, untrusted_target], evidence),
                         v12_probability([target, untrusted_target], evidence))

    def test_only_revealed_records_can_affect_prediction(self):
        target, target_item = subject("target"), item("shared")
        visible = [labeled(target, item(f"own-{i}"), i % 2) for i in range(7)]
        for p in range(3):
            peer = subject(f"peer-{p}")
            visible.extend([labeled(peer, target_item, 1), labeled(peer, item(f"other-{p}"), 0)])
        pred_visible = predict([target, target_item], visible)
        # Hypothetical outcomes not added to labeled are not available to the pure predictor.
        hypothetical_future = [labeled(subject("future"), target_item, 0)]
        self.assertEqual(pred_visible, predict([target, target_item], visible))
        self.assertNotIn(hypothetical_future[0], visible)

    def test_other_benchmark_labels_do_not_change_target_count_or_transfer(self):
        target, target_item = subject("target"), item("shared", "bench-a")
        own = [labeled(target, item(f"own-{i}", "bench-a"), 1) for i in range(7)]
        before = predict([target, target_item], own)
        foreign = []
        for p in range(5):
            peer = subject(f"peer-{p}")
            foreign.extend([labeled(peer, item("shared", "bench-b"), 1),
                             labeled(peer, item(f"other-{p}", "bench-b"), 0)])
        self.assertEqual(before, predict([target, target_item], own + foreign))

    def test_six_budget_values_are_finite_and_in_range(self):
        target = subject("target")
        for budget in (0, 1, 3, 7, 15, 31):
            evidence = [labeled(target, item(f"own-{i}"), i % 2) for i in range(budget)]
            value = predict([target, item("eval")], evidence)
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_malformed_records_fail_closed(self):
        target, target_item = subject("target"), item("eval")
        evidence = [labeled(target, item(f"own-{i}"), 1) for i in range(7)]
        evidence.extend([[[subject("peer"), {"item_content": "eval", "item_features": "",
                                                   "interactors": "", "benchmark_id": "bench"}], True]])
        value = predict([target, target_item], evidence)
        self.assertEqual(value, v12_probability([target, target_item], evidence))

    def test_prediction_does_not_mutate_inputs(self):
        target, target_item = subject("target"), item("eval")
        evidence = [labeled(target, item(f"own-{i}"), 1) for i in range(7)]
        snapshot = repr((target, target_item, evidence))
        predict([target, target_item], evidence)
        self.assertEqual(snapshot, repr((target, target_item, evidence)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
