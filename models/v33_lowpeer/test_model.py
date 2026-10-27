"""Synthetic behavior checks for the frozen V33 Low-peer model."""

import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("lowpeer", Path(__file__).with_name("model.py"))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def subject(name):
    return {"normalized_name": name, "provider": "example", "release_date": "",
            "access_date": "", "harness": "example", "reasoning_effort": "",
            "harness_version": "", "subject_features_extra": ""}


def item(content, benchmark="b", features=""):
    return {"item_content": content, "item_features": features,
            "interactors": "", "benchmark_id": benchmark}


def label(who, what, value):
    return [[who, what], value]


class LowPeerTests(unittest.TestCase):
    def setUp(self):
        self.target_subject = subject("target")
        self.target_item = item("target")
        self.pair = [self.target_subject, self.target_item]
        self.history = [label(self.target_subject, item(f"history {i}"), int(i < 4))
                        for i in range(7)]

    def peers(self, count):
        evidence = []
        for index in range(count):
            peer = subject(f"peer {index}")
            evidence.extend([label(peer, self.target_item, 1),
                             label(peer, item(f"peer baseline {index}"), 0)])
        return evidence

    def test_peer_coefficients_one_two_and_three_or_more(self):
        for count, coefficient in ((1, 1 / 7), (2, 1 / 4), (3, 1 / 3), (5, 1 / 3)):
            with self.subTest(peers=count):
                # Target baseline is 5/9; each peer residual is 1 - 1/3 = 2/3.
                self.assertAlmostEqual(model.predict(self.pair, self.history + self.peers(count)),
                                       5 / 9 + (2 / 3) * coefficient)

    def test_fallback_without_labels_or_with_too_few_target_labels(self):
        self.assertEqual(model.predict(self.pair), 0.5)
        short = self.history[:6] + self.peers(1)
        self.assertEqual(model.predict(self.pair, short), 5 / 8)

    def test_nonmatching_payload_and_mixed_peer_signs_fall_back(self):
        peer_a, peer_b = subject("a"), subject("b")
        mismatch = [label(peer_a, item("target", features="different"), 1),
                    label(peer_a, item("other"), 0)]
        self.assertEqual(model.predict(self.pair, self.history + mismatch), 5 / 9)
        mixed = [label(peer_a, self.target_item, 1), label(peer_a, item("a other"), 0),
                 label(peer_b, self.target_item, 0), label(peer_b, item("b other"), 0)]
        self.assertEqual(model.predict(self.pair, self.history + mixed), 5 / 9)

    def test_target_payload_and_nonbinary_labels_are_excluded(self):
        extra = [label(self.target_subject, dict(self.target_item), 1),
                 label(self.target_subject, item("fractional"), 0.5),
                 label(self.target_subject, item("boolean"), True)]
        self.assertEqual(model.predict(self.pair, self.history + extra), 5 / 9)

    def test_subject_and_benchmark_isolation(self):
        other = [label(subject("unrelated"), item("not the target"), 1),
                 label(self.target_subject, item("other benchmark", "other"), 1)]
        self.assertEqual(model.predict(self.pair, self.history + other), 5 / 9)


if __name__ == "__main__":
    unittest.main()
