import importlib.util
import math
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v38_run_trial", HERE / "run_trial.py")
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def load_module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    assert s and s.loader
    s.loader.exec_module(m)
    return m


v22_impl = load_module("frozen_v22_candidate", HERE.parents[1] / "experiments/v22_binomial_minimax/model_candidate.py")
v12_impl = load_module("frozen_v12_candidate", HERE.parents[1] / "release_staging/v12_beta11_candidate/model.py")


class EstimatorTests(unittest.TestCase):
    def test_budget_zero_matches(self):
        self.assertEqual(module.v12(0, 0), 0.5)
        self.assertEqual(module.v38(0, 0), 0.5)

    def test_budget_one_matches_jeffreys(self):
        self.assertEqual(module.v12(1, 0), 1 / 3)
        self.assertEqual(module.v38(1, 0), 0.25)
        self.assertEqual(module.v12(1, 1), 2 / 3)
        self.assertEqual(module.v38(1, 1), 0.75)

    def test_all_required_counts_and_probability_bounds(self):
        for n in module.BUDGETS:
            for k in (0, n // 2, n):
                p = module.v38(n, k)
                self.assertTrue(0.0 <= p <= 1.0)
                self.assertTrue(math.isfinite(p))

    def test_exact_risk_matches_enumeration_at_n1(self):
        for theta in (0.05, 0.25, 0.5, 0.75, 0.95):
            risk = module.binomial_risk(theta, 1, module.v38)
            expected = theta * (1 - theta) + 0.0625
            self.assertAlmostEqual(risk, expected, places=14)

    def test_minimax_risk_constant_parameter_mse(self):
        # For n=1, V38 risk for estimating theta is 1/16 for all theta.
        for theta in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
            mse = 0.0
            for k in (0, 1):
                prob = (1-theta) if k == 0 else theta
                mse += prob * (module.v38(1, k)-theta)**2
            self.assertAlmostEqual(mse, 1/16, places=14)

    def test_formula_matches_frozen_predictor_implementations(self):
        subject = {"subject_id": "synthetic"}
        labels = [
            [[subject, {"benchmark_id": "b", "item_content": str(i)}], y]
            for i, y in enumerate((1, 0, 1, 1, 0, 0, 1))
        ]
        target = [subject, {"benchmark_id": "b", "item_content": "target"}]
        n, k = 7, 4
        self.assertAlmostEqual(v12_impl.predict(target, labels), module.v12(n, k))
        self.assertAlmostEqual(v22_impl.predict(target, labels), module.v38(n, k))


if __name__ == "__main__":
    unittest.main(verbosity=2)
