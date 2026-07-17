from __future__ import annotations

import importlib.util
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "experiment_design.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_design", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BalancedPairOrdersTests(unittest.TestCase):
    def test_even_schedule_is_exactly_balanced_and_reproducible(self) -> None:
        module = _load_module()
        first = module.balanced_pair_orders(12, seed=17)
        second = module.balanced_pair_orders(12, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first.count("AB"), 6)
        self.assertEqual(first.count("BA"), 6)

    def test_odd_schedule_differs_by_at_most_one(self) -> None:
        module = _load_module()
        orders = module.balanced_pair_orders(5, seed=4)
        self.assertEqual(len(orders), 5)
        self.assertLessEqual(abs(orders.count("AB") - orders.count("BA")), 1)

    def test_does_not_mutate_global_rng_and_rejects_bad_inputs(self) -> None:
        module = _load_module()
        random.seed(99)
        before = random.getstate()
        module.balanced_pair_orders(4, seed=3)
        self.assertEqual(random.getstate(), before)
        for blocks in (0, -1, True, 1.5):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                module.balanced_pair_orders(blocks)
        for seed in (True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                module.balanced_pair_orders(2, seed=seed)


if __name__ == "__main__":
    unittest.main()
