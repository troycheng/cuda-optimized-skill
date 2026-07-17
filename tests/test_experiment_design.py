from __future__ import annotations

import importlib.util
import copy
import math
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


def _formal_design() -> dict:
    return {
        "schema_version": "cuda-evidence/experiment-design-v1",
        "formal": True,
        "schedule": [
            {"pair_id": "p1", "order": "AB"},
            {"pair_id": "p2", "order": "BA"},
            {"pair_id": "p3", "order": "BA"},
            {"pair_id": "p4", "order": "AB"},
        ],
        "experimental_unit": "fresh_process_pair",
        "aggregation": "median_paired_improvement",
        "resampling_unit": "pair",
        "ci": {
            "method": "paired_bootstrap",
            "confidence": 0.95,
            "samples": 10000,
            "seed": 17,
        },
        "min_valid_pairs": 4,
        "wins_required": 3,
        "guardrails": {
            "relative": [
                {
                    "metric": "p99_latency_ms",
                    "comparison": "max_regression",
                    "direction": "lower",
                    "limit_pct": 2.0,
                }
            ],
            "absolute": [
                {
                    "metric": "error_rate_pct",
                    "operator": "<=",
                    "limit": 0.1,
                }
            ],
        },
        "exclusion_policy": "no_exclusion",
        "retry_policy": {
            "role_retries": 0,
            "whole_pair_only": True,
            "allowed_reasons": ["pre_measurement_infrastructure_failure"],
        },
    }


class FrozenExperimentDesignTests(unittest.TestCase):
    def test_valid_design_is_detached_and_exposes_exact_schedule(self) -> None:
        module = _load_module()
        payload = _formal_design()

        validated = module.validate_frozen_design(payload)
        payload["schedule"][0]["order"] = "BA"

        self.assertEqual(module.schedule_orders(validated), ["AB", "BA", "BA", "AB"])
        self.assertEqual(validated["schedule"][0]["order"], "AB")

    def test_formal_design_rejects_unknown_unbalanced_or_incomplete_schedule(self) -> None:
        module = _load_module()
        cases = []
        unknown = _formal_design()
        unknown["surprise"] = True
        cases.append(unknown)
        duplicate = _formal_design()
        duplicate["schedule"][1]["pair_id"] = "p1"
        cases.append(duplicate)
        unbalanced = _formal_design()
        for row in unbalanced["schedule"]:
            row["order"] = "AB"
        cases.append(unbalanced)
        bad_order = _formal_design()
        bad_order["schedule"][0]["order"] = "AA"
        cases.append(bad_order)

        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                module.validate_frozen_design(payload)

    def test_formal_statistics_guardrails_and_retry_contract_are_fail_closed(self) -> None:
        module = _load_module()
        mutators = (
            lambda item: item.update(formal=False),
            lambda item: item.update(experimental_unit=""),
            lambda item: item.update(resampling_unit="request"),
            lambda item: item["ci"].update(confidence=math.nan),
            lambda item: item.update(min_valid_pairs=1),
            lambda item: item.update(wins_required=5),
            lambda item: item["guardrails"].update(relative=[]),
            lambda item: item["guardrails"].update(absolute=[]),
            lambda item: item.update(exclusion_policy="drop_outliers"),
            lambda item: item["retry_policy"].update(role_retries=1),
            lambda item: item["retry_policy"].update(whole_pair_only=False),
            lambda item: item["retry_policy"].update(allowed_reasons=["slow_role"]),
        )

        for mutate in mutators:
            payload = copy.deepcopy(_formal_design())
            mutate(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                module.validate_frozen_design(payload)


if __name__ == "__main__":
    unittest.main()
