from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "budget.py"


def _load_budget():
    module_name = "cuda_optimizer_budget"
    spec = importlib.util.spec_from_file_location(module_name, BUDGET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


budget = _load_budget()


class BudgetPolicyTests(unittest.TestCase):
    def test_balanced_preset_has_expected_limits(self) -> None:
        policy = budget.resolve_budget("balanced")

        self.assertEqual(policy.max_seconds, 3 * 60 * 60)
        self.assertEqual(policy.branches, 8)
        self.assertEqual(policy.max_rounds, 4)
        self.assertEqual(policy.min_pairs, 20)
        self.assertEqual(policy.max_pairs, 100)
        self.assertEqual(policy.outer_candidates, 2)
        self.assertEqual(policy.reserve_seconds, 300)

    def test_overrides_do_not_mutate_presets(self) -> None:
        original = dict(budget.PRESETS)

        resolved = budget.resolve_budget("balanced", branches=3, max_rounds=2)

        self.assertEqual(resolved.branches, 3)
        self.assertEqual(resolved.max_rounds, 2)
        self.assertEqual(dict(budget.PRESETS), original)
        self.assertEqual(budget.PRESETS["balanced"].branches, 8)
        self.assertIsNot(resolved, budget.PRESETS["balanced"])

    def test_budget_policy_is_frozen(self) -> None:
        policy = budget.resolve_budget("quick")

        with self.assertRaises(FrozenInstanceError):
            policy.branches = 99

    def test_custom_zero_max_seconds_names_invalid_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_seconds"):
            budget.resolve_budget("custom", max_seconds=0)

    def test_custom_required_numeric_fields_must_be_positive(self) -> None:
        valid = {
            "max_seconds": 900,
            "branches": 2,
            "max_rounds": 2,
            "min_pairs": 5,
            "max_pairs": 10,
            "outer_candidates": 1,
        }
        for field in valid:
            with self.subTest(field=field):
                overrides = dict(valid)
                overrides[field] = 0
                with self.assertRaisesRegex(ValueError, field):
                    budget.resolve_budget("custom", **overrides)

    def test_custom_requires_all_numeric_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "branches"):
            budget.resolve_budget("custom", max_seconds=900)

    def test_custom_rejects_min_pairs_above_max_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_pairs"):
            budget.resolve_budget(
                "custom",
                max_seconds=900,
                branches=2,
                max_rounds=2,
                min_pairs=11,
                max_pairs=10,
                outer_candidates=1,
            )

    def test_unknown_preset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            budget.resolve_budget("unknown")

    def test_invalid_sanitizer_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sanitizer_mode"):
            budget.resolve_budget("balanced", sanitizer_mode="invalid")

    def test_reserve_must_be_less_than_total_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserve_seconds"):
            budget.resolve_budget("quick", reserve_seconds=2700)


class BudgetClockTests(unittest.TestCase):
    def test_quick_policy_cannot_start_after_execution_deadline(self) -> None:
        clock = budget.BudgetClock(
            policy=budget.resolve_budget("quick"), started_at=100
        )

        self.assertFalse(clock.can_start(now=2500, estimated_seconds=10))

    def test_zero_estimate_can_start_at_execution_deadline(self) -> None:
        clock = budget.BudgetClock(
            policy=budget.resolve_budget("quick"), started_at=100
        )

        self.assertTrue(clock.can_start(now=2500, estimated_seconds=0))

    def test_negative_estimate_is_zero_at_execution_deadline(self) -> None:
        clock = budget.BudgetClock(
            policy=budget.resolve_budget("quick"), started_at=100
        )

        self.assertTrue(clock.can_start(now=2500, estimated_seconds=-10))

    def test_remaining_seconds_is_clamped_to_zero(self) -> None:
        clock = budget.BudgetClock(
            policy=budget.resolve_budget("quick"), started_at=100
        )

        self.assertEqual(clock.remaining_seconds(now=200), 2600)
        self.assertEqual(clock.remaining_seconds(now=3000), 0)


if __name__ == "__main__":
    unittest.main()
