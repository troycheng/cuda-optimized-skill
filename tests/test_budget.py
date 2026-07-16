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

INVALID_TIME_VALUES = (True, False, "100", float("nan"), float("inf"), float("-inf"))


class BudgetPolicyTests(unittest.TestCase):
    def test_presets_have_expected_fields(self) -> None:
        expected = {
            "quick": {
                "name": "quick",
                "max_seconds": 2700,
                "branches": 4,
                "max_rounds": 2,
                "min_pairs": 20,
                "max_pairs": 50,
                "outer_candidates": 1,
                "max_cases": 3,
                "sanitizer_mode": "targeted",
                "reserve_seconds": 300,
            },
            "balanced": {
                "name": "balanced",
                "max_seconds": 3 * 60 * 60,
                "branches": 8,
                "max_rounds": 4,
                "min_pairs": 20,
                "max_pairs": 100,
                "outer_candidates": 2,
                "max_cases": 10,
                "sanitizer_mode": "targeted",
                "reserve_seconds": 300,
            },
            "thorough": {
                "name": "thorough",
                "max_seconds": 10 * 60 * 60,
                "branches": 16,
                "max_rounds": 8,
                "min_pairs": 30,
                "max_pairs": 200,
                "outer_candidates": 3,
                "max_cases": None,
                "sanitizer_mode": "full",
                "reserve_seconds": 300,
            },
        }
        for name, expected_fields in expected.items():
            with self.subTest(name=name):
                policy = budget.resolve_budget(name)
                actual = {
                    field: getattr(policy, field)
                    for field in policy.__dataclass_fields__
                }
                self.assertEqual(actual, expected_fields)

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
    def _assert_invalid_time(self, parameter: str, action) -> None:
        try:
            action()
        except ValueError as error:
            self.assertIn(parameter, str(error))
        except Exception as error:
            self.fail(
                f"{parameter} raised {type(error).__name__}, expected ValueError"
            )
        else:
            self.fail(f"{parameter} did not raise ValueError")

    def test_constructor_rejects_invalid_started_at(self) -> None:
        policy = budget.resolve_budget("quick")
        for value in INVALID_TIME_VALUES:
            with self.subTest(value=repr(value)):
                self._assert_invalid_time(
                    "started_at",
                    lambda value=value: budget.BudgetClock(
                        policy=policy, started_at=value
                    ),
                )

    def test_can_start_rejects_invalid_now(self) -> None:
        clock = budget.BudgetClock(
            policy=budget.resolve_budget("quick"), started_at=100
        )
        for value in INVALID_TIME_VALUES:
            with self.subTest(value=repr(value)):
                self._assert_invalid_time(
                    "now",
                    lambda value=value: clock.can_start(
                        now=value, estimated_seconds=10
                    ),
                )

    def test_remaining_seconds_rejects_invalid_now(self) -> None:
        clock = budget.BudgetClock(
            policy=budget.resolve_budget("quick"), started_at=100
        )
        for value in INVALID_TIME_VALUES:
            with self.subTest(value=repr(value)):
                self._assert_invalid_time(
                    "now", lambda value=value: clock.remaining_seconds(now=value)
                )

    def test_can_start_rejects_invalid_estimated_seconds(self) -> None:
        clock = budget.BudgetClock(
            policy=budget.resolve_budget("quick"), started_at=100
        )
        for value in INVALID_TIME_VALUES:
            with self.subTest(value=repr(value)):
                self._assert_invalid_time(
                    "estimated_seconds",
                    lambda value=value: clock.can_start(
                        now=200, estimated_seconds=value
                    ),
                )

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

    def test_persisted_elapsed_time_survives_clock_rollback(self) -> None:
        clock = budget.BudgetClock(
            policy=budget.resolve_budget("quick"),
            started_at=100,
            elapsed_seconds=50,
        )

        self.assertEqual(clock.elapsed(now=90), 50)
        self.assertEqual(clock.remaining_seconds(now=90), 2650)
        self.assertEqual(clock.elapsed(now=110), 60)
        self.assertEqual(clock.execution_seconds_available(now=110), 2340)

    def test_constructor_rejects_invalid_persisted_elapsed_time(self) -> None:
        policy = budget.resolve_budget("quick")
        for value in (*INVALID_TIME_VALUES, -1):
            with self.subTest(value=repr(value)):
                self._assert_invalid_time(
                    "elapsed_seconds",
                    lambda value=value: budget.BudgetClock(
                        policy=policy,
                        started_at=100,
                        elapsed_seconds=value,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
