from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "decision.py"
)
WORKLOAD_EVALUATE_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "workload_evaluate.py"
)


def _load_decision():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_decision_test", DECISION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_workload_evaluate():
    module_name = "cuda_optimizer_workload_evaluate_decision_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, WORKLOAD_EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _statistics(status: str = "confirmed_win") -> dict:
    return {
        "status": status,
        "statistic": "median_paired_improvement_pct",
        "estimate_pct": 5.0,
        "ci_low_pct": 3.0,
        "ci_high_pct": 7.0,
    }


def _kernel(status: str = "confirmed_win") -> dict:
    return {"status": status, "statistics": _statistics(status)}


def _workload(primary_status: str = "confirmed_win", *, constraints=None) -> dict:
    constraint_results = [] if constraints is None else constraints
    return {
        "status": "evaluated",
        "primary": _statistics(primary_status),
        "objective": {
            "primary_metric": {"name": "latency_ms", "direction": "lower"},
            "min_effect_pct": 1.0,
            "constraints": [
                {"name": constraint["name"].strip(), "max_regression_pct": 5.0}
                for constraint in constraint_results
            ],
        },
        "constraints": constraint_results,
    }


def _pareto() -> dict:
    return {
        "schema": "cuda-kernel-optimizer/pareto-v1",
        "status": "non_dominated",
        "objectives": [
            {"name": "latency", "outcome": "improved"},
            {"name": "memory", "outcome": "regressed"},
        ],
    }


class DecisionTests(unittest.TestCase):
    def test_terminal_statuses_are_exactly_the_public_set(self) -> None:
        module = _load_decision()
        self.assertEqual(
            module.TERMINAL_STATUSES,
            {
                "rejected_compile",
                "rejected_correctness",
                "rejected_constraint",
                "confirmed_loss",
                "inconclusive",
                "kernel_only_win",
                "end_to_end_win",
                "pareto_frontier",
            },
        )

    def test_kernel_rejections_and_loss_propagate(self) -> None:
        module = _load_decision()
        for status in (
            "rejected_compile",
            "rejected_correctness",
            "confirmed_loss",
        ):
            with self.subTest(status=status):
                result = module.decide(mode="full", kernel={"status": status})
                self.assertEqual(result["status"], status)
                self.assertIn("reason", result)

    def test_nonwinning_kernel_evidence_is_inconclusive(self) -> None:
        module = _load_decision()
        for status in ("inconclusive", "invalid", "no_confirmed_kernel_win"):
            with self.subTest(status=status):
                result = module.decide(mode="full", kernel={"status": status})
                self.assertEqual(result["status"], "inconclusive")

    def test_kernel_only_mode_never_claims_end_to_end(self) -> None:
        module = _load_decision()
        kernel = _kernel()
        before = copy.deepcopy(kernel)
        result = module.decide(
            mode="kernel-only",
            kernel=kernel,
            workload=_workload(),
            constraints=[],
        )
        self.assertEqual(result["status"], "kernel_only_win")
        self.assertEqual(result["mode"], "kernel-only")
        self.assertEqual(result["statistics"], kernel["statistics"])
        self.assertNotIn("workload_statistics", result)
        self.assertEqual(kernel, before)

    def test_full_mode_requires_primary_win_and_passed_constraints(self) -> None:
        module = _load_decision()
        constraints = [
            {"name": "memory_mb", "status": "passed"},
            {"name": "p99_ms", "status": "passed"},
        ]
        kernel = _kernel()
        workload = _workload(constraints=constraints)
        before = copy.deepcopy((kernel, workload, constraints))

        result = module.decide(
            mode="full",
            kernel=kernel,
            workload=workload,
            constraints=constraints,
        )

        self.assertEqual(result["status"], "end_to_end_win")
        self.assertEqual(result["statistics"], kernel["statistics"])
        self.assertEqual(result["workload_statistics"], workload["primary"])
        self.assertEqual((kernel, workload, constraints), before)

    def test_constraint_failure_rejects_before_pareto(self) -> None:
        module = _load_decision()
        constraints = [{"name": "memory", "status": "failed"}]
        result = module.decide(
            mode="full",
            kernel=_kernel(),
            workload=_workload(constraints=constraints),
            pareto=_pareto(),
        )
        self.assertEqual(result["status"], "rejected_constraint")

    def test_explicit_constraints_cannot_override_workload_constraints(self) -> None:
        module = _load_decision()
        workload = _workload(
            constraints=[{"name": "memory", "status": "failed"}]
        )
        with self.assertRaisesRegex(ValueError, "conflicting constraints"):
            module.decide(
                mode="full",
                kernel=_kernel(),
                workload=workload,
                constraints=[{"name": "memory", "status": "passed"}],
            )

    def test_matching_explicit_and_workload_constraints_are_accepted(self) -> None:
        module = _load_decision()
        constraints = [{"name": " memory ", "status": "passed"}]
        workload = _workload(
            constraints=[{"name": "memory", "status": "passed"}]
        )
        result = module.decide(
            mode="full",
            kernel=_kernel(),
            workload=workload,
            constraints=constraints,
        )
        self.assertEqual(result["status"], "end_to_end_win")

    def test_explicit_constraints_cannot_complete_truncated_workload(self) -> None:
        module = _load_decision()
        workload = {"status": "evaluated", "primary": _statistics()}
        with self.assertRaisesRegex(ValueError, "incomplete workload evidence"):
            module.decide(
                mode="full",
                kernel=_kernel(),
                workload=workload,
                constraints=[{"name": "memory", "status": "passed"}],
            )

    def test_winning_workload_constraint_names_match_objective_exactly(self) -> None:
        module = _load_decision()
        primary = _statistics()
        primary_metric = {"name": "latency_ms", "direction": "lower"}
        malformed = (
            {
                "status": "evaluated",
                "primary": primary,
                "objective": {
                    "primary_metric": primary_metric,
                    "constraints": [
                        {"name": "memory", "max_regression_pct": 5.0}
                    ],
                },
                "constraints": [],
            },
            {
                "status": "evaluated",
                "primary": primary,
                "objective": {"primary_metric": primary_metric, "constraints": []},
                "constraints": [{"name": "memory", "status": "passed"}],
            },
            {
                "status": "evaluated",
                "primary": primary,
                "objective": {
                    "primary_metric": primary_metric,
                    "constraints": [
                        {"name": "memory", "max_regression_pct": 5.0},
                        {"name": "memory", "max_regression_pct": 5.0},
                    ],
                },
                "constraints": [{"name": "memory", "status": "passed"}],
            },
            {
                "status": "evaluated",
                "primary": primary,
                "objective": {
                    "primary_metric": primary_metric,
                    "constraints": None,
                },
                "constraints": [],
            },
            {
                "status": "evaluated",
                "primary": primary,
                "objective": {"primary_metric": primary_metric, "constraints": []},
                "constraints": None,
            },
        )
        for workload in malformed:
            with self.subTest(workload=workload), self.assertRaises(ValueError):
                module.decide(mode="full", kernel=_kernel(), workload=workload)

    def test_zero_declared_constraints_require_explicit_empty_evidence(self) -> None:
        module = _load_decision()
        workload = _workload(constraints=[])
        result = module.decide(mode="full", kernel=_kernel(), workload=workload)
        self.assertEqual(result["status"], "end_to_end_win")

    def test_inconclusive_constraint_keeps_only_kernel_win(self) -> None:
        module = _load_decision()
        constraints = [{"name": "memory", "status": "inconclusive"}]
        result = module.decide(
            mode="full",
            kernel=_kernel(),
            workload=_workload(constraints=constraints),
        )
        self.assertEqual(result["status"], "kernel_only_win")
        self.assertEqual(result["statistics"]["status"], "confirmed_win")

    def test_missing_failed_or_nonwinning_workload_keeps_only_kernel_win(self) -> None:
        module = _load_decision()
        workloads = (
            None,
            {"status": "workload_failed"},
            _workload("inconclusive"),
            _workload("confirmed_loss"),
            {"status": "evaluated", "constraints": []},
        )
        for workload in workloads:
            with self.subTest(workload=workload):
                result = module.decide(
                    mode="full", kernel=_kernel(), workload=workload
                )
                self.assertEqual(result["status"], "kernel_only_win")

    def test_pareto_cannot_bypass_missing_or_nonwinning_workload(self) -> None:
        module = _load_decision()
        workloads = (
            None,
            {"status": "workload_failed"},
            _workload("inconclusive"),
            _workload("confirmed_loss"),
            {"status": "evaluated", "constraints": []},
        )
        for workload in workloads:
            with self.subTest(workload=workload):
                result = module.decide(
                    mode="full",
                    kernel=_kernel(),
                    workload=workload,
                    pareto=_pareto(),
                )
                self.assertEqual(result["status"], "kernel_only_win")

    def test_empty_constraints_are_all_passed(self) -> None:
        module = _load_decision()
        result = module.decide(
            mode="full", kernel=_kernel(), workload=_workload(), constraints=[]
        )
        self.assertEqual(result["status"], "end_to_end_win")

    def test_explicit_non_dominated_tradeoff_returns_pareto_without_weighting(self) -> None:
        module = _load_decision()
        result = module.decide(
            mode="full",
            kernel=_kernel(),
            workload=_workload(),
            constraints=[],
            pareto=_pareto(),
        )
        self.assertEqual(result["status"], "pareto_frontier")
        self.assertEqual(result["pareto"], _pareto())
        self.assertNotIn("score", result)
        self.assertNotIn("weight", repr(result).lower())

    def test_status_only_win_evidence_is_rejected(self) -> None:
        module = _load_decision()
        with self.assertRaisesRegex(ValueError, "kernel.statistics"):
            module.decide(
                mode="full",
                kernel={"status": "confirmed_win"},
                workload=_workload(),
            )
        workload = _workload()
        workload["primary"] = {"status": "confirmed_win"}
        with self.assertRaisesRegex(ValueError, "workload.primary"):
            module.decide(mode="full", kernel=_kernel(), workload=workload)

    def test_win_statistics_require_complete_finite_nonboolean_fields(self) -> None:
        module = _load_decision()
        bad_values = (math.nan, math.inf, -math.inf, True, 10**400)
        for field in ("estimate_pct", "ci_low_pct", "ci_high_pct"):
            for bad in bad_values:
                with self.subTest(source="kernel", field=field, bad=bad):
                    kernel = _kernel()
                    kernel["statistics"][field] = bad
                    with self.assertRaises(ValueError):
                        module.decide(
                            mode="full", kernel=kernel, workload=_workload()
                        )
                with self.subTest(source="primary", field=field, bad=bad):
                    workload = _workload()
                    workload["primary"][field] = bad
                    with self.assertRaises(ValueError):
                        module.decide(
                            mode="full", kernel=_kernel(), workload=workload
                        )
        for missing in (
            "statistic",
            "estimate_pct",
            "ci_low_pct",
            "ci_high_pct",
        ):
            with self.subTest(source="kernel", missing=missing):
                kernel = _kernel()
                del kernel["statistics"][missing]
                with self.assertRaises(ValueError):
                    module.decide(mode="full", kernel=kernel, workload=_workload())
            with self.subTest(source="primary", missing=missing):
                workload = _workload()
                del workload["primary"][missing]
                with self.assertRaises(ValueError):
                    module.decide(mode="full", kernel=_kernel(), workload=workload)

    def test_all_evidence_must_be_detached_strict_json(self) -> None:
        module = _load_decision()
        cyclic = []
        cyclic.append(cyclic)
        malformed = (
            {"kernel": {"status": "inconclusive", "extra": cyclic}},
            {"kernel": {"status": "inconclusive", "extra": object()}},
            {
                "kernel": _kernel(),
                "workload": {"status": "workload_failed", "extra": {1: "bad"}},
            },
            {
                "kernel": _kernel(),
                "workload": _workload(),
                "constraints": [
                    {"name": "memory", "status": "passed", "extra": {1, 2}}
                ],
            },
        )
        for kwargs in malformed:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module.decide(mode="full", **kwargs)

        kernel = _kernel()
        result = module.decide(mode="full", kernel=kernel, workload=_workload())
        result["evidence"]["kernel"]["statistics"]["estimate_pct"] = 999.0
        self.assertEqual(kernel["statistics"]["estimate_pct"], 5.0)
        json.dumps(result, allow_nan=False)

    def test_real_evaluate_pairs_output_is_end_to_end_state_compatible(self) -> None:
        evaluator = _load_workload_evaluate()
        decision = _load_decision()
        objective = {
            "primary_metric": {"name": "latency_ms", "direction": "lower"},
            "min_effect_pct": 1.0,
            "constraints": [
                {"name": "memory_mb", "max_regression_pct": 5.0}
            ],
        }
        workload_spec = evaluator.WorkloadSpec(
            kind="python",
            source="unused.py",
            objective=objective,
            cases=(),
            source_hash="0" * 64,
        )

        def runner(spec, *, candidate, role, case, timeout):
            return {
                "role": role,
                "case": {},
                "validation": True,
                "benchmark": {
                    "latency_ms": 100.0 if role == "baseline" else 90.0,
                    "memory_mb": 100.0,
                },
                "objective": objective,
            }

        workload = evaluator.evaluate_pairs(
            workload_spec,
            "baseline.py",
            "candidate.py",
            blocks=3,
            retries=0,
            bootstrap_samples=20,
            runner=runner,
        )
        result = decision.decide(mode="full", kernel=_kernel(), workload=workload)

        self.assertEqual(result["status"], "end_to_end_win")
        required = {
            "statistic",
            "estimate_pct",
            "ci_low_pct",
            "ci_high_pct",
            "status",
        }
        self.assertTrue(required.issubset(result["statistics"]))
        self.assertTrue(required.issubset(result["workload_statistics"]))
        json.dumps(result, allow_nan=False)

    def test_mode_kernel_workload_and_statuses_are_strict(self) -> None:
        module = _load_decision()
        calls = (
            {"mode": "FULL", "kernel": _kernel()},
            {"mode": True, "kernel": _kernel()},
            {"mode": "kernel_only", "kernel": _kernel()},
            {"mode": "full", "kernel": []},
            {"mode": "full", "kernel": {}},
            {"mode": "full", "kernel": {"status": True}},
            {"mode": "full", "kernel": {"status": "end_to_end_win"}},
            {
                "mode": "full",
                "kernel": _kernel(),
                "workload": {"status": True},
            },
            {
                "mode": "full",
                "kernel": _kernel(),
                "workload": {"status": "unknown"},
            },
        )
        for kwargs in calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module.decide(**kwargs)

    def test_constraints_require_unique_names_and_literal_statuses(self) -> None:
        module = _load_decision()
        malformed = (
            {},
            [{"name": "", "status": "passed"}],
            [{"name": "memory", "status": True}],
            [{"name": "memory", "status": "unknown"}],
            [
                {"name": "memory", "status": "passed"},
                {"name": "memory", "status": "failed"},
            ],
        )
        for constraints in malformed:
            with self.subTest(constraints=constraints), self.assertRaises(ValueError):
                module.decide(
                    mode="full",
                    kernel=_kernel(),
                    workload=_workload(),
                    constraints=constraints,
                )

    def test_pareto_schema_is_strict_and_requires_an_actual_tradeoff(self) -> None:
        module = _load_decision()
        malformed = (
            {},
            {**_pareto(), "weight": 0.5},
            {**_pareto(), "schema": "v2"},
            {**_pareto(), "status": "dominated"},
            {
                **_pareto(),
                "objectives": [
                    {"name": "latency", "outcome": "improved"},
                    {"name": "memory", "outcome": "improved"},
                ],
            },
            {
                **_pareto(),
                "objectives": [
                    {"name": "latency", "outcome": "improved", "weight": 1.0},
                    {"name": "memory", "outcome": "regressed"},
                ],
            },
        )
        for pareto in malformed:
            with self.subTest(pareto=pareto), self.assertRaises(ValueError):
                module.decide(
                    mode="full",
                    kernel=_kernel(),
                    workload=_workload(),
                    constraints=[],
                    pareto=pareto,
                )

    def test_kernel_only_win_is_accepted_as_inner_win(self) -> None:
        module = _load_decision()
        result = module.decide(
            mode="kernel-only",
            kernel={"status": "kernel_only_win", "statistics": _statistics()},
        )
        self.assertEqual(result["status"], "kernel_only_win")
        self.assertEqual(result["statistics"]["status"], "confirmed_win")


if __name__ == "__main__":
    unittest.main()
