from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "decision.py"
)


def _load_decision():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_decision_test", DECISION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
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
    return {
        "status": "evaluated",
        "primary": _statistics(primary_status),
        "constraints": [] if constraints is None else constraints,
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

    def test_explicit_constraints_are_allowed_when_workload_omits_them(self) -> None:
        module = _load_decision()
        workload = {"status": "evaluated", "primary": _statistics()}
        result = module.decide(
            mode="full",
            kernel=_kernel(),
            workload=workload,
            constraints=[{"name": "memory", "status": "passed"}],
        )
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

    def test_status_only_win_evidence_does_not_fabricate_ci_values(self) -> None:
        module = _load_decision()
        result = module.decide(
            mode="full",
            kernel={"status": "confirmed_win"},
            workload={
                "status": "evaluated",
                "primary": {"status": "confirmed_win"},
                "constraints": [],
            },
        )
        self.assertEqual(result["status"], "end_to_end_win")
        self.assertEqual(result["statistics"], {"status": "confirmed_win"})
        self.assertEqual(
            result["workload_statistics"], {"status": "confirmed_win"}
        )
        self.assertNotIn("ci_low_pct", result["statistics"])

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
