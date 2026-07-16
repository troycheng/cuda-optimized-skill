from __future__ import annotations

import copy
import importlib.util
import json
import math
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
WORKLOAD_EVALUATE_PATH = SCRIPT_DIR / "workload_evaluate.py"


def _load_workload_evaluate():
    module_name = "cuda_optimizer_workload_evaluate_test"
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


def _objective(
    *,
    primary: str = "latency_ms",
    direction: str = "lower",
    min_effect_pct: float = 1.0,
    constraints: list[dict] | None = None,
) -> dict:
    return {
        "primary_metric": {"name": primary, "direction": direction},
        "min_effect_pct": min_effect_pct,
        "constraints": (
            [{"name": "memory_mb", "max_regression_pct": 5.0}]
            if constraints is None
            else constraints
        ),
    }


def _workload(module, *, objective: dict | None = None, cases=()):
    return module.WorkloadSpec(
        kind="python",
        source="unused-by-fake-runner.py",
        objective=_objective() if objective is None else objective,
        cases=tuple(cases),
        source_hash="0" * 64,
    )


def _observation(workload, *, role, case, metrics, validation=True):
    return {
        "role": role,
        "case": {} if case is None else copy.deepcopy(case),
        "validation": copy.deepcopy(validation),
        "benchmark": copy.deepcopy(metrics),
        "objective": copy.deepcopy(dict(workload.objective)),
    }


class MeasureCandidateTests(unittest.TestCase):
    def test_default_timeout_preserves_python_runner_none_contract(self) -> None:
        module = _load_workload_evaluate()
        workload = _workload(module)

        def runner(spec, *, candidate, role, case, timeout):
            self.assertIsNone(timeout)
            return _observation(
                spec,
                role=role,
                case=case,
                metrics={"latency_ms": 90.0, "memory_mb": 100.0},
            )

        result = module.measure_candidate(
            workload, "candidate.py", retries=0, runner=runner
        )
        self.assertEqual(result["status"], "measured")

    def test_two_transient_failures_then_success_reports_three_attempts(self) -> None:
        module = _load_workload_evaluate()
        workload = _workload(module)
        calls = 0

        def runner(spec, *, candidate, role, case, timeout):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("secret=do-not-record " + "x" * 5000)
            return _observation(
                spec,
                role=role,
                case=case,
                metrics={"latency_ms": 90.0, "memory_mb": 100.0},
                validation={"valid": True},
            )

        result = module.measure_candidate(
            workload,
            {"path": "candidate.py"},
            retries=2,
            timeout=12.5,
            runner=runner,
        )

        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["metrics"]["latency_ms"], 90.0)
        self.assertEqual(
            [record["status"] for record in result["attempt_records"]],
            ["failed", "failed", "success"],
        )
        self.assertTrue(
            all(
                set(record) == {"attempt", "status", "error_type", "error"}
                for record in result["attempt_records"]
            )
        )
        serialized = json.dumps(result)
        self.assertNotIn("do-not-record", serialized)
        self.assertLess(len(serialized), 3000)

    def test_persistent_exception_returns_one_structured_failed_result(self) -> None:
        module = _load_workload_evaluate()
        calls = 0

        def runner(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("credential token must stay private")

        result = module.measure_candidate(
            _workload(module), "candidate.py", retries=1, runner=runner
        )

        self.assertEqual(calls, 2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], 2)
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["failure"]["error_type"], "TimeoutError")
        self.assertNotIn("credential", json.dumps(result))

    def test_retry_count_is_a_nonnegative_literal_integer(self) -> None:
        module = _load_workload_evaluate()
        workload = _workload(module)
        for retries in (True, -1, 1.0, "1"):
            with self.subTest(retries=retries), self.assertRaisesRegex(
                ValueError, "retries"
            ):
                module.measure_candidate(
                    workload, "candidate.py", retries=retries, runner=lambda: None
                )

    def test_zero_retries_runs_exactly_one_attempt(self) -> None:
        module = _load_workload_evaluate()
        calls = 0

        def runner(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("transient")

        result = module.measure_candidate(
            _workload(module), "candidate.py", retries=0, runner=runner
        )
        self.assertEqual(calls, 1)
        self.assertEqual(result["attempts"], 1)

    def test_keyboard_interrupt_and_system_exit_are_never_retried(self) -> None:
        module = _load_workload_evaluate()
        for error in (KeyboardInterrupt(), SystemExit(9)):
            calls = 0

            def runner(*args, **kwargs):
                nonlocal calls
                calls += 1
                raise error

            with self.subTest(error=type(error).__name__), self.assertRaises(
                type(error)
            ):
                module.measure_candidate(
                    _workload(module), "candidate.py", retries=3, runner=runner
                )
            self.assertEqual(calls, 1)

    def test_success_requires_five_fields_passing_validation_and_mapping_metrics(
        self,
    ) -> None:
        module = _load_workload_evaluate()
        workload = _workload(module)
        valid = _observation(
            workload,
            role="candidate",
            case=None,
            metrics={"latency_ms": 90.0},
        )
        invalid_results = []
        for missing in ("role", "case", "validation", "benchmark", "objective"):
            result = copy.deepcopy(valid)
            del result[missing]
            invalid_results.append(result)
        invalid_results.extend(
            [
                {**copy.deepcopy(valid), "validation": False},
                {**copy.deepcopy(valid), "validation": {"valid": False}},
                {**copy.deepcopy(valid), "validation": {"valid": 1}},
                {**copy.deepcopy(valid), "benchmark": [1.0]},
            ]
        )

        for raw in invalid_results:
            with self.subTest(raw=raw):
                result = module.measure_candidate(
                    workload,
                    "candidate.py",
                    retries=0,
                    runner=lambda *args, raw=raw, **kwargs: raw,
                )
                self.assertEqual(result["status"], "failed")

    def test_candidate_case_and_returned_metrics_are_detached(self) -> None:
        module = _load_workload_evaluate()
        workload = _workload(module)
        candidate = {"nested": [1]}
        case = {"shape": [16]}
        shared_metrics = {"latency_ms": 90.0, "nested": {"v": 1}}

        def runner(spec, *, candidate, role, case, timeout):
            candidate["nested"].append(2)
            case["shape"].append(32)
            return _observation(
                spec, role=role, case=case, metrics=shared_metrics, validation=True
            )

        result = module.measure_candidate(
            workload, candidate, case=case, retries=0, runner=runner
        )
        result["metrics"]["nested"]["v"] = 9

        self.assertEqual(candidate, {"nested": [1]})
        self.assertEqual(case, {"shape": [16]})
        self.assertEqual(shared_metrics["nested"]["v"], 1)


class EvaluatePairsTests(unittest.TestCase):
    def test_seed_controls_ab_ba_order_without_touching_global_rng(self) -> None:
        module = _load_workload_evaluate()
        workload = _workload(
            module,
            cases=({"shape": [16]}, {"shape": [32]}),
        )
        calls = []

        def runner(spec, *, candidate, role, case, timeout):
            calls.append((role, copy.deepcopy(case)))
            metrics = (
                {"latency_ms": 100.0, "memory_mb": 100.0}
                if role == "baseline"
                else {"latency_ms": 90.0, "memory_mb": 100.0}
            )
            return _observation(
                spec, role=role, case=case, metrics=metrics, validation=True
            )

        random.seed(20260716)
        global_state = random.getstate()
        first = module.evaluate_pairs(
            workload,
            "baseline.py",
            "candidate.py",
            blocks=8,
            seed=0,
            bootstrap_samples=100,
            retries=0,
            runner=runner,
        )
        second = module.evaluate_pairs(
            workload,
            "baseline.py",
            "candidate.py",
            blocks=8,
            seed=0,
            bootstrap_samples=100,
            retries=0,
            runner=runner,
        )

        self.assertEqual(random.getstate(), global_state)
        self.assertEqual(
            [pair["order"] for pair in first["pairs"]],
            [pair["order"] for pair in second["pairs"]],
        )
        self.assertEqual(
            [pair["case"] for pair in first["pairs"]],
            [
                {"shape": [16]},
                {"shape": [32]},
                {"shape": [16]},
                {"shape": [32]},
                {"shape": [16]},
                {"shape": [32]},
                {"shape": [16]},
                {"shape": [32]},
            ],
        )
        self.assertEqual(set(pair["order"] for pair in first["pairs"]), {"AB", "BA"})
        first_run_calls = calls[:16]
        for index, pair in enumerate(first["pairs"]):
            observed_roles = [role for role, _ in first_run_calls[index * 2 : index * 2 + 2]]
            expected_roles = (
                ["baseline", "candidate"]
                if pair["order"] == "AB"
                else ["candidate", "baseline"]
            )
            self.assertEqual(observed_roles, expected_roles)
            self.assertEqual(
                first_run_calls[index * 2][1], first_run_calls[index * 2 + 1][1]
            )

    def test_no_cases_uses_none_for_each_pair(self) -> None:
        module = _load_workload_evaluate()

        def runner(spec, *, candidate, role, case, timeout):
            self.assertIsNone(case)
            metrics = {"latency_ms": 100.0, "memory_mb": 100.0}
            if role == "candidate":
                metrics["latency_ms"] = 90.0
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            _workload(module),
            "baseline.py",
            "candidate.py",
            blocks=2,
            retries=0,
            bootstrap_samples=50,
            runner=runner,
        )
        self.assertEqual([pair["case"] for pair in result["pairs"]], [None, None])

    def test_partial_win_then_persistent_failure_is_only_workload_failed(self) -> None:
        module = _load_workload_evaluate()
        workload = _workload(module, cases=({"id": 0}, {"id": 1}))

        def runner(spec, *, candidate, role, case, timeout):
            if role == "candidate" and case["id"] == 1:
                raise RuntimeError("persistent")
            metrics = {"latency_ms": 100.0, "memory_mb": 100.0}
            if role == "candidate":
                metrics["latency_ms"] = 80.0
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            workload,
            "baseline.py",
            "candidate.py",
            blocks=2,
            retries=1,
            bootstrap_samples=50,
            runner=runner,
        )

        self.assertEqual(result["status"], "workload_failed")
        self.assertEqual(len(result["pairs"]), 2)
        self.assertTrue(result["pairs"][0]["valid"])
        self.assertFalse(result["pairs"][1]["valid"])
        self.assertEqual(result["pairs"][1]["attempts"]["candidate"], 2)
        self.assertNotIn("confirmed_win", json.dumps(result))

    def test_complete_pairs_classify_primary_and_constraints(self) -> None:
        module = _load_workload_evaluate()

        def runner(spec, *, candidate, role, case, timeout):
            metrics = (
                {"latency_ms": 100.0, "memory_mb": 100.0}
                if role == "baseline"
                else {"latency_ms": 90.0, "memory_mb": 103.0}
            )
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            _workload(module),
            "baseline.py",
            "candidate.py",
            blocks=4,
            retries=0,
            confidence=0.95,
            bootstrap_samples=100,
            runner=runner,
        )

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["primary"]["status"], "confirmed_win")
        self.assertEqual(result["primary"]["valid_pairs"], 4)
        self.assertEqual(result["constraints"][0]["status"], "passed")
        self.assertAlmostEqual(result["constraints"][0]["estimate_pct"], 3.0)
        self.assertEqual(result["objective"], _objective())

    def test_higher_primary_direction_is_respected(self) -> None:
        module = _load_workload_evaluate()
        objective = _objective(
            primary="throughput",
            direction="higher",
            constraints=[],
        )

        def runner(spec, *, candidate, role, case, timeout):
            metrics = {"throughput": 100.0 if role == "baseline" else 110.0}
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            _workload(module, objective=objective),
            "baseline.py",
            "candidate.py",
            blocks=3,
            retries=0,
            bootstrap_samples=50,
            runner=runner,
        )
        self.assertEqual(result["primary"]["status"], "confirmed_win")

    def test_missing_nonfinite_boolean_and_zero_metrics_fail_the_workload(self) -> None:
        module = _load_workload_evaluate()
        bad_candidate_metrics = (
            {"memory_mb": 100.0},
            {"latency_ms": math.nan, "memory_mb": 100.0},
            {"latency_ms": math.inf, "memory_mb": 100.0},
            {"latency_ms": True, "memory_mb": 100.0},
        )
        for bad_metrics in bad_candidate_metrics:
            with self.subTest(metrics=bad_metrics):

                def runner(spec, *, candidate, role, case, timeout):
                    metrics = (
                        {"latency_ms": 100.0, "memory_mb": 100.0}
                        if role == "baseline"
                        else bad_metrics
                    )
                    return _observation(
                        spec, role=role, case=case, metrics=metrics
                    )

                result = module.evaluate_pairs(
                    _workload(module),
                    "baseline.py",
                    "candidate.py",
                    blocks=1,
                    retries=0,
                    bootstrap_samples=20,
                    runner=runner,
                )
                self.assertEqual(result["status"], "workload_failed")
                self.assertNotIn("confirmed_win", json.dumps(result))
                json.dumps(result, allow_nan=False)

        for zero_metric in ("latency_ms", "memory_mb"):
            with self.subTest(zero_metric=zero_metric):

                def runner(spec, *, candidate, role, case, timeout):
                    metrics = {"latency_ms": 100.0, "memory_mb": 100.0}
                    if role == "baseline":
                        metrics[zero_metric] = 0.0
                    return _observation(
                        spec, role=role, case=case, metrics=metrics
                    )

                result = module.evaluate_pairs(
                    _workload(module),
                    "baseline.py",
                    "candidate.py",
                    blocks=1,
                    retries=0,
                    bootstrap_samples=20,
                    runner=runner,
                )
                self.assertEqual(result["status"], "workload_failed")

    def test_constraint_ci_equal_to_cap_passes(self) -> None:
        module = _load_workload_evaluate()

        def runner(spec, *, candidate, role, case, timeout):
            metrics = (
                {"latency_ms": 100.0, "memory_mb": 100.0}
                if role == "baseline"
                else {"latency_ms": 90.0, "memory_mb": 105.0}
            )
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            _workload(module),
            "baseline.py",
            "candidate.py",
            blocks=3,
            retries=0,
            bootstrap_samples=50,
            runner=runner,
        )
        constraint = result["constraints"][0]
        self.assertEqual(constraint["ci_high_pct"], 5.0)
        self.assertEqual(constraint["status"], "passed")

    def test_constraint_ci_crossing_cap_is_inconclusive(self) -> None:
        module = _load_workload_evaluate()
        workload = _workload(module, cases=({"id": 0}, {"id": 1}))

        def runner(spec, *, candidate, role, case, timeout):
            memory = 100.0
            if role == "candidate" and case["id"] == 1:
                memory = 110.0
            metrics = {
                "latency_ms": 100.0 if role == "baseline" else 90.0,
                "memory_mb": memory,
            }
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            workload,
            "baseline.py",
            "candidate.py",
            blocks=2,
            retries=0,
            confidence=0.9,
            bootstrap_samples=200,
            runner=runner,
        )
        constraint = result["constraints"][0]
        self.assertLessEqual(constraint["ci_low_pct"], 5.0)
        self.assertGreater(constraint["ci_high_pct"], 5.0)
        self.assertEqual(constraint["status"], "inconclusive")

    def test_finite_metrics_with_overflowing_regression_fail_json_safely(self) -> None:
        module = _load_workload_evaluate()

        def runner(spec, *, candidate, role, case, timeout):
            metrics = {
                "latency_ms": 100.0 if role == "baseline" else 90.0,
                "memory_mb": 1e-308 if role == "baseline" else 1e308,
            }
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            _workload(module),
            "baseline.py",
            "candidate.py",
            blocks=2,
            retries=0,
            bootstrap_samples=20,
            runner=runner,
        )

        self.assertEqual(result["status"], "workload_failed")
        self.assertTrue(all(not pair["valid"] for pair in result["pairs"]))
        for pair in result["pairs"]:
            error = pair["metric_errors"][0]
            self.assertEqual(error["error_type"], "ValueError")
            self.assertLessEqual(len(error["reason"]), 512)
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("confirmed_win", serialized)

    def test_primary_median_overflow_becomes_json_safe_workload_failure(self) -> None:
        module = _load_workload_evaluate()
        objective = _objective(
            primary="throughput",
            direction="higher",
            constraints=[],
        )

        def runner(spec, *, candidate, role, case, timeout):
            metrics = {
                "throughput": 1e-306 if role == "baseline" else 1.0,
            }
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            _workload(module, objective=objective),
            "baseline.py",
            "candidate.py",
            blocks=2,
            retries=0,
            bootstrap_samples=20,
            runner=runner,
        )

        self.assertEqual(result["status"], "workload_failed")
        self.assertTrue(all(not pair["valid"] for pair in result["pairs"]))
        self.assertTrue(all("invalid_reason" in pair for pair in result["pairs"]))
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("confirmed_win", serialized)

    def test_constraint_median_overflow_becomes_json_safe_workload_failure(
        self,
    ) -> None:
        module = _load_workload_evaluate()

        def runner(spec, *, candidate, role, case, timeout):
            metrics = {
                "latency_ms": 100.0 if role == "baseline" else 90.0,
                "memory_mb": 1e-306 if role == "baseline" else 1.0,
            }
            return _observation(spec, role=role, case=case, metrics=metrics)

        result = module.evaluate_pairs(
            _workload(module),
            "baseline.py",
            "candidate.py",
            blocks=2,
            retries=0,
            bootstrap_samples=20,
            runner=runner,
        )

        self.assertEqual(result["status"], "workload_failed")
        self.assertTrue(all(not pair["valid"] for pair in result["pairs"]))
        self.assertTrue(all("invalid_reason" in pair for pair in result["pairs"]))
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("confirmed_win", serialized)

    def test_blocks_are_positive_literal_integers(self) -> None:
        module = _load_workload_evaluate()
        for blocks in (True, 0, -1, 1.0, "1"):
            with self.subTest(blocks=blocks), self.assertRaisesRegex(
                ValueError, "blocks"
            ):
                module.evaluate_pairs(
                    _workload(module),
                    "baseline.py",
                    "candidate.py",
                    blocks=blocks,
                    runner=lambda *args, **kwargs: None,
                )

    def test_evaluation_does_not_mutate_candidates_cases_or_objective(self) -> None:
        module = _load_workload_evaluate()
        baseline = {"path": ["baseline.py"]}
        candidate = {"path": ["candidate.py"]}
        cases = ({"shape": [16]},)
        objective = _objective()
        workload = _workload(module, objective=objective, cases=cases)
        before = (copy.deepcopy(baseline), copy.deepcopy(candidate), copy.deepcopy(objective))

        def runner(spec, *, candidate, role, case, timeout):
            candidate["path"].append("mutated")
            case["shape"].append(32)
            metrics = {"latency_ms": 100.0, "memory_mb": 100.0}
            if role == "candidate":
                metrics["latency_ms"] = 90.0
            return _observation(spec, role=role, case=case, metrics=metrics)

        module.evaluate_pairs(
            workload,
            baseline,
            candidate,
            blocks=1,
            retries=0,
            bootstrap_samples=20,
            runner=runner,
        )
        self.assertEqual((baseline, candidate, objective), before)


if __name__ == "__main__":
    unittest.main()
