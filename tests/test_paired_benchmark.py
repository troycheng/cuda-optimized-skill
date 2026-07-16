from __future__ import annotations

import copy
import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
PAIRED_BENCHMARK_PATH = SCRIPTS / "paired_benchmark.py"


def _load_paired_benchmark():
    scripts = str(SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_paired_benchmark", PAIRED_BENCHMARK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PairedBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _dependencies(timings=None, telemetry_readings=None):
        calls = {"prepare": [], "warm": [], "measure": []}
        timing_values = iter(timings or [10.0, 9.0] * 20)
        telemetry_values = iter(
            telemetry_readings
            or [
                {"available": True, "temperature_c": 60, "sm_clock_mhz": 2000},
                {"available": True, "temperature_c": 61, "sm_clock_mhz": 2010},
            ]
            * 20
        )

        def prepare(solution_file, **kwargs):
            state = {
                "name": solution_file,
                "counter": 0,
                "dims_seen": kwargs["dims"],
            }
            calls["prepare"].append((solution_file, kwargs, state))
            return state

        def warm(state, warmup):
            calls["warm"].append((state["name"], warmup))

        def measure(state):
            state["counter"] += 1
            calls["measure"].append((state["name"], state["counter"]))
            return next(timing_values)

        def read_telemetry():
            return next(telemetry_values)

        return calls, prepare, warm, measure, read_telemetry

    def test_fixed_seed_is_reproducible_and_contains_both_orders(self) -> None:
        paired = _load_paired_benchmark()

        def run_once():
            deps = self._dependencies()
            return paired.run_paired(
                "baseline.py",
                "candidate.py",
                backend="triton",
                dims={"n": 128},
                ptr_size=0,
                arch="sm_120",
                nvcc_bin="nvcc",
                seed=7,
                blocks=8,
                warmup=2,
                prepare_fn=deps[1],
                warm_fn=deps[2],
                measure_fn=deps[3],
                telemetry_reader=deps[4],
            )

        first = run_once()
        second = run_once()
        first_orders = [item["order"] for item in first["pairs"]]
        second_orders = [item["order"] for item in second["pairs"]]
        self.assertEqual(first_orders, second_orders)
        self.assertEqual(set(first_orders), {"AB", "BA"})

    def test_pair_order_rng_does_not_change_global_random_state(self) -> None:
        paired = _load_paired_benchmark()
        deps = self._dependencies()
        random.seed(9182)
        state_before = random.getstate()

        paired.run_paired(
            "baseline.py",
            "candidate.py",
            backend="triton",
            dims={},
            ptr_size=0,
            arch="sm_120",
            nvcc_bin="nvcc",
            seed=7,
            blocks=3,
            warmup=0,
            prepare_fn=deps[1],
            warm_fn=deps[2],
            measure_fn=deps[3],
            telemetry_reader=deps[4],
        )

        self.assertEqual(random.getstate(), state_before)

    def test_each_solution_is_measured_once_per_block_without_state_cross_talk(self) -> None:
        paired = _load_paired_benchmark()
        calls, prepare, warm, measure, read_telemetry = self._dependencies()
        dims = {"n": 128, "nested": {"tile": 16}}
        original_dims = copy.deepcopy(dims)

        result = paired.run_paired(
            "baseline.py",
            "candidate.py",
            backend="triton",
            dims=dims,
            ptr_size=0,
            arch="sm_120",
            nvcc_bin="nvcc",
            seed=7,
            blocks=4,
            warmup=3,
            prepare_fn=prepare,
            warm_fn=warm,
            measure_fn=measure,
            telemetry_reader=read_telemetry,
        )

        self.assertEqual(calls["warm"], [("baseline.py", 3), ("candidate.py", 3)])
        self.assertEqual(
            sorted(name for name, _count in calls["measure"]),
            ["baseline.py"] * 4 + ["candidate.py"] * 4,
        )
        self.assertEqual([entry[2]["counter"] for entry in calls["prepare"]], [4, 4])
        self.assertIsNot(calls["prepare"][0][2], calls["prepare"][1][2])
        self.assertIsNot(
            calls["prepare"][0][1]["dims"], calls["prepare"][1][1]["dims"]
        )
        self.assertEqual(dims, original_dims)
        self.assertEqual(len(result["pairs"]), 4)

    def test_telemetry_validation_applies_to_the_whole_pair(self) -> None:
        paired = _load_paired_benchmark()
        readings = [
            {"available": True, "temperature_c": 60, "sm_clock_mhz": 2000},
            {"available": True, "temperature_c": 67, "sm_clock_mhz": 2140},
        ]
        deps = self._dependencies(telemetry_readings=readings)

        result = paired.run_paired(
            "baseline.py",
            "candidate.py",
            backend="triton",
            dims={},
            ptr_size=0,
            arch="sm_120",
            nvcc_bin="nvcc",
            seed=1,
            blocks=1,
            warmup=0,
            prepare_fn=deps[1],
            warm_fn=deps[2],
            measure_fn=deps[3],
            telemetry_reader=deps[4],
        )

        pair = result["pairs"][0]
        self.assertFalse(pair["valid"])
        self.assertEqual(
            pair["invalid_reasons"], ["temperature_delta", "clock_delta"]
        )
        self.assertEqual(pair["telemetry"]["before"], readings[0])
        self.assertEqual(pair["telemetry"]["after"], readings[1])

    def test_missing_telemetry_is_preserved_without_fabricating_invalidity(self) -> None:
        paired = _load_paired_benchmark()
        readings = [
            {"available": False, "reason": "nvidia_smi_unavailable"},
            {"available": False, "reason": "nvidia_smi_unavailable"},
        ]
        deps = self._dependencies(telemetry_readings=readings)

        result = paired.run_paired(
            "baseline.py",
            "candidate.py",
            backend="triton",
            dims={},
            ptr_size=0,
            arch="sm_120",
            nvcc_bin="nvcc",
            seed=1,
            blocks=1,
            warmup=0,
            prepare_fn=deps[1],
            warm_fn=deps[2],
            measure_fn=deps[3],
            telemetry_reader=deps[4],
        )

        pair = result["pairs"][0]
        self.assertTrue(pair["valid"])
        self.assertEqual(pair["invalid_reasons"], [])
        self.assertEqual(pair["telemetry"]["status"], "unknown")
        self.assertFalse(pair["telemetry"]["before"]["available"])

    def test_invalid_counts_and_seed_are_rejected_before_preparing(self) -> None:
        paired = _load_paired_benchmark()
        for parameter, value in (
            ("blocks", 0),
            ("blocks", -1),
            ("blocks", True),
            ("blocks", 1.5),
            ("warmup", -1),
            ("warmup", True),
            ("warmup", 1.5),
            ("seed", True),
            ("seed", 1.5),
        ):
            kwargs = {"blocks": 1, "warmup": 0, "seed": 1}
            kwargs[parameter] = value
            called = []
            with self.subTest(parameter=parameter, value=value), self.assertRaisesRegex(
                ValueError, parameter
            ):
                paired.run_paired(
                    "baseline.py",
                    "candidate.py",
                    backend="triton",
                    dims={},
                    ptr_size=0,
                    arch="sm_120",
                    nvcc_bin="nvcc",
                    prepare_fn=lambda *_args, **_kwargs: called.append(True),
                    **kwargs,
                )
            self.assertEqual(called, [])

    def test_invalid_timings_are_named_value_errors(self) -> None:
        paired = _load_paired_benchmark()
        for value in (0, -1, True, "1", math.nan, math.inf, -math.inf):
            deps = self._dependencies(timings=[value, 1.0])
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "timing"
            ):
                paired.run_paired(
                    "baseline.py",
                    "candidate.py",
                    backend="triton",
                    dims={},
                    ptr_size=0,
                    arch="sm_120",
                    nvcc_bin="nvcc",
                    seed=1,
                    blocks=1,
                    warmup=0,
                    prepare_fn=deps[1],
                    warm_fn=deps[2],
                    measure_fn=deps[3],
                    telemetry_reader=deps[4],
                )

    def test_invalid_configuration_objects_are_rejected(self) -> None:
        paired = _load_paired_benchmark()
        base = dict(
            backend="triton",
            dims={},
            ptr_size=0,
            arch="sm_120",
            nvcc_bin="nvcc",
            seed=1,
            blocks=1,
            warmup=0,
        )
        cases = (
            ("dims", [], "dims"),
            ("ptr_size", True, "ptr_size"),
            ("ptr_size", -1, "ptr_size"),
            ("max_temperature_delta_c", math.nan, "max_temperature_delta_c"),
            ("max_clock_delta_pct", math.inf, "max_clock_delta_pct"),
        )
        for name, value, message in cases:
            kwargs = dict(base)
            kwargs[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                paired.run_paired("baseline.py", "candidate.py", **kwargs)


if __name__ == "__main__":
    unittest.main()
