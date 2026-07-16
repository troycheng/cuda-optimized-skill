from __future__ import annotations

import importlib.util
import json
import math
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_benchmark", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeEvent:
    def __init__(self, cuda, is_start: bool) -> None:
        self.cuda = cuda
        self.is_start = is_start

    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end_event) -> float:
        del end_event
        return self.cuda.elapsed_values.pop(0)


class _FakeCuda:
    def __init__(self, elapsed_values) -> None:
        self.elapsed_values = list(elapsed_values)
        self.event_count = 0
        self.sync_count = 0

    def Event(self, enable_timing: bool):
        self.assert_timing = enable_timing
        event = _FakeEvent(self, is_start=self.event_count % 2 == 0)
        self.event_count += 1
        return event

    def synchronize(self) -> None:
        self.sync_count += 1


class BenchmarkTests(unittest.TestCase):
    def test_help_works_without_site_packages(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", str(BENCHMARK_PATH), "--help"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_timing_returns_independent_samples(self) -> None:
        benchmark = _load_benchmark()
        fake_cuda = _FakeCuda([1.0, 2.0, 3.0])
        calls = []
        values = benchmark._time_iterations(
            lambda: calls.append("kernel"), warmup=0, repeat=3, cuda=fake_cuda
        )
        self.assertEqual(values, [1.0, 2.0, 3.0])
        self.assertEqual(calls, ["kernel", "kernel", "kernel"])

    def test_even_sample_median_averages_middle_values(self) -> None:
        benchmark = _load_benchmark()
        _average, median, _minimum, _maximum = benchmark._stats([1.0, 2.0, 100.0, 200.0])
        self.assertEqual(median, 51.0)

    def test_stats_dict_preserves_samples_and_distribution(self) -> None:
        benchmark = _load_benchmark()
        samples = [1.0, 2.0, 3.0, 4.0]
        result = benchmark._stats_dict(samples)
        self.assertEqual(result["samples_ms"], samples)
        self.assertEqual(result["median_ms"], 2.5)
        self.assertAlmostEqual(result["p95_ms"], 4.0)
        self.assertAlmostEqual(result["stddev_ms"], statistics.pstdev(samples))
        self.assertAlmostEqual(
            result["cv_pct"], statistics.pstdev(samples) / 2.5 * 100
        )

    def test_prepare_solution_delegates_to_existing_backend_setup(self) -> None:
        benchmark = _load_benchmark()
        expected = {"state": "prepared"}
        calls = []

        def setup(*args, **kwargs):
            calls.append((args, kwargs))
            return expected

        benchmark._require_torch = lambda: None
        benchmark._setup_backend = setup
        result = benchmark.prepare_solution(
            "kernel.py",
            backend="triton",
            dims={"n": 128},
            ptr_size=0,
            arch="sm_120",
            nvcc_bin="nvcc",
            seed=17,
        )

        self.assertIs(result, expected)
        self.assertEqual(
            calls,
            [
                (
                    ("kernel.py", "triton", {"n": 128}, 0, "sm_120", "nvcc"),
                    {"seed": 17},
                )
            ],
        )

    def test_prepare_solution_initializes_torch_before_real_triton_setup(self) -> None:
        benchmark = _load_benchmark()
        self.assertIsNone(benchmark.torch)
        initialized = []

        class FakeTensor:
            pass

        fake_torch = SimpleNamespace(Tensor=FakeTensor)

        def require_torch():
            initialized.append(True)
            benchmark.torch = fake_torch
            return fake_torch

        benchmark._require_torch = require_torch
        source = (
            "def setup(**kwargs):\n"
            "    return {'inputs': {'n': kwargs['n']}, 'outputs': []}\n"
            "def run_kernel(**kwargs):\n"
            "    return None\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            solution = Path(tmp) / "kernel.py"
            solution.write_text(source, encoding="utf-8")
            state = benchmark.prepare_solution(
                str(solution),
                backend="triton",
                dims={"n": 128},
                ptr_size=0,
                arch="sm_120",
                nvcc_bin="nvcc",
                seed=17,
            )

        self.assertEqual(initialized, [True])
        self.assertEqual(state["backend"], "triton")
        self.assertEqual(state["reference_inputs"], {"n": 128})

    def test_warm_solution_resets_before_every_call_and_synchronizes(self) -> None:
        benchmark = _load_benchmark()
        events = []
        benchmark.torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: events.append("sync"))
        )
        benchmark._reset_tensor_inputs = lambda state: events.append(
            f"reset:{state['name']}"
        )
        state = {"name": "candidate", "callable": lambda: events.append("call")}

        benchmark.warm_solution(state, 3)

        self.assertEqual(
            events,
            ["reset:candidate", "call"] * 3 + ["sync"],
        )

    def test_warm_solution_rejects_invalid_counts(self) -> None:
        benchmark = _load_benchmark()
        for value in (-1, True, 1.5, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "warmup"
            ):
                benchmark.warm_solution({"callable": lambda: None}, value)

    def test_measure_once_resets_and_returns_one_valid_timing(self) -> None:
        benchmark = _load_benchmark()
        events = []
        state = {"callable": lambda: events.append("unexpected direct call")}
        cuda = object()
        benchmark._reset_tensor_inputs = lambda actual: events.append(
            ("reset", actual)
        )

        def time_iterations(fn, *, warmup, repeat, cuda):
            self.assertIs(fn, state["callable"])
            self.assertEqual((warmup, repeat), (0, 1))
            self.assertIs(cuda, cuda_arg)
            return [0.25]

        cuda_arg = cuda
        benchmark._time_iterations = time_iterations
        self.assertEqual(benchmark.measure_once(state, cuda=cuda), 0.25)
        self.assertEqual(events, [("reset", state)])

    def test_measure_once_rejects_invalid_elapsed_time(self) -> None:
        benchmark = _load_benchmark()
        benchmark._reset_tensor_inputs = lambda _state: None
        for value in (0, 0.0, -1, True, "1", math.nan, math.inf, -math.inf):
            benchmark._time_iterations = lambda *_args, **_kwargs: [value]
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "timing"
            ):
                benchmark.measure_once({"callable": lambda: None})

    def test_setup_failure_json_keeps_existing_top_level_schema(self) -> None:
        benchmark = _load_benchmark()
        benchmark.torch = SimpleNamespace(
            cuda=SimpleNamespace(
                current_device=lambda: 0,
                get_device_name=lambda _index: "fake-gpu",
            )
        )
        benchmark._setup_backend = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("setup broke")
        )
        expected_keys = {
            "solution_file",
            "cu_file",
            "backend",
            "ref_file",
            "has_reference",
            "dims",
            "warmup",
            "repeat",
            "ptr_size_override",
            "gpu_index",
            "gpu_name",
            "arch",
            "seed",
            "correctness",
            "kernel",
            "reference",
            "speedup_vs_reference",
            "error",
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            with self.assertRaisesRegex(ValueError, "setup broke"):
                benchmark.run(
                    "kernel.cu",
                    "",
                    {},
                    0,
                    1,
                    0,
                    "sm_120",
                    1e-4,
                    1e-3,
                    42,
                    json_out=str(output),
                    backend="cuda",
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(set(payload), expected_keys)
        self.assertEqual(payload["error"]["code"], "setup_value_error")


if __name__ == "__main__":
    unittest.main()
