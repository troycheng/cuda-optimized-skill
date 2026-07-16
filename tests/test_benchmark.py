from __future__ import annotations

import importlib.util
import statistics
import subprocess
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
