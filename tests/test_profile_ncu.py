from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cuda_optimizer_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeProfiler:
    def __init__(self, calls) -> None:
        self.calls = calls

    def cudaProfilerStart(self) -> None:
        self.calls.append("start")

    def cudaProfilerStop(self) -> None:
        self.calls.append("stop")


class _FakeCuda:
    def __init__(self, calls) -> None:
        self.calls = calls

    def synchronize(self) -> None:
        self.calls.append("sync")


class ProfileNcuTests(unittest.TestCase):
    def test_ncu_profiles_only_the_explicit_target_range(self) -> None:
        profile_ncu = _load("profile_ncu")
        cmd = profile_ncu._build_profile_command(
            ncu_bin="ncu",
            rep_path="out.ncu-rep",
            benchmark_py="benchmark.py",
            solution="kernel.py",
            dims={"M": 128},
            warmup=3,
            launch_count=1,
        )
        self.assertEqual(cmd[cmd.index("--profile-from-start") + 1], "off")
        self.assertEqual(cmd[cmd.index("--launch-count") + 1], "1")
        self.assertIn("--profile-only", cmd)
        self.assertIn("--target-processes", cmd)

    def test_profile_target_uses_start_then_one_call_then_stop(self) -> None:
        benchmark = _load("benchmark")
        calls = []
        benchmark._profile_target_once(
            lambda: calls.append("kernel"),
            profiler=_FakeProfiler(calls),
            cuda=_FakeCuda(calls),
        )
        self.assertEqual(calls, ["start", "kernel", "sync", "stop"])

    def test_ncu_metric_query_does_not_claim_counter_permission(self) -> None:
        check_env = _load("check_env")
        with mock.patch.object(check_env.shutil, "which", return_value="/usr/bin/ncu"), mock.patch.object(
            check_env,
            "_run",
            side_effect=[(0, "NVIDIA Nsight Compute 13.3", ""), (0, "metric", "")],
        ):
            result = check_env._detect_ncu()
        self.assertTrue(result["metrics_query_available"])
        self.assertIsNone(result["can_read_counters"])

    def test_long_form_csv_is_normalized(self) -> None:
        profile_ncu = _load("profile_ncu")
        text = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"target","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","75"\n'
        )
        rows = profile_ncu._parse_ncu_csv(text)
        self.assertEqual(rows[0]["Metric Name"], "dram__throughput.avg.pct_of_peak_sustained_elapsed")

    def test_wide_form_csv_selects_target_kernel(self) -> None:
        profile_ncu = _load("profile_ncu")
        text = (
            '"Kernel Name","gpu__time_duration.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed"\n'
            '"rng_setup","100","10"\n'
            '"target_kernel","20","75"\n'
        )
        rows = profile_ncu._parse_ncu_csv(text, kernel_name_hints=["target_kernel"])
        self.assertTrue(rows)
        self.assertEqual({row["Kernel Name"] for row in rows}, {"target_kernel"})


if __name__ == "__main__":
    unittest.main()
