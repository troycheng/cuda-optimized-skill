from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "telemetry.py"
)


def _load_telemetry():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_telemetry", TELEMETRY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReadGpuTelemetryTests(unittest.TestCase):
    def test_queries_all_metrics_once_and_parses_first_gpu(self) -> None:
        telemetry = _load_telemetry()
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="61, 2100, 312.5, 4096, 87\n",
            stderr="",
        )

        with mock.patch.object(
            telemetry.subprocess, "run", return_value=completed
        ) as run:
            result = telemetry.read_gpu_telemetry(timeout=1.25)

        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "nvidia-smi")
        self.assertEqual(
            command[1],
            "--query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu",
        )
        self.assertEqual(command[2], "--format=csv,noheader,nounits")
        self.assertEqual(
            result,
            {
                "available": True,
                "temperature_c": 61.0,
                "sm_clock_mhz": 2100.0,
                "power_w": 312.5,
                "memory_used_mb": 4096.0,
                "gpu_utilization_pct": 87.0,
            },
        )

    def test_command_failures_are_recorded_without_raising(self) -> None:
        telemetry = _load_telemetry()
        cases = (
            (FileNotFoundError("missing"), "unavailable"),
            (subprocess.TimeoutExpired("nvidia-smi", 1), "timeout"),
        )
        for error, reason in cases:
            with self.subTest(error=type(error).__name__), mock.patch.object(
                telemetry.subprocess, "run", side_effect=error
            ):
                result = telemetry.read_gpu_telemetry()
            self.assertFalse(result["available"])
            self.assertIn(reason, result["reason"])
            self.assertNotIn("temperature_c", result)

    def test_nonzero_exit_and_parse_failure_are_unavailable(self) -> None:
        telemetry = _load_telemetry()
        cases = (
            (
                subprocess.CompletedProcess([], 9, stdout="", stderr="denied"),
                "exit_9",
            ),
            (
                subprocess.CompletedProcess([], 0, stdout="not,numeric\n", stderr=""),
                "parse_error",
            ),
        )
        for completed, reason in cases:
            with self.subTest(reason=reason), mock.patch.object(
                telemetry.subprocess, "run", return_value=completed
            ):
                result = telemetry.read_gpu_telemetry()
            self.assertFalse(result["available"])
            self.assertIn(reason, result["reason"])

    def test_physically_invalid_command_values_are_unavailable(self) -> None:
        telemetry = _load_telemetry()
        for stdout in (
            "60, -1, 300, 4096, 80\n",
            "60, 2100, 300, 4096, 101\n",
        ):
            completed = subprocess.CompletedProcess(
                [], 0, stdout=stdout, stderr=""
            )
            with self.subTest(stdout=stdout), mock.patch.object(
                telemetry.subprocess, "run", return_value=completed
            ):
                result = telemetry.read_gpu_telemetry()
            self.assertFalse(result["available"])
            self.assertIn("parse_error", result["reason"])

    def test_timeout_must_be_a_positive_finite_real(self) -> None:
        telemetry = _load_telemetry()
        for value in (0, -1, True, "1", math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "timeout"
            ):
                telemetry.read_gpu_telemetry(timeout=value)


class ValidateBlockTests(unittest.TestCase):
    @staticmethod
    def reading(temperature: float = 60, clock: float = 2000) -> dict:
        return {
            "available": True,
            "temperature_c": temperature,
            "sm_clock_mhz": clock,
        }

    def test_temperature_and_clock_drift_invalidate_whole_block(self) -> None:
        telemetry = _load_telemetry()
        result = telemetry.validate_block(
            self.reading(60, 2000),
            self.reading(66, 2120),
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["invalid_reasons"], ["temperature_delta", "clock_delta"]
        )
        self.assertEqual(result["telemetry_status"], "available")

    def test_threshold_equality_remains_valid(self) -> None:
        telemetry = _load_telemetry()
        result = telemetry.validate_block(
            self.reading(60, 2000),
            self.reading(65, 2100),
            max_temperature_delta_c=5,
            max_clock_delta_pct=5,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["invalid_reasons"], [])
        self.assertEqual(result["temperature_delta_c"], 5.0)
        self.assertEqual(result["clock_delta_pct"], 5.0)

    def test_unavailable_telemetry_is_unknown_but_not_invalid(self) -> None:
        telemetry = _load_telemetry()
        before = {"available": False, "reason": "nvidia_smi_unavailable"}
        after = self.reading()
        originals = copy.deepcopy((before, after))

        result = telemetry.validate_block(before, after)

        self.assertTrue(result["valid"])
        self.assertEqual(result["invalid_reasons"], [])
        self.assertEqual(result["telemetry_status"], "unknown")
        self.assertIn("before_unavailable", result["unknown_reasons"])
        self.assertEqual((before, after), originals)

    def test_metrics_without_available_flag_still_gate_real_drift(self) -> None:
        telemetry = _load_telemetry()
        result = telemetry.validate_block(
            {"temperature_c": 60, "sm_clock_mhz": 2500},
            {"temperature_c": 67, "sm_clock_mhz": 2250},
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["invalid_reasons"], ["temperature_delta", "clock_delta"]
        )
        self.assertEqual(result["telemetry_status"], "available")

    def test_empty_mapping_is_unknown_not_invalid(self) -> None:
        telemetry = _load_telemetry()
        result = telemetry.validate_block({}, {})

        self.assertTrue(result["valid"])
        self.assertEqual(result["telemetry_status"], "unknown")
        self.assertEqual(
            result["unknown_reasons"], ["before_no_metrics", "after_no_metrics"]
        )

    def test_missing_or_zero_clock_is_unknown_without_dividing_by_zero(self) -> None:
        telemetry = _load_telemetry()
        for before in (self.reading(clock=0), {"available": True, "temperature_c": 60}):
            with self.subTest(before=before):
                result = telemetry.validate_block(before, self.reading(61, 2000))
            self.assertTrue(result["valid"])
            self.assertIsNone(result["clock_delta_pct"])
            self.assertIn("clock_delta_unknown", result["unknown_reasons"])

    def test_invalid_thresholds_are_named_value_errors(self) -> None:
        telemetry = _load_telemetry()
        for name in ("max_temperature_delta_c", "max_clock_delta_pct"):
            for value in (-1, True, "5", math.nan, math.inf, -math.inf):
                kwargs = {name: value}
                with self.subTest(name=name, value=value), self.assertRaisesRegex(
                    ValueError, name
                ):
                    telemetry.validate_block(
                        self.reading(), self.reading(), **kwargs
                    )

    def test_invalid_existing_numeric_values_are_rejected(self) -> None:
        telemetry = _load_telemetry()
        for field in ("temperature_c", "sm_clock_mhz"):
            for value in (True, "60", math.nan, math.inf, -math.inf):
                before = self.reading()
                before[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    ValueError, field
                ):
                    telemetry.validate_block(before, self.reading())

    def test_extreme_nonnegative_temperature_delta_is_invalid(self) -> None:
        telemetry = _load_telemetry()
        result = telemetry.validate_block(
            self.reading(temperature=0),
            self.reading(temperature=1e308),
        )

        self.assertFalse(result["valid"])
        self.assertIn("temperature_delta", result["invalid_reasons"])
        self.assertTrue(math.isfinite(result["temperature_delta_c"]))

    def test_extreme_finite_clock_delta_is_invalid_and_json_safe(self) -> None:
        telemetry = _load_telemetry()
        result = telemetry.validate_block(
            self.reading(clock=1e-308),
            self.reading(clock=1e308),
        )

        self.assertFalse(result["valid"])
        self.assertIn("clock_delta", result["invalid_reasons"])
        self.assertTrue(result["clock_delta_capped"])
        self.assertTrue(math.isfinite(result["clock_delta_pct"]))
        self.assertGreater(result["clock_delta_pct"], 5)
        self.assertIsInstance(json.dumps(result, allow_nan=False), str)

    def test_negative_physical_metrics_and_utilization_above_100_are_rejected(
        self,
    ) -> None:
        telemetry = _load_telemetry()
        for field in (
            "temperature_c",
            "sm_clock_mhz",
            "power_w",
            "memory_used_mb",
            "gpu_utilization_pct",
        ):
            before = self.reading()
            before[field] = -0.1
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field
            ):
                telemetry.validate_block(before, self.reading())

        before = self.reading()
        before["gpu_utilization_pct"] = 100.1
        with self.assertRaisesRegex(ValueError, "gpu_utilization_pct"):
            telemetry.validate_block(before, self.reading())

    def test_inputs_must_be_mappings_and_available_must_be_bool(self) -> None:
        telemetry = _load_telemetry()
        with self.assertRaisesRegex(ValueError, "before"):
            telemetry.validate_block([], self.reading())
        with self.assertRaisesRegex(ValueError, "after.available"):
            telemetry.validate_block(self.reading(), {"available": 1})


if __name__ == "__main__":
    unittest.main()
