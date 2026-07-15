from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOFLINE_PATH = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "roofline.py"


def _load_roofline():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_roofline", ROOFLINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NCU_FIXTURE = {
    "degraded": False,
    "compute": [
        {
            "name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "value": 30.0,
        }
    ],
    "memory": [
        {
            "name": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "value": 60.0,
        }
    ],
    "latency": [
        {
            "name": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "value": 25.0,
        }
    ],
}


class RooflineTests(unittest.TestCase):
    def test_single_axis_does_not_allocate_to_zero_gap_axes(self) -> None:
        roofline = _load_roofline()
        self.assertEqual(
            roofline.allocate_budget(1.0, 0.0, 0.0),
            {"compute": 2, "memory": 0, "latency": 0},
        )

    def test_negligible_gaps_allocate_no_methods(self) -> None:
        roofline = _load_roofline()
        self.assertEqual(
            roofline.allocate_budget(0.05, 0.05, 0.05),
            {"compute": 0, "memory": 0, "latency": 0},
        )

    def test_equal_gaps_split_the_three_method_budget(self) -> None:
        roofline = _load_roofline()
        self.assertEqual(
            roofline.allocate_budget(1.0, 1.0, 1.0),
            {"compute": 1, "memory": 1, "latency": 1},
        )

    def test_dominant_axis_cap_preserves_smaller_evidenced_axis(self) -> None:
        roofline = _load_roofline()
        self.assertEqual(
            roofline.allocate_budget(10.0, 1.0, 0.0),
            {"compute": 2, "memory": 1, "latency": 0},
        )

    def test_unknown_peak_data_is_reported_as_heuristic(self) -> None:
        roofline = _load_roofline()
        result = roofline.compute_deltas(
            NCU_FIXTURE,
            {"gpus": [{"sm_arch": "sm_120"}]},
        )
        self.assertEqual(result["analysis_model"], "utilization_gap")
        self.assertEqual(result["analysis_quality"], "heuristic")
        self.assertIsNone(result["ai_ridge"])
        self.assertIsNone(result["arithmetic_intensity"])

    def test_explicit_workload_and_peaks_enable_measured_roofline(self) -> None:
        roofline = _load_roofline()
        ncu = {
            **NCU_FIXTURE,
            "workload": {
                "flops": 2.0e12,
                "bytes": 1.0e11,
                "kernel_time_ms": 10.0,
            },
        }
        env = {
            "gpus": [
                {
                    "sm_arch": "sm_90",
                    "peak_flops_tflops": 1000.0,
                    "peak_bw_gbs": 3000.0,
                }
            ]
        }
        result = roofline.compute_deltas(ncu, env)
        self.assertEqual(result["analysis_quality"], "measured_roofline")
        self.assertAlmostEqual(result["arithmetic_intensity"], 20.0)
        self.assertAlmostEqual(result["achieved_tflops"], 200.0)
        self.assertEqual(result["roofline_bound"], "bandwidth")

    def test_missing_axis_metric_is_not_invented_as_a_full_gap(self) -> None:
        roofline = _load_roofline()
        result = roofline.compute_deltas(
            {"degraded": False, "compute": [], "memory": [], "latency": []},
            {"gpus": [{"sm_arch": "sm_120"}]},
        )
        self.assertEqual(result["delta_compute"], 0.0)
        self.assertEqual(result["delta_memory"], 0.0)
        self.assertEqual(result["delta_latency"], 0.0)
        self.assertEqual(result["analysis_quality"], "unavailable")


if __name__ == "__main__":
    unittest.main()
