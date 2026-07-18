from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_readiness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReadinessTests(unittest.TestCase):
    def test_source_only_stops_at_static_hypotheses(self) -> None:
        report = load_module().assess({"source_available": True})
        self.assertEqual(report["claim_ceiling"], "static_hypotheses")
        self.assertFalse(report["can_start_mutation"])
        self.assertIn("correctness_reference", report["missing"])
        self.assertIn("stable_kernel_benchmark", report["missing"])
        self.assertNotIn("representative_workload", report["missing"])
        self.assertNotIn("serving_benchmark", report["missing"])

    def test_kernel_environment_allows_only_kernel_claim(self) -> None:
        report = load_module().assess(
            {
                "source_available": True,
                "correctness_reference": True,
                "compile_command": True,
                "stable_kernel_benchmark": True,
            }
        )
        self.assertEqual(report["claim_ceiling"], "kernel_performance")
        self.assertTrue(report["can_start_mutation"])
        self.assertEqual(report["missing"], [])

    def test_workload_target_requests_workload_but_not_serving_foundation(self) -> None:
        report = load_module().assess(
            {
                "requested_claim": "workload",
                "source_available": True,
                "correctness_reference": True,
                "compile_command": True,
                "stable_kernel_benchmark": True,
            }
        )
        self.assertEqual(report["claim_ceiling"], "kernel_performance")
        self.assertEqual(report["missing"], ["representative_workload"])
        self.assertNotIn("serving_benchmark", report["missing"])

    def test_serving_claim_requires_every_lower_layer(self) -> None:
        report = load_module().assess(
            {
                "source_available": True,
                "requested_claim": "serving",
                "correctness_reference": True,
                "compile_command": True,
                "stable_kernel_benchmark": True,
                "representative_workload": True,
                "serving_benchmark": True,
            }
        )
        self.assertEqual(report["claim_ceiling"], "serving_performance")
        self.assertEqual(report["host_change_policy"], "recommend_only")

    def test_invalid_requested_claim_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_module().assess({"requested_claim": "unbounded"})


if __name__ == "__main__":
    unittest.main()
