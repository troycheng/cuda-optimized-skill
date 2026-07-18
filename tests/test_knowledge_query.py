from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "knowledge_query.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_knowledge", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class KnowledgeQueryTests(unittest.TestCase):
    def test_query_returns_small_arch_compatible_method_set(self) -> None:
        result = load_module().query(arch="sm_120", axis="compute", limit=3)
        self.assertLessEqual(len(result["methods"]), 3)
        self.assertEqual(result["arch"], "sm_120")
        self.assertNotIn(
            "compute.gemm_softmax_interleave",
            {item["id"] for item in result["methods"]},
        )
        for item in result["methods"]:
            self.assertNotIn("typical_speedup", item)
            self.assertEqual(item["applicability"], "unverified")

    def test_observed_bad_metric_ranks_matching_method_first(self) -> None:
        result = load_module().query(
            arch="sm_120",
            axis="compute",
            observed_metrics={
                "sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak": 10
            },
            limit=3,
        )
        self.assertEqual(result["methods"][0]["id"], "compute.tensor_core")
        self.assertEqual(
            result["methods"][0]["applicability"], "observed_bad_trigger"
        )

    def test_unknown_arch_fails_closed_instead_of_numeric_inheritance(self) -> None:
        with self.assertRaises(ValueError):
            load_module().query(arch="sm_999", axis="memory", limit=3)

    def test_workload_query_routes_non_kernel_bottlenecks(self) -> None:
        result = load_module().query(
            arch="sm_120", layer="workload", bottleneck="framework", limit=2
        )
        self.assertLessEqual(len(result["methods"]), 2)
        self.assertTrue(result["methods"])
        self.assertTrue(all(item["layer"] == "workload" for item in result["methods"]))


if __name__ == "__main__":
    unittest.main()
