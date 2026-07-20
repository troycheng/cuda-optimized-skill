from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py"


def _load():
    name = "cuda_optimizer_diagnostic_knowledge_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DiagnosticKnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load()

    def test_routes_a_small_framework_launch_context(self) -> None:
        diagnosis = {
            "primary_category": "framework",
            "ranked_categories": [{"category": "framework", "score": 100}],
        }
        execution_map = {
            "nodes": [
                {"node_id": "cpu", "layer": "cpu", "label": "cudaLaunchKernel"},
                {"node_id": "gpu", "layer": "gpu", "label": "decode_attention"},
            ]
        }
        result = self.module.route_cards(diagnosis, execution_map, limit=3)
        self.assertLessEqual(len(result["cards"]), 3)
        self.assertEqual(result["cards"][0]["id"], "diagnostic.framework.launch-gaps")
        self.assertEqual(result["promotion_authority"], "none")
        self.assertTrue(result["cards"][0]["distinguishing_question"])
        self.assertTrue(result["cards"][0]["counter_signals"])

    def test_inconclusive_diagnosis_routes_cross_layer_triage(self) -> None:
        result = self.module.route_cards(
            {"primary_category": None, "ranked_categories": []},
            {"nodes": []},
            limit=3,
        )
        self.assertEqual(result["cards"][0]["id"], "diagnostic.cross-layer.triage")

    def test_cards_are_hints_not_direction_evidence(self) -> None:
        result = self.module.route_cards(
            {
                "primary_category": "kernel",
                "ranked_categories": [{"category": "kernel", "score": 100}],
            },
            {"nodes": [{"node_id": "gpu", "layer": "gpu", "label": "gemm"}]},
            limit=3,
        )
        self.assertEqual(result["promotion_authority"], "none")
        self.assertTrue(all(card["status"] == "routing_only" for card in result["cards"]))


if __name__ == "__main__":
    unittest.main()
