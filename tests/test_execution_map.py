from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

from tests.test_analysis_epoch import epoch_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/cuda-kernel-optimizer/scripts/execution_map.py"
OLD_SCHEMA = ROOT / "skills/cuda-kernel-optimizer/templates/execution_path.schema.json"
LAYERS = (
    "cpu",
    "gpu",
    "framework",
    "transfer",
    "communication",
    "io",
    "synchronization",
    "idle",
)


def _load():
    name = "cuda_optimizer_execution_map_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def evidence_catalog() -> dict:
    return {
        "ev-cpu": {
            "epoch_id": "epoch-0001",
            "kind": "os_runtime",
            "artifact_sha256": "3" * 64,
        },
        "ev-gpu": {
            "epoch_id": "epoch-0001",
            "kind": "nsys_timeline",
            "artifact_sha256": "4" * 64,
        },
        "ev-edge": {
            "epoch_id": "epoch-0001",
            "kind": "nsys_timeline",
            "artifact_sha256": "5" * 64,
        },
    }


def map_fixture(module) -> dict:
    epoch = epoch_fixture()
    coverage = []
    for layer in LAYERS:
        if layer in {"cpu", "gpu"}:
            coverage.append({"layer": layer, "status": "observed", "reason": None})
        else:
            coverage.append(
                {
                    "layer": layer,
                    "status": "not_observed",
                    "reason": "not present in the bounded trace window",
                }
            )
    return {
        "schema_version": "cuda-optimizer/execution-map-v1",
        "map_id": "map-0001",
        "epoch_id": epoch["epoch_id"],
        "epoch_sha256": module.epoch_digest(epoch),
        "identities": copy.deepcopy(epoch["identities"]),
        "window": {
            "start_us": 0.0,
            "end_us": 1000.0,
            "boundary_ambiguous": False,
        },
        "coverage": coverage,
        "nodes": [
            {
                "node_id": "cpu-launch",
                "layer": "cpu",
                "lane": "thread-7",
                "kind": "cuda_api",
                "label": "cudaLaunchKernel",
                "duration_us": 900.0,
                "occurrences": 4,
                "timing_status": "observed",
                "first_start_us": 0.0,
                "last_end_us": 900.0,
                "attribution_status": "explained",
                "evidence_ids": ["ev-cpu"],
            },
            {
                "node_id": "gpu-kernel",
                "layer": "gpu",
                "lane": "stream-0",
                "kind": "kernel",
                "label": "decode_attention",
                "duration_us": 900.0,
                "occurrences": 4,
                "timing_status": "observed",
                "first_start_us": 100.0,
                "last_end_us": 1000.0,
                "attribution_status": "not_applicable",
                "evidence_ids": ["ev-gpu"],
            },
        ],
        "edges": [
            {
                "source": "cpu-launch",
                "target": "gpu-kernel",
                "relation": "calls",
                "overlap_us": None,
                "evidence_ids": ["ev-edge"],
            }
        ],
        "hot_path": ["cpu-launch", "gpu-kernel"],
        "uncovered_intervals": [],
        "conclusion_level": "observed",
    }


class ExecutionMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load()
        self.epoch = epoch_fixture()
        self.catalog = evidence_catalog()

    def validate(self, value: dict) -> dict:
        return self.module.validate_execution_map(
            value, epoch=self.epoch, evidence_catalog=self.catalog
        )

    def test_overlapping_cpu_and_gpu_lane_time_is_valid(self) -> None:
        value = map_fixture(self.module)
        result = self.validate(value)
        self.assertEqual(result["execution_map"], value)
        self.assertFalse(result["requires_unmodeled_hypothesis"])
        self.assertEqual(result["window_duration_us"], 1000.0)

    def test_transfer_overlap_is_explicit_and_bounded(self) -> None:
        value = map_fixture(self.module)
        next(x for x in value["coverage"] if x["layer"] == "transfer").update(
            {"status": "observed", "reason": None}
        )
        value["nodes"].append(
            {
                "node_id": "h2d-copy",
                "layer": "transfer",
                "lane": "copy-stream-0",
                "kind": "h2d",
                "label": "input copy",
                "duration_us": 200.0,
                "occurrences": 1,
                "timing_status": "observed",
                "first_start_us": 200.0,
                "last_end_us": 400.0,
                "attribution_status": "explained",
                "evidence_ids": ["ev-gpu"],
            }
        )
        value["edges"].append(
            {
                "source": "gpu-kernel",
                "target": "h2d-copy",
                "relation": "overlaps",
                "overlap_us": 200.0,
                "evidence_ids": ["ev-edge"],
            }
        )
        result = self.validate(value)
        overlap = result["execution_map"]["edges"][1]
        self.assertEqual(overlap["overlap_us"], 200.0)

        value["edges"][-1]["overlap_us"] = 201.0
        with self.assertRaisesRegex(self.module.ValidationError, "overlap"):
            self.validate(value)

    def test_partial_overlap_and_serial_transfer_are_distinguishable(self) -> None:
        value = map_fixture(self.module)
        next(x for x in value["coverage"] if x["layer"] == "transfer").update(
            {"status": "observed", "reason": None}
        )
        for node_id, start, end in (
            ("partial-copy", 850.0, 950.0),
            ("serial-copy", 950.0, 1000.0),
        ):
            value["nodes"].append(
                {
                    "node_id": node_id,
                    "layer": "transfer",
                    "lane": f"copy-{node_id}",
                    "kind": "h2d",
                    "label": node_id,
                    "duration_us": end - start,
                    "occurrences": 1,
                    "timing_status": "observed",
                    "first_start_us": start,
                    "last_end_us": end,
                    "attribution_status": "explained",
                    "evidence_ids": ["ev-gpu"],
                }
            )
        value["edges"].extend(
            [
                {
                    "source": "gpu-kernel",
                    "target": "partial-copy",
                    "relation": "overlaps",
                    "overlap_us": 100.0,
                    "evidence_ids": ["ev-edge"],
                },
                {
                    "source": "partial-copy",
                    "target": "serial-copy",
                    "relation": "precedes",
                    "overlap_us": None,
                    "evidence_ids": ["ev-edge"],
                },
            ]
        )
        self.validate(value)

        value["edges"][1]["overlap_us"] = 0.0
        with self.assertRaisesRegex(self.module.ValidationError, "positive"):
            self.validate(value)

    def test_missing_node_timing_forces_inconclusive_without_fake_overlap(self) -> None:
        value = map_fixture(self.module)
        node = value["nodes"][1]
        node.update(
            {
                "timing_status": "unavailable",
                "first_start_us": None,
                "last_end_us": None,
            }
        )
        value["conclusion_level"] = "inconclusive"
        result = self.validate(value)
        self.assertTrue(result["requires_unmodeled_hypothesis"])

        value["edges"].append(
            {
                "source": "cpu-launch",
                "target": "gpu-kernel",
                "relation": "overlaps",
                "overlap_us": 10.0,
                "evidence_ids": ["ev-edge"],
            }
        )
        with self.assertRaisesRegex(self.module.ValidationError, "timing"):
            self.validate(value)

    def test_epoch_and_controller_identities_are_bound(self) -> None:
        value = map_fixture(self.module)
        value["epoch_id"] = "epoch-0002"
        with self.assertRaisesRegex(self.module.ValidationError, "epoch_id"):
            self.validate(value)
        value = map_fixture(self.module)
        value["identities"]["environment_sha256"] = "9" * 64
        with self.assertRaisesRegex(self.module.ValidationError, "environment"):
            self.validate(value)

    def test_missing_layer_is_not_silently_treated_as_zero(self) -> None:
        value = map_fixture(self.module)
        item = next(x for x in value["coverage"] if x["layer"] == "communication")
        item["reason"] = None
        with self.assertRaisesRegex(self.module.ValidationError, "reason"):
            self.validate(value)

        value = map_fixture(self.module)
        value["coverage"] = [
            item for item in value["coverage"] if item["layer"] != "communication"
        ]
        with self.assertRaisesRegex(self.module.ValidationError, "all layers"):
            self.validate(value)

    def test_observed_layer_requires_node_and_unobserved_layer_rejects_node(self) -> None:
        value = map_fixture(self.module)
        value["nodes"] = [n for n in value["nodes"] if n["layer"] != "gpu"]
        with self.assertRaisesRegex(self.module.ValidationError, "observed.*gpu"):
            self.validate(value)

        value = map_fixture(self.module)
        value["nodes"][0]["layer"] = "framework"
        with self.assertRaisesRegex(self.module.ValidationError, "not_observed"):
            self.validate(value)

    def test_evidence_from_another_epoch_is_rejected(self) -> None:
        catalog = evidence_catalog()
        catalog["ev-gpu"]["epoch_id"] = "epoch-0000"
        with self.assertRaisesRegex(self.module.ValidationError, "current epoch"):
            self.module.validate_execution_map(
                map_fixture(self.module), epoch=self.epoch, evidence_catalog=catalog
            )

    def test_unknown_edge_and_dangling_nodes_are_rejected(self) -> None:
        value = map_fixture(self.module)
        value["edges"][0]["relation"] = "probably_calls"
        with self.assertRaisesRegex(self.module.ValidationError, "relation"):
            self.validate(value)
        value = map_fixture(self.module)
        value["edges"][0]["target"] = "missing"
        with self.assertRaisesRegex(self.module.ValidationError, "unknown node"):
            self.validate(value)

    def test_unexplained_idle_unknown_dependency_and_gaps_require_meta_hypothesis(self) -> None:
        cases = []
        idle = map_fixture(self.module)
        next(x for x in idle["coverage"] if x["layer"] == "idle").update(
            {"status": "observed", "reason": None}
        )
        idle["nodes"].append(
            {
                "node_id": "idle-gap",
                "layer": "idle",
                "lane": "gpu-0",
                "kind": "idle_gap",
                "label": "unattributed GPU idle",
                "duration_us": 50.0,
                "occurrences": 1,
                "timing_status": "observed",
                "first_start_us": 900.0,
                "last_end_us": 950.0,
                "attribution_status": "unexplained",
                "evidence_ids": ["ev-gpu"],
            }
        )
        cases.append(idle)
        unknown_edge = map_fixture(self.module)
        unknown_edge["edges"][0]["relation"] = "unknown_dependency"
        cases.append(unknown_edge)
        uncovered = map_fixture(self.module)
        uncovered["uncovered_intervals"] = [
            {"start_us": 400.0, "end_us": 450.0, "reason": "trace buffer loss"}
        ]
        cases.append(uncovered)
        for value in cases:
            with self.subTest(value=value):
                self.assertTrue(self.validate(value)["requires_unmodeled_hypothesis"])

    def test_window_boundary_must_match_controller_epoch(self) -> None:
        value = map_fixture(self.module)
        value["window"]["boundary_ambiguous"] = True
        with self.assertRaisesRegex(self.module.ValidationError, "boundary"):
            self.validate(value)

    def test_old_v2_5_execution_path_schema_is_unchanged(self) -> None:
        digest = hashlib.sha256(OLD_SCHEMA.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "0b39e6e961e4bcf19a997bcef261a3c64bc3cf871d04d32724d1ffbec3f43cc3",
        )


if __name__ == "__main__":
    unittest.main()
