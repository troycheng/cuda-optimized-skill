from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

from tests.test_analysis_epoch import epoch_fixture
from tests.test_execution_map import evidence_catalog, map_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/cuda-kernel-optimizer/scripts/hypothesis_space.py"
EXECUTION_MAP_SCRIPT = ROOT / "skills/cuda-kernel-optimizer/scripts/execution_map.py"


def _load(path: Path, name: str):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def hypothesis_fixture(module, map_module) -> dict:
    epoch = epoch_fixture()
    execution_map = map_fixture(map_module)
    return {
        "schema_version": "cuda-optimizer/hypothesis-set-v1",
        "set_id": "hypotheses-0001",
        "epoch_id": epoch["epoch_id"],
        "epoch_sha256": map_module.epoch_digest(epoch),
        "execution_map_sha256": map_module.execution_map_digest(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog()
        ),
        "hypotheses": [
            {
                "hypothesis_id": "h-framework-gap",
                "kind": "mechanism",
                "scope_node_ids": ["cpu-launch", "gpu-kernel"],
                "statement": "CPU launch serialization delays the GPU kernel.",
                "mechanism": "framework_launch_overhead",
                "disposition": "active",
                "confidence": "plausible",
                "support_evidence_ids": ["ev-cpu"],
                "oppose_evidence_ids": [],
                "missing_evidence_kinds": ["framework_trace"],
                "falsification_question": "Does removing the launch gap leave GPU idle unchanged?",
            },
            {
                "hypothesis_id": "h-kernel-bound",
                "kind": "mechanism",
                "scope_node_ids": ["gpu-kernel"],
                "statement": "The kernel body dominates the critical GPU lane.",
                "mechanism": "kernel_execution",
                "disposition": "active",
                "confidence": "inconclusive",
                "support_evidence_ids": [],
                "oppose_evidence_ids": [],
                "missing_evidence_kinds": ["ncu_kernel"],
                "falsification_question": "Does kernel-level evidence show no dominant stall?",
            },
        ],
        "relationships": [
            {
                "relation": "exclusive",
                "left": "h-framework-gap",
                "right": "h-kernel-bound",
            }
        ],
    }


class HypothesisSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.map_module = _load(
            EXECUTION_MAP_SCRIPT, "cuda_optimizer_execution_map_hypothesis_test"
        )
        self.module = _load(SCRIPT, "cuda_optimizer_hypothesis_space_test")
        self.epoch = epoch_fixture()
        self.catalog = evidence_catalog()
        self.execution_map = map_fixture(self.map_module)

    def validate(self, value: dict, *, execution_map: dict | None = None, catalog=None):
        return self.module.validate_hypothesis_set(
            value,
            epoch=self.epoch,
            execution_map=self.execution_map if execution_map is None else execution_map,
            evidence_catalog=self.catalog if catalog is None else catalog,
        )

    def test_valid_set_is_canonical_and_hash_bound(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        value["hypotheses"].reverse()
        result = self.validate(value)
        self.assertEqual(
            [item["hypothesis_id"] for item in result["hypothesis_set"]["hypotheses"]],
            ["h-framework-gap", "h-kernel-bound"],
        )
        self.assertEqual(len(result["hypothesis_set_sha256"]), 64)
        self.assertEqual(result["active_hypothesis_ids"], ["h-framework-gap", "h-kernel-bound"])

    def test_closed_contract_rejects_probability_and_model_score(self) -> None:
        for field, value in (("probability", 0.93), ("model_score", 9)):
            item = hypothesis_fixture(self.module, self.map_module)
            item["hypotheses"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                self.module.ValidationError, "unknown"
            ):
                self.validate(item)

    def test_scope_and_evidence_references_must_be_current_and_real(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        value["hypotheses"][0]["scope_node_ids"] = ["invented-node"]
        with self.assertRaisesRegex(self.module.ValidationError, "scope.*unknown"):
            self.validate(value)

        value = hypothesis_fixture(self.module, self.map_module)
        value["hypotheses"][0]["support_evidence_ids"] = ["invented-evidence"]
        with self.assertRaisesRegex(self.module.ValidationError, "unknown evidence"):
            self.validate(value)

        catalog = evidence_catalog()
        catalog["ev-cpu"]["epoch_id"] = "epoch-0000"
        with self.assertRaisesRegex(self.module.ValidationError, "current epoch"):
            self.validate(hypothesis_fixture(self.module, self.map_module), catalog=catalog)

    def test_single_evidence_kind_cannot_support_direction(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        item = value["hypotheses"][0]
        item["confidence"] = "direction_supported"
        item["support_evidence_ids"] = ["ev-gpu", "ev-edge"]
        with self.assertRaisesRegex(self.module.ValidationError, "independent evidence kinds"):
            self.validate(value)

        item["support_evidence_ids"] = ["ev-cpu", "ev-gpu"]
        item["missing_evidence_kinds"] = []
        self.assertEqual(
            self.validate(value)["hypothesis_set"]["hypotheses"][0]["confidence"],
            "direction_supported",
        )

    def test_direction_requires_resolved_gaps_and_no_opposing_evidence(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        item = value["hypotheses"][0]
        item["confidence"] = "direction_supported"
        item["support_evidence_ids"] = ["ev-cpu", "ev-gpu"]
        with self.assertRaisesRegex(self.module.ValidationError, "missing evidence"):
            self.validate(value)

        item["missing_evidence_kinds"] = []
        item["oppose_evidence_ids"] = ["ev-edge"]
        with self.assertRaisesRegex(self.module.ValidationError, "opposing evidence"):
            self.validate(value)

    def test_exclusive_hypotheses_cannot_both_support_a_direction(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        for item in value["hypotheses"]:
            item["confidence"] = "direction_supported"
            item["support_evidence_ids"] = ["ev-cpu", "ev-gpu"]
            item["missing_evidence_kinds"] = []
        with self.assertRaisesRegex(self.module.ValidationError, "exclusive.*direction"):
            self.validate(value)

    def test_ambiguous_epoch_cannot_support_direction(self) -> None:
        epoch = epoch_fixture()
        epoch.update(
            {
                "epoch_id": "epoch-0002",
                "sequence": 2,
                "trigger": "conservative_boundary",
                "parent_epoch_id": "epoch-0001",
                "boundary_ambiguous": True,
            }
        )
        catalog = evidence_catalog()
        for item in catalog.values():
            item["epoch_id"] = "epoch-0002"
        execution_map = map_fixture(self.map_module)
        execution_map["epoch_id"] = "epoch-0002"
        execution_map["epoch_sha256"] = self.map_module.epoch_digest(epoch)
        execution_map["window"]["boundary_ambiguous"] = True
        value = hypothesis_fixture(self.module, self.map_module)
        value["epoch_id"] = "epoch-0002"
        value["epoch_sha256"] = self.map_module.epoch_digest(epoch)
        value["execution_map_sha256"] = self.map_module.execution_map_digest(
            execution_map, epoch=epoch, evidence_catalog=catalog
        )
        value["hypotheses"][0]["confidence"] = "direction_supported"
        value["hypotheses"][0]["support_evidence_ids"] = ["ev-cpu", "ev-gpu"]
        with self.assertRaisesRegex(self.module.ValidationError, "ambiguous"):
            self.module.validate_hypothesis_set(
                value,
                epoch=epoch,
                execution_map=execution_map,
                evidence_catalog=catalog,
            )

    def test_unmodeled_gap_requires_active_meta_hypothesis(self) -> None:
        execution_map = map_fixture(self.map_module)
        execution_map["edges"][0]["relation"] = "unknown_dependency"
        value = hypothesis_fixture(self.module, self.map_module)
        value["execution_map_sha256"] = self.map_module.execution_map_digest(
            execution_map, epoch=self.epoch, evidence_catalog=self.catalog
        )
        with self.assertRaisesRegex(self.module.ValidationError, "unmodeled"):
            self.validate(value, execution_map=execution_map)

        value["hypotheses"].append(
            {
                "hypothesis_id": "h-unmodeled",
                "kind": "unmodeled",
                "scope_node_ids": ["cpu-launch", "gpu-kernel"],
                "statement": "An unmodeled dependency may explain the observed gap.",
                "mechanism": "unmodeled_dependency",
                "disposition": "active",
                "confidence": "inconclusive",
                "support_evidence_ids": [],
                "oppose_evidence_ids": [],
                "missing_evidence_kinds": ["os_runtime"],
                "falsification_question": "Can a bounded OS-runtime trace account for the gap?",
            }
        )
        self.assertIn("h-unmodeled", self.validate(value, execution_map=execution_map)["active_hypothesis_ids"])

    def test_every_hypothesis_needs_a_falsification_question(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        value["hypotheses"][0]["falsification_question"] = ""
        with self.assertRaisesRegex(self.module.ValidationError, "falsification"):
            self.validate(value)

    def test_symmetric_relationships_are_canonical_and_not_conflicting(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        value["relationships"][0].update(
            {"left": "h-kernel-bound", "right": "h-framework-gap"}
        )
        with self.assertRaisesRegex(self.module.ValidationError, "canonical"):
            self.validate(value)

        value = hypothesis_fixture(self.module, self.map_module)
        value["relationships"].append(
            {
                "relation": "coexists_with",
                "left": "h-framework-gap",
                "right": "h-kernel-bound",
            }
        )
        with self.assertRaisesRegex(self.module.ValidationError, "conflicting"):
            self.validate(value)

    def test_depends_on_must_be_acyclic(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        value["relationships"] = [
            {
                "relation": "depends_on",
                "left": "h-framework-gap",
                "right": "h-kernel-bound",
            },
            {
                "relation": "depends_on",
                "left": "h-kernel-bound",
                "right": "h-framework-gap",
            },
        ]
        with self.assertRaisesRegex(self.module.ValidationError, "cycle"):
            self.validate(value)

    def test_rejected_hypothesis_needs_opposing_evidence(self) -> None:
        value = hypothesis_fixture(self.module, self.map_module)
        value["hypotheses"][0]["disposition"] = "rejected"
        with self.assertRaisesRegex(self.module.ValidationError, "opposing evidence"):
            self.validate(value)
        value["hypotheses"][0]["oppose_evidence_ids"] = ["ev-gpu"]
        value["hypotheses"][0]["confidence"] = "inconclusive"
        result = self.validate(value)
        self.assertNotIn("h-framework-gap", result["active_hypothesis_ids"])


if __name__ == "__main__":
    unittest.main()
