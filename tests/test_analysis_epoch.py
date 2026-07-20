from __future__ import annotations

import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/cuda-kernel-optimizer/scripts/analysis_epoch.py"


def _load():
    name = "cuda_optimizer_analysis_epoch_test"
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


def epoch_fixture() -> dict:
    return {
        "schema_version": "cuda-optimizer/analysis-epoch-v1",
        "epoch_id": "epoch-0001",
        "sequence": 1,
        "trigger": "initial",
        "parent_epoch_id": None,
        "started_at": 1000.0,
        "identities": {
            "workload_contract_sha256": "a" * 64,
            "environment_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "analysis_policy_sha256": "d" * 64,
        },
        "source": {
            "profiler": "nsys",
            "profiler_version": "2026.3.1",
            "export_schema": "sqlite-3.1.0",
            "adapter_id": "nsys-execution-map",
            "adapter_version": "1.0.0",
            "adapter_sha256": "e" * 64,
        },
        "regime": {
            "shape_distribution_sha256": "f" * 64,
            "dynamic_branch_sha256": "1" * 64,
            "execution_regime_sha256": "2" * 64,
        },
        "boundary_ambiguous": False,
    }


class AnalysisEpochTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load()

    def test_valid_epoch_is_detached_and_digest_is_canonical(self) -> None:
        value = epoch_fixture()
        admitted = self.module.validate_epoch(value)
        self.assertEqual(admitted, value)
        self.assertIsNot(admitted, value)
        reversed_value = dict(reversed(list(value.items())))
        self.assertEqual(
            self.module.epoch_digest(value),
            self.module.epoch_digest(reversed_value),
        )
        value["identities"]["source_sha256"] = "9" * 64
        self.assertEqual(admitted["identities"]["source_sha256"], "c" * 64)

    def test_closed_contract_and_finite_time_are_required(self) -> None:
        extra = epoch_fixture()
        extra["confidence"] = 0.99
        with self.assertRaisesRegex(self.module.ValidationError, "unknown"):
            self.module.validate_epoch(extra)
        non_finite = epoch_fixture()
        non_finite["started_at"] = math.nan
        with self.assertRaisesRegex(self.module.ValidationError, "finite"):
            self.module.validate_epoch(non_finite)

    def test_source_version_schema_and_adapter_identity_are_required(self) -> None:
        for field in ("profiler_version", "export_schema", "adapter_sha256"):
            value = epoch_fixture()
            del value["source"][field]
            with self.subTest(field=field), self.assertRaises(
                self.module.ValidationError
            ):
                self.module.validate_epoch(value)

    def test_controller_identities_must_match(self) -> None:
        expected = copy.deepcopy(epoch_fixture()["identities"])
        expected["environment_sha256"] = "9" * 64
        with self.assertRaisesRegex(self.module.ValidationError, "environment"):
            self.module.validate_epoch(
                epoch_fixture(), expected_identities=expected
            )

    def test_initial_and_child_epoch_lineage_is_consistent(self) -> None:
        invalid = epoch_fixture()
        invalid["parent_epoch_id"] = "epoch-0000"
        with self.assertRaisesRegex(self.module.ValidationError, "initial"):
            self.module.validate_epoch(invalid)

        child = epoch_fixture()
        child.update(
            {
                "epoch_id": "epoch-0002",
                "sequence": 2,
                "trigger": "workload_regime_change",
                "parent_epoch_id": "epoch-0001",
            }
        )
        self.assertEqual(self.module.validate_epoch(child), child)
        child["parent_epoch_id"] = None
        with self.assertRaisesRegex(self.module.ValidationError, "parent"):
            self.module.validate_epoch(child)

    def test_conservative_boundary_must_remain_ambiguous(self) -> None:
        value = epoch_fixture()
        value.update(
            {
                "epoch_id": "epoch-0002",
                "sequence": 2,
                "trigger": "conservative_boundary",
                "parent_epoch_id": "epoch-0001",
            }
        )
        with self.assertRaisesRegex(self.module.ValidationError, "ambiguous"):
            self.module.validate_epoch(value)
        value["boundary_ambiguous"] = True
        self.assertEqual(self.module.validate_epoch(value), value)


if __name__ == "__main__":
    unittest.main()
