from __future__ import annotations

import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "workload_diagnosis.py"
)
POLICY_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "references"
    / "workload_diagnosis_policy.json"
)


def _load_diagnosis():
    module_name = "cuda_optimizer_workload_diagnosis_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workload diagnosis: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _probe(
    metrics: dict,
    *,
    probe_id: str = "timeline",
    kind: str = "timeline",
    status: str = "ok",
    issues: list | None = None,
) -> dict:
    return {
        "schema_version": "cuda-workload-optimizer/probe-v1",
        "probe_id": probe_id,
        "kind": kind,
        "status": status,
        "metrics": metrics,
        "issues": [] if issues is None else issues,
        "artifacts": [],
    }


class ProbeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diagnosis = _load_diagnosis()

    def test_valid_probe_is_detached(self) -> None:
        probe = _probe({"gpu_busy_pct": 80.0, "kernel_time_pct": 70})
        normalized = self.diagnosis.validate_probe(probe)
        self.assertEqual(normalized, probe)
        self.assertIsNot(normalized, probe)
        probe["metrics"]["gpu_busy_pct"] = 1
        self.assertEqual(normalized["metrics"]["gpu_busy_pct"], 80.0)

    def test_probe_rejects_unknown_fields_metrics_and_non_finite_percentages(self) -> None:
        cases = []
        unknown = _probe({"gpu_busy_pct": 80})
        unknown["extra"] = True
        cases.append(unknown)
        unknown_metric = _probe({"made_up_pct": 30})
        cases.append(unknown_metric)
        for value in (-1, 101, True, math.inf, math.nan):
            cases.append(_probe({"gpu_busy_pct": value}))

        for probe in cases:
            with self.subTest(probe=probe), self.assertRaises(
                self.diagnosis.DiagnosisError
            ):
                self.diagnosis.validate_probe(probe)

    def test_probe_issues_and_artifacts_are_closed_and_hashed(self) -> None:
        valid = _probe(
            {},
            status="failed",
            issues=[
                {
                    "id": "environment:ncu-permission",
                    "category": "environment",
                    "severity": "error",
                    "message": "ERR_NVGPUCTRPERM",
                }
            ],
        )
        valid["artifacts"] = [{"name": "timeline.qdrep", "sha256": "a" * 64}]
        self.assertEqual(self.diagnosis.validate_probe(valid), valid)

        invalid = copy.deepcopy(valid)
        invalid["issues"][0]["command"] = ["sudo", "ncu"]
        with self.assertRaisesRegex(self.diagnosis.DiagnosisError, "unknown"):
            self.diagnosis.validate_probe(invalid)

        invalid_hash = copy.deepcopy(valid)
        invalid_hash["artifacts"][0]["sha256"] = "short"
        with self.assertRaisesRegex(self.diagnosis.DiagnosisError, "sha256"):
            self.diagnosis.validate_probe(invalid_hash)


class DeterministicDiagnosisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diagnosis = _load_diagnosis()
        self.policy = self.diagnosis.load_policy(POLICY_PATH)

    def _classify(self, metrics: dict, **kwargs) -> dict:
        return self.diagnosis.diagnose([_probe(metrics, **kwargs)], self.policy)

    def test_classifies_each_supported_bottleneck_from_evidence(self) -> None:
        cases = {
            "kernel": {"gpu_busy_pct": 92, "kernel_time_pct": 78},
            "framework": {
                "gpu_busy_pct": 52,
                "launch_gap_pct": 42,
                "cuda_api_time_pct": 31,
            },
            "cpu_data": {
                "gpu_busy_pct": 45,
                "cpu_busy_pct": 92,
                "data_wait_pct": 48,
            },
            "transfer": {"gpu_busy_pct": 55, "transfer_time_pct": 45},
            "communication": {
                "gpu_busy_pct": 60,
                "communication_time_pct": 46,
            },
            "io": {"gpu_busy_pct": 40, "io_wait_pct": 47},
        }
        for category, metrics in cases.items():
            with self.subTest(category=category):
                result = self._classify(metrics)
                self.assertEqual(result["status"], "classified")
                self.assertEqual(result["primary_category"], category)
                self.assertEqual(result["ranked_categories"][0]["category"], category)
                self.assertTrue(result["matched_rules"])

    def test_environment_failure_is_classified_without_invented_zero_metrics(self) -> None:
        issue = {
            "id": "environment:toolchain",
            "category": "environment",
            "severity": "error",
            "message": "CUDA compiler unavailable",
        }
        result = self._classify(
            {}, kind="environment", status="failed", issues=[issue]
        )
        self.assertEqual(result["primary_category"], "environment")
        self.assertEqual(result["diagnosis_ids"], ["environment:toolchain"])
        self.assertEqual(result["coverage"]["known_metrics"], [])

    def test_close_competing_scores_return_mixed(self) -> None:
        result = self._classify(
            {
                "gpu_busy_pct": 88,
                "kernel_time_pct": 72,
                "transfer_time_pct": 44,
            }
        )
        self.assertEqual(result["status"], "mixed")
        self.assertEqual(result["primary_category"], "mixed")
        self.assertEqual(
            {item["category"] for item in result["ranked_categories"][:2]},
            {"kernel", "transfer"},
        )

    def test_insufficient_evidence_is_explicit(self) -> None:
        result = self._classify({"gpu_busy_pct": 15})
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["primary_category"])
        self.assertEqual(result["confidence"], "inconclusive")
        self.assertTrue(result["suggested_probes"])

    def test_result_binds_policy_digest_rule_provenance_and_evidence_paths(self) -> None:
        result = self._classify(
            {"gpu_busy_pct": 90, "kernel_time_pct": 75}
        )
        self.assertEqual(len(result["policy_digest"]), 64)
        self.assertEqual(result["policy_schema_version"], self.policy["schema_version"])
        match = result["matched_rules"][0]
        self.assertIn("rule_id", match)
        self.assertIn("conditions", match)
        self.assertIn("evidence_paths", match)
        self.assertTrue(
            all(path.startswith("probes/") for path in match["evidence_paths"])
        )
        self.assertEqual(result["coverage"]["probe_count"], 1)


if __name__ == "__main__":
    unittest.main()
