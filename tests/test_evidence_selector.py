from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

from tests.test_analysis_epoch import epoch_fixture
from tests.test_execution_map import evidence_catalog, map_fixture
from tests.test_hypothesis_space import hypothesis_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills/cuda-kernel-optimizer/scripts"


def _load(filename: str, name: str):
    path = SCRIPTS / filename
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


def catalog_fixture() -> dict:
    return {
        "schema_version": "cuda-optimizer/evidence-action-catalog-v1",
        "catalog_id": "default-v1",
        "actions": [
            {
                "action_id": "ncu-targeted",
                "evidence_kind": "ncu_kernel",
                "required_capability_ids": ["ncu.counter_access"],
                "cost": "high",
                "perturbation": "high",
                "risk": "low",
                "control_scope": "read_only",
                "repeatable": True,
            },
            {
                "action_id": "framework-targeted",
                "evidence_kind": "framework_trace",
                "required_capability_ids": ["pytorch.profiler"],
                "cost": "low",
                "perturbation": "low",
                "risk": "none",
                "control_scope": "read_only",
                "repeatable": False,
            },
            {
                "action_id": "os-runtime-targeted",
                "evidence_kind": "os_runtime",
                "required_capability_ids": ["nsys.timeline"],
                "cost": "medium",
                "perturbation": "medium",
                "risk": "none",
                "control_scope": "read_only",
                "repeatable": True,
            },
        ],
    }


def policy_fixture() -> dict:
    return {
        "schema_version": "cuda-optimizer/evidence-selection-policy-v1",
        "max_cost": "high",
        "max_perturbation": "high",
        "max_risk": "low",
        "remaining_profile_actions": 2,
        "available_capability_ids": [
            "ncu.counter_access",
            "nsys.timeline",
            "pytorch.profiler",
        ],
    }


def request_fixture() -> dict:
    return {
        "schema_version": "cuda-optimizer/evidence-request-set-v1",
        "request_set_id": "requests-0001",
        "epoch_id": "epoch-0001",
        "epoch_sha256": "",
        "hypothesis_set_sha256": "",
        "requests": [
            {
                "request_id": "req-framework",
                "action_id": "framework-targeted",
                "question": "Is the launch gap the cause rather than kernel execution?",
                "target_hypothesis_ids": ["h-framework-gap", "h-kernel-bound"],
                "exclusive_pairs": [
                    {"left": "h-framework-gap", "right": "h-kernel-bound"}
                ],
                "outcomes": [
                    {
                        "outcome_id": "gap-present",
                        "supports": ["h-framework-gap"],
                        "opposes": ["h-kernel-bound"],
                    },
                    {
                        "outcome_id": "kernel-dominant",
                        "supports": ["h-kernel-bound"],
                        "opposes": ["h-framework-gap"],
                    },
                ],
            },
            {
                "request_id": "req-ncu",
                "action_id": "ncu-targeted",
                "question": "Can kernel counters falsify kernel execution as the bottleneck?",
                "target_hypothesis_ids": ["h-kernel-bound"],
                "exclusive_pairs": [],
                "outcomes": [
                    {"outcome_id": "stall-found", "supports": ["h-kernel-bound"], "opposes": []},
                    {"outcome_id": "no-stall", "supports": [], "opposes": ["h-kernel-bound"]},
                ],
            },
        ],
    }


class EvidenceSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.map_module = _load("execution_map.py", "selector_execution_map_test")
        self.hypothesis_module = _load("hypothesis_space.py", "selector_hypothesis_test")
        self.module = _load("evidence_selector.py", "evidence_selector_test")
        self.epoch = epoch_fixture()
        self.evidence = evidence_catalog()
        self.execution_map = map_fixture(self.map_module)
        hypothesis = hypothesis_fixture(self.hypothesis_module, self.map_module)
        self.hypothesis_result = self.hypothesis_module.validate_hypothesis_set(
            hypothesis,
            epoch=self.epoch,
            execution_map=self.execution_map,
            evidence_catalog=self.evidence,
        )

    def requests(self) -> dict:
        value = request_fixture()
        value["epoch_sha256"] = self.map_module.epoch_digest(self.epoch)
        value["hypothesis_set_sha256"] = self.hypothesis_result["hypothesis_set_sha256"]
        return value

    def select(
        self, value: dict, *, catalog=None, policy=None, history=(), completed_actions=()
    ):
        return self.module.select_evidence_request(
            value,
            epoch=self.epoch,
            hypothesis_result=self.hypothesis_result,
            evidence_catalog=self.evidence,
            action_catalog=catalog_fixture() if catalog is None else catalog,
            policy=policy_fixture() if policy is None else policy,
            request_history=list(history),
            completed_action_ids=list(completed_actions),
        )

    def test_exclusive_discrimination_wins_before_lower_level_falsification(self) -> None:
        result = self.select(self.requests())
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_request"]["request_id"], "req-framework")
        self.assertEqual(result["selected_request"]["controller_action"]["cost"], "low")
        self.assertNotIn("probability", str(result).lower())

    def test_model_cannot_supply_cost_risk_or_information_score(self) -> None:
        for field, raw in (("cost", "low"), ("risk", "none"), ("information_gain", 99)):
            value = self.requests()
            value["requests"][0][field] = raw
            with self.subTest(field=field), self.assertRaisesRegex(
                self.module.ValidationError, "unknown"
            ):
                self.select(value)

    def test_catalog_cost_breaks_equal_coverage_ties_not_model_claims(self) -> None:
        value = self.requests()
        clone = copy.deepcopy(value["requests"][0])
        clone.update({"request_id": "req-os", "action_id": "os-runtime-targeted"})
        value["requests"] = [clone, value["requests"][0]]
        result = self.select(value)
        self.assertEqual(result["selected_request"]["request_id"], "req-framework")

    def test_missing_capability_yields_evidence_gap_without_host_action(self) -> None:
        value = self.requests()
        value["requests"] = [value["requests"][1]]
        policy = policy_fixture()
        policy["available_capability_ids"].remove("ncu.counter_access")
        result = self.select(value, policy=policy)
        self.assertEqual(result["status"], "evidence_gap")
        self.assertIsNone(result["selected_request"])
        self.assertEqual(result["missing_capability_ids"], ["ncu.counter_access"])
        self.assertNotIn("sudo", str(result).lower())

    def test_renaming_request_cannot_bypass_equivalent_history(self) -> None:
        first = self.select(self.requests())
        signature = first["selected_request"]["request_signature"]
        value = self.requests()
        value["requests"] = [value["requests"][0]]
        value["requests"][0]["request_id"] = "req-framework-renamed"
        result = self.select(value, history=[signature])
        self.assertEqual(result["status"], "evidence_gap")
        self.assertEqual(result["rejections"][0]["reason"], "equivalent_request_already_attempted")

    def test_non_repeatable_action_cannot_be_selected_twice(self) -> None:
        value = self.requests()
        value["requests"] = [value["requests"][0]]
        result = self.select(
            value, completed_actions=["framework-targeted"]
        )
        self.assertEqual(result["status"], "evidence_gap")
        self.assertEqual(
            result["rejections"][0]["reason"], "action_is_not_repeatable"
        )

    def test_request_must_change_at_least_one_hypothesis(self) -> None:
        value = self.requests()
        value["requests"][0]["outcomes"] = [
            {"outcome_id": "same-a", "supports": [], "opposes": []},
            {"outcome_id": "same-b", "supports": [], "opposes": []},
        ]
        with self.assertRaisesRegex(self.module.ValidationError, "change"):
            self.select(value)

    def test_pair_claim_must_match_an_exclusive_relationship(self) -> None:
        value = self.requests()
        value["requests"][1]["exclusive_pairs"] = [
            {"left": "h-framework-gap", "right": "h-kernel-bound"}
        ]
        with self.assertRaisesRegex(self.module.ValidationError, "target"):
            self.select(value)

    def test_budget_exhaustion_stops_without_consuming_another_action(self) -> None:
        policy = policy_fixture()
        policy["remaining_profile_actions"] = 0
        result = self.select(self.requests(), policy=policy)
        self.assertEqual(result["status"], "evidence_gap")
        self.assertEqual(result["gap_reason"], "profile_budget_exhausted")

    def test_request_id_is_final_stable_tie_breaker(self) -> None:
        value = self.requests()
        first = value["requests"][0]
        second = copy.deepcopy(first)
        first["request_id"] = "req-z"
        second["request_id"] = "req-a"
        value["requests"] = [first, second]
        result = self.select(value)
        self.assertEqual(result["selected_request"]["request_id"], "req-a")


if __name__ == "__main__":
    unittest.main()
