import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "run_skill_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cuda_skill_eval", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _suite() -> dict:
    arms = [
        "no_skill",
        "v2.9",
        "v3_random_planner",
        "v3_shuffled_registry",
        "v3_full",
    ]
    prompt_ref = "tests/evals/v3/README.md"
    prompt_sha256 = hashlib.sha256((ROOT / prompt_ref).read_bytes()).hexdigest()
    return {
        "schema_version": "cuda-skill-eval/suite-v1",
        "suite_id": "v3-control",
        "experiment": {
            "arms": arms,
            "replicates": 3,
            "seed_policy": "paired_fixed",
            "aggregation": "median",
            "release_gate": {
                "candidate_arm": "v3_full",
                "baseline_arm": "v2.9",
                "ablation_arms": ["v3_random_planner", "v3_shuffled_registry"],
                "must_pass_all_scenarios": True,
            },
        },
        "scenarios": [
            {
                "id": "wrong-kernel-bottleneck",
                "category": "direction",
                "fixture": "tests/gpu/sm120/fixtures/workload_probe.py",
                "claim_ceiling": "workload",
                "budget": {"max_seconds": 600, "max_candidates": 4},
                "required_events": ["workload_profiled", "non_kernel_direction"],
                "forbidden_events": ["unsupported_promotion"],
                "prompt_ref": prompt_ref,
                "prompt_sha256": prompt_sha256,
                "prompt_id": "wrong-kernel-bottleneck",
                "required_arms": arms,
                "oracle": "ledger_and_artifacts_v1",
            },
            {
                "id": "resume-after-interrupt",
                "category": "recovery",
                "fixture": "tests/test_state_schema.py",
                "claim_ceiling": "kernel",
                "budget": {"max_seconds": 300, "max_candidates": 2},
                "required_events": ["checkpoint_restored"],
                "forbidden_events": ["repeated_failed_candidate"],
                "prompt_ref": prompt_ref,
                "prompt_sha256": prompt_sha256,
                "prompt_id": "resume-after-interrupt",
                "required_arms": arms,
                "oracle": "ledger_and_artifacts_v1",
            },
        ],
    }


def _result(scenario_id: str, events: list[str], **overrides) -> dict:
    prompt_sha256 = _suite()["scenarios"][0]["prompt_sha256"]
    value = {
        "scenario_id": scenario_id,
        "status": "completed",
        "elapsed_seconds": 120.0,
        "gpu_seconds": 30.0,
        "candidates": 2,
        "valid_candidates": 1,
        "first_valid_hypothesis_seconds": 15.0,
        "end_to_end_gain_pct": 3.0,
        "correctness_violations": 0,
        "policy_violations": 0,
        "events": events,
        "event_evidence": [
            {"name": name, "ledger_sequence": index + 1, "source_sha256": "e" * 64}
            for index, name in enumerate(events)
        ],
        "run_identity": {
            "model_identity": "test-model@fixed",
            "prompt_sha256": prompt_sha256,
            "skill_sha256": "d" * 64,
            "contract_sha256": "c" * 64,
            "environment_sha256": "b" * 64,
            "seed": 7,
            "replicate": 1,
        },
    }
    value.update(overrides)
    return value


class SkillEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eval = _load_module()

    def test_validate_suite_requires_existing_contained_fixtures(self) -> None:
        validated = self.eval.validate_suite(_suite(), ROOT)
        self.assertEqual(validated["suite_id"], "v3-control")
        changed = _suite()
        changed["scenarios"][0]["fixture"] = "../outside.py"
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.eval.validate_suite(changed, ROOT)

    def test_repository_v3_suite_is_a_valid_five_arm_experiment(self) -> None:
        suite = json.loads((ROOT / "tests/evals/v3/scenarios.json").read_text("utf-8"))
        validated = self.eval.validate_suite(suite, ROOT)
        self.assertEqual(validated["experiment"]["arms"], [
            "no_skill",
            "v2.9",
            "v3_random_planner",
            "v3_shuffled_registry",
            "v3_full",
        ])
        self.assertIn(
            "combined-long-run-faults",
            {item["id"] for item in validated["scenarios"]},
        )
        scenarios = {item["id"]: item for item in validated["scenarios"]}
        self.assertEqual(
            scenarios["noisy-environment"]["fixture"],
            "tests/test_stability_calibration.py",
        )
        self.assertIn(
            "audit_cadence_enforced",
            scenarios["combined-long-run-faults"]["required_events"],
        )

    def test_suite_requires_the_preregistered_five_arm_matrix_and_prompt_hash(self) -> None:
        changed = _suite()
        changed["experiment"]["arms"].remove("v3_shuffled_registry")
        with self.assertRaisesRegex(ValueError, "arms|matrix"):
            self.eval.validate_suite(changed, ROOT)
        changed = _suite()
        changed["scenarios"][0]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prompt"):
            self.eval.validate_suite(changed, ROOT)

    def test_score_requires_one_result_per_scenario(self) -> None:
        suite = self.eval.validate_suite(_suite(), ROOT)
        results = [
            _result(
                "wrong-kernel-bottleneck",
                ["workload_profiled", "non_kernel_direction"],
            )
        ]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.eval.score_results(suite, results, mode="v2.9")

    def test_score_reports_policy_failures_and_efficiency_without_hiding_them(self) -> None:
        suite = self.eval.validate_suite(_suite(), ROOT)
        results = [
            _result(
                "wrong-kernel-bottleneck",
                ["workload_profiled", "unsupported_promotion"],
                elapsed_seconds=90.0,
                gpu_seconds=20.0,
                candidates=4,
                valid_candidates=1,
                policy_violations=1,
            ),
            _result(
                "resume-after-interrupt",
                ["checkpoint_restored"],
                elapsed_seconds=30.0,
                gpu_seconds=0.0,
                candidates=1,
                valid_candidates=1,
                end_to_end_gain_pct=None,
            ),
        ]
        score = self.eval.score_results(suite, results, mode="v2.9")
        self.assertEqual(score["mode"], "v2.9")
        self.assertEqual(score["scenarios_total"], 2)
        self.assertEqual(score["scenarios_passed"], 1)
        self.assertEqual(score["policy_violations"], 1)
        self.assertEqual(score["required_events_missing"], 1)
        self.assertEqual(score["forbidden_events_seen"], 1)
        self.assertAlmostEqual(score["valid_candidate_rate"], 2 / 5)
        self.assertEqual(score["gpu_seconds"], 20.0)
        self.assertEqual(len(score["run_identities"]), 2)

    def test_rejects_unknown_fields_nonfinite_metrics_and_duplicate_results(self) -> None:
        suite = self.eval.validate_suite(_suite(), ROOT)
        good = _result(
            "wrong-kernel-bottleneck",
            ["workload_profiled", "non_kernel_direction"],
        )
        second = _result("resume-after-interrupt", ["checkpoint_restored"])
        with self.subTest("unknown"):
            changed = dict(good, surprise=True)
            with self.assertRaisesRegex(ValueError, "unknown"):
                self.eval.score_results(suite, [changed, second], mode="v3_full")
        with self.subTest("nonfinite"):
            changed = dict(good, elapsed_seconds=float("nan"))
            with self.assertRaisesRegex(ValueError, "finite"):
                self.eval.score_results(suite, [changed, second], mode="v3_full")
        with self.subTest("duplicate"):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.eval.score_results(suite, [good, good], mode="v3_full")
        with self.subTest("unbound event"):
            changed = dict(good, event_evidence=[])
            with self.assertRaisesRegex(ValueError, "event.*evidence"):
                self.eval.score_results(suite, [changed, second], mode="v3_full")

    def test_cli_output_is_deterministic_and_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_path = root / "suite.json"
            result_path = root / "results.json"
            out = root / "score.json"
            suite_path.write_text(json.dumps(_suite()), "utf-8")
            result_path.write_text(
                json.dumps(
                    [
                        _result(
                            "wrong-kernel-bottleneck",
                            ["workload_profiled", "non_kernel_direction"],
                        ),
                        _result("resume-after-interrupt", ["checkpoint_restored"]),
                    ]
                ),
                "utf-8",
            )
            self.eval.run_score(suite_path, result_path, out, mode="v3_full", root=ROOT)
            first = out.read_bytes()
            with self.assertRaisesRegex(ValueError, "exists|create"):
                self.eval.run_score(suite_path, result_path, out, mode="v3_full", root=ROOT)
            self.assertIn(b'"schema_version": "cuda-skill-eval/score-v1"', first)


if __name__ == "__main__":
    unittest.main()
