import importlib.util
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
    return {
        "schema_version": "cuda-skill-eval/suite-v1",
        "suite_id": "v3-control",
        "scenarios": [
            {
                "id": "wrong-kernel-bottleneck",
                "category": "direction",
                "fixture": "tests/gpu/sm120/fixtures/workload_probe.py",
                "claim_ceiling": "workload",
                "budget": {"max_seconds": 600, "max_candidates": 4},
                "required_events": ["workload_profiled", "non_kernel_direction"],
                "forbidden_events": ["unsupported_promotion"],
            },
            {
                "id": "resume-after-interrupt",
                "category": "recovery",
                "fixture": "tests/test_state_schema.py",
                "claim_ceiling": "kernel",
                "budget": {"max_seconds": 300, "max_candidates": 2},
                "required_events": ["checkpoint_restored"],
                "forbidden_events": ["repeated_failed_candidate"],
            },
        ],
    }


def _result(scenario_id: str, events: list[str], **overrides) -> dict:
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
                self.eval.score_results(suite, [changed, second], mode="v3")
        with self.subTest("nonfinite"):
            changed = dict(good, elapsed_seconds=float("nan"))
            with self.assertRaisesRegex(ValueError, "finite"):
                self.eval.score_results(suite, [changed, second], mode="v3")
        with self.subTest("duplicate"):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.eval.score_results(suite, [good, good], mode="v3")

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
            self.eval.run_score(suite_path, result_path, out, mode="v3", root=ROOT)
            first = out.read_bytes()
            with self.assertRaisesRegex(ValueError, "exists|create"):
                self.eval.run_score(suite_path, result_path, out, mode="v3", root=ROOT)
            self.assertIn(b'"schema_version": "cuda-skill-eval/score-v1"', first)


if __name__ == "__main__":
    unittest.main()
