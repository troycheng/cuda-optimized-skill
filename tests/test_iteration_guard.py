from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "cuda-kernel-optimizer"
SCRIPTS = SKILL / "scripts"
GUARD = SCRIPTS / "iteration_guard.py"
sys.path.insert(0, str(SCRIPTS))

import iteration_guard  # noqa: E402


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _registry(*, fallback: bool = True) -> dict:
    paths = [
        {
            "id": "paired-kernel",
            "version": "1",
            "definition_sha256": "a" * 64,
            "status": "validated",
        }
    ]
    if fallback:
        paths.append(
            {
                "id": "event-fallback",
                "version": "1",
                "definition_sha256": "b" * 64,
                "status": "validated",
            }
        )
    return {
        "schema_version": "cuda-optimizer/measurement-path-registry-v1",
        "paths": paths,
    }


def _path_binding(registry: dict) -> dict:
    selected = registry["paths"][0]
    return {
        "id": selected["id"],
        "version": selected["version"],
        "definition_sha256": selected["definition_sha256"],
        "registry_sha256": _digest(registry),
    }


def _identity(record: dict) -> dict:
    return {
        "baseline_snapshot_sha256": record["baseline"]["snapshot_sha256"],
        "candidate_snapshot_sha256": record["candidate"]["candidate_snapshot_sha256"],
        "environment_sha256": record["baseline"]["environment_sha256"],
        "measurement_path": {
            key: record["measurement_path"][key]
            for key in ("id", "version", "definition_sha256")
        },
    }


def _record(*, verdict: str = "confirmed_loss") -> dict:
    registry = _registry()
    record = {
        "schema_version": "cuda-optimizer/performance-iteration-v1",
        "round_id": "iter-131-fast32",
        "lineage_id": "fast32-selector",
        "hypothesis": {
            "statement": "Fusing class eligibility into the mask kernel lowers latency.",
            "mechanism": "fuse-class-eligibility",
            "target_metric": "latency_us",
            "direction": "lower",
            "minimum_effect_pct": 1.0,
            "mutation_scope": ["kernels/fast32.py"],
        },
        "budget": {
            "round_seconds": 2700,
            "infrastructure_seconds": 120,
            "infrastructure_repairs": 0,
        },
        "measurement_path": _path_binding(registry),
        "baseline": {
            "snapshot_sha256": "c" * 64,
            "environment_sha256": "d" * 64,
        },
        "candidate": {
            "candidate_id": "fast32-fused-mask",
            "baseline_snapshot_sha256": "c" * 64,
            "candidate_snapshot_sha256": "e" * 64,
            "environment_sha256": "d" * 64,
            "mechanism": "fuse-class-eligibility",
            "changed_paths": ["kernels/fast32.py"],
        },
        "correctness": None,
        "performance": None,
    }
    identity = _identity(record)
    record["correctness"] = {"status": "passed", **identity}
    record["performance"] = {
        "status": "completed",
        "verdict": verdict,
        "target_metric": "latency_us",
        "direction": "lower",
        "minimum_effect_pct": 1.0,
        "estimate_pct": -2.0 if verdict == "confirmed_loss" else 2.0,
        "ci_low_pct": -3.0 if verdict == "confirmed_loss" else 1.2,
        "ci_high_pct": -1.0 if verdict == "confirmed_loss" else 2.8,
        **identity,
    }
    return record


class IterationGuardClassificationTests(unittest.TestCase):
    def test_completed_loss_is_candidate_evaluated_not_performance_gain(self) -> None:
        registry = _registry()
        record = _record()
        result = iteration_guard.classify_iteration(record, registry)
        self.assertEqual(result["work_class"], "candidate_evaluated")
        self.assertEqual(result["performance_result"], "confirmed_loss")
        self.assertEqual(result["next_action"], "continue_candidate_search")
        self.assertNotIn("performance_gain", result["claims"])

    def test_confirmed_win_is_forwarded_but_never_claimed_as_a_gain(self) -> None:
        registry = _registry()
        record = _record(verdict="confirmed_win")
        result = iteration_guard.classify_iteration(record, registry)
        self.assertEqual(result["work_class"], "candidate_evaluated")
        self.assertEqual(result["performance_result"], "confirmed_win")
        self.assertEqual(result["claims"], ["candidate_evaluated"])
        self.assertNotIn("performance_gain", result["claims"])
        self.assertEqual(result["next_action"], "proceed_to_existing_promotion_gate")

    def test_correctness_failure_completes_a_real_candidate_without_timing(self) -> None:
        registry = _registry()
        record = _record()
        record["correctness"]["status"] = "failed"
        record["performance"] = None
        result = iteration_guard.classify_iteration(record, registry)
        self.assertEqual(result["work_class"], "candidate_evaluated")
        self.assertEqual(result["performance_result"], "correctness_failed")
        self.assertEqual(result["next_action"], "continue_candidate_search")

    def test_candidate_with_missing_measurement_is_blocked_and_uses_fallback(self) -> None:
        registry = _registry()
        record = _record()
        record["performance"] = None
        result = iteration_guard.classify_iteration(record, registry)
        self.assertEqual(result["work_class"], "measurement_blocked")
        self.assertEqual(result["next_action"], "switch_measurement_path")
        self.assertEqual(result["fallback_measurement_path"]["id"], "event-fallback")

    def test_infrastructure_only_is_not_reported_as_optimization(self) -> None:
        registry = _registry()
        record = _record()
        record["candidate"] = None
        record["correctness"] = None
        record["performance"] = None
        result = iteration_guard.classify_iteration(record, registry)
        self.assertEqual(result["work_class"], "infrastructure_only")
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["next_action"], "return_to_candidate")
        self.assertEqual(result["budget"]["infrastructure_cap_seconds"], 405)


class IterationGuardStopRuleTests(unittest.TestCase):
    def test_one_repair_and_fifteen_percent_are_the_hard_caps(self) -> None:
        registry = _registry()
        at_cap = _record()
        at_cap["candidate"] = None
        at_cap["correctness"] = None
        at_cap["performance"] = None
        at_cap["budget"]["infrastructure_seconds"] = 405
        at_cap["budget"]["infrastructure_repairs"] = 1
        self.assertEqual(
            iteration_guard.classify_iteration(at_cap, registry)["next_action"],
            "return_to_candidate",
        )

        over = deepcopy(at_cap)
        over["budget"]["infrastructure_seconds"] = 406
        result = iteration_guard.classify_iteration(over, registry)
        self.assertEqual(result["next_action"], "switch_measurement_path")
        self.assertIn("infrastructure_budget_exceeded", result["reasons"])

        repairs = deepcopy(at_cap)
        repairs["budget"]["infrastructure_repairs"] = 2
        result = iteration_guard.classify_iteration(repairs, registry)
        self.assertEqual(result["next_action"], "switch_measurement_path")
        self.assertIn("infrastructure_repair_limit_exceeded", result["reasons"])

    def test_twenty_minutes_is_an_absolute_cap_for_long_rounds(self) -> None:
        registry = _registry()
        record = _record()
        record["candidate"] = None
        record["correctness"] = None
        record["performance"] = None
        record["budget"]["round_seconds"] = 20000
        result = iteration_guard.classify_iteration(record, registry)
        self.assertEqual(result["budget"]["infrastructure_cap_seconds"], 1200)

    def test_second_consecutive_non_candidate_round_forces_fallback(self) -> None:
        registry = _registry()
        record = _record()
        record["candidate"] = None
        record["correctness"] = None
        record["performance"] = None
        history = [
            {
                "schema_version": "cuda-optimizer/iteration-decision-v1",
                "lineage_id": "fast32-selector",
                "work_class": "measurement_blocked",
            }
        ]
        result = iteration_guard.classify_iteration(record, registry, history)
        self.assertEqual(result["next_action"], "switch_measurement_path")
        self.assertIn("two_consecutive_non_candidate_rounds", result["reasons"])

    def test_other_lineage_does_not_trigger_the_consecutive_stop_rule(self) -> None:
        registry = _registry()
        record = _record()
        record["candidate"] = None
        record["correctness"] = None
        record["performance"] = None
        history = [
            {
                "schema_version": "cuda-optimizer/iteration-decision-v1",
                "lineage_id": "full79-selector",
                "work_class": "infrastructure_only",
            }
        ]
        result = iteration_guard.classify_iteration(record, registry, history)
        self.assertEqual(result["next_action"], "return_to_candidate")
        self.assertNotIn("two_consecutive_non_candidate_rounds", result["reasons"])

    def test_stop_direction_when_no_prevalidated_alternative_exists(self) -> None:
        registry = _registry(fallback=False)
        record = _record()
        record["measurement_path"] = _path_binding(registry)
        record["candidate"] = None
        record["correctness"] = None
        record["performance"] = None
        record["budget"]["infrastructure_repairs"] = 2
        result = iteration_guard.classify_iteration(record, registry)
        self.assertEqual(result["next_action"], "stop_direction")
        self.assertIsNone(result["fallback_measurement_path"])


class IterationGuardValidationTests(unittest.TestCase):
    def test_registry_digest_and_selected_path_are_frozen(self) -> None:
        registry = _registry()
        record = _record()
        record["measurement_path"]["registry_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "registry_sha256"):
            iteration_guard.classify_iteration(record, registry)

        record = _record()
        record["measurement_path"]["definition_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "measurement path"):
            iteration_guard.classify_iteration(record, registry)

    def test_candidate_must_be_a_real_bound_in_scope_change(self) -> None:
        registry = _registry()
        same = _record()
        same["candidate"]["candidate_snapshot_sha256"] = same["baseline"][
            "snapshot_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "must differ"):
            iteration_guard.classify_iteration(same, registry)

        escaped = _record()
        escaped["candidate"]["changed_paths"] = ["../runner.py"]
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            iteration_guard.classify_iteration(escaped, registry)

        out_of_scope = _record()
        out_of_scope["candidate"]["changed_paths"] = ["tools/runner.py"]
        with self.assertRaisesRegex(ValueError, "mutation_scope"):
            iteration_guard.classify_iteration(out_of_scope, registry)

    def test_evidence_identity_metric_and_mechanism_drift_fail_closed(self) -> None:
        registry = _registry()
        cases = []
        identity = _record()
        identity["performance"]["environment_sha256"] = "f" * 64
        cases.append((identity, "environment_sha256"))
        metric = _record()
        metric["performance"]["target_metric"] = "throughput"
        cases.append((metric, "target_metric"))
        mechanism = _record()
        mechanism["candidate"]["mechanism"] = "comment-only"
        cases.append((mechanism, "mechanism"))
        for record, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    iteration_guard.classify_iteration(record, registry)

    def test_confirmed_win_requires_confidence_interval_to_clear_minimum_effect(self) -> None:
        registry = _registry()
        record = _record(verdict="confirmed_win")
        record["performance"]["ci_low_pct"] = 0.5
        with self.assertRaisesRegex(ValueError, "confirmed_win"):
            iteration_guard.classify_iteration(record, registry)

    def test_unknown_keys_booleans_nonfinite_and_duplicate_json_are_rejected(self) -> None:
        registry = _registry()
        unknown = _record()
        unknown["extra"] = True
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            iteration_guard.classify_iteration(unknown, registry)

        boolean = _record()
        boolean["budget"]["round_seconds"] = True
        with self.assertRaisesRegex(ValueError, "round_seconds"):
            iteration_guard.classify_iteration(boolean, registry)

        nonfinite = _record()
        nonfinite["performance"]["estimate_pct"] = math.inf
        with self.assertRaisesRegex(ValueError, "estimate_pct"):
            iteration_guard.classify_iteration(nonfinite, registry)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                iteration_guard.load_json_strict(path)


class IterationGuardCliTests(unittest.TestCase):
    def test_help_and_check_work_without_site_packages_or_external_commands(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(GUARD), "--help"],
            cwd=ROOT,
            env={"PATH": ""},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("check", help_result.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = _registry()
            record = _record(verdict="confirmed_win")
            registry_path = root / "registry.json"
            record_path = root / "record.json"
            output_path = root / "decision.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            record_path.write_text(json.dumps(record), encoding="utf-8")
            command = [
                sys.executable,
                str(GUARD),
                "check",
                "--record",
                str(record_path),
                "--registry",
                str(registry_path),
                "--out",
                str(output_path),
            ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={"PATH": ""},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(decision["performance_result"], "confirmed_win")
            self.assertEqual(
                decision["next_action"], "proceed_to_existing_promotion_gate"
            )

            repeated = subprocess.run(
                command,
                cwd=ROOT,
                env={"PATH": ""},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("exists", repeated.stderr)

    def test_history_jsonl_is_strict_and_changes_the_stop_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = _registry()
            record = _record()
            record["candidate"] = None
            record["correctness"] = None
            record["performance"] = None
            paths = {
                "registry": root / "registry.json",
                "record": root / "record.json",
                "history": root / "history.jsonl",
                "out": root / "decision.json",
            }
            paths["registry"].write_text(json.dumps(registry), encoding="utf-8")
            paths["record"].write_text(json.dumps(record), encoding="utf-8")
            paths["history"].write_text(
                json.dumps(
                    {
                        "schema_version": "cuda-optimizer/iteration-decision-v1",
                        "lineage_id": "fast32-selector",
                        "work_class": "infrastructure_only",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(GUARD),
                    "check",
                    "--record",
                    str(paths["record"]),
                    "--registry",
                    str(paths["registry"]),
                    "--history",
                    str(paths["history"]),
                    "--out",
                    str(paths["out"]),
                ],
                cwd=ROOT,
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(paths["out"].read_text())["next_action"],
                "switch_measurement_path",
            )


if __name__ == "__main__":
    unittest.main()
