from __future__ import annotations

import hashlib
import json
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

import evidence_protocol  # noqa: E402
import iteration_guard  # noqa: E402
from tests.test_evidence_protocol import _build_attempt, _file_sha, _write_json  # noqa: E402


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _registry(*, fallback: bool = True, alias: bool = False) -> dict:
    runner_sha256 = hashlib.sha256(b"print('runner')\n").hexdigest()
    paths = [
        {
            "id": "paired-kernel",
            "version": "1",
            "definition_sha256": runner_sha256,
            "status": "validated",
        }
    ]
    if fallback:
        paths.append(
            {
                "id": "event-fallback",
                "version": "1",
                "definition_sha256": runner_sha256 if alias else "b" * 64,
                "status": "validated",
            }
        )
    return {
        "schema_version": "cuda-optimizer/measurement-path-registry-v1",
        "paths": paths,
    }


def _path(registry: dict, index: int = 0) -> dict:
    return {
        key: registry["paths"][index][key]
        for key in ("id", "version", "definition_sha256")
    }


def _anchor(registry: dict | None = None) -> dict:
    registry = registry or _registry()
    return iteration_guard.freeze_lineage(
        registry,
        baseline_source_sha256="c" * 64,
        environment_sha256="d" * 64,
        initial_measurement_path=_path(registry),
    )


def _record(
    anchor: dict,
    *,
    candidate_declared: bool = True,
    evidence_manifest_sha256: str | None = None,
    round_index: int = 1,
    previous_decision_sha256: str | None = None,
) -> dict:
    return {
        "schema_version": "cuda-optimizer/performance-iteration-v1",
        "round_id": f"iter-{round_index:04d}-fast32",
        "round_index": round_index,
        "anchor_sha256": _digest(anchor),
        "previous_decision_sha256": previous_decision_sha256,
        "hypothesis": {
            "statement": "Fusing class eligibility into the mask kernel lowers latency.",
            "mechanism": "fuse-class-eligibility",
            "target_metric": "latency_us",
            "direction": "lower",
            "minimum_effect_pct": 1.0,
            "mutation_scope": ["source.cu"],
        },
        "budget": {
            "round_seconds": 2700,
            "infrastructure_seconds": 120,
            "infrastructure_repairs": 0,
        },
        "measurement_path": dict(anchor["initial_measurement_path"]),
        "candidate_declared": candidate_declared,
        "evidence_manifest_sha256": evidence_manifest_sha256,
    }


def _closure(
    root: Path,
    anchor: dict,
    record: dict,
    *,
    verdict: str = "confirmed_loss",
    state: str = "valid",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    attempt = _build_attempt(root, state=state, claim_layer="isolated_operator")
    performance_path = root / "performance.json"
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["status"] = verdict
    performance["promotional_eligible"] = verdict == "confirmed_win"
    _write_json(performance_path, performance)
    binding_path = root / "iteration-binding.json"
    _write_json(
        binding_path,
        iteration_guard.make_iteration_binding(
            anchor, record, source_path="source.cu"
        ),
    )
    attempt_payload = json.loads(attempt.read_text(encoding="utf-8"))
    attempt_payload["artifacts"].append(
        {
            "id": "iteration_binding",
            "kind": "iteration_binding",
            "path": binding_path.name,
        }
    )
    _write_json(attempt, attempt_payload)
    evidence_protocol.seal_attempt(attempt, root / "seal.json")
    evidence_protocol.audit_seal(root / "seal.json", root / "audit.json")
    evidence_protocol.decide_attempt(
        root / "seal.json",
        root / "audit.json",
        root / "decision.json",
        root / "manifest.json",
    )
    return root / "manifest.json"


class IterationGuardEvidenceTests(unittest.TestCase):
    def test_real_sealed_loss_is_candidate_evaluated_without_gain_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            anchor = _anchor()
            record = _record(anchor)
            manifest = _closure(Path(tmp), anchor, record, verdict="confirmed_loss")
            record = _record(anchor, evidence_manifest_sha256=_file_sha(manifest))
            result = iteration_guard.classify_iteration(
                record, anchor, evidence_manifest=manifest
            )
            self.assertEqual(result["work_class"], "candidate_evaluated")
            self.assertEqual(result["performance_result"], "confirmed_loss")
            self.assertEqual(result["next_action"], "continue_candidate_search")
            self.assertNotIn("claims", result)

    def test_confirmed_win_is_forwarded_only_to_existing_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            anchor = _anchor()
            record = _record(anchor)
            manifest = _closure(Path(tmp), anchor, record, verdict="confirmed_win")
            record = _record(anchor, evidence_manifest_sha256=_file_sha(manifest))
            result = iteration_guard.classify_iteration(
                record, anchor, evidence_manifest=manifest
            )
            self.assertEqual(result["performance_result"], "confirmed_win")
            self.assertEqual(result["next_action"], "proceed_to_existing_promotion_gate")

    def test_confirmed_win_from_a_retained_v25_attempt_is_not_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            anchor = _anchor()
            record = _record(anchor)
            manifest = _closure(
                Path(tmp), anchor, record, verdict="confirmed_win", state="partial"
            )
            record = _record(anchor, evidence_manifest_sha256=_file_sha(manifest))
            result = iteration_guard.classify_iteration(
                record, anchor, evidence_manifest=manifest
            )
            self.assertEqual(result["performance_result"], "confirmed_win")
            self.assertEqual(result["next_action"], "continue_candidate_search")

    def test_rewritten_decision_is_rejected_even_when_closure_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = _anchor()
            provisional = _record(anchor)
            manifest = _closure(
                root,
                anchor,
                provisional,
                verdict="confirmed_win",
                state="partial",
            )
            decision_path = root / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision"] = "promote"
            decision_path.chmod(0o600)
            _write_json(decision_path, decision)
            closure = json.loads(manifest.read_text(encoding="utf-8"))
            closure["evidence_refs"]["decision"]["sha256"] = _file_sha(decision_path)
            manifest.chmod(0o600)
            _write_json(manifest, closure)
            record = _record(anchor, evidence_manifest_sha256=_file_sha(manifest))
            with self.assertRaisesRegex(ValueError, "semantic|decision"):
                iteration_guard.classify_iteration(
                    record,
                    anchor,
                    evidence_manifest=manifest,
                )

    def test_inline_or_forged_digest_cannot_create_candidate_evaluated(self) -> None:
        anchor = _anchor()
        record = _record(anchor, evidence_manifest_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "sealed V2.5 evidence"):
            iteration_guard.classify_iteration(record, anchor)
        forged = deepcopy(record)
        forged["performance"] = {"status": "confirmed_win"}
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            iteration_guard.classify_iteration(forged, anchor)

    def test_tampered_sealed_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = _anchor()
            record = _record(anchor)
            manifest = _closure(root, anchor, record)
            (root / "source.cu").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity|audit"):
                iteration_guard.load_v25_closure(manifest, _file_sha(manifest))

    def test_candidate_source_must_differ_from_frozen_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = _registry()
            initial_anchor = _anchor(registry)
            initial_record = _record(initial_anchor)
            manifest = _closure(root, initial_anchor, initial_record)
            anchor = iteration_guard.freeze_lineage(
                registry,
                baseline_source_sha256=_file_sha(root / "source.cu"),
                environment_sha256="d" * 64,
                initial_measurement_path=_path(registry),
            )
            record = _record(anchor, evidence_manifest_sha256=_file_sha(manifest))
            with self.assertRaisesRegex(ValueError, "baseline"):
                iteration_guard.classify_iteration(
                    record, anchor, evidence_manifest=manifest
                )

    def test_closure_is_bound_to_environment_path_and_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            anchor = _anchor()
            original = _record(anchor)
            manifest = _closure(Path(tmp), anchor, original)

            changed_environment = deepcopy(anchor)
            changed_environment["environment_sha256"] = "f" * 64
            changed_record = _record(
                changed_environment,
                evidence_manifest_sha256=_file_sha(manifest),
            )
            with self.assertRaisesRegex(ValueError, "iteration binding"):
                iteration_guard.classify_iteration(
                    changed_record,
                    changed_environment,
                    evidence_manifest=manifest,
                )

            changed_hypothesis = _record(
                anchor,
                evidence_manifest_sha256=_file_sha(manifest),
            )
            changed_hypothesis["hypothesis"]["minimum_effect_pct"] = 2.0
            with self.assertRaisesRegex(ValueError, "iteration binding"):
                iteration_guard.classify_iteration(
                    changed_hypothesis,
                    anchor,
                    evidence_manifest=manifest,
                )


class IterationGuardBudgetAndChainTests(unittest.TestCase):
    def test_budget_caps_and_twenty_minute_absolute_limit(self) -> None:
        anchor = _anchor()
        record = _record(anchor, candidate_declared=False)
        result = iteration_guard.classify_iteration(record, anchor)
        self.assertEqual(result["budget"]["infrastructure_cap_seconds"], 405)
        self.assertEqual(result["next_action"], "return_to_candidate")

        record["budget"]["round_seconds"] = 20000
        record["budget"]["infrastructure_seconds"] = 1201
        record["budget"]["infrastructure_repairs"] = 2
        result = iteration_guard.classify_iteration(record, anchor)
        self.assertEqual(result["budget"]["infrastructure_cap_seconds"], 1200)
        self.assertIn("infrastructure_budget_exceeded", result["reasons"])
        self.assertIn("infrastructure_repair_limit_exceeded", result["reasons"])

    def test_second_non_candidate_round_requires_bound_previous_decision(self) -> None:
        anchor = _anchor()
        first_record = _record(anchor, candidate_declared=False)
        first = iteration_guard.classify_iteration(first_record, anchor)
        second_record = _record(
            anchor,
            candidate_declared=True,
            round_index=2,
            previous_decision_sha256=_digest(first),
        )
        second = iteration_guard.classify_iteration(second_record, anchor, previous=first)
        self.assertEqual(second["next_action"], "switch_measurement_path")
        self.assertIn("two_consecutive_non_candidate_rounds", second["reasons"])

        with self.assertRaisesRegex(ValueError, "previous"):
            iteration_guard.classify_iteration(second_record, anchor, previous=None)

    def test_previous_decision_from_other_anchor_or_reordered_round_is_rejected(self) -> None:
        anchor = _anchor()
        other = _anchor(_registry(fallback=False))
        previous = iteration_guard.classify_iteration(
            _record(other, candidate_declared=False), other
        )
        record = _record(
            anchor,
            candidate_declared=False,
            round_index=2,
            previous_decision_sha256=_digest(previous),
        )
        with self.assertRaisesRegex(ValueError, "anchor|registry"):
            iteration_guard.classify_iteration(record, anchor, previous=previous)

    def test_previous_switch_or_stop_action_is_mandatory(self) -> None:
        anchor = _anchor()
        first_record = _record(anchor, candidate_declared=True)
        first = iteration_guard.classify_iteration(first_record, anchor)
        self.assertEqual(first["next_action"], "switch_measurement_path")
        second_record = _record(
            anchor,
            candidate_declared=True,
            round_index=2,
            previous_decision_sha256=_digest(first),
        )
        with self.assertRaisesRegex(ValueError, "fallback"):
            iteration_guard.classify_iteration(second_record, anchor, previous=first)
        second_record["measurement_path"] = first["fallback_measurement_path"]
        iteration_guard.classify_iteration(second_record, anchor, previous=first)

        no_fallback_anchor = _anchor(_registry(fallback=False))
        stopped_record = _record(no_fallback_anchor, candidate_declared=True)
        stopped = iteration_guard.classify_iteration(stopped_record, no_fallback_anchor)
        self.assertEqual(stopped["next_action"], "stop_direction")
        after_stop = _record(
            no_fallback_anchor,
            candidate_declared=False,
            round_index=2,
            previous_decision_sha256=_digest(stopped),
        )
        with self.assertRaisesRegex(ValueError, "stopped"):
            iteration_guard.classify_iteration(
                after_stop, no_fallback_anchor, previous=stopped
            )

    def test_fallback_requires_a_different_frozen_definition(self) -> None:
        registry = _registry(alias=True)
        anchor = _anchor(registry)
        record = _record(anchor, candidate_declared=True)
        result = iteration_guard.classify_iteration(record, anchor)
        self.assertEqual(result["next_action"], "stop_direction")
        self.assertIsNone(result["fallback_measurement_path"])


class IterationGuardValidationTests(unittest.TestCase):
    def test_anchor_identity_is_derived_and_record_is_closed(self) -> None:
        anchor = _anchor()
        record = _record(anchor, candidate_declared=False)
        record["anchor_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "anchor_sha256"):
            iteration_guard.classify_iteration(record, anchor)

        record = _record(anchor, candidate_declared=False)
        record["lineage_id"] = "reset-me"
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            iteration_guard.classify_iteration(record, anchor)

    def test_measurement_path_must_come_from_frozen_anchor(self) -> None:
        anchor = _anchor()
        record = _record(anchor, candidate_declared=False)
        record["measurement_path"]["definition_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "measurement path"):
            iteration_guard.classify_iteration(record, anchor)


class IterationGuardCliTests(unittest.TestCase):
    def test_binding_command_creates_sealable_context_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = _anchor()
            record = _record(anchor)
            anchor_path = root / "iteration-anchor.json"
            record_path = root / "round-0001.json"
            out = root / "iteration-binding.json"
            anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
            record_path.write_text(json.dumps(record), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(GUARD),
                    "binding",
                    "--anchor",
                    str(anchor_path),
                    "--record",
                    str(record_path),
                    "--source-path",
                    "source.cu",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                env={"PATH": ""},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(out.read_text(encoding="utf-8"))["schema_version"],
                "cuda-optimizer/iteration-binding-v1",
            )

    def test_init_and_check_use_canonical_create_once_run_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = _registry()
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            anchor_path = root / "iteration-anchor.json"
            init = subprocess.run(
                [
                    sys.executable,
                    str(GUARD),
                    "init",
                    "--registry",
                    str(registry_path),
                    "--baseline-source-sha256",
                    "c" * 64,
                    "--environment-sha256",
                    "d" * 64,
                    "--measurement-path",
                    "paired-kernel@1",
                    "--out",
                    str(anchor_path),
                ],
                cwd=ROOT,
                env={"PATH": ""},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(init.returncode, 0, init.stderr)
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))

            provisional = _record(anchor)
            manifest = _closure(
                root / "evidence", anchor, provisional, verdict="confirmed_win"
            )
            record = _record(anchor, evidence_manifest_sha256=_file_sha(manifest))
            record_path = root / "round-0001.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            out = root / "round-0001-decision.json"
            check = subprocess.run(
                [
                    sys.executable,
                    str(GUARD),
                    "check",
                    "--anchor",
                    str(anchor_path),
                    "--record",
                    str(record_path),
                    "--evidence-manifest",
                    str(manifest),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertEqual(json.loads(out.read_text())["performance_result"], "confirmed_win")

            repeated = subprocess.run(
                check.args,
                cwd=ROOT,
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
