from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
SUMMARY_PATH = SCRIPTS / "evidence_summary.py"
LEDGER_PATH = SCRIPTS / "evidence_ledger.py"
CONTRACT_SHA = "a" * 64
ENVIRONMENT_SHA = "b" * 64
REFERENCE_SHA = "1" * 64
TARGET_SHA = "2" * 64
WORKLOAD_SHA = "9" * 64
ADAPTER_SHA = "e" * 64
ADAPTER_REQUEST_SHA = "0" * 64
RUN_ID = "run-1"
LEDGER_ID = "ledger-1"
SEAL_KEY = b"s" * 32
SUMMARY_LIMITS = {
    "run_id": RUN_ID,
    "ledger_id": LEDGER_ID,
    "max_observations": 16,
    "context_budget_bytes": 10000,
    "controller_seal_key": SEAL_KEY,
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload(
    artifact: Path,
    *,
    observation_id: str = "obs-correctness",
    kind: str = "correctness_reference",
    recorded_at: float = 100.0,
    environment_sha256: str = ENVIRONMENT_SHA,
) -> dict:
    producers = {
        "correctness_reference": "correctness-reference-adapter",
        "dispatch_identity": "dispatch-identity-adapter",
        "target_compile_probe": "compiler-evidence-adapter",
        "candidate_correctness": "candidate-correctness-adapter",
        "paired_measurement": "paired-measurement-adapter",
        "workload_replay": "workload-replay-adapter",
    }
    subjects = {
        "correctness_reference": {"reference_sha256": "1" * 64},
        "dispatch_identity": {"target_sha256": "2" * 64},
        "target_compile_probe": {"target_sha256": "2" * 64},
        "candidate_correctness": {
            "candidate_id": "candidate-1",
            "candidate_sha256": "3" * 64,
        },
        "paired_measurement": {
            "candidate_id": "candidate-1",
            "candidate_sha256": "3" * 64,
        },
        "workload_replay": {
            "candidate_id": "candidate-1",
            "candidate_sha256": "3" * 64,
        },
    }
    results = {
        "correctness_reference": {"oracle_sha256": "4" * 64, "cases_total": 8},
        "dispatch_identity": {"dispatch_sha256": "5" * 64, "cases_total": 8},
        "target_compile_probe": {
            "arch": "sm_120",
            "binary_sha256": "6" * 64,
            "compiler_sha256": "7" * 64,
        },
        "candidate_correctness": {
            "reference_sha256": "1" * 64,
            "cases_total": 8,
            "cases_passed": 8,
        },
        "paired_measurement": {
            "samples_sha256": "8" * 64,
            "pairs_total": 9,
            "decision": "PASS",
        },
        "workload_replay": {
            "workload_sha256": "9" * 64,
            "constraints_passed": True,
            "objective_gate_passed": True,
        },
    }
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "cuda-optimizer/gate-evidence-v1",
                "kind": kind,
                "producer": {
                    "id": producers[kind],
                    "version": "1.0.0",
                    "implementation_sha256": ADAPTER_SHA,
                },
                "adapter_request_sha256": ADAPTER_REQUEST_SHA,
                "contract_sha256": CONTRACT_SHA,
                "environment_sha256": environment_sha256,
                "recorded_at": recorded_at,
                "status": "PASS",
                "subject": subjects[kind],
                "result": results[kind],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    raw = artifact.read_bytes()
    return {
        "observation_id": observation_id,
        "artifact": {
            "path": artifact.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    }


class EvidenceSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = _load(SUMMARY_PATH, "cuda_v3_evidence_summary")
        cls.ledger = _load(LEDGER_PATH, "cuda_v3_summary_ledger")

    def _append(self, ledger: Path, payload: dict) -> None:
        self.summary._append_controller_gate_observation(
            ledger,
            artifact_root=ledger.parent,
            contract_sha256=CONTRACT_SHA,
            environment_sha256=ENVIRONMENT_SHA,
            run_id=RUN_ID,
            ledger_id=LEDGER_ID,
            observation_id=payload["observation_id"],
            artifact=payload["artifact"],
            adapter_implementation_sha256=ADAPTER_SHA,
            adapter_request_sha256=ADAPTER_REQUEST_SHA,
            as_of=110.0,
            max_age_seconds=60.0,
            controller_seal_key=SEAL_KEY,
        )

    def test_summary_uses_only_verified_hash_bound_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "correctness.json"
            artifact.write_text('{"passed":true}\n', encoding="utf-8")
            ledger = root / "ledger"
            self._append(ledger, _payload(artifact))
            self.ledger.append_event(
                ledger,
                event_type="candidate_registered",
                contract_sha256=CONTRACT_SHA,
                payload={"candidate_id": "ignored"},
            )

            result = self.summary.build_summary(
                ledger,
                artifact_root=root,
                contract_sha256=CONTRACT_SHA,
                environment_sha256=ENVIRONMENT_SHA,
                as_of=110.0,
                max_age_seconds=60.0,
                **SUMMARY_LIMITS,
            )
            with self.assertRaisesRegex(ValueError, "attestation|seal"):
                self.summary.build_summary(
                    ledger,
                    artifact_root=root,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    run_id=RUN_ID,
                    ledger_id=LEDGER_ID,
                    as_of=110.0,
                    max_age_seconds=60.0,
                    max_observations=16,
                    context_budget_bytes=10000,
                    controller_seal_key=b"x" * 32,
                )

        self.assertEqual(result["schema_version"], "cuda-optimizer/observation-summary-v1")
        self.assertEqual(result["contract_sha256"], CONTRACT_SHA)
        self.assertEqual(result["environment_sha256"], ENVIRONMENT_SHA)
        self.assertEqual(len(result["observations"]), 1)
        observation = result["observations"][0]
        self.assertEqual(observation["freshness"], "current")
        self.assertEqual(observation["age_seconds"], 10.0)
        self.assertEqual(len(result["summary_sha256"]), 64)

    def test_stale_and_future_evidence_cannot_satisfy_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "ledger"
            for name, kind, recorded_at in (
                ("reference.json", "correctness_reference", 1.0),
                ("dispatch.json", "dispatch_identity", 200.0),
                ("compile.json", "target_compile_probe", 100.0),
            ):
                artifact = root / name
                artifact.write_text(name, encoding="utf-8")
                self._append(
                    ledger,
                    _payload(
                        artifact,
                        observation_id="obs-" + kind,
                        kind=kind,
                        recorded_at=recorded_at,
                    ),
                )
            summary = self.summary.build_summary(
                ledger,
                artifact_root=root,
                contract_sha256=CONTRACT_SHA,
                environment_sha256=ENVIRONMENT_SHA,
                as_of=110.0,
                max_age_seconds=60.0,
                **SUMMARY_LIMITS,
            )
            resolution = self.summary.resolve_gate_requirements(
                summary,
                {
                    "pre_execution": [
                        "correctness_reference",
                        "dispatch_identity",
                        "target_compile_probe",
                    ],
                    "promotion": [
                        "candidate_correctness",
                        "paired_measurement",
                        "workload_replay",
                    ],
                },
                ledger_path=ledger,
                artifact_root=root,
                expected_run_id=RUN_ID,
                expected_ledger_id=LEDGER_ID,
                expected_contract_sha256=CONTRACT_SHA,
                expected_environment_sha256=ENVIRONMENT_SHA,
                current_as_of=110.0,
                max_age_seconds=60.0,
                expected_ledger_tail_sha256=summary["ledger_tail_sha256"],
                expected_reference_sha256=REFERENCE_SHA,
                expected_target_sha256=TARGET_SHA,
                expected_workload_sha256=WORKLOAD_SHA,
                expected_arch="sm_120",
                controller_seal_key=SEAL_KEY,
            )

        by_id = {item["observation_id"]: item for item in summary["observations"]}
        self.assertEqual(by_id["obs-correctness_reference"]["freshness"], "stale")
        self.assertEqual(by_id["obs-dispatch_identity"]["freshness"], "future")
        self.assertEqual(by_id["obs-target_compile_probe"]["freshness"], "current")
        self.assertFalse(resolution["pre_execution"]["satisfied"])
        missing = resolution["pre_execution"]["missing_gates"]
        self.assertEqual(missing, ["correctness_reference", "dispatch_identity"])

    def test_gate_resolution_recomputes_freshness_at_controller_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "ledger"
            gates = [
                "correctness_reference",
                "dispatch_identity",
                "target_compile_probe",
            ]
            for index, kind in enumerate(gates):
                artifact = root / f"{kind}.json"
                self._append(
                    ledger,
                    _payload(
                        artifact,
                        observation_id=f"obs-{kind}",
                        kind=kind,
                        recorded_at=100.0 + index,
                    ),
                )
            old = self.summary.build_summary(
                ledger,
                artifact_root=root,
                contract_sha256=CONTRACT_SHA,
                environment_sha256=ENVIRONMENT_SHA,
                as_of=110.0,
                max_age_seconds=60.0,
                **SUMMARY_LIMITS,
            )
            resolution = self.summary.resolve_gate_requirements(
                old,
                {
                    "pre_execution": gates,
                    "promotion": [
                        "candidate_correctness",
                        "paired_measurement",
                        "workload_replay",
                    ],
                },
                ledger_path=ledger,
                artifact_root=root,
                expected_run_id=RUN_ID,
                expected_ledger_id=LEDGER_ID,
                expected_contract_sha256=CONTRACT_SHA,
                expected_environment_sha256=ENVIRONMENT_SHA,
                current_as_of=200.0,
                max_age_seconds=60.0,
                expected_ledger_tail_sha256=old["ledger_tail_sha256"],
                expected_reference_sha256=REFERENCE_SHA,
                expected_target_sha256=TARGET_SHA,
                expected_workload_sha256=WORKLOAD_SHA,
                expected_arch="sm_120",
                controller_seal_key=SEAL_KEY,
            )

        self.assertFalse(resolution["pre_execution"]["satisfied"])
        self.assertEqual(resolution["pre_execution"]["missing_gates"], gates)

    def test_gate_resolution_matches_controller_target_and_candidate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "ledger"
            artifact = root / "compile.json"
            self._append(
                ledger,
                _payload(
                    artifact,
                    observation_id="obs-compile",
                    kind="target_compile_probe",
                ),
            )
            summary = self.summary.build_summary(
                ledger,
                artifact_root=root,
                contract_sha256=CONTRACT_SHA,
                environment_sha256=ENVIRONMENT_SHA,
                as_of=110.0,
                max_age_seconds=60.0,
                **SUMMARY_LIMITS,
            )
            resolution = self.summary.resolve_gate_requirements(
                summary,
                {
                    "pre_execution": [
                        "correctness_reference",
                        "dispatch_identity",
                        "target_compile_probe",
                    ],
                    "promotion": [
                        "candidate_correctness",
                        "paired_measurement",
                        "workload_replay",
                    ],
                },
                ledger_path=ledger,
                artifact_root=root,
                expected_run_id=RUN_ID,
                expected_ledger_id=LEDGER_ID,
                expected_contract_sha256=CONTRACT_SHA,
                expected_environment_sha256=ENVIRONMENT_SHA,
                current_as_of=110.0,
                max_age_seconds=60.0,
                expected_ledger_tail_sha256=summary["ledger_tail_sha256"],
                expected_reference_sha256=REFERENCE_SHA,
                expected_target_sha256="f" * 64,
                expected_workload_sha256=WORKLOAD_SHA,
                expected_arch="sm_120",
                controller_seal_key=SEAL_KEY,
            )

        self.assertIn(
            "target_compile_probe", resolution["pre_execution"]["missing_gates"]
        )

    def test_gate_resolution_rejects_wrong_arch_and_candidate_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "ledger"
            for name, kind in (
                ("compile.json", "target_compile_probe"),
                ("candidate.json", "candidate_correctness"),
            ):
                artifact = root / name
                self._append(
                    ledger,
                    _payload(
                        artifact,
                        observation_id="obs-" + kind,
                        kind=kind,
                    ),
                )
            summary = self.summary.build_summary(
                ledger,
                artifact_root=root,
                contract_sha256=CONTRACT_SHA,
                environment_sha256=ENVIRONMENT_SHA,
                as_of=110.0,
                max_age_seconds=60.0,
                **SUMMARY_LIMITS,
            )
            resolution = self.summary.resolve_gate_requirements(
                summary,
                {
                    "pre_execution": [
                        "correctness_reference",
                        "dispatch_identity",
                        "target_compile_probe",
                    ],
                    "promotion": [
                        "candidate_correctness",
                        "paired_measurement",
                        "workload_replay",
                    ],
                },
                ledger_path=ledger,
                artifact_root=root,
                expected_run_id=RUN_ID,
                expected_ledger_id=LEDGER_ID,
                expected_contract_sha256=CONTRACT_SHA,
                expected_environment_sha256=ENVIRONMENT_SHA,
                current_as_of=110.0,
                max_age_seconds=60.0,
                expected_ledger_tail_sha256=summary["ledger_tail_sha256"],
                expected_reference_sha256=REFERENCE_SHA,
                expected_target_sha256=TARGET_SHA,
                expected_workload_sha256=WORKLOAD_SHA,
                expected_candidate_id="candidate-1",
                expected_candidate_sha256="f" * 64,
                expected_arch="sm_121",
                controller_seal_key=SEAL_KEY,
            )

        self.assertIn(
            "target_compile_probe", resolution["pre_execution"]["missing_gates"]
        )
        self.assertIn(
            "candidate_correctness", resolution["promotion"]["missing_gates"]
        )

    def test_tampering_identity_mismatch_and_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "observation.json"
            ledger = root / "ledger"
            self._append(ledger, _payload(artifact))
            original = artifact.read_bytes()

            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash|size|identity"):
                self.summary.build_summary(
                    ledger,
                    artifact_root=root,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    as_of=110.0,
                    max_age_seconds=60.0,
                    **SUMMARY_LIMITS,
                )

            artifact.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "environment|attestation"):
                self.summary.build_summary(
                    ledger,
                    artifact_root=root,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256="c" * 64,
                    as_of=110.0,
                    max_age_seconds=60.0,
                    **SUMMARY_LIMITS,
                )

            external = root / "external.json"
            external.write_bytes(original)
            artifact.unlink()
            artifact.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                self.summary.build_summary(
                    ledger,
                    artifact_root=root,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    as_of=110.0,
                    max_age_seconds=60.0,
                    **SUMMARY_LIMITS,
                )

    def test_summary_observation_count_and_context_bytes_are_hard_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "ledger"
            for index in range(2):
                artifact = root / f"obs-{index}.json"
                artifact.write_text("x" * 100, encoding="utf-8")
                self._append(
                    ledger,
                    _payload(
                        artifact,
                        observation_id=f"obs-{index}",
                        kind="target_compile_probe",
                    ),
                )
            with self.assertRaisesRegex(ValueError, "observation.*limit|limit.*observation"):
                self.summary.build_summary(
                    ledger,
                    artifact_root=root,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    run_id=RUN_ID,
                    ledger_id=LEDGER_ID,
                    as_of=110.0,
                    max_age_seconds=60.0,
                    max_observations=1,
                    context_budget_bytes=10000,
                    controller_seal_key=SEAL_KEY,
                )
            with self.assertRaisesRegex(ValueError, "context.*budget|budget.*context"):
                self.summary.build_summary(
                    ledger,
                    artifact_root=root,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    run_id=RUN_ID,
                    ledger_id=LEDGER_ID,
                    as_of=110.0,
                    max_age_seconds=60.0,
                    max_observations=2,
                    context_budget_bytes=100,
                    controller_seal_key=SEAL_KEY,
                )

    def test_gate_kind_and_pass_status_are_derived_from_strict_artifact_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "fake.json"
            artifact.write_text('{"passed":true}', encoding="utf-8")
            raw = artifact.read_bytes()
            ledger = root / "ledger"
            fake_payload = {
                "observation_id": "obs-fake",
                "artifact": {
                    "path": artifact.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                },
            }
            with self.assertRaisesRegex(ValueError, "reserved|adapter"):
                self.ledger.append_event(
                    ledger,
                    event_type="observation_sealed",
                    contract_sha256=CONTRACT_SHA,
                    payload=fake_payload,
                )
            self.ledger._append_reserved_event(
                ledger,
                event_type="observation_sealed",
                contract_sha256=CONTRACT_SHA,
                payload=fake_payload,
            )
            with self.assertRaisesRegex(
                ValueError, "gate evidence|schema|artifact|attestation"
            ):
                self.summary.build_summary(
                    ledger,
                    artifact_root=root,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    as_of=110.0,
                    max_age_seconds=60.0,
                    **SUMMARY_LIMITS,
                )

            valid_payload = _payload(
                artifact,
                kind="candidate_correctness",
                observation_id="obs-bad-correctness",
            )
            content = json.loads(artifact.read_text(encoding="utf-8"))
            content["result"]["cases_passed"] = 7
            artifact.write_text(json.dumps(content, sort_keys=True), encoding="utf-8")
            raw = artifact.read_bytes()
            valid_payload["artifact"]["sha256"] = hashlib.sha256(raw).hexdigest()
            valid_payload["artifact"]["size_bytes"] = len(raw)
            ledger2 = root / "ledger2"
            with self.assertRaisesRegex(ValueError, "correctness|PASS|cases"):
                self.summary._append_controller_gate_observation(
                    ledger2,
                    artifact_root=root,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    run_id=RUN_ID,
                    ledger_id=LEDGER_ID,
                    observation_id=valid_payload["observation_id"],
                    artifact=valid_payload["artifact"],
                    adapter_implementation_sha256=ADAPTER_SHA,
                    adapter_request_sha256=ADAPTER_REQUEST_SHA,
                    as_of=110.0,
                    max_age_seconds=60.0,
                    controller_seal_key=SEAL_KEY,
                )

    def test_gate_resolution_rejects_unbound_or_forged_summary(self) -> None:
        with self.assertRaisesRegex(ValueError, "summary|digest|schema|controller-owned"):
            self.summary.resolve_gate_requirements(
                {
                    "schema_version": "cuda-optimizer/observation-summary-v1",
                    "observations": [
                        {"kind": "correctness_reference", "freshness": "current"}
                    ],
                },
                {
                    "pre_execution": ["correctness_reference"],
                    "promotion": [],
                },
            )


if __name__ == "__main__":
    unittest.main()
