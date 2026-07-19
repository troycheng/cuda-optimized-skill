from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
SEAL_KEY = b"long-run-controller-secret" * 2


def _load(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cuda_{name}_recovery", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _draft(project: Path, environment: Path) -> dict:
    return {
        "schema_version": "cuda-optimizer/workload-contract-draft-v1",
        "run_id": "recovery-test",
        "parent_run": None,
        "requested_claim": "workload",
        "project_root": str(project),
        "artifacts": [{"role": "workload_manifest", "path": "workload.json"}],
        "workload": {
            "argv": ["python3", "workload.py"],
            "input_distribution": "fixture",
            "representative_cases": ["case-1"],
        },
        "objective": {
            "metric": "latency",
            "unit": "ms",
            "direction": "lower",
            "aggregation": "median",
            "minimum_practical_effect_pct": 1.0,
            "constraints": ["correctness"],
        },
        "budget": {"preset": "quick", "max_seconds": 600, "max_candidates": 2},
        "stability": {
            "confidence": 0.95,
            "power": 0.8,
            "bootstrap_samples": 2000,
            "min_valid_pairs": 4,
            "seed": 17,
            "audit_every_candidates": 1,
        },
        "mutation": {
            "project_paths": ["kernels"],
            "environment_root": str(environment),
            "host_policy": "recommend_only",
        },
        "evidence": {"max_age_seconds": 60},
    }


def _proposal(candidate_id: str = "candidate-1") -> dict:
    return {
        "schema_version": "cuda-optimizer/candidate-proposal-v1",
        "candidate_id": candidate_id,
        "observation_id": f"obs-{candidate_id}",
        "observation_summary_sha256": "d" * 64,
        "capability_query_sha256": "e" * 64,
        "hypothesis": "A bounded layout change should reduce latency.",
        "expected_metric": {"name": "latency", "direction": "lower"},
        "expected_effect_pct": 2.0,
        "kill_gate": "Kill when paired latency is not measurably lower.",
        "estimated_cost_seconds": 60,
        "capability_ids": ["test.capability"],
        "paths": ["kernels/op.py"],
    }


class LongRunRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = _load("workload_contract")
        cls.control = _load("run_control")
        cls.ledger = _load("evidence_ledger")
        cls.admission = _load("planner_admission")
        cls.stability = _load("stability_calibration")

    def _admission(
        self,
        contract_path: Path,
        *,
        now: float,
        age: float,
        proposal: dict | None = None,
    ) -> dict:
        proposal = _proposal() if proposal is None else proposal
        contract = json.loads(contract_path.read_text("utf-8"))
        gates = []
        for index, gate in enumerate(
            ("correctness_reference", "dispatch_identity", "target_compile_probe")
        ):
            gates.append(
                {
                    "gate": gate,
                    "satisfied": True,
                    "observation_ids": [f"obs-gate-{index}"],
                    "artifact_sha256s": [str(index + 1) * 64],
                }
            )
        return self.admission.seal_admission(
            {
                "schema_version": "cuda-optimizer/planner-admission-v1",
                "status": "ADMITTED",
                "run_id": "recovery-test",
                "ledger_id": "ledger-1",
                "contract_sha256": contract["contract_sha256"],
                "environment_sha256": "b" * 64,
                "candidate_id": proposal["candidate_id"],
                "observation_id": proposal["observation_id"],
                "observation_summary_sha256": proposal["observation_summary_sha256"],
                "capability_query_sha256": proposal["capability_query_sha256"],
                "capability_ids": proposal["capability_ids"],
                "admitted_at": now,
                "evidence_age_seconds": age,
                "pre_execution": {
                    "satisfied": True,
                    "missing_gates": [],
                    "gates": gates,
                },
            },
            controller_seal_key=SEAL_KEY,
        )

    def _run(self, root: Path) -> tuple[Path, Path, Path]:
        project = root / "project"
        project.mkdir()
        (project / "kernels").mkdir()
        (project / "workload.json").write_text('{"name":"fixture"}\n', "utf-8")
        environment = root / "env"
        environment.mkdir()
        run_dir = root / "run"
        run_dir.mkdir()
        contract_path = run_dir / "contract.json"
        self.contract.freeze_contract(_draft(project, environment), contract_path)
        return project, run_dir, contract_path

    def _exploring(self, run_dir: Path, contract_path: Path) -> dict:
        self.control.initialize_run(contract_path, run_dir, now=0.0)
        self.control.transition_run(contract_path, run_dir, "freeze", now=1.0)
        self.control.transition_run(contract_path, run_dir, "calibrate", now=2.0)
        calibration = self.stability.calibrate(
            contract_path=contract_path,
            blocks=[
                {"pair_id": f"pair-{index}", "first": 100.0, "second": value, "valid": True}
                for index, value in enumerate((100.1, 99.9, 100.2, 99.8), 1)
            ],
            hard_guardrails_passed=True,
            environment_sha256="b" * 64,
            source_sha256="c" * 64,
            recorded_at=3.0,
            controller_seal_key=SEAL_KEY,
        )
        return self.control.apply_run_calibration(
            contract_path,
            run_dir,
            calibration,
            now=3.0,
            controller_seal_key=SEAL_KEY,
        )

    def test_persistent_state_cannot_claim_green_without_controller_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, run_dir, contract_path = self._run(root)
            self.control.initialize_run(contract_path, run_dir, now=0.0)
            self.control.transition_run(contract_path, run_dir, "freeze", now=1.0)
            self.control.transition_run(contract_path, run_dir, "calibrate", now=2.0)
            with self.assertRaisesRegex(ValueError, "calibration|derived|stability"):
                self.control.transition_run(
                    contract_path,
                    run_dir,
                    "start_exploration",
                    now=3.0,
                    environment_state="green",
                    measurable=True,
                    controller_seal_key=SEAL_KEY,
                )
            loaded = self.control.load_run(
                contract_path, run_dir, controller_seal_key=SEAL_KEY
            )
            self.assertEqual(loaded["state"]["phase"], "CALIBRATING")
            self.assertEqual(loaded["event_count"], 3)

    def test_tampered_calibration_cannot_advance_or_append_to_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, run_dir, contract_path = self._run(root)
            self.control.initialize_run(contract_path, run_dir, now=0.0)
            self.control.transition_run(contract_path, run_dir, "freeze", now=1.0)
            calibrating = self.control.transition_run(
                contract_path, run_dir, "calibrate", now=2.0
            )
            calibration = self.stability.calibrate(
                contract_path=contract_path,
                blocks=[
                    {"pair_id": f"pair-{index}", "first": 100.0, "second": 100.0, "valid": True}
                    for index in range(1, 5)
                ],
                hard_guardrails_passed=True,
                environment_sha256="b" * 64,
                source_sha256="c" * 64,
                recorded_at=3.0,
                controller_seal_key=SEAL_KEY,
            )
            calibration["environment_state"] = "green" if calibration["environment_state"] != "green" else "yellow"
            with self.assertRaisesRegex(ValueError, "attestation|controller"):
                self.control.apply_run_calibration(
                    contract_path,
                    run_dir,
                    calibration,
                    now=3.0,
                    controller_seal_key=SEAL_KEY,
                )
            loaded = self.control.load_run(
                contract_path, run_dir, controller_seal_key=SEAL_KEY
            )
            self.assertEqual(loaded, calibrating)

    def test_contract_cadence_requires_attested_periodic_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, run_dir, contract_path = self._run(root)
            self._exploring(run_dir, contract_path)
            first = _proposal("candidate-1")
            self.control.register_run_candidate(
                contract_path,
                run_dir,
                first,
                admission=self._admission(
                    contract_path, now=10.0, age=1.0, proposal=first
                ),
                controller_seal_key=SEAL_KEY,
                now=10.0,
            )
            self.control.resolve_run_candidate(
                contract_path,
                run_dir,
                candidate_id="candidate-1",
                outcome="KILL",
                correctness_ok=True,
                performance_gate_passed=False,
                now=20.0,
                controller_seal_key=SEAL_KEY,
            )
            second = _proposal("candidate-2")
            with self.assertRaisesRegex(ValueError, "audit|cadence"):
                self.control.register_run_candidate(
                    contract_path,
                    run_dir,
                    second,
                    admission=self._admission(
                        contract_path, now=21.0, age=1.0, proposal=second
                    ),
                    controller_seal_key=SEAL_KEY,
                    now=21.0,
                )

            self.control.transition_run(
                contract_path,
                run_dir,
                "audit",
                now=21.0,
                controller_seal_key=SEAL_KEY,
            )
            records = self.ledger.verify_ledger(run_dir / "ledger")
            anchor = next(
                record["payload"]["calibration"]
                for record in records
                if record["event_type"] == "stability_calibrated"
            )
            audit = self.stability.audit(
                anchor,
                contract_path=contract_path,
                blocks=[
                    {"pair_id": f"audit-{index}", "first": 100.0, "second": value, "valid": True}
                    for index, value in enumerate((100.1, 99.9, 100.2, 99.8), 1)
                ],
                hard_guardrails_passed=True,
                environment_sha256="b" * 64,
                source_sha256="c" * 64,
                recorded_at=22.0,
                controller_seal_key=SEAL_KEY,
            )
            resumed = self.control.apply_run_stability_audit(
                contract_path,
                run_dir,
                audit,
                now=22.0,
                controller_seal_key=SEAL_KEY,
            )
            self.assertEqual(resumed["state"]["phase"], "EXPLORING")
            registered = self.control.register_run_candidate(
                contract_path,
                run_dir,
                second,
                admission=self._admission(
                    contract_path, now=23.0, age=1.0, proposal=second
                ),
                controller_seal_key=SEAL_KEY,
                now=23.0,
            )
            self.assertEqual(
                registered["state"]["active_candidate"]["candidate_id"],
                "candidate-2",
            )

    def test_ledger_replay_rejects_candidate_past_audit_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, run_dir, contract_path = self._run(root)
            self._exploring(run_dir, contract_path)
            first = _proposal("candidate-1")
            self.control.register_run_candidate(
                contract_path,
                run_dir,
                first,
                admission=self._admission(
                    contract_path, now=10.0, age=1.0, proposal=first
                ),
                controller_seal_key=SEAL_KEY,
                now=10.0,
            )
            resolved = self.control.resolve_run_candidate(
                contract_path,
                run_dir,
                candidate_id="candidate-1",
                outcome="KILL",
                correctness_ok=True,
                performance_gate_passed=False,
                now=20.0,
                controller_seal_key=SEAL_KEY,
            )
            second = _proposal("candidate-2")
            admission = self._admission(
                contract_path, now=21.0, age=1.0, proposal=second
            )
            illegal_state = self.control.register_candidate(
                resolved["state"],
                second,
                admission=admission,
                controller_seal_key=SEAL_KEY,
                now=21.0,
            )
            records = self.ledger.verify_ledger(run_dir / "ledger")
            records.append(
                {
                    "event_type": "candidate_registered",
                    "payload": {
                        "proposal": second,
                        "admission": admission,
                        "now": 21.0,
                        "state": illegal_state,
                    },
                }
            )
            contract = self.contract.verify_frozen_contract(contract_path)
            with self.assertRaisesRegex(ValueError, "audit|cadence"):
                self.control._replay_records(
                    contract,
                    records,
                    contract_path=contract_path,
                    controller_seal_key=SEAL_KEY,
                )

    def test_reload_replays_every_event_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, run_dir, contract_path = self._run(root)
            self._exploring(run_dir, contract_path)
            registered = self.control.register_run_candidate(
                contract_path,
                run_dir,
                _proposal(),
                admission=self._admission(contract_path, now=10.0, age=5.0),
                controller_seal_key=SEAL_KEY,
                now=10.0,
            )
            self.assertEqual(registered["state"]["active_candidate"]["candidate_id"], "candidate-1")

            reloaded = self.control.load_run(
                contract_path, run_dir, controller_seal_key=SEAL_KEY
            )
            self.assertEqual(reloaded, registered)
            resolved = self.control.resolve_run_candidate(
                contract_path,
                run_dir,
                candidate_id="candidate-1",
                outcome="KILL",
                correctness_ok=True,
                performance_gate_passed=False,
                now=20.0,
                controller_seal_key=SEAL_KEY,
            )
            self.assertEqual(resolved["state"]["candidate_history"][0]["outcome"], "KILL")
            self.assertEqual(resolved["event_count"], 6)

    def test_stale_writer_cannot_overwrite_a_newer_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, run_dir, contract_path = self._run(root)
            initialized = self.control.initialize_run(contract_path, run_dir, now=0.0)
            stale_tail = initialized["tail_sha256"]
            self.control.transition_run(contract_path, run_dir, "freeze", now=1.0)
            with self.assertRaisesRegex(ValueError, "stale|tail"):
                self.control.transition_run(
                    contract_path,
                    run_dir,
                    "freeze",
                    now=1.0,
                    expected_tail_sha256=stale_tail,
                )

    def test_child_contract_can_verify_a_parent_with_admitted_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, run_dir, contract_path = self._run(root)
            self._exploring(run_dir, contract_path)
            self.control.register_run_candidate(
                contract_path,
                run_dir,
                _proposal(),
                admission=self._admission(contract_path, now=10.0, age=5.0),
                controller_seal_key=SEAL_KEY,
                now=10.0,
            )
            self.control.resolve_run_candidate(
                contract_path,
                run_dir,
                candidate_id="candidate-1",
                outcome="KILL",
                correctness_ok=True,
                performance_gate_passed=False,
                now=20.0,
                controller_seal_key=SEAL_KEY,
            )
            stopped = self.control.transition_run(
                contract_path,
                run_dir,
                "stop",
                now=21.0,
                reason="recalibration_required",
                controller_seal_key=SEAL_KEY,
            )
            parent = json.loads(contract_path.read_text("utf-8"))
            child = _draft(project, root / "env")
            child["run_id"] = "recovery-child"
            child["parent_run"] = {
                "run_id": parent["run_id"],
                "contract_sha256": parent["contract_sha256"],
                "ledger_tail_sha256": stopped["tail_sha256"],
            }
            frozen = self.contract.freeze_contract(
                child,
                root / "child-contract.json",
                parent_contract_path=contract_path,
                parent_run_dir=run_dir,
                controller_seal_key=SEAL_KEY,
            )

        self.assertEqual(frozen["parent_run"], child["parent_run"])

    def test_contract_artifact_drift_blocks_resume_before_state_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, run_dir, contract_path = self._run(root)
            initialized = self.control.initialize_run(contract_path, run_dir, now=0.0)
            (project / "workload.json").write_text('{"name":"changed"}\n', "utf-8")
            with self.assertRaisesRegex(ValueError, "changed|identity|sha256"):
                self.control.transition_run(contract_path, run_dir, "freeze", now=1.0)
            records = self.ledger.verify_ledger(run_dir / "ledger")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["record_sha256"], initialized["tail_sha256"])

    def test_resealed_but_illegal_state_snapshot_fails_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _project, run_dir, contract_path = self._run(root)
            initialized = self.control.initialize_run(contract_path, run_dir, now=0.0)
            records = self.ledger.verify_ledger(run_dir / "ledger")
            forged_state = dict(initialized["state"], phase="EXPLORING")
            self.ledger.append_event(
                run_dir / "ledger",
                event_type="state_transition",
                contract_sha256=initialized["state"]["contract_sha256"],
                expected_previous_sha256=records[-1]["record_sha256"],
                payload={
                    "action": "freeze",
                    "now": 1.0,
                    "arguments": {},
                    "state": forged_state,
                },
            )
            with self.assertRaisesRegex(ValueError, "replay|state|EXPLORING"):
                self.control.load_run(contract_path, run_dir)


if __name__ == "__main__":
    unittest.main()
