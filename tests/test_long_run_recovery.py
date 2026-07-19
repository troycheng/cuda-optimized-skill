import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


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
        "mutation": {
            "project_paths": ["kernels"],
            "environment_root": str(environment),
            "host_policy": "recommend_only",
        },
        "evidence": {"max_age_seconds": 60},
    }


def _proposal() -> dict:
    return {
        "schema_version": "cuda-optimizer/candidate-proposal-v1",
        "candidate_id": "candidate-1",
        "observation_id": "obs-1",
        "hypothesis": "A bounded layout change should reduce latency.",
        "expected_metric": {"name": "latency", "direction": "lower"},
        "expected_effect_pct": 2.0,
        "kill_gate": "Kill when paired latency is not measurably lower.",
        "estimated_cost_seconds": 60,
        "capability_ids": [],
        "paths": ["kernels/op.py"],
    }


class LongRunRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = _load("workload_contract")
        cls.control = _load("run_control")
        cls.ledger = _load("evidence_ledger")

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
        return self.control.transition_run(
            contract_path,
            run_dir,
            "start_exploration",
            now=3.0,
            environment_state="green",
            measurable=True,
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
                evidence_age_seconds=5.0,
                now=10.0,
            )
            self.assertEqual(registered["state"]["active_candidate"]["candidate_id"], "candidate-1")

            reloaded = self.control.load_run(contract_path, run_dir)
            self.assertEqual(reloaded, registered)
            resolved = self.control.resolve_run_candidate(
                contract_path,
                run_dir,
                candidate_id="candidate-1",
                outcome="KILL",
                correctness_ok=True,
                performance_gate_passed=False,
                now=20.0,
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
