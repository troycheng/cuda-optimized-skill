import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "run_control.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cuda_run_control_v3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _contract(**overrides) -> dict:
    value = {
        "schema_version": "cuda-optimizer/workload-contract-v1",
        "contract_sha256": "a" * 64,
        "budget": {
            "preset": "balanced",
            "max_seconds": 1000.0,
            "max_candidates": 2,
        },
        "evidence": {"max_age_seconds": 60.0},
        "objective": {"metric": "request_latency", "direction": "lower"},
        "mutation": {"project_paths": ["kernels"]},
        "project_root": str(ROOT),
    }
    value.update(overrides)
    return value


def _proposal(candidate_id: str = "candidate-1", **overrides) -> dict:
    value = {
        "schema_version": "cuda-optimizer/candidate-proposal-v1",
        "candidate_id": candidate_id,
        "observation_id": "obs-1",
        "hypothesis": "Coalescing the selected load should reduce request latency.",
        "expected_metric": {"name": "request_latency", "direction": "lower"},
        "expected_effect_pct": 2.0,
        "kill_gate": "Kill when paired latency does not improve measurably.",
        "estimated_cost_seconds": 120.0,
        "capability_ids": ["triton.coalesced-load"],
        "paths": ["kernels/attention.py"],
    }
    value.update(overrides)
    return value


def _exploring(control, *, now: float = 0.0) -> dict:
    state = control.initialize_state(_contract(), now=now)
    state = control.advance(state, "freeze", now=now)
    state = control.advance(state, "calibrate", now=now + 1)
    return control.advance(
        state,
        "start_exploration",
        now=now + 2,
        environment_state="green",
        measurable=True,
    )


class RunControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.control = _load_module()

    def test_legal_state_path_is_deterministic_and_illegal_transition_is_rejected(self) -> None:
        state = self.control.initialize_state(_contract(), now=10.0)
        self.assertEqual(state["phase"], "INIT")
        original = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, "transition"):
            self.control.advance(state, "start_exploration", now=11.0)
        self.assertEqual(state, original)

        for action, phase in (
            ("freeze", "FROZEN"),
            ("calibrate", "CALIBRATING"),
        ):
            state = self.control.advance(state, action, now=11.0)
            self.assertEqual(state["phase"], phase)
        state = self.control.advance(
            state,
            "start_exploration",
            now=12.0,
            environment_state="green",
            measurable=True,
        )
        state = self.control.advance(state, "audit", now=20.0)
        state = self.control.advance(state, "audit_pass", now=21.0)
        state = self.control.advance(state, "converge", now=22.0)
        state = self.control.advance(state, "stop", now=23.0, reason="no_eligible_direction")
        self.assertEqual(state["phase"], "STOPPED")
        self.assertEqual(state["stop_reason"], "no_eligible_direction")

    def test_candidate_is_preregistered_and_resolved_without_mutating_input(self) -> None:
        state = _exploring(self.control)
        before = copy.deepcopy(state)
        registered = self.control.register_candidate(
            state,
            _proposal(),
            contract_sha256="a" * 64,
            evidence_age_seconds=10.0,
            now=10.0,
        )
        self.assertEqual(state, before)
        self.assertEqual(registered["active_candidate"]["candidate_id"], "candidate-1")
        self.assertEqual(registered["candidates_started"], 1)

        resolved = self.control.resolve_candidate(
            registered,
            candidate_id="candidate-1",
            outcome="KILL",
            correctness_ok=True,
            performance_gate_passed=False,
            now=20.0,
        )
        self.assertIsNone(resolved["active_candidate"])
        self.assertIsNone(resolved["champion_candidate_id"])
        self.assertEqual(resolved["candidate_history"][0]["outcome"], "KILL")

    def test_pass_cannot_bypass_correctness_or_performance_gate(self) -> None:
        state = self.control.register_candidate(
            _exploring(self.control),
            _proposal(),
            contract_sha256="a" * 64,
            evidence_age_seconds=0.0,
            now=3.0,
        )
        for correctness, performance in ((False, True), (True, False), (True, True)):
            with self.subTest(correctness=correctness, performance=performance):
                with self.assertRaisesRegex(ValueError, "PASS|gate|correctness"):
                    self.control.resolve_candidate(
                        state,
                        candidate_id="candidate-1",
                        outcome="PASS",
                        correctness_ok=correctness,
                        performance_gate_passed=performance,
                        now=4.0,
                    )

    def test_promotion_is_fail_closed_until_verified_evidence_adapter_is_connected(self) -> None:
        state = self.control.register_candidate(
            _exploring(self.control),
            _proposal(),
            contract_sha256="a" * 64,
            evidence_age_seconds=0.0,
            now=3.0,
        )
        with self.assertRaisesRegex(ValueError, "verified evidence|PASS"):
            self.control.resolve_candidate(
                state,
                candidate_id="candidate-1",
                outcome="PASS",
                correctness_ok=True,
                performance_gate_passed=True,
                now=4.0,
            )

    def test_stale_evidence_drift_and_budget_stop_new_candidates(self) -> None:
        state = _exploring(self.control)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.control.register_candidate(
                state,
                _proposal(),
                contract_sha256="a" * 64,
                evidence_age_seconds=61.0,
                now=3.0,
            )
        with self.assertRaisesRegex(ValueError, "contract"):
            self.control.register_candidate(
                state,
                _proposal(),
                contract_sha256="b" * 64,
                evidence_age_seconds=1.0,
                now=3.0,
            )
        stopped = self.control.register_candidate(
            state,
            _proposal(),
            contract_sha256="a" * 64,
            evidence_age_seconds=1.0,
            now=1001.0,
        )
        self.assertEqual(stopped["phase"], "STOPPED")
        self.assertEqual(stopped["stop_reason"], "time_budget_exhausted")

        drifted = self.control.advance(state, "drift", now=5.0, reason="source_identity_changed")
        self.assertEqual(drifted["phase"], "DRIFTED")
        with self.assertRaisesRegex(ValueError, "EXPLORING"):
            self.control.register_candidate(
                drifted,
                _proposal(),
                contract_sha256="a" * 64,
                evidence_age_seconds=1.0,
                now=6.0,
            )

    def test_candidate_must_match_frozen_objective_and_mutation_roots(self) -> None:
        state = _exploring(self.control)
        wrong_metric = _proposal(
            expected_metric={"name": "throughput", "direction": "higher"}
        )
        with self.assertRaisesRegex(ValueError, "objective|metric"):
            self.control.register_candidate(
                state,
                wrong_metric,
                contract_sha256="a" * 64,
                evidence_age_seconds=0.0,
                now=3.0,
            )
        with self.assertRaisesRegex(ValueError, "mutation|allowed|paths"):
            self.control.register_candidate(
                state,
                _proposal(paths=["runtime/outside.py"]),
                contract_sha256="a" * 64,
                evidence_age_seconds=0.0,
                now=3.0,
            )

    def test_nested_symlink_cannot_escape_an_allowed_mutation_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            kernels = project / "kernels"
            outside = Path(tmp) / "outside"
            kernels.mkdir(parents=True)
            outside.mkdir()
            (kernels / "link").symlink_to(outside, target_is_directory=True)
            contract = _contract(project_root=str(project))
            state = self.control.initialize_state(contract, now=0.0)
            state = self.control.advance(state, "freeze", now=1.0)
            state = self.control.advance(state, "calibrate", now=2.0)
            state = self.control.advance(
                state,
                "start_exploration",
                now=3.0,
                environment_state="green",
                measurable=True,
            )
            with self.assertRaisesRegex(ValueError, "symlink|unsafe|mutation"):
                self.control.register_candidate(
                    state,
                    _proposal(paths=["kernels/link/out.py"]),
                    contract_sha256="a" * 64,
                    evidence_age_seconds=0.0,
                    now=4.0,
                )
    def test_yellow_pauses_for_audit_and_red_stops(self) -> None:
        state = _exploring(self.control)
        yellow = self.control.advance(
            state, "environment_yellow", now=4.0, reason="baseline_noise_increased"
        )
        self.assertEqual(yellow["phase"], "AUDITING")
        red = self.control.advance(
            state, "environment_red", now=4.0, reason="thermal_state_not_comparable"
        )
        self.assertEqual(red["phase"], "STOPPED")
        self.assertEqual(red["stop_reason"], "thermal_state_not_comparable")

        calibrating = self.control.advance(
            self.control.advance(
                self.control.initialize_state(_contract(), now=0.0),
                "freeze",
                now=1.0,
            ),
            "calibrate",
            now=2.0,
        )
        paused = self.control.advance(
            calibrating,
            "environment_yellow",
            now=3.0,
            reason="baseline_noise_increased",
        )
        self.assertEqual(paused["phase"], "CALIBRATING")
        with self.assertRaisesRegex(ValueError, "transition"):
            self.control.advance(paused, "audit_pass", now=4.0)

    def test_drifted_run_cannot_refreeze_or_mix_contracts(self) -> None:
        drifted = self.control.advance(
            _exploring(self.control),
            "drift",
            now=4.0,
            reason="source_identity_changed",
        )
        with self.assertRaisesRegex(ValueError, "transition"):
            self.control.advance(
                drifted,
                "refreeze",
                now=5.0,
                new_contract_sha256="b" * 64,
            )

    def test_candidate_resolved_after_deadline_cannot_promote_and_stops_run(self) -> None:
        state = self.control.register_candidate(
            _exploring(self.control),
            _proposal(estimated_cost_seconds=10.0),
            contract_sha256="a" * 64,
            evidence_age_seconds=0.0,
            now=900.0,
        )
        stopped = self.control.resolve_candidate(
            state,
            candidate_id="candidate-1",
            outcome="DEFERRED",
            correctness_ok=None,
            performance_gate_passed=None,
            now=1001.0,
        )
        self.assertEqual(stopped["phase"], "STOPPED")
        self.assertEqual(stopped["stop_reason"], "time_budget_exhausted")

    def test_idle_deadline_and_candidate_count_exhaustion_persist_stop_state(self) -> None:
        state = _exploring(self.control)
        timed_out = self.control.advance(state, "audit", now=1001.0)
        self.assertEqual(timed_out["phase"], "STOPPED")
        self.assertEqual(timed_out["stop_reason"], "time_budget_exhausted")

        one_candidate_contract = _contract(
            budget={"preset": "quick", "max_seconds": 1000.0, "max_candidates": 1}
        )
        state = self.control.initialize_state(one_candidate_contract, now=0.0)
        state = self.control.advance(state, "freeze", now=1.0)
        state = self.control.advance(state, "calibrate", now=2.0)
        state = self.control.advance(
            state,
            "start_exploration",
            now=3.0,
            environment_state="green",
            measurable=True,
        )
        state = self.control.register_candidate(
            state,
            _proposal(),
            contract_sha256="a" * 64,
            evidence_age_seconds=0.0,
            now=4.0,
        )
        state = self.control.resolve_candidate(
            state,
            candidate_id="candidate-1",
            outcome="KILL",
            correctness_ok=True,
            performance_gate_passed=False,
            now=5.0,
        )
        stopped = self.control.register_candidate(
            state,
            _proposal(candidate_id="candidate-2"),
            contract_sha256="a" * 64,
            evidence_age_seconds=0.0,
            now=6.0,
        )
        self.assertEqual(stopped["phase"], "STOPPED")
        self.assertEqual(stopped["stop_reason"], "candidate_budget_exhausted")

    def test_proposal_is_closed_and_cannot_set_control_or_promotion_fields(self) -> None:
        valid = self.control.validate_candidate_proposal(_proposal())
        self.assertEqual(valid["candidate_id"], "candidate-1")
        for field in ("budget", "promotion", "contract_sha256"):
            proposal = _proposal()
            proposal[field] = "forbidden"
            with self.subTest(field), self.assertRaisesRegex(ValueError, "unknown"):
                self.control.validate_candidate_proposal(proposal)

    def test_persisted_state_rejects_forged_candidate_counts_and_promotion(self) -> None:
        state = _exploring(self.control)
        state["candidates_started"] = 1
        with self.assertRaisesRegex(ValueError, "candidates_started|history"):
            self.control.register_candidate(
                state,
                _proposal(),
                contract_sha256="a" * 64,
                evidence_age_seconds=0.0,
                now=3.0,
            )

        registered = self.control.register_candidate(
            _exploring(self.control),
            _proposal(),
            contract_sha256="a" * 64,
            evidence_age_seconds=0.0,
            now=3.0,
        )
        registered["active_candidate"]["promotion"] = True
        with self.assertRaisesRegex(ValueError, "unknown|active_candidate"):
            self.control.resolve_candidate(
                registered,
                candidate_id="candidate-1",
                outcome="PASS",
                correctness_ok=True,
                performance_gate_passed=True,
                now=4.0,
            )

        killed = self.control.resolve_candidate(
            self.control.register_candidate(
                _exploring(self.control),
                _proposal(),
                contract_sha256="a" * 64,
                evidence_age_seconds=0.0,
                now=3.0,
            ),
            candidate_id="candidate-1",
            outcome="KILL",
            correctness_ok=True,
            performance_gate_passed=False,
            now=4.0,
        )
        killed["champion_candidate_id"] = "candidate-1"
        with self.assertRaisesRegex(ValueError, "champion"):
            self.control.advance(killed, "audit", now=5.0)


if __name__ == "__main__":
    unittest.main()
