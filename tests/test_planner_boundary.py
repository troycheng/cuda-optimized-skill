from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
MODULE_PATH = SCRIPTS / "planner_boundary.py"
CONTRACT_SHA = "a" * 64
ENVIRONMENT_SHA = "b" * 64
REFERENCE_SHA = "1" * 64
TARGET_SHA = "2" * 64
WORKLOAD_SHA = "9" * 64
ADAPTER_SHA = "e" * 64
REQUEST_SHA = "0" * 64
SEAL_KEY = b"planner-controller-secret" * 2


def _load():
    spec = importlib.util.spec_from_file_location("cuda_v3_planner_boundary", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _gate(kind: str, recorded_at: float = 100.0) -> dict:
    producers = {
        "correctness_reference": "correctness-reference-adapter",
        "dispatch_identity": "dispatch-identity-adapter",
        "target_compile_probe": "compiler-evidence-adapter",
    }
    subjects = {
        "correctness_reference": {"reference_sha256": REFERENCE_SHA},
        "dispatch_identity": {"target_sha256": TARGET_SHA},
        "target_compile_probe": {"target_sha256": TARGET_SHA},
    }
    results = {
        "correctness_reference": {"oracle_sha256": "4" * 64, "cases_total": 8},
        "dispatch_identity": {"dispatch_sha256": "5" * 64, "cases_total": 8},
        "target_compile_probe": {
            "arch": "sm_120",
            "binary_sha256": "6" * 64,
            "compiler_sha256": "7" * 64,
        },
    }
    return {
        "schema_version": "cuda-optimizer/gate-evidence-v1",
        "kind": kind,
        "producer": {
            "id": producers[kind],
            "version": "1.0.0",
            "implementation_sha256": ADAPTER_SHA,
        },
        "adapter_request_sha256": REQUEST_SHA,
        "contract_sha256": CONTRACT_SHA,
        "environment_sha256": ENVIRONMENT_SHA,
        "recorded_at": recorded_at,
        "status": "PASS",
        "subject": subjects[kind],
        "result": results[kind],
    }


def _diagnostic(kind: str, signals: list[str], recorded_at: float = 100.0) -> dict:
    producer = {
        "nsys_timeline": "nsys-timeline-adapter",
        "pytorch_profile": "pytorch-profile-adapter",
    }[kind]
    return {
        "schema_version": "cuda-optimizer/diagnostic-evidence-v1",
        "kind": kind,
        "producer": {
            "id": producer,
            "version": "1.0.0",
            "implementation_sha256": ADAPTER_SHA,
        },
        "adapter_request_sha256": REQUEST_SHA,
        "contract_sha256": CONTRACT_SHA,
        "environment_sha256": ENVIRONMENT_SHA,
        "recorded_at": recorded_at,
        "subject": {"target_sha256": TARGET_SHA},
        "report": {"artifact_sha256": "8" * 64, "events_total": 12},
        "signals": signals,
    }


class PlannerBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.planner = _load()

    def _snapshot(
        self,
        root: Path,
        *,
        diagnostic_time: float = 100.0,
        include_compile_gate: bool = True,
    ):
        ledger = root / "ledger"
        observations = [
            ("obs-reference", _gate("correctness_reference")),
            ("obs-dispatch", _gate("dispatch_identity")),
            (
                "obs-nsys",
                _diagnostic(
                    "nsys_timeline", ["launch_gap_short_context"], diagnostic_time
                ),
            ),
            ("obs-pytorch", _diagnostic("pytorch_profile", [], diagnostic_time)),
        ]
        if include_compile_gate:
            observations.insert(2, ("obs-compile", _gate("target_compile_probe")))
        for observation_id, evidence in observations:
            raw = (json.dumps(evidence, sort_keys=True) + "\n").encode()
            artifact = root / f"{observation_id}.json"
            artifact.write_bytes(raw)
            self.planner._SUMMARY._append_controller_gate_observation(
                ledger,
                artifact_root=root,
                contract_sha256=CONTRACT_SHA,
                environment_sha256=ENVIRONMENT_SHA,
                run_id="run-1",
                ledger_id="ledger-1",
                observation_id=observation_id,
                artifact={
                    "path": artifact.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                },
                adapter_implementation_sha256=ADAPTER_SHA,
                adapter_request_sha256=REQUEST_SHA,
                as_of=100.0,
                max_age_seconds=60.0,
                controller_seal_key=SEAL_KEY,
            )
        summary = self.planner._SUMMARY.build_summary(
            ledger,
            artifact_root=root,
            contract_sha256=CONTRACT_SHA,
            environment_sha256=ENVIRONMENT_SHA,
            run_id="run-1",
            ledger_id="ledger-1",
            as_of=110.0,
            max_age_seconds=60.0,
            max_observations=16,
            context_budget_bytes=20000,
            controller_seal_key=SEAL_KEY,
        )
        policy = {
            "arch": "sm_120",
            "task": "decode_attention",
            "layer": "kernel",
            "framework_versions": {"triton": "3.4.0"},
            "as_of": "2026-07-19",
            "max_review_age_days": 365,
            "context_budget_bytes": 12000,
            "limit": 3,
        }
        query = self.planner._CAPABILITY_QUERY.query(
            signals=["launch_gap_short_context"],
            available_evidence=["nsys_timeline", "pytorch_profile"],
            **policy,
        )
        proposal = {
            "schema_version": "cuda-optimizer/candidate-proposal-v1",
            "candidate_id": "candidate-1",
            "observation_id": "obs-nsys",
            "observation_summary_sha256": summary["summary_sha256"],
            "capability_query_sha256": query["query_sha256"],
            "hypothesis": "Reduce launch overhead for short decode contexts.",
            "expected_metric": {"name": "latency_ms", "direction": "lower"},
            "expected_effect_pct": 3.0,
            "kill_gate": "p95 latency does not improve",
            "estimated_cost_seconds": 60.0,
            "capability_ids": ["triton.decode-attention-gqa"],
            "paths": ["src/kernel.py"],
        }
        return ledger, summary, policy, query, proposal

    def _admit(self, root: Path, ledger: Path, summary, policy, query, proposal):
        return self.planner.validate_candidate_admission(
            proposal,
            capability_query=query,
            observation_summary=summary,
            capability_policy=policy,
            ledger_path=ledger,
            artifact_root=root,
            controller_seal_key=SEAL_KEY,
            expected_run_id="run-1",
            expected_ledger_id="ledger-1",
            expected_contract_sha256=CONTRACT_SHA,
            expected_environment_sha256=ENVIRONMENT_SHA,
            expected_reference_sha256=REFERENCE_SHA,
            expected_target_sha256=TARGET_SHA,
            expected_workload_sha256=WORKLOAD_SHA,
            current_as_of=110.0,
            max_age_seconds=60.0,
        )

    def test_current_diagnostics_query_and_preexecution_gates_admit_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = self._snapshot(root)
            admission = self._admit(root, *values)

        self.assertEqual(admission["status"], "ADMITTED")
        self.assertTrue(admission["pre_execution"]["satisfied"])
        self.assertEqual(admission["capability_ids"], ["triton.decode-attention-gqa"])
        self.assertEqual(admission["observation_id"], "obs-nsys")

    def test_registration_cannot_bypass_admission_with_hash_shaped_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, summary, policy, query, proposal = self._snapshot(root)
            state = self.planner._RUN_CONTROL.initialize_state(
                {
                    "schema_version": "cuda-optimizer/workload-contract-v1",
                    "contract_sha256": CONTRACT_SHA,
                    "budget": {
                        "preset": "balanced",
                        "max_seconds": 1000.0,
                        "max_candidates": 4,
                    },
                    "evidence": {"max_age_seconds": 60.0},
                    "objective": {"metric": "latency_ms", "direction": "lower"},
                    "mutation": {"project_paths": ["src"]},
                    "project_root": str(root),
                },
                now=0.0,
            )
            state = self.planner._RUN_CONTROL.advance(state, "freeze", now=0.0)
            state = self.planner._RUN_CONTROL.advance(state, "calibrate", now=1.0)
            state = self.planner._RUN_CONTROL.advance(
                state,
                "start_exploration",
                now=2.0,
                environment_state="green",
                measurable=True,
            )
            result = self.planner.register_candidate(
                state,
                proposal,
                now=110.0,
                capability_query=query,
                observation_summary=summary,
                capability_policy=policy,
                ledger_path=ledger,
                artifact_root=root,
                controller_seal_key=SEAL_KEY,
                expected_run_id="run-1",
                expected_ledger_id="ledger-1",
                expected_contract_sha256=CONTRACT_SHA,
                expected_environment_sha256=ENVIRONMENT_SHA,
                expected_reference_sha256=REFERENCE_SHA,
                expected_target_sha256=TARGET_SHA,
                expected_workload_sha256=WORKLOAD_SHA,
                current_as_of=110.0,
                max_age_seconds=60.0,
            )

        self.assertEqual(result["state"]["active_candidate"]["candidate_id"], "candidate-1")
        self.assertEqual(result["admission"]["status"], "ADMITTED")
        self.assertEqual(result["admission"]["evidence_age_seconds"], 10.0)

    def test_registration_time_and_controller_attestation_are_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, summary, policy, query, proposal = self._snapshot(root)
            state = {
                "schema_version": "cuda-optimizer/run-control-v1",
                "phase": "EXPLORING",
                "contract_sha256": CONTRACT_SHA,
                "started_at": 0.0,
                "updated_at": 2.0,
                "max_seconds": 20000.0,
                "max_candidates": 4,
                "max_evidence_age_seconds": 60.0,
                "objective_metric": "latency_ms",
                "objective_direction": "lower",
                "mutation_paths": ["src"],
                "project_root": str(root),
                "candidates_started": 0,
                "active_candidate": None,
                "candidate_history": [],
                "champion_candidate_id": None,
                "environment_state": "green",
                "measurable": True,
                "stop_reason": None,
                "drift_reason": None,
                "audit_reason": None,
            }
            inputs = {
                "capability_query": query,
                "observation_summary": summary,
                "capability_policy": policy,
                "ledger_path": ledger,
                "artifact_root": root,
                "controller_seal_key": SEAL_KEY,
                "expected_run_id": "run-1",
                "expected_ledger_id": "ledger-1",
                "expected_contract_sha256": CONTRACT_SHA,
                "expected_environment_sha256": ENVIRONMENT_SHA,
                "expected_reference_sha256": REFERENCE_SHA,
                "expected_target_sha256": TARGET_SHA,
                "expected_workload_sha256": WORKLOAD_SHA,
                "current_as_of": 110.0,
                "max_age_seconds": 60.0,
            }
            with self.assertRaisesRegex(ValueError, "time|snapshot"):
                self.planner.register_candidate(
                    state, proposal, now=10000.0, **inputs
                )
            admission = self.planner.validate_candidate_admission(proposal, **inputs)
            forged = dict(admission)
            forged["controller_attestation"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "attestation|admission"):
                self.planner._RUN_CONTROL.register_candidate(
                    state,
                    proposal,
                    admission=forged,
                    controller_seal_key=SEAL_KEY,
                    now=110.0,
                )

    def test_forged_query_inputs_or_hash_only_proposal_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, summary, policy, query, proposal = self._snapshot(root)
            forged = dict(query)
            forged["observed_signals"] = ["gqa_head_ratio"]
            unsigned = dict(forged)
            unsigned.pop("query_sha256")
            forged["query_sha256"] = hashlib.sha256(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            proposal = dict(proposal)
            proposal["capability_query_sha256"] = forged["query_sha256"]
            with self.assertRaisesRegex(ValueError, "query|signal|evidence|replay"):
                self._admit(root, ledger, summary, policy, forged, proposal)

    def test_stale_diagnostics_and_missing_preexecution_gate_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, summary, policy, query, proposal = self._snapshot(
                root, diagnostic_time=10.0
            )
            with self.assertRaisesRegex(ValueError, "diagnostic|signal|evidence|current"):
                self._admit(root, ledger, summary, policy, query, proposal)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, summary, policy, query, proposal = self._snapshot(
                root, include_compile_gate=False
            )
            with self.assertRaisesRegex(ValueError, "pre-execution|gate"):
                self._admit(root, ledger, summary, policy, query, proposal)

    def test_unselected_or_evidenceless_capability_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger, summary, policy, query, proposal = self._snapshot(root)
            invalid = dict(proposal)
            invalid["capability_ids"] = ["cuda.unselected"]
            with self.assertRaisesRegex(ValueError, "capability"):
                self._admit(root, ledger, summary, policy, query, invalid)
            invalid = dict(proposal)
            invalid["capability_ids"] = []
            with self.assertRaisesRegex(ValueError, "capability"):
                self._admit(root, ledger, summary, policy, query, invalid)


if __name__ == "__main__":
    unittest.main()
