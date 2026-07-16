from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "state.py"


def _load_state():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_state", STATE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StateSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _load_state()

    def _valid_state(self) -> dict:
        return {
            "schema_version": 2,
            "run_dir": "/tmp/run",
            "input_hash": "abc",
            "budget": {},
            "candidates": {},
        }

    def test_validate_state_rejects_non_dict_legacy_missing_and_invalid_schema(self) -> None:
        for payload in ([], {}, {"schema_version": 1}, {"schema_version": "2"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError) as caught:
                    self.state.validate_state(payload)
                message = str(caught.exception)
                if isinstance(payload, dict):
                    self.assertIn("schema_version", message)
                    self.assertIn("start a new v2.2 run", message)

    def test_validate_state_requires_v2_keys_and_does_not_mutate_valid_payload(self) -> None:
        valid = self._valid_state()
        before = copy.deepcopy(valid)
        result = self.state.validate_state(valid)
        self.assertEqual(result, valid)
        self.assertEqual(valid, before)

        for key in ("run_dir", "input_hash", "budget", "candidates"):
            invalid = self._valid_state()
            del invalid[key]
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    self.state.validate_state(invalid)

    def test_state_command_reader_rejects_legacy_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({"run_dir": tmp}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"start a new v2\.2 run"):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.state.cmd_show(argparse.Namespace(state=str(path)))

    def test_cmd_init_creates_v2_state_manifest_and_preserves_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            ref = root / "ref.py"
            env_path = root / "env.json"
            baseline.write_text("baseline\n", encoding="utf-8")
            ref.write_text("reference\n", encoding="utf-8")
            env_path.write_text(json.dumps({"gpu": "test"}), encoding="utf-8")
            args = argparse.Namespace(
                baseline=str(baseline),
                ref=str(ref),
                iterations=3,
                ncu_num=5,
                branches=4,
                dims='{"M": 128}',
                env=str(env_path),
                noise_threshold_pct=2.0,
                ptr_size=8,
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.state.cmd_init(args)

            output = json.loads(stdout.getvalue())
            run_dir = Path(output["run_dir"])
            state = json.loads(Path(output["state"]).read_text("utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))

            self.assertEqual(state["schema_version"], 2)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(state["input_hash"], manifest["input_hash"])
            self.assertEqual(state["budget"], manifest["budget"])
            self.assertEqual(state["workload"], None)
            self.assertEqual(state["best_kernel_statistics"], None)
            self.assertEqual(state["best_workload_statistics"], None)
            self.assertEqual(state["candidates"], {})
            self.assertEqual(state["frontier"], [])
            self.assertEqual(state["history"], [])

            expected_legacy = {
                "baseline_file",
                "baseline_file_original",
                "ref_file",
                "env",
                "env_path",
                "iterations_total",
                "ncu_num",
                "branches",
                "noise_threshold_pct",
                "ptr_size",
                "dims",
                "selected_methods",
                "effective_methods",
                "ineffective_methods",
                "implementation_failed_methods",
                "roofline_history",
                "best_metric_ms",
                "best_ncu_rep",
            }
            self.assertTrue(expected_legacy.issubset(state))
            self.assertEqual(state["best_file"], state["baseline_file"])
            self.assertEqual(
                manifest["inputs"]["baseline"]["path"], str(baseline.resolve())
            )
            self.assertEqual(manifest["inputs"]["ref"]["path"], str(ref.resolve()))
            for name in ("workload", "baseline", "candidates"):
                self.assertTrue((run_dir / name).is_dir())
            for iteration in range(1, 4):
                self.assertTrue((run_dir / f"iterv{iteration}").is_dir())

    def test_record_decision_is_idempotent_and_does_not_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            iter_dir = run_dir / "iterv1"
            iter_dir.mkdir(parents=True)
            best = run_dir / "best.py"
            best.write_text("# best\n", encoding="utf-8")
            state_path = run_dir / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "run_dir": str(run_dir),
                        "input_hash": "abc",
                        "budget": {},
                        "candidates": {},
                        "best_file": str(best),
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            decision = iter_dir / "decision.json"
            decision.write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "candidate_file": None,
                        "statistics": None,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                state=str(state_path), iter=1, decision=str(decision)
            )
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.state.cmd_record_decision(args)
            updated = json.loads(state_path.read_text("utf-8"))
            self.assertEqual(updated["best_file"], str(best))
            records = [
                item
                for item in updated["history"]
                if item.get("event") == "decision_record"
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "no_confirmed_kernel_win")


class StateDecisionPromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _load_state()

    def _fixture(
        self,
        root: Path,
        *,
        mode: str,
        decision_status: object = "inconclusive",
        statistics_status: object = None,
        workload_statistics_status: object = None,
        candidate_override: str | None = None,
        write_decision: bool = True,
        average_ms: float = 0.1,
    ) -> tuple[argparse.Namespace, Path, Path, dict]:
        run_dir = root / "run"
        iter_dir = run_dir / "iterv1"
        iter_dir.mkdir(parents=True)
        best = run_dir / "baseline" / "best.py"
        best.parent.mkdir(parents=True)
        best.write_text("# best\n", encoding="utf-8")
        candidate = iter_dir / "kernel.py"
        candidate.write_text("# candidate\n", encoding="utf-8")
        state_path = run_dir / "state.json"
        payload = {
            "schema_version": 2,
            "run_dir": str(run_dir),
            "input_hash": "abc",
            "budget": {},
            "candidates": {},
            "mode": mode,
            "workload": (
                {
                    "kind": "command",
                    "source": ["/bin/echo"],
                    "cases": [],
                    "source_hash": "b" * 64,
                    "objective": {
                        "primary_metric": {
                            "name": "latency_ms",
                            "direction": "lower",
                        },
                        "min_effect_pct": 1.0,
                        "constraints": [],
                    },
                }
                if mode == "full"
                else None
            ),
            "best_file": str(best),
            "best_metric_ms": 10.0,
            "best_kernel_statistics": None,
            "best_workload_statistics": None,
            "noise_threshold_pct": 2.0,
            "selected_methods": [],
            "effective_methods": [],
            "ineffective_methods": [],
            "implementation_failed_methods": [],
            "history": [],
            "roofline_history": [],
            "frontier": [],
        }
        state_path.write_text(json.dumps(payload), encoding="utf-8")
        bench = iter_dir / "bench.json"
        bench.write_text(
            json.dumps(
                {
                    "correctness": {"passed": True},
                    "kernel": {"average_ms": average_ms},
                    "reference": {"average_ms": 20.0},
                }
            ),
            encoding="utf-8",
        )
        methods = iter_dir / "methods.json"
        methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
        decision = iter_dir / "decision.json"
        if statistics_status is None:
            statistics_status = (
                "confirmed_win"
                if decision_status in {"kernel_only_win", "end_to_end_win"}
                else decision_status
            )
        if workload_statistics_status is None:
            workload_statistics_status = (
                "confirmed_win"
                if decision_status == "end_to_end_win"
                else statistics_status
            )
        evidence_values = {
            "confirmed_win": (3.0, 2.0, 4.0),
            "confirmed_loss": (-3.0, -4.0, -2.0),
            "inconclusive": (0.0, -1.0, 1.0),
        }
        estimate, ci_low, ci_high = evidence_values.get(
            statistics_status, (3.0, 2.0, 4.0)
        )
        if statistics_status == "confirmed_win":
            ci_low = ci_high = estimate
        statistics = {
            "statistic": "median_paired_improvement_pct",
            "direction": "lower",
            "min_effect_pct": 1.0,
            "confidence": 0.95,
            "estimate_pct": estimate,
            "ci_low_pct": ci_low,
            "ci_high_pct": ci_high,
            "status": statistics_status,
            "valid_pairs": 3,
            "invalid_pairs": 0,
            "improvements_pct": [estimate, estimate, estimate],
        }
        has_kernel_statistics = decision_status not in {
            "no_confirmed_kernel_win",
            "workload_failed",
            "invalid",
        }
        has_workload_statistics = decision_status == "end_to_end_win"
        if write_decision:
            candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
            decision_payload = {
                "status": decision_status,
                "candidate_id": "b1",
                "candidate_file": candidate_override or str(candidate),
                "candidate_sha256": (
                    candidate_sha256
                    if decision_status
                    in {"confirmed_win", "kernel_only_win", "end_to_end_win"}
                    else None
                ),
                "statistics": statistics if has_kernel_statistics else None,
                "workload_statistics": (
                    {
                        **statistics,
                        "status": workload_statistics_status,
                    }
                    if has_workload_statistics
                    else None
                ),
            }
            if decision_status in {
                "confirmed_win",
                "kernel_only_win",
                "end_to_end_win",
            }:
                for kind in (
                    "kernel",
                    *( ("workload",) if decision_status == "end_to_end_win" else () ),
                ):
                    samples = iter_dir / kind / "paired_samples.jsonl"
                    samples.parent.mkdir(parents=True, exist_ok=True)
                    records = [
                        {
                            "schema_version": 2,
                            "kind": kind,
                            "input_hash": "abc",
                            "iteration": 1,
                            "candidate_id": "b1",
                            "candidate_file": str(candidate.resolve()),
                            "candidate_sha256": candidate_sha256,
                            "classifier": (
                                {
                                    "direction": "lower",
                                    "min_effect_pct": 1.0,
                                    "confidence": 0.95,
                                    "bootstrap_samples": 20,
                                    "seed": 0,
                                }
                                if kind == "kernel"
                                else {
                                    "objective": payload["workload"]["objective"],
                                    "objective_sha256": hashlib.sha256(
                                        json.dumps(
                                            payload["workload"]["objective"],
                                            sort_keys=True,
                                            separators=(",", ":"),
                                        ).encode("utf-8")
                                    ).hexdigest(),
                                    "confidence": 0.95,
                                    "bootstrap_samples": 20,
                                    "seed": 0,
                                }
                            ),
                            "pair_index": index,
                            "pair": (
                                {"baseline": 100.0, "candidate": 97.0, "valid": True}
                                if kind == "kernel"
                                else {
                                    "block": index,
                                    "order": "AB",
                                    "case": None,
                                    "baseline_metrics": {"latency_ms": 100.0},
                                    "candidate_metrics": {"latency_ms": 97.0},
                                    "valid": True,
                                    "attempts": {"baseline": 1, "candidate": 1},
                                    "attempt_records": {
                                        "baseline": [],
                                        "candidate": [],
                                    },
                                }
                            ),
                        }
                        for index in range(3)
                    ]
                    samples.write_text(
                        "".join(
                            json.dumps(record, separators=(",", ":")) + "\n"
                            for record in records
                        ),
                        encoding="utf-8",
                    )
                    decision_payload[f"{kind}_paired_samples"] = {
                        "schema_version": 2,
                        "kind": kind,
                        "path": str(samples.resolve()),
                        "sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
                        "pairs": 3,
                        "input_hash": "abc",
                        "iteration": 1,
                        "candidate_id": "b1",
                        "candidate_file": str(candidate.resolve()),
                        "candidate_sha256": candidate_sha256,
                        "classifier": records[0]["classifier"],
                    }
            decision.write_text(
                json.dumps(decision_payload),
                encoding="utf-8",
            )
        args = argparse.Namespace(
            state=str(state_path),
            iter=1,
            kernel=str(candidate),
            bench=str(bench),
            methods_json=str(methods),
            attribution=None,
            sass_check=None,
            retries=0,
            skip_validation=True,
            allow_ineffective=False,
            decision=None,
        )
        return args, state_path, candidate, payload

    def _run_update(self, args: argparse.Namespace) -> dict:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.state.cmd_update(args)
        return json.loads(stdout.getvalue())

    def _attach_sass_status(self, args: argparse.Namespace, status: str) -> None:
        methods_path = Path(args.methods_json)
        methods_path.write_text(
            json.dumps({"methods": [{"id": "vectorize"}]}), encoding="utf-8"
        )
        sass_path = methods_path.parent / "sass_check.json"
        sass_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "checks": [
                        {
                            "method_id": "vectorize",
                            "status": status,
                            "verified": status == "passed",
                            "patterns_missing": ["LDG"] if status == "failed" else [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        args.sass_check = str(sass_path)

    def test_unavailable_or_not_applicable_sass_is_not_implementation_failed(self) -> None:
        for sass_status in ("unavailable", "not_applicable"):
            with self.subTest(status=sass_status), tempfile.TemporaryDirectory() as tmp:
                args, state_path, _candidate, _before = self._fixture(
                    Path(tmp), mode="kernel-only", decision_status="confirmed_win"
                )
                self._attach_sass_status(args, sass_status)
                self._run_update(args)
                updated = json.loads(state_path.read_text("utf-8"))

            self.assertEqual(updated["implementation_failed_methods"], [])
            self.assertEqual(updated["effective_methods"][0]["id"], "vectorize")

    def test_only_failed_sass_status_is_implementation_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, _candidate, _before = self._fixture(
                Path(tmp), mode="kernel-only", decision_status="confirmed_win"
            )
            self._attach_sass_status(args, "failed")
            self._run_update(args)
            updated = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(updated["implementation_failed_methods"][0]["id"], "vectorize")
        self.assertEqual(updated["effective_methods"], [])

    def test_full_mode_inner_kernel_win_does_not_promote_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, _candidate, before = self._fixture(
                Path(tmp), mode="full", decision_status="confirmed_win"
            )
            output = self._run_update(args)
            updated = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(updated["best_file"], before["best_file"])
        self.assertEqual(updated["best_metric_ms"], before["best_metric_ms"])
        self.assertFalse(output["improved"])
        self.assertEqual(output["status"], "kernel_only_win")
        self.assertEqual(updated["history"][-1]["status"], "kernel_only_win")

    def test_kernel_only_confirmed_win_promotes_with_kernel_only_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, candidate, _before = self._fixture(
                Path(tmp), mode="kernel-only", decision_status="confirmed_win"
            )
            output = self._run_update(args)
            updated = json.loads(state_path.read_text("utf-8"))
            candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()

        self.assertEqual(updated["best_file"], str(candidate.resolve()))
        self.assertTrue(output["improved"])
        self.assertEqual(output["status"], "kernel_only_win")
        self.assertEqual(updated["best_kernel_statistics"]["estimate_pct"], 3.0)
        binding = updated["candidates"]["iter-1"]
        self.assertEqual(binding["candidate_file"], str(candidate.resolve()))
        self.assertEqual(binding["candidate_sha256"], candidate_hash)

    def test_winning_decision_without_raw_pairs_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, _candidate, before = self._fixture(
                Path(tmp), mode="kernel-only", decision_status="confirmed_win"
            )
            decision_path = Path(tmp) / "run" / "iterv1" / "decision.json"
            decision = json.loads(decision_path.read_text("utf-8"))
            decision.pop("kernel_paired_samples")
            decision_path.write_text(json.dumps(decision), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "kernel_paired_samples"):
                self._run_update(args)
            unchanged = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(unchanged["best_file"], before["best_file"])
        self.assertNotIn("terminal_decision", unchanged)

    def test_raw_pair_content_must_recompute_the_declared_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, _candidate, before = self._fixture(
                Path(tmp), mode="kernel-only", decision_status="confirmed_win"
            )
            decision_path = Path(tmp) / "run" / "iterv1" / "decision.json"
            decision = json.loads(decision_path.read_text("utf-8"))
            evidence = decision["kernel_paired_samples"]
            artifact = Path(evidence["path"])
            records = [
                json.loads(line) for line in artifact.read_text("utf-8").splitlines()
            ]
            for record in records:
                record["pair"]["candidate"] = 96.0
            artifact.write_text(
                "".join(
                    json.dumps(record, separators=(",", ":")) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            evidence["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            decision_path.write_text(json.dumps(decision), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "recompute|statistics"):
                self._run_update(args)
            unchanged = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(unchanged["best_file"], before["best_file"])
        self.assertNotIn("terminal_decision", unchanged)

    def test_full_mode_end_to_end_win_is_the_only_global_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, candidate, _before = self._fixture(
                Path(tmp), mode="full", decision_status="end_to_end_win"
            )
            output = self._run_update(args)
            updated = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(updated["best_file"], str(candidate.resolve()))
        self.assertTrue(output["improved"])
        self.assertEqual(output["status"], "end_to_end_win")
        self.assertEqual(
            updated["best_workload_statistics"]["statistic"],
            "median_paired_improvement_pct",
        )

    def test_update_persists_one_bound_terminal_evidence_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, candidate, _before = self._fixture(
                Path(tmp), mode="full", decision_status="end_to_end_win"
            )
            run_dir = Path(tmp) / "run"
            iter_dir = run_dir / "iterv1"
            candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()

            decision_path = iter_dir / "decision.json"
            decision = json.loads(decision_path.read_text("utf-8"))
            decision.update(candidate_id="b1", constraints=[])
            decision_path.write_text(json.dumps(decision), encoding="utf-8")
            decision_sha = hashlib.sha256(decision_path.read_bytes()).hexdigest()

            bench_path = Path(args.bench)
            bench = json.loads(bench_path.read_text("utf-8"))
            bench["compiler_evidence"] = {
                "status": "available",
                "stages": ["source", "ptx", "sass"],
            }
            bench_path.write_text(json.dumps(bench), encoding="utf-8")
            sass_path = iter_dir / "sass_check.json"
            sass_path.write_text(
                json.dumps({"status": "passed", "checks": []}),
                encoding="utf-8",
            )
            args.sass_check = str(sass_path)
            checkpoint_path = run_dir / "checkpoint.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "input_hash": "abc",
                        "iteration": 1,
                        "stage": "decision",
                        "status": "in_progress",
                        "stage_evidence": {
                            "candidate_sanitizer": {
                                "status": "passed",
                                "coverage": "complete",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            self._run_update(args)
            updated = json.loads(state_path.read_text("utf-8"))

        terminal = updated["terminal_decision"]
        self.assertEqual(terminal["input_hash"], "abc")
        self.assertEqual(terminal["iteration"], 1)
        self.assertEqual(terminal["status"], "end_to_end_win")
        self.assertEqual(terminal["candidate_sha256"], candidate_sha)
        self.assertEqual(terminal["decision_sha256"], decision_sha)
        self.assertEqual(terminal["kernel_paired_samples"]["pairs"], 3)
        self.assertEqual(terminal["workload_paired_samples"]["pairs"], 3)
        self.assertEqual(terminal["compiler_evidence"]["status"], "available")
        self.assertEqual(terminal["sass"]["status"], "passed")
        self.assertEqual(terminal["sanitizer"]["coverage"], "complete")
        self.assertEqual(terminal["resume"]["stage"], "decision")

    def test_faster_average_with_non_win_decision_never_promotes(self) -> None:
        for status in (
            "inconclusive",
            "confirmed_loss",
            "no_confirmed_kernel_win",
            "workload_failed",
            "invalid",
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                args, state_path, _candidate, before = self._fixture(
                    Path(tmp),
                    mode="kernel-only",
                    decision_status=status,
                    average_ms=0.001,
                )
                output = self._run_update(args)
                updated = json.loads(state_path.read_text("utf-8"))
                self.assertEqual(updated["best_file"], before["best_file"])
                self.assertEqual(updated["best_metric_ms"], before["best_metric_ms"])
                self.assertFalse(output["improved"])
                self.assertEqual(output["status"], status)

    def test_missing_decision_never_falls_back_to_average(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, _candidate, before = self._fixture(
                Path(tmp),
                mode="kernel-only",
                write_decision=False,
                average_ms=0.001,
            )
            with self.assertRaisesRegex(ValueError, "decision.json.*missing"):
                self._run_update(args)
            unchanged = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(unchanged["best_file"], before["best_file"])

    def test_malformed_or_unknown_status_is_diagnostic_and_non_mutating(self) -> None:
        for status in (None, True, 1, 1.5, "surprise_win"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                args, state_path, _candidate, before = self._fixture(
                    Path(tmp), mode="kernel-only", decision_status=status
                )
                with self.assertRaisesRegex(ValueError, "decision.*status"):
                    self._run_update(args)
                unchanged = json.loads(state_path.read_text("utf-8"))
                self.assertEqual(unchanged["best_file"], before["best_file"])

    def test_candidate_path_mismatch_is_rejected_before_state_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, _candidate, before = self._fixture(
                Path(tmp),
                mode="kernel-only",
                decision_status="confirmed_win",
                candidate_override=str(Path(tmp) / "other.py"),
            )
            with self.assertRaisesRegex(ValueError, "candidate_file.*kernel"):
                self._run_update(args)
            unchanged = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(unchanged["best_file"], before["best_file"])

    def test_no_win_candidate_contract_accepts_null_but_rejects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision_path = root / "decision.json"
            decision_path.write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "candidate_file": None,
                        "statistics": None,
                    }
                ),
                encoding="utf-8",
            )
            decision, status, statistics, workload = self.state._load_decision(
                str(decision_path), candidate_file=None
            )
            self.assertEqual(decision["status"], "no_confirmed_kernel_win")
            self.assertEqual(status, "no_confirmed_kernel_win")
            self.assertIsNone(statistics)
            self.assertIsNone(workload)

            conflicting = root / "kernel.py"
            decision_path.write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "candidate_file": str(conflicting),
                        "statistics": None,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "candidate_file"):
                self.state._load_decision(str(decision_path), candidate_file=None)

    def test_win_candidate_missing_tampered_or_bad_hash_never_writes_state(self) -> None:
        for case in ("missing", "missing_hash", "tampered", "bad_hash"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                args, state_path, candidate, _before = self._fixture(
                    Path(tmp), mode="kernel-only", decision_status="confirmed_win"
                )
                original_state = state_path.read_bytes()
                decision_path = Path(tmp) / "run" / "iterv1" / "decision.json"
                if case == "missing":
                    candidate.unlink()
                elif case == "missing_hash":
                    decision = json.loads(decision_path.read_text("utf-8"))
                    decision.pop("candidate_sha256")
                    decision_path.write_text(json.dumps(decision), encoding="utf-8")
                elif case == "tampered":
                    candidate.write_text("# changed\n", encoding="utf-8")
                else:
                    decision = json.loads(decision_path.read_text("utf-8"))
                    decision["candidate_sha256"] = "not-a-sha"
                    decision_path.write_text(json.dumps(decision), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "candidate|sha256|hash"):
                    self._run_update(args)
                self.assertEqual(state_path.read_bytes(), original_state)

    def test_candidate_must_be_regular_non_symlink_in_current_iteration(self) -> None:
        for case in ("outside", "symlink", "directory"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args, state_path, candidate, _before = self._fixture(
                    root, mode="kernel-only", decision_status="confirmed_win"
                )
                original_state = state_path.read_bytes()
                decision_path = root / "run" / "iterv1" / "decision.json"
                if case == "outside":
                    candidate = root / "outside.py"
                    candidate.write_text("# outside\n", encoding="utf-8")
                elif case == "symlink":
                    real = root / "real.py"
                    real.write_text("# real\n", encoding="utf-8")
                    candidate.unlink()
                    candidate.symlink_to(real)
                else:
                    candidate.unlink()
                    candidate.mkdir()
                args.kernel = str(candidate)
                decision = json.loads(decision_path.read_text("utf-8"))
                decision["candidate_file"] = str(candidate)
                if candidate.is_file():
                    decision["candidate_sha256"] = hashlib.sha256(
                        candidate.read_bytes()
                    ).hexdigest()
                decision_path.write_text(json.dumps(decision), encoding="utf-8")

                with self.assertRaisesRegex(
                    ValueError, "iteration|candidate|symlink|regular"
                ):
                    self._run_update(args)
                self.assertEqual(state_path.read_bytes(), original_state)

    def test_decision_must_be_current_iteration_regular_file(self) -> None:
        for case in ("other_iteration", "symlink"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args, state_path, _candidate, _before = self._fixture(
                    root, mode="kernel-only", decision_status="confirmed_win"
                )
                original_state = state_path.read_bytes()
                decision_path = root / "run" / "iterv1" / "decision.json"
                if case == "other_iteration":
                    other = root / "run" / "iterv2" / "decision.json"
                    other.parent.mkdir()
                    other.write_bytes(decision_path.read_bytes())
                    args.decision = str(other)
                else:
                    external = root / "external-decision.json"
                    external.write_bytes(decision_path.read_bytes())
                    decision_path.unlink()
                    decision_path.symlink_to(external)

                with self.assertRaisesRegex(ValueError, "decision.*iteration|symlink"):
                    self._run_update(args)
                self.assertEqual(state_path.read_bytes(), original_state)

    def test_candidate_swap_before_state_write_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, candidate, _before = self._fixture(
                Path(tmp), mode="kernel-only", decision_status="confirmed_win"
            )
            original_state = state_path.read_bytes()
            original_verify = self.state._verify_candidate_binding

            def swap_then_verify(binding):
                candidate.write_text("# swapped before write\n", encoding="utf-8")
                return original_verify(binding)

            with mock.patch.object(
                self.state, "_verify_candidate_binding", side_effect=swap_then_verify
            ):
                with self.assertRaisesRegex(ValueError, "changed|sha256|hash"):
                    self._run_update(args)

            self.assertEqual(state_path.read_bytes(), original_state)

    def test_confirmed_win_with_conflicting_statistics_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, _candidate, before = self._fixture(
                Path(tmp), mode="kernel-only", decision_status="confirmed_win"
            )
            decision_path = Path(tmp) / "run" / "iterv1" / "decision.json"
            decision = json.loads(decision_path.read_text("utf-8"))
            decision["statistics"]["status"] = "confirmed_loss"
            decision_path.write_text(json.dumps(decision), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "statistics.status"):
                self._run_update(args)
            unchanged = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(unchanged["best_file"], before["best_file"])

    def test_decision_candidate_symlink_alias_does_not_match_regular_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, state_path, candidate, _before = self._fixture(
                root, mode="kernel-only", decision_status="confirmed_win"
            )
            original_state = state_path.read_bytes()
            alias = candidate.with_name("alias.py")
            alias.symlink_to(candidate)
            decision_path = root / "run" / "iterv1" / "decision.json"
            decision = json.loads(decision_path.read_text("utf-8"))
            decision["candidate_file"] = str(alias)
            decision_path.write_text(json.dumps(decision), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "candidate_file|symlink"):
                self._run_update(args)

            self.assertEqual(state_path.read_bytes(), original_state)

    def test_terminal_wins_require_confirmed_kernel_and_workload_evidence(self) -> None:
        cases = (
            ("confirmed_win", "confirmed_loss", "confirmed_win"),
            ("kernel_only_win", "confirmed_loss", "confirmed_win"),
            ("end_to_end_win", "confirmed_loss", "confirmed_win"),
            ("end_to_end_win", "confirmed_win", "confirmed_loss"),
        )
        for decision_status, kernel_status, workload_status in cases:
            with self.subTest(
                decision_status=decision_status,
                kernel_status=kernel_status,
                workload_status=workload_status,
            ), tempfile.TemporaryDirectory() as tmp:
                args, state_path, _candidate, before = self._fixture(
                    Path(tmp),
                    mode="full" if decision_status == "end_to_end_win" else "kernel-only",
                    decision_status=decision_status,
                    statistics_status=kernel_status,
                    workload_statistics_status=workload_status,
                )
                with self.assertRaisesRegex(
                    ValueError, "statistics.status|confirmed_win evidence"
                ):
                    self._run_update(args)
                unchanged = json.loads(state_path.read_text("utf-8"))
                self.assertEqual(unchanged["best_file"], before["best_file"])

    def test_nested_evidence_status_must_be_a_known_string(self) -> None:
        for nested_field in ("statistics", "workload_statistics"):
            for status in (True, 1, 1.5, "surprise_win"):
                with self.subTest(
                    nested_field=nested_field, status=status
                ), tempfile.TemporaryDirectory() as tmp:
                    kwargs = {
                        "statistics_status": "confirmed_win",
                        "workload_statistics_status": "confirmed_win",
                    }
                    kwargs[f"{nested_field}_status"] = status
                    args, state_path, _candidate, before = self._fixture(
                        Path(tmp),
                        mode="full",
                        decision_status="end_to_end_win",
                        **kwargs,
                    )
                    with self.assertRaisesRegex(
                        ValueError, rf"{nested_field}\.status"
                    ):
                        self._run_update(args)
                    unchanged = json.loads(state_path.read_text("utf-8"))
                    self.assertEqual(unchanged["best_file"], before["best_file"])


if __name__ == "__main__":
    unittest.main()
