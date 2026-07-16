from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATE_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "orchestrate.py"
)


def _load_orchestrate():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_orchestrate", ORCHESTRATE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CloseIterationDecisionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[SimpleNamespace, Path, Path]:
        run_dir = root / "run"
        iter_dir = run_dir / "iterv1"
        iter_dir.mkdir(parents=True)
        state_path = run_dir / "state.json"
        state_path.write_text(
            json.dumps({"run_dir": str(run_dir), "best_file": str(root / "best.py")}),
            encoding="utf-8",
        )
        (iter_dir / "methods.json").write_text(
            json.dumps({"methods": []}), encoding="utf-8"
        )
        args = SimpleNamespace(
            run_dir=str(run_dir),
            iter=1,
            benchmark=str(root / "benchmark.py"),
            warmup=3,
            repeat=8,
            retries=0,
        )
        return args, state_path, iter_dir

    def test_no_confirmed_result_is_a_normal_terminal_short_circuit(self) -> None:
        orchestrate = _load_orchestrate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, state_path, iter_dir = self._fixture(root)
            branch_payload = {
                "iter": 1,
                "status": "no_confirmed_kernel_win",
                "champion": None,
                "completed_comparisons": 2,
            }
            (iter_dir / "branch_results.json").write_text(
                json.dumps(branch_payload), encoding="utf-8"
            )
            branch_result = SimpleNamespace(
                returncode=0,
                stdout="[triton setup]\n n: int [input]\n" + json.dumps(branch_payload),
                stderr="",
            )
            runner = mock.Mock(return_value=branch_result)

            with mock.patch.object(orchestrate, "_run", runner):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    orchestrate.cmd_close_iter(args)

            output = json.loads(stdout.getvalue())

        self.assertEqual(runner.call_count, 1)
        self.assertEqual(output["iter"], 1)
        self.assertEqual(output["status"], "no_confirmed_kernel_win")
        self.assertEqual(output["decision"], str(iter_dir / "decision.json"))
        self.assertEqual(output["state"], str(state_path))
        self.assertIn("next_step", output)

    def test_confirmed_artifact_with_noisy_stdout_continues_to_champion_path(self) -> None:
        orchestrate = _load_orchestrate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _state_path, iter_dir = self._fixture(root)
            selected = iter_dir / "kernel.py"
            stale = iter_dir / "kernel.cu"
            selected.write_text("# champion\n", encoding="utf-8")
            stale.write_text("// stale\n", encoding="utf-8")
            (iter_dir / "branch_results.json").write_text(
                json.dumps(
                    {
                        "iter": 1,
                        "status": "shortlist_ready",
                        "champion": {"branch_index": 1},
                        "selected_kernel": str(selected),
                    }
                ),
                encoding="utf-8",
            )
            (iter_dir / "decision.json").write_text(
                json.dumps(
                    {
                        "status": "confirmed_win",
                        "candidate_file": str(selected),
                    }
                ),
                encoding="utf-8",
            )
            (iter_dir / "bench.json").write_text(
                json.dumps({"correctness": {"passed": True}}), encoding="utf-8"
            )
            branch_result = SimpleNamespace(
                returncode=0,
                stdout="CUDA setup log\n{not the authority}",
                stderr="",
            )
            ok = SimpleNamespace(returncode=0, stdout="", stderr="")
            runner = mock.Mock(
                side_effect=[
                    branch_result,
                    ok,
                    ok,
                    RuntimeError("state update reached"),
                ]
            )

            with mock.patch.object(orchestrate, "_run", runner):
                with self.assertRaisesRegex(RuntimeError, "state update reached"):
                    orchestrate.cmd_close_iter(args)

        self.assertEqual(runner.call_count, 4)
        update_command = runner.call_args_list[3].args[0]
        self.assertEqual(update_command[update_command.index("--kernel") + 1], str(selected))

    def test_selected_kernel_identity_rejects_malformed_escape_symlink_and_directory(self) -> None:
        orchestrate = _load_orchestrate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.py"
            valid.write_text("# valid\n", encoding="utf-8")
            cases = (
                "missing",
                "outside",
                "symlink",
                "directory",
                "decision_mismatch",
                "decision_symlink_alias",
            )
            for case in cases:
                with self.subTest(case=case):
                    case_root = root / case
                    args, _state_path, iter_dir = self._fixture(case_root)
                    selected = iter_dir / "kernel.py"
                    selected_value = str(selected)
                    decision_value = selected_value
                    if case == "missing":
                        selected_value = None
                    elif case == "outside":
                        selected = case_root / "outside.py"
                        selected.write_text("# outside\n", encoding="utf-8")
                        selected_value = str(selected)
                        decision_value = selected_value
                    elif case == "symlink":
                        selected.symlink_to(valid)
                    elif case == "directory":
                        selected.mkdir()
                    elif case == "decision_mismatch":
                        selected.write_text("# selected\n", encoding="utf-8")
                        decision_value = str(iter_dir / "other.py")
                    elif case == "decision_symlink_alias":
                        selected.write_text("# selected\n", encoding="utf-8")
                        alias = iter_dir / "alias.py"
                        alias.symlink_to(selected)
                        decision_value = str(alias)

                    (iter_dir / "branch_results.json").write_text(
                        json.dumps(
                            {
                                "status": "shortlist_ready",
                                "selected_kernel": selected_value,
                            }
                        ),
                        encoding="utf-8",
                    )
                    (iter_dir / "decision.json").write_text(
                        json.dumps(
                            {
                                "status": "confirmed_win",
                                "candidate_file": decision_value,
                            }
                        ),
                        encoding="utf-8",
                    )
                    branch_result = SimpleNamespace(
                        returncode=0, stdout="noise", stderr=""
                    )
                    runner = mock.Mock(return_value=branch_result)
                    with mock.patch.object(orchestrate, "_run", runner):
                        with self.assertRaisesRegex(
                            SystemExit, "selected_kernel|candidate_file"
                        ):
                            orchestrate.cmd_close_iter(args)
                    self.assertEqual(runner.call_count, 1)

    def test_missing_malformed_or_unknown_branch_artifact_fails_clearly(self) -> None:
        orchestrate = _load_orchestrate()
        cases = (
            (None, "missing"),
            ("{", "malformed"),
            (json.dumps({"status": "surprise"}), "status"),
        )
        for artifact, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args, _state_path, iter_dir = self._fixture(root)
                if artifact is not None:
                    (iter_dir / "branch_results.json").write_text(
                        artifact, encoding="utf-8"
                    )
                branch_result = SimpleNamespace(
                    returncode=0,
                    stdout="noisy setup output\n{}",
                    stderr="",
                )
                runner = mock.Mock(return_value=branch_result)

                with mock.patch.object(orchestrate, "_run", runner):
                    with self.assertRaisesRegex(SystemExit, rf"branch_results.*{expected}"):
                        orchestrate.cmd_close_iter(args)

                self.assertEqual(runner.call_count, 1)

    def test_rc2_keeps_all_branches_failed_semantics(self) -> None:
        orchestrate = _load_orchestrate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _state_path, _iter_dir = self._fixture(root)
            branch_result = SimpleNamespace(returncode=2, stdout="", stderr="")
            runner = mock.Mock(return_value=branch_result)

            with mock.patch.object(orchestrate, "_run", runner):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    with self.assertRaises(SystemExit) as caught:
                        orchestrate.cmd_close_iter(args)

            output = json.loads(stdout.getvalue())

        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(output["status"], "all_branches_failed")

    def test_budgeted_full_close_routes_only_confirmed_shortlist_through_outer_loop(self) -> None:
        orchestrate = _load_orchestrate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, state_path, iter_dir = self._fixture(root)
            baseline = root / "best.py"
            baseline.write_text("# best\n", encoding="utf-8")
            confirmed = iter_dir / "branches" / "b1" / "kernel.py"
            rejected = iter_dir / "branches" / "b2" / "kernel.py"
            confirmed.parent.mkdir(parents=True)
            rejected.parent.mkdir(parents=True)
            confirmed.write_text("# confirmed\n", encoding="utf-8")
            rejected.write_text("# rejected\n", encoding="utf-8")
            selected = iter_dir / "kernel.py"
            selected.write_text("# confirmed\n", encoding="utf-8")
            (iter_dir / "bench.json").write_text(
                json.dumps({"correctness": {"passed": True}}), encoding="utf-8"
            )
            state = {
                "schema_version": 2,
                "run_dir": str((root / "run").resolve()),
                "input_hash": "frozen",
                "budget": {
                    "name": "quick", "max_seconds": 2700, "branches": 4,
                    "max_rounds": 1, "min_pairs": 20, "max_pairs": 50,
                    "outer_candidates": 1, "max_cases": 3,
                    "sanitizer_mode": "targeted", "reserve_seconds": 300,
                },
                "mode": "full",
                "workload": {
                    "kind": "command", "source": ["/bin/echo"],
                    "objective": {
                        "primary_metric": {"name": "latency", "direction": "lower"},
                        "min_effect_pct": 0.5, "constraints": [],
                    },
                    "cases": [], "source_hash": "a" * 64,
                },
                "started_at": orchestrate.time.time(),
                "confidence": 0.95,
                "best_file": str(baseline),
                "iterations_total": 1,
            }
            state_path.write_text(json.dumps(state), encoding="utf-8")
            statistics = {
                "status": "confirmed_win",
                "statistic": "median_paired_improvement_pct",
                "direction": "lower",
                "min_effect_pct": 0.5,
                "confidence": 0.95,
                "estimate_pct": 4.0,
                "ci_low_pct": 1.0,
                "ci_high_pct": 5.0,
                "valid_pairs": 20,
                "invalid_pairs": 0,
                "improvements_pct": [4.0] * 20,
            }
            branch_payload = {
                "status": "shortlist_ready",
                "selected_kernel": str(selected),
                "champion": {
                    "status": "confirmed_win", "kernel": str(confirmed),
                    "statistics": statistics,
                },
                "shortlist": [
                    {
                        "status": "confirmed_win", "kernel": str(confirmed),
                        "statistics": statistics,
                    },
                    {
                        "status": "confirmed_loss", "kernel": str(rejected),
                        "statistics": {"estimate_pct": 99.0},
                    },
                ],
            }
            (iter_dir / "branch_results.json").write_text(
                json.dumps(branch_payload), encoding="utf-8"
            )
            (iter_dir / "decision.json").write_text(
                json.dumps({"status": "confirmed_win", "candidate_file": str(selected)}),
                encoding="utf-8",
            )
            branch_result = SimpleNamespace(returncode=0, stdout="", stderr="")
            ok = SimpleNamespace(returncode=0, stdout="", stderr="")
            runner = mock.Mock(side_effect=[branch_result, ok, ok])
            workload_result = {
                "status": "evaluated",
                "objective": state["workload"]["objective"],
                "primary": statistics,
                "constraints": [],
                "pairs": [
                    {
                        "block": 0,
                        "order": "AB",
                        "baseline_metrics": {"latency": 2.0},
                        "candidate_metrics": {"latency": 1.0},
                        "valid": True,
                    }
                ],
            }
            workload_pairs = mock.Mock(return_value=workload_result)
            apply = mock.Mock(
                return_value={"returncode": 0, "decision_path": str(iter_dir / "decision.json")}
            )

            with mock.patch.object(orchestrate, "_run", runner), \
                    mock.patch.object(orchestrate.workload_evaluate, "evaluate_pairs", workload_pairs), \
                    mock.patch.object(orchestrate, "apply_decision", apply), \
                    contextlib.redirect_stdout(io.StringIO()):
                orchestrate.cmd_close_iter(args)
            persisted_workload_samples = Path(
                apply.call_args.args[0]["workload_paired_samples"]["path"]
            ).is_file()

        self.assertEqual(workload_pairs.call_count, 1)
        self.assertEqual(workload_pairs.call_args.args[1], str(baseline))
        self.assertEqual(workload_pairs.call_args.args[2], str(confirmed.resolve()))
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(apply.call_args.args[0]["status"], "end_to_end_win")
        evidence = apply.call_args.args[0]["workload_paired_samples"]
        self.assertEqual(evidence["input_hash"], "frozen")
        self.assertEqual(evidence["iteration"], 1)
        self.assertTrue(persisted_workload_samples)

    def test_budgeted_kernel_only_close_never_calls_workload_and_builds_terminal_win(self) -> None:
        orchestrate = _load_orchestrate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, state_path, iter_dir = self._fixture(root)
            selected = iter_dir / "kernel.py"
            selected.write_text("# selected\n", encoding="utf-8")
            (iter_dir / "bench.json").write_text(
                json.dumps({"correctness": {"passed": True}}), encoding="utf-8"
            )
            statistics = {
                "status": "confirmed_win",
                "statistic": "median_paired_improvement_pct",
                "direction": "lower",
                "min_effect_pct": 0.5,
                "confidence": 0.95,
                "estimate_pct": 3.0,
                "ci_low_pct": 1.0,
                "ci_high_pct": 4.0,
                "valid_pairs": 20,
                "invalid_pairs": 0,
                "improvements_pct": [3.0] * 20,
            }
            budget = {
                "name": "quick", "max_seconds": 2700, "branches": 4,
                "max_rounds": 1, "min_pairs": 20, "max_pairs": 50,
                "outer_candidates": 1, "max_cases": 3,
                "sanitizer_mode": "targeted", "reserve_seconds": 300,
            }
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2, "run_dir": str((root / "run").resolve()),
                        "input_hash": "frozen", "budget": budget,
                        "mode": "kernel-only", "workload": None,
                        "best_file": str(root / "best.py"), "iterations_total": 1,
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "status": "shortlist_ready",
                "selected_kernel": str(selected),
                "champion": {
                    "status": "confirmed_win", "kernel": str(selected),
                    "statistics": statistics,
                },
                "shortlist": [],
            }
            (iter_dir / "branch_results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            (iter_dir / "decision.json").write_text(
                json.dumps({"status": "confirmed_win", "candidate_file": str(selected)}),
                encoding="utf-8",
            )
            runner = mock.Mock(
                side_effect=[
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                ]
            )
            workload_pairs = mock.Mock(side_effect=AssertionError("must not run"))
            apply = mock.Mock(
                return_value={"returncode": 0, "decision_path": str(iter_dir / "decision.json")}
            )

            with mock.patch.object(orchestrate, "_run", runner), \
                    mock.patch.object(orchestrate.workload_evaluate, "evaluate_pairs", workload_pairs), \
                    mock.patch.object(orchestrate, "apply_decision", apply), \
                    contextlib.redirect_stdout(io.StringIO()):
                orchestrate.cmd_close_iter(args)

        workload_pairs.assert_not_called()
        self.assertEqual(apply.call_args.args[0]["status"], "kernel_only_win")


class BudgetedParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrate = _load_orchestrate()

    def test_setup_defaults_to_balanced_without_overriding_the_preset(self) -> None:
        args = self.orchestrate.build_parser().parse_args(
            ["setup", "--baseline", "kernel.py", "--ref", "ref.py", "--dims", "{}"]
        )
        self.assertEqual(args.budget, "balanced")
        self.assertIsNone(args.max_seconds)
        self.assertIsNone(args.max_rounds)
        self.assertIsNone(args.branches)
        policy = self.orchestrate.resolve_setup_policy(args)
        self.assertEqual(policy.name, "balanced")
        self.assertEqual(policy.branches, 8)
        self.assertEqual(policy.max_rounds, 4)

    def test_custom_requires_every_positive_budget_field(self) -> None:
        parser = self.orchestrate.build_parser()
        args = parser.parse_args(
            [
                "setup", "--baseline", "kernel.py", "--ref", "ref.py", "--dims", "{}",
                "--budget", "custom", "--max-seconds", "600", "--max-rounds", "2",
                "--branches", "3", "--min-pairs", "4", "--max-pairs", "5",
                "--outer-candidates", "1",
            ]
        )
        policy = self.orchestrate.resolve_setup_policy(args)
        self.assertEqual(policy.max_seconds, 600)
        self.assertEqual(policy.max_pairs, 5)

        missing = parser.parse_args(
            [
                "setup", "--baseline", "kernel.py", "--ref", "ref.py", "--dims", "{}",
                "--budget", "custom", "--max-seconds", "60",
            ]
        )
        with self.assertRaisesRegex(ValueError, "custom budget requires"):
            self.orchestrate.resolve_setup_policy(missing)

    def test_removed_noise_option_has_a_migration_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                self.orchestrate.build_parser().parse_args(
                    [
                        "setup", "--baseline", "kernel.py", "--ref", "ref.py",
                        "--dims", "{}", "--noise-threshold-pct", "2",
                    ]
                )
        message = stderr.getvalue()
        self.assertIn("removed", message)
        self.assertIn("--min-effect-pct", message)

    def test_api_rejects_nonfinite_confidence_and_boolean_min_effect(self) -> None:
        parser = self.orchestrate.build_parser()
        args = parser.parse_args(
            ["setup", "--baseline", "a", "--ref", "b", "--dims", "{}"]
        )
        args.confidence = math.nan
        with self.assertRaisesRegex(ValueError, "confidence"):
            self.orchestrate.resolve_setup_policy(args)
        args.confidence = 0.95
        args.min_effect_pct = True
        with self.assertRaisesRegex(ValueError, "min_effect"):
            self.orchestrate.resolve_setup_policy(args)

    def test_setup_freezes_full_workload_and_places_every_artifact_in_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            output_root = root / "runs"
            sources.mkdir()
            output_root.mkdir()
            baseline = sources / "kernel.py"
            ref = sources / "ref.py"
            baseline.write_text("# kernel\n", encoding="utf-8")
            ref.write_text("# ref\n", encoding="utf-8")
            spec = self.orchestrate.WorkloadSpec(
                kind="command",
                source=["/bin/echo", "ok"],
                objective={
                    "primary_metric": {"name": "latency", "direction": "lower"},
                    "min_effect_pct": 0.5,
                    "constraints": [],
                },
                cases=({"n": 1},),
                source_hash="a" * 64,
            )
            args = self.orchestrate.build_parser().parse_args(
                [
                    "setup", "--baseline", str(baseline), "--ref", str(ref),
                    "--dims", '{"N": 1}', "--output-root", str(output_root),
                    "--workload-cmd", "echo ok", "--objective", "objective.json",
                ]
            )
            normalized = mock.Mock(return_value=spec)
            preflight_run = mock.Mock(return_value={"ok": True, "errors": []})

            def check_env(command, **_kwargs):
                if Path(command[1]).name == "check_env.py":
                    env_path = Path(command[command.index("--out") + 1])
                    env_path.write_text(json.dumps({"gpu": "mock"}), encoding="utf-8")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if Path(command[1]).name == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                if Path(command[1]).name == "run_iteration.py":
                    state_path = Path(command[command.index("--state") + 1])
                    state = json.loads(state_path.read_text("utf-8"))
                    bench = Path(state["run_dir"]) / "baseline" / "bench.json"
                    bench.write_text(
                        json.dumps(
                            {
                                "correctness": {"passed": True},
                                "kernel": {"average_ms": 1.25},
                            }
                        ),
                        encoding="utf-8",
                    )
                    state["best_metric_ms"] = 1.25
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                raise AssertionError(f"unexpected setup command: {command}")

            with mock.patch.object(self.orchestrate, "normalize_workload", normalized), \
                    mock.patch.object(self.orchestrate.preflight, "run", preflight_run), \
                    mock.patch.object(self.orchestrate, "_run", side_effect=check_env) as runner, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.orchestrate.cmd_setup(args)

            output = json.loads(stdout.getvalue())
            run_dir = Path(output["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
            state = json.loads((run_dir / "state.json").read_text("utf-8"))
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            baseline_bench_hash = self.orchestrate.sha256_file(
                run_dir / "baseline" / "bench.json"
            )
            workload_spec_mode = stat.S_IMODE(
                (run_dir / "workload" / "spec.json").stat().st_mode
            ) if (run_dir / "workload" / "spec.json").exists() else None
            env_in_run = (run_dir / "env.json").is_file()
            env_at_source = (sources / "env.json").exists()

        normalized.assert_called_once_with(
            workload=None,
            workload_cmd="echo ok",
            workload_manifest=None,
            objective="objective.json",
        )
        preflight_run.assert_called_once()
        self.assertIs(preflight_run.call_args.args[4], spec)
        self.assertEqual(runner.call_count, 3)
        state_command = runner.call_args_list[1].args[0]
        self.assertIn("--output-root", state_command)
        self.assertIn("--budget-json", state_command)
        self.assertIn("--workload-file", state_command)
        self.assertNotIn("--workload-json", state_command)
        self.assertEqual(workload_spec_mode, 0o600)
        self.assertEqual(run_dir.parent, output_root.resolve())
        self.assertEqual(output["mode"], "full")
        self.assertEqual(manifest["workload"]["source_hash"], "a" * 64)
        self.assertEqual(state["workload"], manifest["workload"])
        self.assertEqual(state["input_hash"], manifest["input_hash"])
        self.assertEqual(checkpoint["input_hash"], state["input_hash"])
        self.assertEqual(checkpoint["stage"], "baseline")
        self.assertEqual(checkpoint["status"], "stage_complete")
        self.assertEqual(
            self.orchestrate.resume(
                checkpoint, input_hash=state["input_hash"]
            )["next_stage"],
            "candidate_correctness",
        )
        self.assertEqual(
            checkpoint["stage_evidence"]["baseline"]["status"], "passed"
        )
        baseline_evidence = checkpoint["stage_evidence"]["baseline"]
        self.assertEqual(baseline_evidence["metric_ms"], 1.25)
        self.assertTrue(baseline_evidence["correctness_passed"])
        self.assertEqual(
            baseline_evidence["bench_sha256"],
            baseline_bench_hash,
        )
        self.assertEqual(state["iterations_total"], 4)
        self.assertEqual(state["branches"], 8)
        self.assertTrue(env_in_run)
        self.assertFalse(env_at_source)

    def test_setup_keeps_secret_workload_snapshot_out_of_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "kernel.py"
            ref = root / "ref.py"
            baseline.write_text("# kernel\n", encoding="utf-8")
            ref.write_text("# ref\n", encoding="utf-8")
            secret = "API_TOKEN=super-secret-value"
            spec = self.orchestrate.WorkloadSpec(
                kind="command",
                source=["/bin/echo", secret],
                objective={
                    "primary_metric": {"name": "latency", "direction": "lower"},
                    "min_effect_pct": 0.5,
                    "constraints": [],
                },
                cases=(),
                source_hash="a" * 64,
            )
            args = self.orchestrate.build_parser().parse_args(
                [
                    "setup", "--baseline", str(baseline), "--ref", str(ref),
                    "--dims", "{}", "--output-root", str(root),
                    "--workload-cmd", "ignored", "--objective", "ignored.json",
                ]
            )
            commands = []

            def runner(command, **_kwargs):
                commands.append(list(command))
                script = Path(command[1]).name
                if script == "check_env.py":
                    Path(command[command.index("--out") + 1]).write_text("{}")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if script == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                if script == "run_iteration.py":
                    state_path = Path(command[command.index("--state") + 1])
                    state = json.loads(state_path.read_text("utf-8"))
                    bench = Path(state["run_dir"]) / "baseline" / "bench.json"
                    bench.write_text(json.dumps({"correctness": {"passed": True}, "kernel": {"average_ms": 1.0}}))
                    state["best_metric_ms"] = 1.0
                    state_path.write_text(json.dumps(state))
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                raise AssertionError(script)

            with mock.patch.object(
                self.orchestrate, "normalize_workload", return_value=spec
            ), mock.patch.object(
                self.orchestrate.preflight,
                "run",
                return_value={"ok": True, "errors": []},
            ), mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_setup(args)

            flattened = "\n".join(" ".join(command) for command in commands)
            self.assertNotIn(secret, flattened)
            self.assertNotIn("--workload-json", flattened)

    def test_run_log_redacts_sensitive_flags_and_secret_patterns(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(
            self.orchestrate.subprocess, "run", return_value=completed
        ), contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.orchestrate._run(
                [
                    "tool", "--token", "flag-secret",
                    "API_TOKEN=pattern-secret", "--safe", "visible",
                ]
            )
        logged = stderr.getvalue()
        self.assertNotIn("flag-secret", logged)
        self.assertNotIn("pattern-secret", logged)
        self.assertIn("visible", logged)

    def test_run_hard_timeout_kills_the_entire_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child_pid_path = root / "child.pid"
            script = root / "hang.py"
            script.write_text(
                "\n".join(
                    [
                        "import signal, subprocess, sys, time",
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                        "child = subprocess.Popen([sys.executable, '-c', "
                        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])",
                        f"open({str(child_pid_path)!r}, 'w').write(str(child.pid))",
                        "time.sleep(60)",
                    ]
                ),
                encoding="utf-8",
            )

            result = self.orchestrate._run(
                [sys.executable, str(script)],
                capture_output=True,
                hard_timeout=0.2,
                term_grace=0.1,
            )

            self.assertTrue(result.timed_out)
            self.assertEqual(result.returncode, 124)
            child_pid = int(child_pid_path.read_text("utf-8"))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            else:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.fail("timed-out grandchild process was left running")

    def test_run_waits_for_group_when_term_exits_leader_but_not_grandchild(self) -> None:
        survivors = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for attempt in range(3):
                child_pid_path = root / f"child-{attempt}.pid"
                script = root / f"leader-{attempt}.py"
                script.write_text(
                    "\n".join(
                        [
                            "import subprocess, sys, time",
                            "child = subprocess.Popen([sys.executable, '-c', "
                            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'], "
                            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
                            f"open({str(child_pid_path)!r}, 'w').write(str(child.pid))",
                            "time.sleep(60)",
                        ]
                    ),
                    encoding="utf-8",
                )

                result = self.orchestrate._run(
                    [sys.executable, str(script)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    hard_timeout=0.2,
                    term_grace=0.15,
                )

                self.assertTrue(result.timed_out)
                child_pid = int(child_pid_path.read_text("utf-8"))
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    survivors.append(child_pid)

        for pid in survivors:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.assertEqual(survivors, [], "TERM-ignoring grandchildren survived _run")

    def test_setup_rejects_bad_baseline_evidence_and_cleans_run(self) -> None:
        bad_payloads = (
            {"correctness": {"passed": False}, "kernel": {"average_ms": 1.0}},
            {"correctness": {"passed": True}, "kernel": {"average_ms": 0.0}},
            {"correctness": {"passed": 1}, "kernel": {"average_ms": 1.0}},
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline = root / "kernel.py"
                ref = root / "ref.py"
                baseline.write_text("# kernel\n", encoding="utf-8")
                ref.write_text("# ref\n", encoding="utf-8")
                allocated = root.resolve() / "allocated-run"
                allocated.mkdir()
                args = self.orchestrate.build_parser().parse_args(
                    [
                        "setup", "--baseline", str(baseline), "--ref", str(ref),
                        "--dims", "{}", "--output-root", str(root),
                    ]
                )

                def runner(command, **_kwargs):
                    script = Path(command[1]).name
                    if script == "check_env.py":
                        Path(command[command.index("--out") + 1]).write_text("{}")
                        return SimpleNamespace(returncode=0, stdout="", stderr="")
                    if script == "state.py":
                        return subprocess.run(command, capture_output=True, text=True)
                    if script == "run_iteration.py":
                        state_path = Path(command[command.index("--state") + 1])
                        state = json.loads(state_path.read_text("utf-8"))
                        bench = Path(state["run_dir"]) / "baseline" / "bench.json"
                        bench.write_text(json.dumps(payload), encoding="utf-8")
                        return SimpleNamespace(returncode=0, stdout="", stderr="")
                    raise AssertionError(script)

                with mock.patch.object(
                    self.orchestrate, "_allocate_run_dir", return_value=allocated
                ), mock.patch.object(
                    self.orchestrate, "normalize_workload", return_value=None
                ), mock.patch.object(
                    self.orchestrate.preflight,
                    "run",
                    return_value={"ok": True, "errors": []},
                ), mock.patch.object(
                    self.orchestrate, "_run", side_effect=runner
                ), self.assertRaisesRegex(ValueError, "baseline"):
                    self.orchestrate.cmd_setup(args)
                self.assertFalse(allocated.exists())

    def test_setup_aliases_and_removed_options_are_explicit(self) -> None:
        parser = self.orchestrate.build_parser()
        alias = parser.parse_args(
            [
                "setup", "--baseline", "a", "--ref", "b", "--dims", "{}",
                "--iterations", "3",
            ]
        )
        self.assertEqual(self.orchestrate.resolve_setup_policy(alias).max_rounds, 3)
        conflict = parser.parse_args(
            [
                "setup", "--baseline", "a", "--ref", "b", "--dims", "{}",
                "--iterations", "3", "--max-rounds", "2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "iterations.*max-rounds"):
            self.orchestrate.resolve_setup_policy(conflict)
        for option in ("--env-out", "--noise-threshold-pct"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    parser.parse_args(
                        [
                            "setup", "--baseline", "a", "--ref", "b",
                            "--dims", "{}", option, "value",
                        ]
                    )
                self.assertIn("removed", stderr.getvalue())

    def test_output_root_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.orchestrate.validate_output_root(alias, baseline=root / "kernel.py")

    def test_generic_json_reader_rejects_symlinked_parent_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            real = root / "real"
            real.mkdir()
            payload = real / "payload.json"
            payload.write_text('{"ok":true}', encoding="utf-8")
            linked = root / "linked"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                self.orchestrate._read(linked / "payload.json")


class LifecycleIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrate = _load_orchestrate()

    def _statistics(self) -> dict:
        return {
            "status": "confirmed_win",
            "statistic": "median_paired_improvement_pct",
            "direction": "lower",
            "min_effect_pct": 0.5,
            "confidence": 0.95,
            "estimate_pct": 3.0,
            "ci_low_pct": 3.0,
            "ci_high_pct": 3.0,
            "valid_pairs": 20,
            "invalid_pairs": 0,
            "improvements_pct": [3.0] * 20,
        }

    def _setup(self, root: Path) -> tuple[Path, Path]:
        baseline = root / "kernel.py"
        ref = root / "ref.py"
        baseline.write_text("# baseline\n", encoding="utf-8")
        ref.write_text("# reference\n", encoding="utf-8")
        args = self.orchestrate.build_parser().parse_args(
            [
                "setup", "--baseline", str(baseline), "--ref", str(ref),
                "--dims", "{}", "--output-root", str(root), "--budget", "quick",
            ]
        )

        def setup_runner(command, **_kwargs):
            if Path(command[1]).name == "check_env.py":
                env_path = Path(command[command.index("--out") + 1])
                env_path.write_text("{}", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if Path(command[1]).name == "state.py":
                return subprocess.run(command, capture_output=True, text=True)
            if Path(command[1]).name == "run_iteration.py":
                state_path = Path(command[command.index("--state") + 1])
                state = json.loads(state_path.read_text("utf-8"))
                bench = Path(state["run_dir"]) / "baseline" / "bench.json"
                bench.write_text(
                    json.dumps(
                        {
                            "correctness": {"passed": True},
                            "kernel": {"average_ms": 1.0},
                        }
                    ),
                    encoding="utf-8",
                )
                state["best_metric_ms"] = 1.0
                state_path.write_text(json.dumps(state), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(f"unexpected setup command: {command}")

        with mock.patch.object(self.orchestrate, "normalize_workload", return_value=None), \
                mock.patch.object(
                    self.orchestrate.preflight,
                    "run",
                    return_value={"ok": True, "errors": []},
                ), \
                mock.patch.object(self.orchestrate, "_run", side_effect=setup_runner), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.orchestrate.cmd_setup(args)
        output = json.loads(stdout.getvalue())
        return Path(output["run_dir"]), Path(output["state"])

    def _close_args(self, run_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(
            run_dir=str(run_dir), iter=1, benchmark="benchmark.py",
            warmup=1, repeat=1, retries=0,
        )

    def _write_mock_ncu_top(self, command, run_dir: Path) -> None:
        if Path(command[1]).name != "profile_ncu.py":
            return
        iter_dir = run_dir / "iterv1"
        kernel = next(iter(iter_dir.glob("kernel.*")))
        (iter_dir / "ncu_top.json").write_text(
            json.dumps({"profiled_file": str(kernel.resolve()), "axes": {}}),
            encoding="utf-8",
        )

    def _write_kernel_pair_evidence(
        self, run_dir: Path, selected: Path
    ) -> dict:
        state = json.loads((run_dir / "state.json").read_text("utf-8"))
        return self.orchestrate.write_paired_samples(
            run_dir / "iterv1" / "kernel-pairs" / "paired_samples.jsonl",
            [
                {
                    "block": index,
                    "baseline": 100.0,
                    "candidate": 97.0,
                    "valid": True,
                }
                for index in range(20)
            ],
            kind="kernel",
            input_hash=state["input_hash"],
            iteration=1,
            candidate_id="1",
            candidate_file=selected,
            classifier_config={
                "direction": "lower",
                "min_effect_pct": 0.5,
                "confidence": 0.95,
                "bootstrap_samples": 10000,
                "seed": 0,
            },
        )

    def _write_winner_artifacts(self, run_dir: Path) -> Path:
        iter_dir = run_dir / "iterv1"
        selected = iter_dir / "kernel.py"
        selected.write_text("# candidate\n", encoding="utf-8")
        (iter_dir / "methods.json").write_text(
            json.dumps({"methods": []}), encoding="utf-8"
        )
        (iter_dir / "bench.json").write_text(
            json.dumps(
                {
                    "correctness": {"passed": True},
                    "kernel": {"average_ms": 1.0},
                    "reference": {"average_ms": 2.0},
                }
            ),
            encoding="utf-8",
        )
        statistics = self._statistics()
        paired_samples = self._write_kernel_pair_evidence(run_dir, selected)
        branch_payload = {
            "status": "shortlist_ready",
            "selected_kernel": str(selected),
            "champion": {
                "status": "confirmed_win",
                "kernel": str(selected),
                "branch_index": 1,
                "statistics": statistics,
                "paired_samples": paired_samples,
            },
            "shortlist": [],
            "valid_branches": 1,
            "completed_comparisons": 1,
        }
        (iter_dir / "branch_results.json").write_text(
            json.dumps(branch_payload), encoding="utf-8"
        )
        (iter_dir / "decision.json").write_text(
            json.dumps(
                {
                    "status": "confirmed_win",
                    "candidate_file": str(selected),
                    "candidate_sha256": self.orchestrate.sha256_file(selected),
                    "candidate_id": "1",
                    "statistics": statistics,
                    "kernel_paired_samples": paired_samples,
                }
            ),
            encoding="utf-8",
        )
        return selected

    def _position_before_decision(
        self, run_dir: Path, state_path: Path, *, exhausted: bool
    ) -> dict:
        state = json.loads(state_path.read_text("utf-8"))
        checkpoint = json.loads(
            (run_dir / "checkpoint.json").read_text("utf-8")
        )
        checkpoint.update(
            iteration=1,
            stage="workload_paired",
            stage_index=self.orchestrate.STAGES.index("workload_paired"),
            status="stage_complete",
            candidate_status=None,
        )
        if exhausted:
            checkpoint["budget"] = {
                "elapsed_seconds": state["budget"]["max_seconds"],
                "remaining_seconds": 0.0,
            }
        self.orchestrate.ArtifactStore(run_dir).write_checkpoint(checkpoint)
        return state

    def test_real_setup_no_win_close_resume_and_finalize_records_all_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, state_path = self._setup(root)
            iter_dir = run_dir / "iterv1"
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": []}), encoding="utf-8"
            )
            branch_payload = {
                "status": "no_confirmed_kernel_win",
                "champion": None,
                "shortlist": [],
                "valid_branches": 2,
                "completed_comparisons": 2,
            }
            (iter_dir / "branch_results.json").write_text(
                json.dumps(branch_payload), encoding="utf-8"
            )
            (iter_dir / "decision.json").write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "candidate_file": None,
                        "statistics": None,
                    }
                ),
                encoding="utf-8",
            )
            close_args = SimpleNamespace(
                run_dir=str(run_dir), iter=1, benchmark="benchmark.py",
                warmup=1, repeat=1, retries=0,
            )

            def lifecycle_runner(command, **_kwargs):
                if Path(command[1]).name in {"state.py", "summarize.py"}:
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                self.orchestrate,
                "_run",
                side_effect=lifecycle_runner,
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(close_args)

            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            restored = self.orchestrate.resume(
                checkpoint, input_hash=json.loads(state_path.read_text("utf-8"))["input_hash"]
            )
            self.assertEqual(checkpoint["stage"], "decision")
            self.assertEqual(restored["next_stage"], "complete")
            evidence = checkpoint["stage_evidence"]
            self.assertEqual(evidence["candidate_correctness"]["status"], "passed")
            self.assertEqual(evidence["candidate_paired"]["status"], "completed")
            self.assertEqual(evidence["candidate_profile"]["status"], "not_applicable")
            self.assertEqual(evidence["candidate_sanitizer"]["status"], "deferred")
            self.assertEqual(evidence["workload_paired"]["status"], "not_applicable")
            self.assertEqual(evidence["decision"]["status"], "no_confirmed_kernel_win")
            state_after_close = json.loads(state_path.read_text("utf-8"))
            self.assertEqual(
                state_after_close["terminal_decision"]["status"],
                "no_confirmed_kernel_win",
            )
            self.assertEqual(
                state_after_close["terminal_decision"]["resume"]["iteration"], 1
            )
            self.assertIsNone(
                state_after_close["terminal_decision"]["resume"]["candidate_id"]
            )
            self.assertEqual(
                state_after_close["terminal_decision"]["resume"]["candidate_status"],
                "no_confirmed_kernel_win",
            )

            with mock.patch.object(
                self.orchestrate,
                "_run",
                side_effect=lifecycle_runner,
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_finalize(SimpleNamespace(run_dir=str(run_dir)))
            complete = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            self.assertEqual(
                self.orchestrate.resume(
                    complete, input_hash=complete["input_hash"]
                )["next_stage"],
                "complete",
            )
            final_state = json.loads(state_path.read_text("utf-8"))
            self.assertEqual(final_state["terminal_decision"]["status"], "no_confirmed_kernel_win")
            self.assertEqual(final_state["terminal_decision"]["resume"]["stage"], "complete")
            self.assertEqual(final_state["terminal_decision"]["resume"]["iteration"], 1)
            self.assertEqual(final_state["terminal_decision"]["resume"]["candidate_status"], "no_confirmed_kernel_win")
            self.assertTrue((run_dir / "summary.md").is_file())

    def test_no_win_decision_budget_denial_never_calls_state_producer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path = self._setup(Path(tmp))
            iter_dir = run_dir / "iterv1"
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": []}), encoding="utf-8"
            )
            (iter_dir / "branch_results.json").write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "champion": None,
                        "shortlist": [],
                        "completed_comparisons": 2,
                    }
                ),
                encoding="utf-8",
            )
            (iter_dir / "decision.json").write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "candidate_file": None,
                        "statistics": None,
                    }
                ),
                encoding="utf-8",
            )
            original_state = self._position_before_decision(
                run_dir, state_path, exhausted=True
            )
            runner = mock.Mock(
                return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
            )

            with mock.patch.object(
                self.orchestrate, "_run", runner
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))

            runner.assert_not_called()
            self.assertEqual(
                json.loads(state_path.read_text("utf-8"))["history"],
                original_state["history"],
            )
            decision = json.loads((iter_dir / "decision.json").read_text("utf-8"))
            self.assertEqual(decision["status"], "inconclusive")
            self.assertTrue(decision["budget_exhausted"])
            checkpoint = json.loads(
                (run_dir / "checkpoint.json").read_text("utf-8")
            )
            self.assertEqual(checkpoint["stage"], "decision")
            self.assertEqual(checkpoint["status"], "budget_exhausted")
            self.assertEqual(checkpoint["candidate_status"], "inconclusive")

    def test_winning_decision_budget_denial_never_calls_apply_or_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path = self._setup(Path(tmp))
            selected = self._write_winner_artifacts(run_dir)
            state = json.loads(state_path.read_text("utf-8"))
            terminal = {
                "status": "kernel_only_win",
                "candidate_file": str(selected.resolve()),
                "candidate_sha256": self.orchestrate.sha256_file(selected),
                "statistics": self._statistics(),
            }
            self.orchestrate._write_workload_result_artifact(
                run_dir / "iterv1",
                state=state,
                terminal_decision=terminal,
                candidate_id="1",
            )
            original_state = self._position_before_decision(
                run_dir, state_path, exhausted=True
            )
            applied = mock.Mock(
                return_value={"returncode": 0, "stdout": "", "stderr": ""}
            )

            with mock.patch.object(
                self.orchestrate, "apply_decision", applied
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))

            applied.assert_not_called()
            current_state = json.loads(state_path.read_text("utf-8"))
            self.assertEqual(current_state["best_file"], original_state["best_file"])
            self.assertEqual(current_state["history"], original_state["history"])
            decision = json.loads(
                (run_dir / "iterv1" / "decision.json").read_text("utf-8")
            )
            self.assertEqual(decision["status"], "inconclusive")
            self.assertEqual(decision["candidate_file"], str(selected.resolve()))
            self.assertEqual(
                decision["candidate_sha256"],
                self.orchestrate.sha256_file(selected),
            )
            self.assertEqual(decision["kernel_evidence"]["status"], "kernel_only_win")
            checkpoint = json.loads(
                (run_dir / "checkpoint.json").read_text("utf-8")
            )
            self.assertEqual(checkpoint["stage"], "decision")
            self.assertEqual(checkpoint["status"], "budget_exhausted")
            self.assertEqual(checkpoint["candidate_status"], "inconclusive")

    def test_no_win_decision_producer_timeout_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, state_path = self._setup(root)
            iter_dir = run_dir / "iterv1"
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": []}), encoding="utf-8"
            )
            (iter_dir / "branch_results.json").write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "champion": None,
                        "shortlist": [],
                        "completed_comparisons": 2,
                    }
                ),
                encoding="utf-8",
            )
            (iter_dir / "decision.json").write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "candidate_file": None,
                        "statistics": None,
                    }
                ),
                encoding="utf-8",
            )
            original_state = self._position_before_decision(
                run_dir, state_path, exhausted=False
            )
            fake_scripts = root / "fake-scripts"
            fake_scripts.mkdir()
            child_pid_path = root / "state-child.pid"
            (fake_scripts / "state.py").write_text(
                "\n".join(
                    [
                        "import json, signal, subprocess, sys, time",
                        "from pathlib import Path",
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                        "child = subprocess.Popen([sys.executable, '-c', "
                        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'], "
                        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
                        f"Path({str(child_pid_path)!r}).write_text(str(child.pid))",
                        "state_path = Path(sys.argv[sys.argv.index('--state') + 1])",
                        "state = json.loads(state_path.read_text())",
                        "state['history'].append({'event': 'partial-timeout-write'})",
                        "state_path.write_text(json.dumps(state))",
                        "time.sleep(60)",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                self.orchestrate, "SCRIPT_DIR", fake_scripts
            ), mock.patch.object(
                self.orchestrate, "_hard_timeout_seconds", return_value=0.2
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))

            self.assertEqual(
                json.loads(state_path.read_text("utf-8"))["history"],
                original_state["history"],
            )
            child_pid = int(child_pid_path.read_text("utf-8"))
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            else:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.fail("timed-out state producer descendant survived")
            decision = json.loads((iter_dir / "decision.json").read_text("utf-8"))
            self.assertEqual(decision["status"], "inconclusive")
            checkpoint = json.loads(
                (run_dir / "checkpoint.json").read_text("utf-8")
            )
            self.assertEqual(checkpoint["stage"], "decision")
            self.assertEqual(checkpoint["status"], "budget_exhausted")
            self.assertEqual(checkpoint["candidate_status"], "inconclusive")

    def test_winning_decision_producer_timeout_is_inconclusive_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path = self._setup(Path(tmp))
            selected = self._write_winner_artifacts(run_dir)
            state = json.loads(state_path.read_text("utf-8"))
            terminal = {
                "status": "kernel_only_win",
                "candidate_file": str(selected.resolve()),
                "candidate_sha256": self.orchestrate.sha256_file(selected),
                "statistics": self._statistics(),
            }
            self.orchestrate._write_workload_result_artifact(
                run_dir / "iterv1",
                state=state,
                terminal_decision=terminal,
                candidate_id="1",
            )
            original_state = self._position_before_decision(
                run_dir, state_path, exhausted=False
            )
            applied = mock.Mock(
                return_value={
                    "returncode": 124,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": True,
                }
            )

            with mock.patch.object(
                self.orchestrate, "apply_decision", applied
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))

            self.assertGreater(applied.call_args.kwargs["hard_timeout"], 0.0)
            current_state = json.loads(state_path.read_text("utf-8"))
            self.assertEqual(current_state["best_file"], original_state["best_file"])
            self.assertEqual(current_state["history"], original_state["history"])
            decision = json.loads(
                (run_dir / "iterv1" / "decision.json").read_text("utf-8")
            )
            self.assertEqual(decision["status"], "inconclusive")
            self.assertEqual(decision["candidate_file"], str(selected.resolve()))
            checkpoint = json.loads(
                (run_dir / "checkpoint.json").read_text("utf-8")
            )
            self.assertEqual(checkpoint["stage"], "decision")
            self.assertEqual(checkpoint["status"], "budget_exhausted")
            self.assertEqual(checkpoint["candidate_status"], "inconclusive")

    def test_no_win_round_one_resumes_and_runs_branch_for_round_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path = self._setup(Path(tmp))

            def write_no_win(iteration: int) -> None:
                iter_dir = run_dir / f"iterv{iteration}"
                (iter_dir / "methods.json").write_text(
                    json.dumps({"methods": []}), encoding="utf-8"
                )
                (iter_dir / "branch_results.json").write_text(
                    json.dumps(
                        {
                            "status": "no_confirmed_kernel_win",
                            "champion": None,
                            "shortlist": [],
                            "completed_comparisons": 2,
                        }
                    ),
                    encoding="utf-8",
                )
                (iter_dir / "decision.json").write_text(
                    json.dumps(
                        {
                            "status": "no_confirmed_kernel_win",
                            "candidate_file": None,
                            "statistics": None,
                        }
                    ),
                    encoding="utf-8",
                )

            def runner(command, **_kwargs):
                if Path(command[1]).name == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            write_no_win(1)
            with mock.patch.object(self.orchestrate, "_run", side_effect=runner), \
                    contextlib.redirect_stdout(io.StringIO()) as first_stdout:
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))
            self.assertEqual(json.loads(first_stdout.getvalue())["next_iter"], 2)
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            restored = self.orchestrate.resume(
                checkpoint,
                input_hash=checkpoint["input_hash"],
                max_rounds=2,
            )
            self.assertEqual(restored["next_stage"], "candidate_correctness")
            self.assertEqual(restored["next_iteration"], 2)
            history = json.loads(state_path.read_text("utf-8"))["history"]
            self.assertEqual(history[-1]["status"], "no_confirmed_kernel_win")

            write_no_win(2)
            calls = []

            def second_runner(command, **kwargs):
                calls.append(Path(command[1]).name)
                return runner(command, **kwargs)

            args = self._close_args(run_dir)
            args.iter = 2
            with mock.patch.object(
                self.orchestrate, "_run", side_effect=second_runner
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(args)
            self.assertEqual(calls.count("branch_explore.py"), 1)
            final_checkpoint = json.loads(
                (run_dir / "checkpoint.json").read_text("utf-8")
            )
            self.assertEqual(final_checkpoint["iteration"], 2)
            self.assertEqual(
                self.orchestrate.resume(
                    final_checkpoint,
                    input_hash=final_checkpoint["input_hash"],
                    max_rounds=2,
                )["next_stage"],
                "complete",
            )

    def test_real_setup_winning_close_records_ordered_stage_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, _state_path = self._setup(root)
            iter_dir = run_dir / "iterv1"
            selected = iter_dir / "kernel.py"
            selected.write_text("# candidate\n", encoding="utf-8")
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": []}), encoding="utf-8"
            )
            (iter_dir / "bench.json").write_text(
                json.dumps(
                    {
                        "correctness": {"passed": True},
                        "kernel": {"average_ms": 1.0},
                        "reference": {"average_ms": 2.0},
                    }
                ),
                encoding="utf-8",
            )
            statistics = self._statistics()
            paired_samples = self._write_kernel_pair_evidence(run_dir, selected)
            branch_payload = {
                "status": "shortlist_ready",
                "selected_kernel": str(selected),
                "champion": {
                    "status": "confirmed_win",
                    "kernel": str(selected),
                    "branch_index": 1,
                    "statistics": statistics,
                    "paired_samples": paired_samples,
                },
                "shortlist": [],
                "valid_branches": 1,
                "completed_comparisons": 1,
            }
            (iter_dir / "branch_results.json").write_text(
                json.dumps(branch_payload), encoding="utf-8"
            )
            (iter_dir / "decision.json").write_text(
                json.dumps(
                    {
                        "status": "confirmed_win",
                        "candidate_file": str(selected),
                        "candidate_sha256": self.orchestrate.sha256_file(selected),
                        "candidate_id": "1",
                        "statistics": statistics,
                        "kernel_paired_samples": paired_samples,
                    }
                ),
                encoding="utf-8",
            )
            close_args = SimpleNamespace(
                run_dir=str(run_dir), iter=1, benchmark="benchmark.py",
                warmup=1, repeat=1, retries=0,
            )

            def close_runner(command, **kwargs):
                script = Path(command[1]).name
                self._write_mock_ncu_top(command, run_dir)
                if script == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(self.orchestrate, "_run", side_effect=close_runner), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(close_args)

            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            self.assertEqual(checkpoint["stage"], "decision")
            self.assertEqual(checkpoint["candidate_id"], "1")
            evidence = checkpoint["stage_evidence"]
            self.assertEqual(evidence["candidate_correctness"]["status"], "passed")
            self.assertEqual(evidence["candidate_paired"]["status"], "passed")
            self.assertEqual(evidence["candidate_profile"]["status"], "passed")
            self.assertEqual(evidence["candidate_sanitizer"]["status"], "deferred")
            self.assertEqual(evidence["workload_paired"]["status"], "not_applicable")
            self.assertEqual(evidence["decision"]["status"], "kernel_only_win")

    def test_expired_before_branch_writes_inconclusive_checkpoint_without_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, state_path = self._setup(root)
            state = json.loads(state_path.read_text("utf-8"))
            state_path.write_text(json.dumps(state), encoding="utf-8")
            checkpoint_path = run_dir / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
            checkpoint["budget"] = {
                "elapsed_seconds": state["budget"]["max_seconds"],
                "remaining_seconds": 0.0,
            }
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            iter_dir = run_dir / "iterv1"
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": []}), encoding="utf-8"
            )
            runner = mock.Mock()
            close_args = SimpleNamespace(
                run_dir=str(run_dir), iter=1, benchmark="benchmark.py",
                warmup=1, repeat=1, retries=0,
            )
            with mock.patch.object(self.orchestrate, "_run", runner), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(close_args)

            runner.assert_not_called()
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            validated = self.orchestrate._validate_checkpoint(
                checkpoint, input_hash=state["input_hash"]
            )
            self.assertEqual(validated["stage"], "candidate_correctness")
            self.assertEqual(validated["status"], "budget_exhausted")
            self.assertIsNone(validated["candidate_id"])
            self.assertEqual(validated["candidate_status"], "inconclusive")
            self.assertEqual(
                self.orchestrate.resume(
                    validated, input_hash=state["input_hash"]
                )["next_stage"],
                "candidate_correctness",
            )

    def test_branch_hard_timeout_is_durably_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path = self._setup(Path(tmp))
            state = json.loads(state_path.read_text("utf-8"))
            iter_dir = run_dir / "iterv1"
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": []}), encoding="utf-8"
            )
            timed_out = SimpleNamespace(
                returncode=124, stdout="", stderr="", timed_out=True
            )
            close_args = SimpleNamespace(
                run_dir=str(run_dir), iter=1, benchmark="benchmark.py",
                warmup=1, repeat=1, retries=0,
            )

            with mock.patch.object(
                self.orchestrate, "_run", return_value=timed_out
            ) as runner, contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.orchestrate.cmd_close_iter(close_args)

            self.assertGreater(runner.call_args.kwargs["hard_timeout"], 0.0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "budget_exhausted")
            checkpoint = json.loads(
                (run_dir / "checkpoint.json").read_text("utf-8")
            )
            self.assertEqual(checkpoint["stage"], "candidate_correctness")
            self.assertEqual(checkpoint["status"], "budget_exhausted")
            self.assertEqual(checkpoint["candidate_status"], "inconclusive")
            self.assertEqual(
                checkpoint["stage_evidence"]["candidate_correctness"]["status"],
                "inconclusive",
            )
            self.assertEqual(checkpoint["input_hash"], state["input_hash"])

    def test_resume_reverifies_frozen_workload_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, state_path = self._setup(root)
            adapter = root / "workload.py"
            adapter.write_text(
                "\n".join(
                    [
                        "def prepare(candidate): return None",
                        "def validate(candidate): return True",
                        "def benchmark(candidate): return {'latency_ms': 1.0}",
                        "def metrics(): return {'primary_metric': {'name': 'latency_ms', 'direction': 'lower'}, 'min_effect_pct': 1.0, 'constraints': []}",
                        "def cleanup(): return None",
                    ]
                ),
                encoding="utf-8",
            )
            spec = self.orchestrate.normalize_workload(workload=adapter)
            snapshot = self.orchestrate._workload_snapshot(spec)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest["mode"] = "full"
            manifest["workload"] = snapshot
            manifest["input_hash"] = self.orchestrate._frozen_input_hash(
                manifest,
                workload=snapshot,
                dims=manifest["dims"],
                backend=manifest["backend"],
                budget=manifest["budget"],
                confidence=manifest["confidence"],
                min_effect_pct=manifest["min_effect_pct"],
                ptr_size=manifest["ptr_size"],
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            state = json.loads(state_path.read_text("utf-8"))
            state.update(
                mode="full", workload=snapshot, input_hash=manifest["input_hash"]
            )
            state_path.write_text(json.dumps(state), encoding="utf-8")
            checkpoint_path = run_dir / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
            checkpoint["input_hash"] = manifest["input_hash"]
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            adapter.write_text(
                adapter.read_text("utf-8") + "\n# drifted\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "source_hash"):
                self.orchestrate.cmd_resume(
                    SimpleNamespace(run_dir=str(run_dir))
                )

    def test_correctness_checkpoint_resumes_at_paired_without_rerunning_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path = self._setup(Path(tmp))
            iter_dir = run_dir / "iterv1"
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": []}), encoding="utf-8"
            )
            (iter_dir / "branch_results.json").write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "champion": None,
                        "shortlist": [],
                        "completed_comparisons": 2,
                    }
                ),
                encoding="utf-8",
            )
            (iter_dir / "decision.json").write_text(
                json.dumps(
                    {
                        "status": "no_confirmed_kernel_win",
                        "candidate_file": None,
                        "statistics": None,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            checkpoint = self.orchestrate.transition_checkpoint(
                checkpoint, "candidate_correctness", status="ready"
            )
            checkpoint = self.orchestrate.transition_checkpoint(
                checkpoint,
                "candidate_correctness",
                status="stage_complete",
                evidence={"status": "passed"},
            )
            self.orchestrate.ArtifactStore(run_dir).write_checkpoint(checkpoint)
            self.assertEqual(
                self.orchestrate.resume(
                    checkpoint,
                    input_hash=json.loads(state_path.read_text("utf-8"))["input_hash"],
                )["next_stage"],
                "candidate_paired",
            )
            runner = mock.Mock(
                return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
            )
            with mock.patch.object(self.orchestrate, "_run", runner), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))

            scripts = [Path(call.args[0][1]).name for call in runner.call_args_list]
            self.assertNotIn("branch_explore.py", scripts)
            completed = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            self.assertEqual(completed["stage"], "decision")

    def test_profile_checkpoint_resumes_at_sanitizer_without_branch_or_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _state_path = self._setup(Path(tmp))
            self._write_winner_artifacts(run_dir)

            def stop_after_profile(command, **_kwargs):
                script = Path(command[1]).name
                self._write_mock_ncu_top(command, run_dir)
                if script == "sass_check.py":
                    raise RuntimeError("stop after profile")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                self.orchestrate, "_run", side_effect=stop_after_profile
            ), self.assertRaisesRegex(RuntimeError, "stop after profile"):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))
            interrupted = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            self.assertEqual(interrupted["stage"], "candidate_sanitizer")
            self.assertEqual(
                interrupted["stage_evidence"]["candidate_profile"]["status"],
                "passed",
            )
            self.assertTrue((run_dir / "iterv1" / "selected_candidate.json").is_file())

            calls = []

            def resumed_runner(command, **_kwargs):
                calls.append(Path(command[1]).name)
                if Path(command[1]).name == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                self.orchestrate, "_run", side_effect=resumed_runner
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))
            self.assertNotIn("branch_explore.py", calls)
            self.assertNotIn("profile_ncu.py", calls)

    def test_workload_checkpoint_resumes_at_decision_without_workload_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _state_path = self._setup(Path(tmp))
            self._write_winner_artifacts(run_dir)
            def first_runner(command, **_kwargs):
                self._write_mock_ncu_top(command, run_dir)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                self.orchestrate, "_run", side_effect=first_runner
            ), mock.patch.object(
                self.orchestrate,
                "apply_decision",
                side_effect=RuntimeError("stop before decision"),
            ), self.assertRaisesRegex(RuntimeError, "stop before decision"):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))
            interrupted = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            self.assertEqual(interrupted["stage"], "decision")
            self.assertEqual(interrupted["status"], "in_progress")
            self.assertTrue((run_dir / "iterv1" / "workload_result.json").is_file())

            workload = mock.Mock(side_effect=AssertionError("workload reran"))
            applied = mock.Mock(
                return_value={"returncode": 0, "stdout": "", "stderr": ""}
            )
            with mock.patch.object(
                self.orchestrate, "evaluate_outer_candidate", workload
            ), mock.patch.object(
                self.orchestrate, "apply_decision", applied
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))
            workload.assert_not_called()
            applied.assert_called_once()

    def test_decision_checkpoint_makes_close_idempotent_without_state_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _state_path = self._setup(Path(tmp))
            self._write_winner_artifacts(run_dir)

            def first_runner(command, **_kwargs):
                self._write_mock_ncu_top(command, run_dir)
                if Path(command[1]).name == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                self.orchestrate, "_run", side_effect=first_runner
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            self.assertEqual(checkpoint["stage"], "decision")
            self.assertEqual(checkpoint["status"], "stage_complete")
            # A completed decision no longer needs intermediate update inputs.
            (run_dir / "iterv1" / "methods.json").unlink()

            runner = mock.Mock()
            applied = mock.Mock()
            with mock.patch.object(self.orchestrate, "_run", runner), \
                    mock.patch.object(self.orchestrate, "apply_decision", applied), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))
            runner.assert_not_called()
            applied.assert_not_called()

    def test_resume_from_profile_checkpoint_rejects_missing_selection_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _state_path = self._setup(Path(tmp))
            self._write_winner_artifacts(run_dir)

            def stop_after_profile(command, **_kwargs):
                self._write_mock_ncu_top(command, run_dir)
                if Path(command[1]).name == "sass_check.py":
                    raise RuntimeError("stop after profile")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                self.orchestrate, "_run", side_effect=stop_after_profile
            ), self.assertRaisesRegex(RuntimeError, "stop after profile"):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))
            selection = run_dir / "iterv1" / "selected_candidate.json"
            self.assertTrue(selection.is_file())
            selection.unlink()

            with self.assertRaisesRegex(ValueError, "selected_candidate.*missing"):
                self.orchestrate.cmd_close_iter(self._close_args(run_dir))


class CheckpointAndResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrate = _load_orchestrate()
        self.budget = self.orchestrate.resolve_budget(
            "custom",
            max_seconds=20,
            max_rounds=1,
            branches=1,
            min_pairs=1,
            max_pairs=1,
            outer_candidates=1,
            reserve_seconds=5,
        )

    def _checkpoint(self, root: Path) -> dict:
        return {
            "schema_version": 2,
            "input_hash": "frozen",
            "run_dir": str(root),
            "iteration": 0,
            "stage": "baseline",
            "stage_index": 0,
            "status": "ready",
            "candidate_id": None,
            "candidate_status": None,
            "budget": {"elapsed_seconds": 0.0, "remaining_seconds": 20.0},
            "updated_at": 100.0,
        }

    def test_checkpoint_iteration_is_strict_and_decision_wraps_one_round(self) -> None:
        invalid = self._checkpoint(Path("/tmp/run"))
        del invalid["iteration"]
        with self.assertRaisesRegex(ValueError, "iteration"):
            self.orchestrate._validate_checkpoint(invalid)
        invalid = self._checkpoint(Path("/tmp/run"))
        invalid["iteration"] = True
        with self.assertRaisesRegex(ValueError, "iteration"):
            self.orchestrate._validate_checkpoint(invalid)

        baseline = self.orchestrate.transition_checkpoint(
            self._checkpoint(Path("/tmp/run")),
            "baseline",
            status="stage_complete",
        )
        first = self.orchestrate.resume(
            baseline, input_hash="frozen", max_rounds=2
        )
        self.assertEqual(first["next_stage"], "candidate_correctness")
        self.assertEqual(first["next_iteration"], 1)

        decision = self._checkpoint(Path("/tmp/run"))
        decision.update(
            stage="decision",
            stage_index=self.orchestrate.STAGES.index("decision"),
            status="stage_complete",
            iteration=1,
        )
        second = self.orchestrate.resume(
            decision, input_hash="frozen", max_rounds=2
        )
        self.assertEqual(second["next_stage"], "candidate_correctness")
        self.assertEqual(second["next_iteration"], 2)
        wrapped = self.orchestrate.transition_checkpoint(
            decision, "candidate_correctness", status="ready", iteration=2
        )
        self.assertEqual(wrapped["iteration"], 2)
        with self.assertRaisesRegex(ValueError, "iteration|wrap"):
            self.orchestrate.transition_checkpoint(
                decision, "candidate_correctness", status="ready", iteration=3
            )

    def test_close_iteration_must_match_checkpoint_next_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            helper = LifecycleIntegrationTests()
            helper.setUp()
            run_dir, _state_path = helper._setup(Path(tmp))
            (run_dir / "iterv2" / "methods.json").write_text(
                json.dumps({"methods": []}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "iteration.*checkpoint"):
                helper.orchestrate.cmd_close_iter(
                    SimpleNamespace(
                        run_dir=str(run_dir), iter=2, benchmark="benchmark.py",
                        warmup=1, repeat=1, retries=0,
                    )
                )

    def test_deadline_stops_and_writes_the_real_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self._checkpoint(root)
            clock = self.orchestrate.BudgetClock(self.budget, started_at=100.0)
            result = self.orchestrate.schedule_next(
                state,
                clock,
                5.1,
                now=110.0,
                run_dir=root,
                candidate_id="c1",
            )
            persisted = json.loads((root / "checkpoint.json").read_text("utf-8"))

        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["candidate_status"], "inconclusive")
        self.assertTrue(result["checkpoint_written"])
        self.assertEqual(persisted["status"], "budget_exhausted")
        self.assertTrue(persisted["checkpoint_written"])

    def test_checkpoint_write_failure_never_reports_success(self) -> None:
        state = self._checkpoint(Path("/tmp/run"))
        clock = self.orchestrate.BudgetClock(self.budget, started_at=100.0)
        store = mock.Mock()
        store.write_checkpoint.side_effect = OSError("disk full")
        result = self.orchestrate.schedule_next(
            state, clock, 10.0, now=110.0, store=store, candidate_id="c1"
        )
        self.assertFalse(result["checkpoint_written"])
        self.assertIn("checkpoint_error", result)

    def test_schedule_next_rejects_completed_current_stage(self) -> None:
        checkpoint = self.orchestrate.transition_checkpoint(
            self._checkpoint(Path("/tmp/run")),
            "baseline",
            status="stage_complete",
        )
        clock = self.orchestrate.BudgetClock(self.budget, started_at=100.0)
        with self.assertRaisesRegex(ValueError, "completed stage"):
            self.orchestrate.schedule_next(
                checkpoint, clock, 1.0, now=101.0, candidate_id=None
            )

    def test_transition_is_ordered_and_resume_skips_completed_stage(self) -> None:
        checkpoint = self._checkpoint(Path("/tmp/run"))
        completed = self.orchestrate.transition_checkpoint(
            checkpoint, "baseline", status="stage_complete", updated_at=101.0
        )
        resumed = self.orchestrate.resume(completed, input_hash="frozen")
        self.assertEqual(resumed["next_stage"], "candidate_correctness")
        self.assertEqual(checkpoint["status"], "ready")

        replay = self.orchestrate.transition_checkpoint(
            completed, "baseline", status="stage_complete", updated_at=102.0
        )
        self.assertEqual(replay["stage"], "baseline")
        with self.assertRaisesRegex(ValueError, "skip|order"):
            self.orchestrate.transition_checkpoint(
                completed, "candidate_paired", status="stage_complete"
            )

    def test_resume_rejects_changed_hash_and_is_idempotent_at_complete(self) -> None:
        checkpoint = self._checkpoint(Path("/tmp/run"))
        with self.assertRaisesRegex(ValueError, "frozen input"):
            self.orchestrate.resume(checkpoint, input_hash="changed")
        checkpoint.update(
            stage="complete",
            stage_index=len(self.orchestrate.STAGES) - 1,
            status="complete",
        )
        resumed = self.orchestrate.resume(checkpoint, input_hash="frozen")
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["next_stage"], "complete")

    def test_stage_exception_runs_cleanup_once_without_marking_complete(self) -> None:
        checkpoint = self._checkpoint(Path("/tmp/run"))
        cleanup = mock.Mock()
        store = mock.Mock()
        store.write_checkpoint.return_value = Path("checkpoint.json")

        def fail():
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.orchestrate.execute_stage(
                checkpoint,
                "baseline",
                fail,
                store=store,
                cleanup=cleanup,
                updated_at=101.0,
            )
        cleanup.assert_called_once_with()
        saved = store.write_checkpoint.call_args.args[0]
        self.assertEqual(saved["stage"], "baseline")
        self.assertEqual(saved["status"], "failed")

    def test_finalize_advances_a_completed_decision_checkpoint_to_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            checkpoint = self._checkpoint(root)
            checkpoint.update(
                stage="decision",
                stage_index=self.orchestrate.STAGES.index("decision"),
                status="stage_complete",
            )
            self.orchestrate.ArtifactStore(root).write_checkpoint(checkpoint)
            (root / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "run_dir": str(root),
                        "input_hash": checkpoint["input_hash"],
                        "budget": {},
                        "candidates": {},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(run_dir=str(root))

            def summarize_after_complete(_command, **_kwargs):
                stored_checkpoint = json.loads(
                    (root / "checkpoint.json").read_text("utf-8")
                )
                stored_state = json.loads((root / "state.json").read_text("utf-8"))
                self.assertEqual(stored_checkpoint["status"], "complete")
                self.assertEqual(stored_state["resume"]["status"], "complete")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                self.orchestrate,
                "_run",
                side_effect=summarize_after_complete,
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_finalize(args)
            persisted = json.loads((root / "checkpoint.json").read_text("utf-8"))

        self.assertEqual(persisted["stage"], "complete")
        self.assertEqual(persisted["status"], "complete")

    def test_cli_resume_rejects_candidate_drift_and_symlink_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            candidate = root / "candidate.py"
            candidate.write_text("# original\n", encoding="utf-8")
            digest = self.orchestrate.sha256_file(candidate)
            state = {
                "schema_version": 2,
                "run_dir": str(root),
                "input_hash": "frozen",
                "budget": {},
                "candidates": {
                    "c1": {
                        "candidate_file": str(candidate),
                        "candidate_sha256": digest,
                    }
                },
            }
            (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
            self.orchestrate.ArtifactStore(root).write_checkpoint(
                self._checkpoint(root)
            )
            candidate.write_text("# drifted\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted"):
                self.orchestrate.cmd_resume(SimpleNamespace(run_dir=str(root)))

            (root / "checkpoint.json").unlink()
            external = root / "external.json"
            external.write_text("{}", encoding="utf-8")
            (root / "checkpoint.json").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.orchestrate.cmd_resume(SimpleNamespace(run_dir=str(root)))


class OuterLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrate = _load_orchestrate()

    def test_select_outer_candidates_is_stable_strict_and_nonmutating(self) -> None:
        items = [
            {"id": "a", "status": "confirmed_win", "statistics": {"estimate_pct": 3.0}},
            {"id": "b", "status": "confirmed_loss", "statistics": {"estimate_pct": 99.0}},
            {"id": "c", "status": "confirmed_win", "statistics": {"estimate_pct": 3.0}},
            {"id": "d", "status": "confirmed_win", "statistics": {"estimate_pct": 2.0}},
        ]
        before = json.loads(json.dumps(items))
        selected = self.orchestrate.select_outer_candidates(items, 2)
        self.assertEqual([item["id"] for item in selected], ["a", "c"])
        self.assertEqual(items, before)
        for bad in (True, 0, -1):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                self.orchestrate.select_outer_candidates(items, bad)
        malformed = [{"status": "confirmed_win", "statistics": {"estimate_pct": math.nan}}]
        with self.assertRaisesRegex(ValueError, "estimate_pct"):
            self.orchestrate.select_outer_candidates(malformed, 1)

    def test_candidate_file_rejects_symlinked_parent_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            real = root / "real"
            real.mkdir()
            candidate = real / "kernel.py"
            candidate.write_text("# candidate\n", encoding="utf-8")
            linked = root / "linked"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                self.orchestrate._candidate_file(
                    {"candidate_file": str(linked / "kernel.py")}
                )

    def test_kernel_only_terminal_never_calls_workload_and_can_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "kernel.py"
            candidate.write_text("# candidate\n", encoding="utf-8")
            kernel = {
                "status": "confirmed_win",
                "candidate_file": str(candidate),
                "statistics": self.orchestrate_test_statistics(),
            }
            evaluator = mock.Mock()
            result = self.orchestrate.evaluate_outer_candidate(
                kernel,
                mode="kernel-only",
                workload_spec=None,
                baseline="baseline.py",
                policy=self.orchestrate.resolve_budget("quick"),
                confidence=0.95,
                evaluator=evaluator,
            )
        evaluator.assert_not_called()
        self.assertEqual(result["status"], "kernel_only_win")

    def test_branch_index_zero_is_a_stable_candidate_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_file = Path(tmp) / "kernel.py"
            candidate_file.write_text("# candidate\n", encoding="utf-8")
            candidate = {
                "branch_index": 0,
                "status": "confirmed_win",
                "candidate_file": str(candidate_file),
                "statistics": self.orchestrate_test_statistics(),
            }

            decision = self.orchestrate.build_terminal_decision(
                mode="kernel-only",
                candidate=candidate,
                decide_fn=lambda **_kwargs: {"status": "kernel_only_win"},
            )

            self.assertEqual(decision["candidate_id"], "0")

    def test_full_outer_evaluation_uses_the_frozen_spec_and_can_end_to_end_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_file = Path(tmp) / "kernel.py"
            candidate_file.write_text("# candidate\n", encoding="utf-8")
            candidate = {
                "status": "confirmed_win",
                "candidate_file": str(candidate_file),
                "statistics": self.orchestrate_test_statistics(),
            }
            spec = self.orchestrate.WorkloadSpec(
                kind="command",
                source=["/bin/echo"],
                objective={
                    "primary_metric": {"name": "latency", "direction": "lower"},
                    "min_effect_pct": 0.5,
                    "constraints": [],
                },
                cases=(),
                source_hash="a" * 64,
            )
            workload_result = {
                "status": "evaluated",
                "objective": dict(spec.objective),
                "primary": self.orchestrate_test_statistics(),
                "constraints": [],
            }
            evaluator = mock.Mock(return_value=workload_result)
            result = self.orchestrate.evaluate_outer_candidate(
                candidate,
                mode="full",
                workload_spec=spec,
                baseline="best.py",
                policy=self.orchestrate.resolve_budget("quick"),
                confidence=0.95,
                evaluator=evaluator,
            )

        self.assertEqual(result["status"], "end_to_end_win")
        self.assertIs(evaluator.call_args.args[0], spec)
        self.assertEqual(evaluator.call_args.args[1], "best.py")
        self.assertEqual(evaluator.call_args.kwargs["blocks"], 50)
        self.assertEqual(result["candidate_sha256"], "2a283e1acf58a80beaf171b3e8df6cbb378e33419c4aac17082b41bee540b84a")

    def test_full_outer_evaluation_persists_bound_workload_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "run" / "iterv2"
            iter_dir.mkdir(parents=True)
            candidate_file = iter_dir / "candidate.py"
            candidate_file.write_text("# candidate\n", encoding="utf-8")
            candidate = {
                "branch_index": 0,
                "status": "confirmed_win",
                "candidate_file": str(candidate_file),
                "statistics": self.orchestrate_test_statistics(),
            }
            spec = self.orchestrate.WorkloadSpec(
                kind="command",
                source=["/bin/echo"],
                objective={
                    "primary_metric": {"name": "latency", "direction": "lower"},
                    "min_effect_pct": 0.5,
                    "constraints": [],
                },
                cases=(),
                source_hash="c" * 64,
            )
            raw_pairs = [
                {
                    "block": 0,
                    "order": "AB",
                    "case": {},
                    "baseline_metrics": {"latency": 2.0},
                    "candidate_metrics": {"latency": 1.0},
                    "valid": True,
                    "attempts": {"baseline": 1, "candidate": 1},
                    "attempt_records": {"baseline": [], "candidate": []},
                }
            ]
            workload_result = {
                "status": "evaluated",
                "objective": dict(spec.objective),
                "primary": self.orchestrate_test_statistics(),
                "constraints": [],
                "pairs": copy.deepcopy(raw_pairs),
            }
            result = self.orchestrate.evaluate_outer_candidate(
                candidate,
                mode="full",
                workload_spec=spec,
                baseline="best.py",
                policy=self.orchestrate.resolve_budget("quick"),
                confidence=0.95,
                evaluator=mock.Mock(return_value=workload_result),
                candidate_root=iter_dir,
                input_hash="a" * 64,
                iteration=2,
            )
            evidence = result["workload_paired_samples"]
            artifact = Path(evidence["path"])
            records = [
                json.loads(line) for line in artifact.read_text("utf-8").splitlines()
            ]

        self.assertEqual(evidence["input_hash"], "a" * 64)
        self.assertEqual(evidence["iteration"], 2)
        self.assertEqual(evidence["candidate_id"], "0")
        self.assertEqual(result["candidate_id"], "0")
        self.assertEqual(evidence["pairs"], 1)
        self.assertEqual(records[0]["pair"], raw_pairs[0])

    def test_full_outer_non_win_matrix_preserves_workload_terminal_evidence(self) -> None:
        objective = {
            "primary_metric": {"name": "latency", "direction": "lower"},
            "min_effect_pct": 1.0,
            "constraints": [{"name": "memory", "max_regression_pct": 2.0}],
        }

        def primary(status: str, estimate: float, low: float, high: float) -> dict:
            return {
                "status": status,
                "statistic": "median_paired_improvement_pct",
                "direction": "lower",
                "min_effect_pct": 1.0,
                "confidence": 0.95,
                "estimate_pct": estimate,
                "ci_low_pct": low,
                "ci_high_pct": high,
                "valid_pairs": 1,
                "invalid_pairs": 0,
                "improvements_pct": [estimate],
            }

        def constraint(status: str, estimate: float, low: float, high: float) -> dict:
            return {
                "name": "memory",
                "max_regression_pct": 2.0,
                "cap_pct": 2.0,
                "estimate_pct": estimate,
                "ci_low_pct": low,
                "ci_high_pct": high,
                "status": status,
                "values_pct": [estimate],
            }

        cases = {
            "primary_loss": {
                "status": "evaluated",
                "objective": objective,
                "primary": primary("confirmed_loss", -2.0, -2.0, -2.0),
                "constraints": [constraint("passed", 0.0, 0.0, 0.0)],
            },
            "primary_inconclusive": {
                "status": "evaluated",
                "objective": objective,
                "primary": primary("inconclusive", 0.0, 0.0, 0.0),
                "constraints": [constraint("passed", 0.0, 0.0, 0.0)],
            },
            "constraint_failed": {
                "status": "evaluated",
                "objective": objective,
                "primary": primary("confirmed_win", 3.0, 3.0, 3.0),
                "constraints": [constraint("failed", 5.0, 5.0, 5.0)],
            },
            "constraint_inconclusive": {
                "status": "evaluated",
                "objective": objective,
                "primary": primary("confirmed_win", 3.0, 3.0, 3.0),
                "constraints": [constraint("inconclusive", 2.0, 2.0, 3.0)],
            },
            "workload_failed": {
                "status": "workload_failed",
                "objective": objective,
                "reason": "one or more workload roles exhausted retries",
                "primary": {
                    "status": "invalid",
                    "statistic": "median_paired_improvement_pct",
                    "estimate_pct": None,
                    "ci_low_pct": None,
                    "ci_high_pct": None,
                },
                "constraints": [],
            },
        }
        for name, workload_result in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                iter_dir = Path(tmp).resolve() / "iterv1"
                iter_dir.mkdir()
                candidate_file = iter_dir / "candidate.py"
                candidate_file.write_text("# candidate\n", encoding="utf-8")
                candidate = {
                    "id": "b1",
                    "status": "confirmed_win",
                    "candidate_file": str(candidate_file),
                    "statistics": self.orchestrate_test_statistics(),
                }
                raw_pair = {
                    "block": 0,
                    "order": "AB",
                    "case": None,
                    "baseline_metrics": {"latency": 100.0, "memory": 100.0},
                    "candidate_metrics": {"latency": 100.0, "memory": 100.0},
                    "valid": name != "workload_failed",
                    "attempts": {"baseline": 1, "candidate": 1},
                    "attempt_records": {"baseline": [], "candidate": []},
                }
                result_payload = copy.deepcopy(workload_result)
                result_payload.update(
                    confidence=0.95,
                    bootstrap_samples=10000,
                    seed=0,
                    pairs=[raw_pair],
                )
                result = self.orchestrate.evaluate_outer_candidate(
                    candidate,
                    mode="full",
                    workload_spec=self.orchestrate.WorkloadSpec(
                        kind="command",
                        source=["/bin/echo"],
                        objective=objective,
                        cases=(),
                        source_hash="c" * 64,
                    ),
                    baseline="best.py",
                    policy=self.orchestrate.resolve_budget("quick"),
                    confidence=0.95,
                    evaluator=mock.Mock(return_value=result_payload),
                    candidate_root=iter_dir,
                    input_hash="a" * 64,
                    iteration=1,
                )

                self.assertEqual(result["workload_status"], workload_result["status"])
                self.assertIn("workload_paired_samples", result)
                if workload_result["status"] == "evaluated":
                    self.assertEqual(
                        result["workload_statistics"], workload_result["primary"]
                    )
                    self.assertEqual(result["constraints"], workload_result["constraints"])
                else:
                    self.assertEqual(
                        result["workload_failure"]["reason"],
                        workload_result["reason"],
                    )

    def test_actual_outer_non_win_matrix_survives_state_update_and_finalize(self) -> None:
        cases = {
            "primary_loss": {"latency": 102.0, "memory": [100.0], "fail": False},
            "primary_inconclusive": {"latency": 100.0, "memory": [100.0], "fail": False},
            "constraint_failed": {"latency": 97.0, "memory": [105.0], "fail": False},
            "constraint_inconclusive": {
                "latency": 97.0,
                "memory": [100.0, 104.0, 100.0],
                "fail": False,
            },
            "workload_failed": {"latency": 97.0, "memory": [100.0], "fail": True},
        }
        expected_terminal = {
            "primary_loss": "kernel_only_win",
            "primary_inconclusive": "kernel_only_win",
            "constraint_failed": "rejected_constraint",
            "constraint_inconclusive": "kernel_only_win",
            "workload_failed": "kernel_only_win",
        }
        for name, outcome in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                run_dir = root / "run"
                iter_dir = run_dir / "iterv1"
                iter_dir.mkdir(parents=True)
                baseline = run_dir / "baseline.py"
                candidate_file = iter_dir / "kernel.py"
                baseline.write_text("# baseline\n", encoding="utf-8")
                candidate_file.write_text("# candidate\n", encoding="utf-8")
                input_hash = "a" * 64
                objective = {
                    "primary_metric": {"name": "latency", "direction": "lower"},
                    "min_effect_pct": 1.0,
                    "constraints": [
                        {"name": "memory", "max_regression_pct": 2.0}
                    ],
                }
                memory_values = outcome["memory"]
                workload_cases = tuple(
                    {"index": index, "memory": value}
                    for index, value in enumerate(memory_values)
                )
                workload_spec = self.orchestrate.WorkloadSpec(
                    kind="command",
                    source=["/bin/echo"],
                    objective=objective,
                    cases=workload_cases,
                    source_hash="b" * 64,
                )
                state_path = run_dir / "state.json"
                state_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "run_dir": str(run_dir),
                            "input_hash": input_hash,
                            "budget": {
                                "name": "custom",
                                "max_seconds": 60,
                                "max_rounds": 1,
                                "branches": 1,
                            },
                            "candidates": {},
                            "mode": "full",
                            "workload": {
                                "kind": "command",
                                "source": ["/bin/echo"],
                                "source_hash": "b" * 64,
                                "cases": list(workload_cases),
                                "objective": objective,
                            },
                            "confidence": 0.95,
                            "min_effect_pct": 1.0,
                            "bootstrap_samples": 10000,
                            "seed": 0,
                            "best_file": str(baseline),
                            "best_metric_ms": 2.0,
                            "best_kernel_statistics": None,
                            "best_workload_statistics": None,
                            "selected_methods": [],
                            "effective_methods": [],
                            "ineffective_methods": [],
                            "implementation_failed_methods": [],
                            "history": [],
                            "roofline_history": [],
                            "frontier": [],
                        }
                    ),
                    encoding="utf-8",
                )
                kernel_pairs = [
                    {"baseline": 100.0, "candidate": 97.0, "valid": True}
                    for _ in range(3)
                ]
                kernel_statistics = self.orchestrate.workload_evaluate.paired_stats.classify_pairs(
                    kernel_pairs,
                    direction="lower",
                    min_effect_pct=1.0,
                    confidence=0.95,
                    bootstrap_samples=10000,
                    seed=0,
                )
                kernel_evidence = self.orchestrate.write_paired_samples(
                    iter_dir / "kernel-pairs.jsonl",
                    kernel_pairs,
                    kind="kernel",
                    input_hash=input_hash,
                    iteration=1,
                    candidate_id="b1",
                    candidate_file=candidate_file,
                    classifier_config={
                        "direction": "lower",
                        "min_effect_pct": 1.0,
                        "confidence": 0.95,
                        "bootstrap_samples": 10000,
                        "seed": 0,
                    },
                )
                candidate = {
                    "id": "b1",
                    "status": "confirmed_win",
                    "candidate_file": str(candidate_file),
                    "statistics": kernel_statistics,
                    "paired_samples": kernel_evidence,
                }

                def workload_runner(_spec, *, role, case=None, **_kwargs):
                    if outcome["fail"] and role == "candidate":
                        raise RuntimeError("workload failed")
                    memory = 100.0 if role == "baseline" else case["memory"]
                    latency = 100.0 if role == "baseline" else outcome["latency"]
                    return {
                        "role": role,
                        "case": case,
                        "validation": True,
                        "benchmark": {"latency": latency, "memory": memory},
                        "objective": objective,
                    }

                policy = self.orchestrate.resolve_budget(
                    "custom",
                    max_seconds=60,
                    max_rounds=1,
                    branches=1,
                    min_pairs=3,
                    max_pairs=3,
                    outer_candidates=1,
                    reserve_seconds=5,
                )
                terminal = self.orchestrate.evaluate_outer_candidate(
                    candidate,
                    mode="full",
                    workload_spec=workload_spec,
                    baseline=str(baseline),
                    policy=policy,
                    confidence=0.95,
                    candidate_root=iter_dir,
                    input_hash=input_hash,
                    iteration=1,
                    retries=0,
                    seed=0,
                    workload_runner=workload_runner,
                )
                self.assertEqual(terminal["status"], expected_terminal[name])
                self.assertIn("workload_paired_samples", terminal)

                bench = iter_dir / "bench.json"
                bench.write_text(
                    json.dumps(
                        {
                            "correctness": {"passed": True},
                            "kernel": {"average_ms": 1.0},
                            "reference": {"average_ms": 2.0},
                            "compiler_evidence": {"status": "not_recorded"},
                        }
                    ),
                    encoding="utf-8",
                )
                methods = iter_dir / "methods.json"
                methods.write_text('{"methods":[]}', encoding="utf-8")
                sass = iter_dir / "sass_check.json"
                sass.write_text('{"status":"passed","checks":[]}', encoding="utf-8")
                applied = self.orchestrate.apply_decision(
                    terminal,
                    run_dir=run_dir,
                    iteration=1,
                    state_path=state_path,
                    kernel=candidate_file,
                    bench=bench,
                    methods_json=methods,
                    sass_check=sass,
                    skip_validation=True,
                    runner=lambda command, **_kwargs: subprocess.run(
                        command, capture_output=True, text=True
                    ),
                )
                self.assertEqual(applied["returncode"], 0, applied["stderr"])
                updated = json.loads(state_path.read_text("utf-8"))
                self.assertEqual(
                    updated["terminal_decision"]["workload_status"],
                    "workload_failed" if name == "workload_failed" else "evaluated",
                )

                self.orchestrate.ArtifactStore(run_dir).write_checkpoint(
                    {
                        "schema_version": 2,
                        "input_hash": input_hash,
                        "run_dir": str(run_dir),
                        "iteration": 1,
                        "stage": "decision",
                        "stage_index": self.orchestrate.STAGES.index("decision"),
                        "status": "stage_complete",
                        "candidate_id": "b1",
                        "candidate_status": terminal["status"],
                        "budget": {"elapsed_seconds": 1.0, "remaining_seconds": 59.0},
                        "updated_at": 1.0,
                        "stage_evidence": {"decision": {"status": terminal["status"]}},
                    }
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.orchestrate.cmd_finalize(SimpleNamespace(run_dir=str(run_dir)))
                self.assertTrue((run_dir / "summary.md").is_file())

    def test_outer_deadline_marks_candidate_inconclusive_without_workload_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_file = Path(tmp) / "kernel.py"
            candidate_file.write_text("# candidate\n", encoding="utf-8")
            candidate = {
                "status": "confirmed_win",
                "candidate_file": str(candidate_file),
                "statistics": self.orchestrate_test_statistics(),
            }
            spec = self.orchestrate.WorkloadSpec(
                kind="command",
                source=["/bin/echo"],
                objective={
                    "primary_metric": {"name": "latency", "direction": "lower"},
                    "min_effect_pct": 0.5,
                    "constraints": [],
                },
                cases=(),
                source_hash="a" * 64,
            )
            policy = self.orchestrate.resolve_budget(
                "custom", max_seconds=20, max_rounds=1, branches=1,
                min_pairs=2, max_pairs=3, outer_candidates=1, reserve_seconds=5,
            )
            clock = self.orchestrate.BudgetClock(policy, started_at=100.0)
            evaluator = mock.Mock()
            decider = mock.Mock(side_effect=AssertionError("decision must not run"))
            with mock.patch.object(
                self.orchestrate.decision_engine, "decide", decider
            ):
                result = self.orchestrate.evaluate_outer_candidate(
                    candidate,
                    mode="full",
                    workload_spec=spec,
                    baseline="best.py",
                    policy=policy,
                    confidence=0.95,
                    evaluator=evaluator,
                    budget_clock=clock,
                    now=116.0,
                    estimated_seconds_per_pair=1.0,
                )

        evaluator.assert_not_called()
        decider.assert_not_called()
        self.assertEqual(result["candidate_status"], "inconclusive")
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["kernel_evidence"]["status"], "confirmed_win")

    def test_outer_blocks_are_capped_by_execution_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_file = Path(tmp) / "kernel.py"
            candidate_file.write_text("# candidate\n", encoding="utf-8")
            candidate = {
                "status": "confirmed_win", "candidate_file": str(candidate_file),
                "statistics": self.orchestrate_test_statistics(),
            }
            spec = self.orchestrate.WorkloadSpec(
                kind="command", source=["/bin/echo"],
                objective={
                    "primary_metric": {"name": "latency", "direction": "lower"},
                    "min_effect_pct": 0.5, "constraints": [],
                },
                cases=(), source_hash="a" * 64,
            )
            policy = self.orchestrate.resolve_budget(
                "custom", max_seconds=20, max_rounds=1, branches=1,
                min_pairs=2, max_pairs=3, outer_candidates=1, reserve_seconds=5,
            )
            clock = self.orchestrate.BudgetClock(policy, started_at=100.0)
            evaluator = mock.Mock(return_value={"status": "workload_failed"})
            self.orchestrate.evaluate_outer_candidate(
                candidate, mode="full", workload_spec=spec, baseline="best.py",
                policy=policy, confidence=0.95, evaluator=evaluator,
                budget_clock=clock, now=108.0, estimated_seconds_per_pair=3.0,
            )

        self.assertEqual(evaluator.call_args.kwargs["blocks"], 2)

    def test_outer_candidate_cannot_escape_the_iteration_before_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iter_dir = root / "run" / "iterv1"
            iter_dir.mkdir(parents=True)
            outside = root / "outside.py"
            outside.write_text("# outside\n", encoding="utf-8")
            candidate = {
                "status": "confirmed_win", "candidate_file": str(outside),
                "statistics": self.orchestrate_test_statistics(),
            }
            spec = self.orchestrate.WorkloadSpec(
                kind="command", source=["/bin/echo"],
                objective={
                    "primary_metric": {"name": "latency", "direction": "lower"},
                    "min_effect_pct": 0.5, "constraints": [],
                },
                cases=(), source_hash="a" * 64,
            )
            evaluator = mock.Mock()
            with self.assertRaisesRegex(ValueError, "iteration"):
                self.orchestrate.evaluate_outer_candidate(
                    candidate, mode="full", workload_spec=spec, baseline="best.py",
                    policy=self.orchestrate.resolve_budget("quick"), confidence=0.95,
                    evaluator=evaluator, candidate_root=iter_dir,
                )

        evaluator.assert_not_called()

    def test_apply_decision_persists_before_explicit_state_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            iter_dir = run_dir / "iterv1"
            iter_dir.mkdir(parents=True)
            kernel = iter_dir / "kernel.py"
            kernel.write_text("# candidate\n", encoding="utf-8")
            payload = {
                "status": "kernel_only_win",
                "candidate_file": str(kernel),
                "candidate_sha256": self.orchestrate.sha256_file(kernel),
                "statistics": self.orchestrate_test_statistics(),
            }

            def inspect(command, **_kwargs):
                decision_path = Path(command[command.index("--decision") + 1])
                self.assertTrue(decision_path.is_file())
                self.assertEqual(json.loads(decision_path.read_text("utf-8")), payload)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = self.orchestrate.apply_decision(
                payload,
                run_dir=run_dir,
                iteration=1,
                state_path=run_dir / "state.json",
                kernel=kernel,
                bench=iter_dir / "bench.json",
                methods_json=iter_dir / "methods.json",
                runner=mock.Mock(side_effect=inspect),
            )

        command = result["command"]
        self.assertIn("--decision", command)
        self.assertEqual(result["decision_path"], str((iter_dir / "decision.json").resolve()))

    def orchestrate_test_statistics(self) -> dict:
        return {
            "status": "confirmed_win",
            "statistic": "median_paired_improvement_pct",
            "direction": "lower",
            "min_effect_pct": 0.5,
            "confidence": 0.95,
            "estimate_pct": 3.0,
            "ci_low_pct": 1.0,
            "ci_high_pct": 4.0,
            "valid_pairs": 20,
            "invalid_pairs": 0,
            "improvements_pct": [3.0] * 20,
        }


if __name__ == "__main__":
    unittest.main()
