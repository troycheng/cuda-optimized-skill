from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import subprocess
import tempfile
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

        self.assertEqual(workload_pairs.call_count, 1)
        self.assertEqual(workload_pairs.call_args.args[1], str(baseline))
        self.assertEqual(workload_pairs.call_args.args[2], str(confirmed.resolve()))
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(apply.call_args.args[0]["status"], "end_to_end_win")

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
        self.assertEqual(runner.call_count, 2)
        state_command = runner.call_args_list[1].args[0]
        self.assertIn("--output-root", state_command)
        self.assertIn("--budget-json", state_command)
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
        self.assertEqual(state["iterations_total"], 4)
        self.assertEqual(state["branches"], 8)
        self.assertTrue(env_in_run)
        self.assertFalse(env_at_source)

    def test_output_root_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.orchestrate.validate_output_root(alias, baseline=root / "kernel.py")


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
            "ci_low_pct": 1.0,
            "ci_high_pct": 4.0,
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
            with mock.patch.object(
                self.orchestrate,
                "_run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
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

            with mock.patch.object(
                self.orchestrate,
                "_run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_finalize(SimpleNamespace(run_dir=str(run_dir)))
            complete = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            self.assertEqual(
                self.orchestrate.resume(
                    complete, input_hash=complete["input_hash"]
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
            branch_payload = {
                "status": "shortlist_ready",
                "selected_kernel": str(selected),
                "champion": {
                    "status": "confirmed_win",
                    "kernel": str(selected),
                    "branch_index": 1,
                    "statistics": statistics,
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
                        "statistics": statistics,
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
            state["started_at"] = 100.0
            state_path.write_text(json.dumps(state), encoding="utf-8")
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
                    mock.patch.object(self.orchestrate.time, "time", return_value=10_000.0), \
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
            "stage": "baseline",
            "stage_index": 0,
            "status": "ready",
            "candidate_id": None,
            "candidate_status": None,
            "budget": {"elapsed_seconds": 0.0, "remaining_seconds": 20.0},
            "updated_at": 100.0,
        }

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
            root = Path(tmp)
            checkpoint = self._checkpoint(root)
            checkpoint.update(
                stage="decision",
                stage_index=self.orchestrate.STAGES.index("decision"),
                status="stage_complete",
            )
            self.orchestrate.ArtifactStore(root).write_checkpoint(checkpoint)
            args = SimpleNamespace(run_dir=str(root))
            with mock.patch.object(
                self.orchestrate,
                "_run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
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
        self.assertEqual(result["candidate_status"], "inconclusive")
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(result["status"], "kernel_only_win")

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
