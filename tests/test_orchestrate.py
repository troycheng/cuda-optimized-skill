from __future__ import annotations

import contextlib
import importlib.util
import io
import json
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


if __name__ == "__main__":
    unittest.main()
