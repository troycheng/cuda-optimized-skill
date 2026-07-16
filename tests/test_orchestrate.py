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
            (iter_dir / "branch_results.json").write_text(
                json.dumps(
                    {
                        "iter": 1,
                        "status": "shortlist_ready",
                        "champion": {"branch_index": 1},
                    }
                ),
                encoding="utf-8",
            )
            (iter_dir / "kernel.py").write_text("# champion\n", encoding="utf-8")
            (iter_dir / "bench.json").write_text(
                json.dumps({"correctness": {"passed": True}}), encoding="utf-8"
            )
            branch_result = SimpleNamespace(
                returncode=0,
                stdout="CUDA setup log\n{not the authority}",
                stderr="",
            )
            runner = mock.Mock(
                side_effect=[branch_result, RuntimeError("profile reached")]
            )

            with mock.patch.object(orchestrate, "_run", runner):
                with self.assertRaisesRegex(RuntimeError, "profile reached"):
                    orchestrate.cmd_close_iter(args)

        self.assertEqual(runner.call_count, 2)
        self.assertTrue(
            any("profile_ncu.py" in str(part) for part in runner.call_args_list[1].args[0])
        )

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
