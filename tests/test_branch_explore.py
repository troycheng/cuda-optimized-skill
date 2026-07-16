from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
BRANCH_EXPLORE_PATH = SCRIPTS / "branch_explore.py"
RUN_ITERATION_PATH = SCRIPTS / "run_iteration.py"
STATE_PATH = SCRIPTS / "state.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_branch_explore():
    return _load(BRANCH_EXPLORE_PATH, "cuda_optimizer_branch_explore")


def _load_run_iteration():
    return _load(RUN_ITERATION_PATH, "cuda_optimizer_run_iteration")


def _bench(passed: bool = True, average: float = 1.0) -> dict:
    return {
        "correctness": {"passed": passed},
        "kernel": {
            "average_ms": average,
            "median_ms": average,
            "p95_ms": average,
            "cv_pct": 1.0,
        },
    }


def _statistics(status: str, estimate: float | None) -> dict:
    return {
        "statistic": "median_paired_improvement_pct",
        "estimate_pct": estimate,
        "ci_low_pct": None if estimate is None else estimate - 0.25,
        "ci_high_pct": None if estimate is None else estimate + 0.25,
        "status": status,
    }


class BranchExploreTests(unittest.TestCase):
    def _state(self, root: Path, branches: int = 2) -> tuple[Path, dict]:
        run_dir = root / "run"
        baseline = root / "best.py"
        baseline.write_text("# best\n", encoding="utf-8")
        ref = root / "reference.py"
        ref.write_text("# ref\n", encoding="utf-8")
        for branch in range(1, branches + 1):
            branch_dir = run_dir / "iterv1" / "branches" / f"b{branch}"
            branch_dir.mkdir(parents=True)
            (branch_dir / "kernel.py").write_text(
                f"# branch {branch}\n", encoding="utf-8"
            )
        payload = {
            "run_dir": str(run_dir),
            "ref_file": str(ref),
            "best_file": str(baseline),
            "dims": {"n": 128, "nested": {"tile": 16}},
            "ptr_size": 0,
            "branches": branches,
            "backend": "triton",
            "env": {
                "primary_sm_arch": "sm_120",
                "nvcc": {"path": "/opt/cuda/bin/nvcc"},
            },
            "seed": 17,
            "budget": {"max_pairs": 24},
            "min_effect_pct": 0.5,
        }
        state_path = root / "state.json"
        state_path.write_text(json.dumps(payload), encoding="utf-8")
        return state_path, payload

    def test_only_confirmed_win_enters_shortlist_and_writes_decision(self) -> None:
        branch_explore = _load_branch_explore()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path, _payload = self._state(root)
            with mock.patch.object(
                branch_explore, "_bench_kernel", side_effect=[_bench(), _bench()]
            ), mock.patch.object(
                branch_explore,
                "_paired_candidate",
                side_effect=[
                    _statistics("inconclusive", 0.2),
                    _statistics("confirmed_win", 3.0),
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    output = branch_explore.run(str(state_path), iteration=1)

            self.assertEqual(output["status"], "shortlist_ready")
            self.assertEqual([item["branch_index"] for item in output["shortlist"]], [2])
            self.assertEqual(output["champion"], output["shortlist"][0])
            self.assertEqual(output["champion"]["status"], "confirmed_win")
            promoted = root / "run" / "iterv1" / "kernel.py"
            self.assertEqual(promoted.read_text("utf-8"), "# branch 2\n")
            decision = json.loads(
                (root / "run" / "iterv1" / "decision.json").read_text("utf-8")
            )
            self.assertEqual(decision["status"], "confirmed_win")
            self.assertEqual(Path(decision["candidate_file"]), promoted)
            self.assertEqual(decision["statistics"], output["champion"]["statistics"])

    def test_no_confirmed_win_keeps_existing_best_and_does_not_copy_candidate(self) -> None:
        branch_explore = _load_branch_explore()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path, payload = self._state(root)
            before_state = state_path.read_bytes()
            before_best = Path(payload["best_file"]).read_bytes()
            with mock.patch.object(
                branch_explore, "_bench_kernel", side_effect=[_bench(), _bench()]
            ), mock.patch.object(
                branch_explore,
                "_paired_candidate",
                side_effect=[
                    _statistics("inconclusive", 0.1),
                    _statistics("confirmed_loss", -2.0),
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    output = branch_explore.run(str(state_path), iteration=1)

            self.assertEqual(output["status"], "no_confirmed_kernel_win")
            self.assertIsNone(output["champion"])
            self.assertEqual(output["shortlist"], [])
            self.assertFalse((root / "run" / "iterv1" / "kernel.py").exists())
            self.assertEqual(Path(payload["best_file"]).read_bytes(), before_best)
            self.assertEqual(state_path.read_bytes(), before_state)
            decision_path = root / "run" / "iterv1" / "decision.json"
            state_module = _load(STATE_PATH, "cuda_optimizer_state_consumer")
            decision, status, statistics, workload_statistics = (
                state_module._load_decision(
                    str(decision_path), candidate_file=None
                )
            )
            self.assertEqual(status, "no_confirmed_kernel_win")
            self.assertIsNone(decision["candidate_file"])
            self.assertIsNone(statistics)
            self.assertIsNone(workload_statistics)
            self.assertEqual(
                state_module._promotion_for(status, "kernel-only"),
                ("no_confirmed_kernel_win", False),
            )

    def test_confirmed_winners_are_stably_sorted_by_estimate(self) -> None:
        branch_explore = _load_branch_explore()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path, _payload = self._state(root, branches=5)
            stats = [
                _statistics("confirmed_win", 2.0),
                _statistics("confirmed_loss", -3.0),
                _statistics("confirmed_win", 4.0),
                _statistics("confirmed_win", 2.0),
                _statistics("inconclusive", 8.0),
            ]
            with mock.patch.object(
                branch_explore, "_bench_kernel", side_effect=[_bench()] * 5
            ), mock.patch.object(branch_explore, "_paired_candidate", side_effect=stats):
                with contextlib.redirect_stdout(io.StringIO()):
                    output = branch_explore.run(str(state_path), iteration=1)

        self.assertEqual(
            [item["branch_index"] for item in output["shortlist"]], [3, 1, 4]
        )

    def test_failed_correctness_and_malformed_statistics_are_safely_excluded(self) -> None:
        branch_explore = _load_branch_explore()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path, _payload = self._state(root, branches=3)
            paired = mock.Mock(
                side_effect=[
                    {"status": "confirmed_win", "estimate_pct": True},
                    _statistics("invalid", 99.0),
                ]
            )
            with mock.patch.object(
                branch_explore,
                "_bench_kernel",
                side_effect=[_bench(False), _bench(), _bench()],
            ), mock.patch.object(branch_explore, "_paired_candidate", paired):
                with contextlib.redirect_stdout(io.StringIO()):
                    output = branch_explore.run(str(state_path), iteration=1)

        self.assertEqual(output["status"], "no_confirmed_kernel_win")
        self.assertEqual(paired.call_count, 2)
        by_branch = {item["branch_index"]: item for item in output["branches"]}
        self.assertEqual(by_branch[1]["correctness"], "failed")
        self.assertEqual(by_branch[2]["status"], "invalid")
        self.assertIn("invalid_statistics", by_branch[2]["error"])
        self.assertNotIn(1, [item["branch_index"] for item in output["shortlist"]])

    def test_paired_seam_receives_frozen_state_inputs_and_budget(self) -> None:
        branch_explore = _load_branch_explore()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path, payload = self._state(root)
            original = copy.deepcopy(payload)
            paired = mock.Mock(side_effect=[
                _statistics("inconclusive", None),
                _statistics("inconclusive", None),
            ])
            with mock.patch.object(
                branch_explore, "_bench_kernel", side_effect=[_bench(), _bench()]
            ), mock.patch.object(branch_explore, "_paired_candidate", paired):
                with contextlib.redirect_stdout(io.StringIO()):
                    branch_explore.run(str(state_path), iteration=1, warmup=3, repeat=7)

            args, kwargs = paired.call_args_list[0]
            self.assertEqual(args[0], payload["best_file"])
            self.assertTrue(args[1].endswith("branches/b1/kernel.py"))
            self.assertEqual(kwargs["backend"], "triton")
            self.assertEqual(kwargs["dims"], payload["dims"])
            self.assertEqual(kwargs["ptr_size"], 0)
            self.assertEqual(kwargs["arch"], "sm_120")
            self.assertEqual(kwargs["nvcc_bin"], "/opt/cuda/bin/nvcc")
            self.assertEqual(kwargs["seed"], 17)
            self.assertEqual(kwargs["blocks"], 24)
            self.assertEqual(kwargs["warmup"], 3)
            self.assertEqual(kwargs["min_effect_pct"], 0.5)
            self.assertEqual(payload, original)
            self.assertEqual(json.loads(state_path.read_text("utf-8")), original)

    def test_cli_treats_completed_but_unconfirmed_comparison_as_normal(self) -> None:
        branch_explore = _load_branch_explore()
        argv = ["branch_explore.py", "--state", "state.json", "--iter", "1"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            branch_explore,
            "run",
            return_value={
                "status": "no_confirmed_kernel_win",
                "valid_branches": 2,
                "completed_comparisons": 2,
            },
        ):
            branch_explore.main()

    def test_cli_keeps_nonzero_exit_when_all_branches_fail(self) -> None:
        branch_explore = _load_branch_explore()
        argv = ["branch_explore.py", "--state", "state.json", "--iter", "1"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            branch_explore,
            "run",
            return_value={
                "status": "no_confirmed_kernel_win",
                "valid_branches": 0,
                "completed_comparisons": 0,
            },
        ), self.assertRaises(SystemExit) as caught:
            branch_explore.main()

        self.assertEqual(caught.exception.code, 2)


class RunIterationDecisionSummaryTests(unittest.TestCase):
    def test_decision_summary_requires_terminal_status_and_candidate_path(self) -> None:
        run_iteration = _load_run_iteration()
        statistics = _statistics("confirmed_win", 3.0)
        invalid = (
            ({"candidate_file": "kernel.py", "statistics": statistics}, "status"),
            (
                {
                    "status": True,
                    "candidate_file": "kernel.py",
                    "statistics": statistics,
                },
                "status",
            ),
            (
                {
                    "status": "surprise_win",
                    "candidate_file": "kernel.py",
                    "statistics": statistics,
                },
                "status",
            ),
            ({"status": "confirmed_win", "statistics": statistics}, "candidate_file"),
        )
        for decision, field in invalid:
            with self.subTest(field=field, decision=decision), self.assertRaisesRegex(
                ValueError, field
            ):
                run_iteration._decision_summary(decision)

    def test_decision_summary_rejects_unknown_or_non_string_evidence_status(self) -> None:
        run_iteration = _load_run_iteration()
        for status in (True, 1, "surprise_win"):
            with self.subTest(status=status), self.assertRaisesRegex(
                ValueError, "statistics.status"
            ):
                run_iteration._decision_summary(
                    {
                        "status": "confirmed_win",
                        "candidate_file": "kernel.py",
                        "statistics": _statistics(status, 3.0),
                    }
                )

    def test_benchmark_summary_uses_unified_decision_statistics(self) -> None:
        run_iteration = _load_run_iteration()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            iter_dir = run_dir / "iterv1"
            iter_dir.mkdir(parents=True)
            kernel = iter_dir / "kernel.py"
            kernel.write_text("# kernel\n", encoding="utf-8")
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps({"run_dir": str(run_dir), "ref_file": str(root / "ref.py")}),
                encoding="utf-8",
            )
            stats = _statistics("confirmed_win", 3.0)
            (iter_dir / "decision.json").write_text(
                json.dumps(
                    {
                        "status": "confirmed_win",
                        "candidate_file": str(kernel),
                        "statistics": stats,
                    }
                ),
                encoding="utf-8",
            )
            bench_payload = _bench(average=0.5)

            def fake_bench(**kwargs):
                Path(kwargs["json_out"]).write_text(
                    json.dumps(bench_payload), encoding="utf-8"
                )
                return 0

            args = type(
                "Args",
                (),
                {
                    "state": str(state_path),
                    "iter": 1,
                    "benchmark": str(ROOT / "fake_benchmark.py"),
                    "warmup": 1,
                    "repeat": 2,
                },
            )()
            with mock.patch.object(run_iteration, "_run_bench", side_effect=fake_bench):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    run_iteration.cmd_benchmark(args)

            summary = json.loads(stdout.getvalue())
            expected = {
                "statistic": "median_paired_improvement_pct",
                "estimate_pct": 3.0,
                "ci_low_pct": 2.75,
                "ci_high_pct": 3.25,
                "status": "confirmed_win",
            }
            self.assertEqual(
                {field: summary[field] for field in expected}, expected
            )
            self.assertEqual(summary["decision"], expected)
            self.assertNotIn("improved", summary)

    def test_candidate_benchmark_requires_decision_before_running(self) -> None:
        run_iteration = _load_run_iteration()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            iter_dir = run_dir / "iterv1"
            iter_dir.mkdir(parents=True)
            (iter_dir / "kernel.py").write_text("# kernel\n", encoding="utf-8")
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps({"run_dir": str(run_dir), "ref_file": str(root / "ref.py")}),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "state": str(state_path),
                    "iter": 1,
                    "benchmark": str(ROOT / "fake_benchmark.py"),
                    "warmup": 1,
                    "repeat": 2,
                },
            )()
            bench = mock.Mock()
            with mock.patch.object(run_iteration, "_run_bench", bench):
                with self.assertRaisesRegex(ValueError, "decision.json.*missing"):
                    run_iteration.cmd_benchmark(args)

            bench.assert_not_called()

    def test_candidate_benchmark_rejects_malformed_decision_before_running(self) -> None:
        run_iteration = _load_run_iteration()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            iter_dir = run_dir / "iterv1"
            iter_dir.mkdir(parents=True)
            (iter_dir / "kernel.py").write_text("# kernel\n", encoding="utf-8")
            (iter_dir / "decision.json").write_text("{", encoding="utf-8")
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps({"run_dir": str(run_dir), "ref_file": str(root / "ref.py")}),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "state": str(state_path),
                    "iter": 1,
                    "benchmark": str(ROOT / "fake_benchmark.py"),
                    "warmup": 1,
                    "repeat": 2,
                },
            )()
            bench = mock.Mock()
            with mock.patch.object(run_iteration, "_run_bench", bench):
                with self.assertRaisesRegex(ValueError, "decision.json.*malformed"):
                    run_iteration.cmd_benchmark(args)

            bench.assert_not_called()


if __name__ == "__main__":
    unittest.main()
