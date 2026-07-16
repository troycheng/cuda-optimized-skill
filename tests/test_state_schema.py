from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


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


class StateDecisionPromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _load_state()

    def _fixture(
        self,
        root: Path,
        *,
        mode: str,
        decision_status: object = "inconclusive",
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
            "workload": {"kind": "command"} if mode == "full" else None,
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
        statistics = {
            "statistic": "median_paired_improvement_pct",
            "estimate_pct": 3.0,
            "ci_low_pct": 2.0,
            "ci_high_pct": 4.0,
            "status": decision_status,
        }
        if write_decision:
            decision.write_text(
                json.dumps(
                    {
                        "status": decision_status,
                        "candidate_file": candidate_override or str(candidate),
                        "statistics": statistics,
                        "workload_statistics": {
                            **statistics,
                            "statistic": "median_paired_workload_improvement_pct",
                        },
                    }
                ),
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

        self.assertEqual(updated["best_file"], str(candidate.resolve()))
        self.assertTrue(output["improved"])
        self.assertEqual(output["status"], "kernel_only_win")
        self.assertEqual(updated["best_kernel_statistics"]["estimate_pct"], 3.0)

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
            "median_paired_workload_improvement_pct",
        )

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

    def test_confirmed_win_with_conflicting_statistics_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, state_path, _candidate, before = self._fixture(
                Path(tmp), mode="kernel-only", decision_status="confirmed_win"
            )
            decision_path = Path(tmp) / "run" / "iterv1" / "decision.json"
            decision = json.loads(decision_path.read_text("utf-8"))
            decision["statistics"]["status"] = "confirmed_loss"
            decision_path.write_text(json.dumps(decision), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "statistics.status.*decision"):
                self._run_update(args)
            unchanged = json.loads(state_path.read_text("utf-8"))

        self.assertEqual(unchanged["best_file"], before["best_file"])


if __name__ == "__main__":
    unittest.main()
