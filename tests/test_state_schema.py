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


if __name__ == "__main__":
    unittest.main()
