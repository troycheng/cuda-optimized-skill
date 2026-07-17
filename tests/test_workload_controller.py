from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "workload_controller.py"
)


def _load_controller():
    module_name = "cuda_optimizer_workload_controller_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workload controller: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _control(root: Path) -> dict:
    project = root / "project"
    return {
        "schema_version": "cuda-workload-optimizer/control-v1",
        "project_root": str(project),
        "workload_manifest": str(project / "workload.json"),
        "baseline_candidate": {"name": "baseline", "revision": "abc123"},
        "budget": "balanced",
        "mutation": {
            "project_paths": ["src", "configs/serve.json"],
            "environment_root": str(root / "environment-copy"),
            "host_policy": "recommend_only",
        },
        "probes": [
            {
                "id": "timeline",
                "kind": "timeline",
                "argv": ["python3", "collect_timeline.py"],
                "timeout_seconds": 30,
            }
        ],
        "reviewer": {
            "argv": ["reviewer-cli", "--json"],
            "timeout_seconds": 20,
        },
    }


def _change_set() -> dict:
    return {
        "schema_version": "cuda-workload-optimizer/change-v1",
        "id": "round-1-dataloader-workers",
        "hypothesis": "data wait dominates GPU idle time",
        "diagnosis_ids": ["cpu_data:data_wait"],
        "scope": "project",
        "candidate": {"name": "dataloader-workers-8", "revision": "worktree"},
        "paths": ["configs/serve.json"],
        "commands": [["python3", "-m", "unittest", "tests.test_config"]],
        "rollback": "restore_frozen_snapshot",
        "expected_metrics": ["data_wait_pct", "gpu_busy_pct", "p50_latency_ms"],
    }


class WorkloadControllerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _load_controller()

    def test_valid_control_and_change_set_are_detached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control = _control(root)
            change = _change_set()

            normalized = self.controller.validate_control_manifest(
                control, root / "control.json"
            )
            normalized_change = self.controller.validate_change_set(change, normalized)

            self.assertEqual(normalized, control)
            self.assertEqual(normalized_change, change)
            self.assertIsNot(normalized, control)
            self.assertIsNot(normalized_change, change)
            control["mutation"]["project_paths"].append("later")
            change["paths"].append("src/later.py")
            self.assertNotIn("later", normalized["mutation"]["project_paths"])
            self.assertNotIn("src/later.py", normalized_change["paths"])

    def test_unknown_keys_are_rejected_at_every_object_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cases = []
            root_unknown = _control(root)
            root_unknown["surprise"] = True
            cases.append(root_unknown)
            mutation_unknown = _control(root)
            mutation_unknown["mutation"]["sudo"] = True
            cases.append(mutation_unknown)
            probe_unknown = _control(root)
            probe_unknown["probes"][0]["shell"] = True
            cases.append(probe_unknown)
            reviewer_unknown = _control(root)
            reviewer_unknown["reviewer"]["callback"] = "run"
            cases.append(reviewer_unknown)

            for value in cases:
                with self.subTest(value=value), self.assertRaisesRegex(
                    self.controller.ValidationError, "unknown"
                ):
                    self.controller.validate_control_manifest(
                        value, root / "control.json"
                    )

            control = self.controller.validate_control_manifest(
                _control(root), root / "control.json"
            )
            change = _change_set()
            change["sudo"] = True
            with self.assertRaisesRegex(self.controller.ValidationError, "unknown"):
                self.controller.validate_change_set(change, control)

    def test_argv_must_be_a_nonempty_array_of_nonempty_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for argv in (
                "python3 collect.py",
                [],
                ["python3", ""],
                ["python3", 3],
            ):
                control = _control(root)
                control["probes"][0]["argv"] = argv
                with self.subTest(argv=argv), self.assertRaisesRegex(
                    self.controller.ValidationError, "argv"
                ):
                    self.controller.validate_control_manifest(
                        control, root / "control.json"
                    )

    def test_project_root_and_manifest_must_be_absolute_and_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            relative = _control(root)
            relative["project_root"] = "project"
            outside = _control(root)
            outside["workload_manifest"] = str(root / "other" / "workload.json")

            for value in (relative, outside):
                with self.subTest(value=value), self.assertRaisesRegex(
                    self.controller.ValidationError, "project_root|workload_manifest"
                ):
                    self.controller.validate_control_manifest(
                        value, root / "control.json"
                    )

    def test_mutation_paths_cannot_escape_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for path in ("../outside", "/tmp/outside", "."):
                control = _control(root)
                control["mutation"]["project_paths"] = [path]
                with self.subTest(path=path), self.assertRaisesRegex(
                    self.controller.ValidationError, "project_paths"
                ):
                    self.controller.validate_control_manifest(
                        control, root / "control.json"
                    )

    def test_environment_root_must_be_absolute_and_outside_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for path in ("environment", str(root / "project" / "env")):
                control = _control(root)
                control["mutation"]["environment_root"] = path
                with self.subTest(path=path), self.assertRaisesRegex(
                    self.controller.ValidationError, "environment_root"
                ):
                    self.controller.validate_control_manifest(
                        control, root / "control.json"
                    )

    def test_host_scope_and_host_mutation_policy_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            unsafe_control = _control(root)
            unsafe_control["mutation"]["host_policy"] = "allow"
            with self.assertRaisesRegex(
                self.controller.ValidationError, "recommend_only"
            ):
                self.controller.validate_control_manifest(
                    unsafe_control, root / "control.json"
                )

            control = self.controller.validate_control_manifest(
                _control(root), root / "control.json"
            )
            change = _change_set()
            change["scope"] = "host"
            with self.assertRaisesRegex(self.controller.ValidationError, "scope"):
                self.controller.validate_change_set(change, control)

    def test_change_set_paths_must_be_covered_by_declared_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control = self.controller.validate_control_manifest(
                _control(root), root / "control.json"
            )
            for path in ("README.md", "../outside", "/tmp/outside"):
                change = _change_set()
                change["paths"] = [path]
                with self.subTest(path=path), self.assertRaisesRegex(
                    self.controller.ValidationError, "paths"
                ):
                    self.controller.validate_change_set(change, control)

    def test_change_set_requires_candidate_and_argv_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control = self.controller.validate_control_manifest(
                _control(root), root / "control.json"
            )
            missing = _change_set()
            del missing["candidate"]
            with self.assertRaisesRegex(
                self.controller.ValidationError, "candidate"
            ):
                self.controller.validate_change_set(missing, control)

            shell_command = _change_set()
            shell_command["commands"] = ["python3 -m unittest"]
            with self.assertRaisesRegex(self.controller.ValidationError, "commands"):
                self.controller.validate_change_set(shell_command, control)

    def test_versions_budgets_timeouts_and_numbers_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutations = (
                ("schema_version", "v2"),
                ("budget", "unlimited"),
            )
            for key, value in mutations:
                control = _control(root)
                control[key] = value
                with self.subTest(key=key), self.assertRaises(
                    self.controller.ValidationError
                ):
                    self.controller.validate_control_manifest(
                        control, root / "control.json"
                    )

            for timeout in (0, -1, True, math.inf, 3601):
                control = _control(root)
                control["probes"][0]["timeout_seconds"] = timeout
                with self.subTest(timeout=timeout), self.assertRaisesRegex(
                    self.controller.ValidationError, "timeout_seconds"
                ):
                    self.controller.validate_control_manifest(
                        control, root / "control.json"
                    )

    def test_load_json_rejects_duplicate_keys_and_non_object_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(
                self.controller.ValidationError, "duplicate"
            ):
                self.controller.load_json_object(duplicate)

            array = root / "array.json"
            array.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(self.controller.ValidationError, "object"):
                self.controller.load_json_object(array)

    def test_validate_cli_returns_zero_for_valid_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control_path = root / "control.json"
            change_path = root / "change.json"
            control_path.write_text(json.dumps(_control(root)), encoding="utf-8")
            change_path.write_text(json.dumps(_change_set()), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "validate",
                    "--control",
                    str(control_path),
                    "--change-set",
                    str(change_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "valid")


class ProbeRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _load_controller()

    def _workspace(self, root: Path) -> tuple[dict, Path]:
        project = root / "project"
        project.mkdir()
        (project / "src").mkdir()
        (project / "configs").mkdir()
        (project / "workload.json").write_text("{}", encoding="utf-8")
        run_dir = root / "run"
        control = _control(root)
        return control, run_dir

    def _probe_script(self, project: Path, body: str) -> Path:
        script = project / "probe.py"
        script.write_text(
            "import json, os, pathlib, sys, time\n"
            + textwrap.dedent(body),
            encoding="utf-8",
        )
        return script

    def _configure(self, control: dict, script: Path, timeout: float = 10) -> dict:
        control["probes"] = [
            {
                "id": "timeline",
                "kind": "timeline",
                "argv": [sys.executable, str(script)],
                "timeout_seconds": timeout,
            }
        ]
        return self.controller.validate_control_manifest(control, script.parent / "c.json")

    def test_run_probe_captures_valid_output_and_required_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir = self._workspace(root)
            script = self._probe_script(
                root / "project",
                """
                output = pathlib.Path(os.environ["CUDA_OPTIMIZER_OUTPUT"])
                assert pathlib.Path(os.environ["CUDA_OPTIMIZER_RUN_DIR"]).is_absolute()
                assert os.environ["CUDA_OPTIMIZER_PROJECT_ROOT"] == str(pathlib.Path.cwd())
                output.write_text(json.dumps({
                    "schema_version": "cuda-workload-optimizer/probe-v1",
                    "probe_id": "timeline",
                    "kind": "timeline",
                    "status": "ok",
                    "metrics": {"gpu_busy_pct": 88, "kernel_time_pct": 72},
                    "issues": [],
                    "artifacts": []
                }))
                """,
            )
            normalized = self._configure(control, script)

            result = self.controller.run_probe(
                normalized["probes"][0], normalized, run_dir
            )

            self.assertEqual(result["status"], "ok")
            stored = json.loads(
                (run_dir / "probes" / "timeline.json").read_text("utf-8")
            )
            self.assertEqual(stored, result)
            execution = json.loads(
                (run_dir / "probes" / "timeline.execution.json").read_text("utf-8")
            )
            self.assertEqual(execution["exit_code"], 0)
            self.assertEqual(len(execution["argv_sha256"]), 64)

    def test_timeout_stops_process_group_and_returns_unavailable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir = self._workspace(root)
            script = self._probe_script(root / "project", "time.sleep(30)\n")
            normalized = self._configure(control, script, timeout=1)

            started = time.monotonic()
            result = self.controller.run_probe(
                normalized["probes"][0], normalized, run_dir
            )

            self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["issues"][0]["id"], "environment:probe-timeout")

    def test_nonzero_exit_and_malformed_output_become_failed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name, body, issue_id in (
                ("exit", "sys.exit(7)\n", "environment:probe-exit"),
                (
                    "json",
                    'pathlib.Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text("not-json")\n',
                    "environment:probe-output",
                ),
            ):
                with self.subTest(name=name):
                    case_root = root / name
                    case_root.mkdir()
                    control, run_dir = self._workspace(case_root)
                    script = self._probe_script(case_root / "project", body)
                    normalized = self._configure(control, script)
                    result = self.controller.run_probe(
                        normalized["probes"][0], normalized, run_dir
                    )
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["issues"][0]["id"], issue_id)

    def test_logs_are_bounded_and_secrets_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir = self._workspace(root)
            script = self._probe_script(
                root / "project",
                """
                print("x" * 2000)
                print("API_TOKEN=" + os.environ.get("API_TOKEN", "missing"), file=sys.stderr)
                pathlib.Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(json.dumps({
                    "schema_version": "cuda-workload-optimizer/probe-v1",
                    "probe_id": "timeline",
                    "kind": "timeline",
                    "status": "ok",
                    "metrics": {"gpu_busy_pct": 70},
                    "issues": [],
                    "artifacts": []
                }))
                """,
            )
            normalized = self._configure(control, script)
            original = os.environ.get("API_TOKEN")
            os.environ["API_TOKEN"] = "super-secret-value"
            try:
                result = self.controller.run_probe(
                    normalized["probes"][0],
                    normalized,
                    run_dir,
                    log_limit_bytes=128,
                )
            finally:
                if original is None:
                    os.environ.pop("API_TOKEN", None)
                else:
                    os.environ["API_TOKEN"] = original

            self.assertEqual(result["status"], "ok")
            execution = json.loads(
                (run_dir / "probes" / "timeline.execution.json").read_text("utf-8")
            )
            self.assertTrue(execution["stdout_truncated"])
            self.assertLessEqual(len(execution["stdout"]), 160)
            self.assertNotIn("super-secret-value", json.dumps(execution))

    def test_probe_and_diagnose_cli_write_machine_readable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir = self._workspace(root)
            script = self._probe_script(
                root / "project",
                """
                pathlib.Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(json.dumps({
                    "schema_version": "cuda-workload-optimizer/probe-v1",
                    "probe_id": "timeline",
                    "kind": "timeline",
                    "status": "ok",
                    "metrics": {"gpu_busy_pct": 90, "kernel_time_pct": 75},
                    "issues": [],
                    "artifacts": []
                }))
                """,
            )
            control = self._configure(control, script)
            control_path = root / "control.json"
            control_path.write_text(json.dumps(control), encoding="utf-8")

            probe_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "probe",
                    "--control",
                    str(control_path),
                    "--run-dir",
                    str(run_dir),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(probe_result.returncode, 0, probe_result.stderr)
            self.assertEqual(json.loads(probe_result.stdout)["probe_count"], 1)

            diagnosis_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "diagnose",
                    "--run-dir",
                    str(run_dir),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(diagnosis_result.returncode, 0, diagnosis_result.stderr)
            self.assertEqual(
                json.loads(diagnosis_result.stdout)["primary_category"], "kernel"
            )
            self.assertTrue((run_dir / "diagnosis.json").is_file())


if __name__ == "__main__":
    unittest.main()
