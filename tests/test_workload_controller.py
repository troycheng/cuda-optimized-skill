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
            for path in (
                "environment",
                str(root / "project" / "env"),
                str(root),
                "/",
                "/etc",
            ):
                control = _control(root)
                control["mutation"]["environment_root"] = path
                with self.subTest(path=path), self.assertRaisesRegex(
                    self.controller.ValidationError, "environment_root"
                ):
                    self.controller.validate_control_manifest(
                        control, root / "control.json"
                    )

    def test_mutation_roots_must_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control = _control(root)
            control["mutation"]["project_paths"] = ["src", "src/generated"]
            with self.assertRaisesRegex(
                self.controller.ValidationError, "overlap"
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


class ReviewerControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _load_controller()

    def test_review_cli_records_skipped_when_no_reviewer_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            project = root / "project"
            project.mkdir()
            (project / "configs").mkdir()
            (project / "src").mkdir()
            (project / "workload.json").write_text("{}", encoding="utf-8")
            control = _control(root)
            del control["reviewer"]
            control_path = root / "control.json"
            control_path.write_text(json.dumps(control), encoding="utf-8")
            change_path = root / "change.json"
            change_path.write_text(json.dumps(_change_set()), encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "diagnosis.json").write_text(
                json.dumps(
                    {
                        "schema_version": "cuda-workload-optimizer/diagnosis-v1",
                        "primary_category": "cpu_data",
                        "confidence": "medium",
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "review",
                    "--control",
                    str(control_path),
                    "--run-dir",
                    str(run_dir),
                    "--change-set",
                    str(change_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "skipped")
            artifact = json.loads((run_dir / "review.json").read_text("utf-8"))
            self.assertEqual(artifact["status"], "skipped")
            self.assertIsNone(artifact["response"])


class WorkloadRoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _load_controller()

    def _workspace(self, root: Path) -> tuple[dict, Path, Path]:
        project = root / "project"
        project.mkdir()
        (project / "configs").mkdir()
        (project / "src").mkdir()
        (project / "configs" / "value.json").write_text(
            '{"workers": 4}\n', encoding="utf-8"
        )
        adapter = project / "adapter.py"
        adapter.write_text(
            textwrap.dedent(
                """
                def prepare(candidate):
                    return None

                def validate(candidate):
                    return {"valid": isinstance(candidate, dict) and "revision" in candidate}

                def benchmark(candidate):
                    revision = candidate["revision"]
                    latency = {
                        "baseline": 100.0,
                        "optimized": 80.0,
                        "slow": 120.0,
                        "constraint_bad": 80.0,
                    }[revision]
                    memory = 120.0 if revision == "constraint_bad" else 100.0
                    return {"p50_latency_ms": latency, "memory_mb": memory}

                def metrics():
                    return {
                        "primary_metric": {"name": "p50_latency_ms", "direction": "lower"},
                        "min_effect_pct": 1.0,
                        "constraints": [
                            {"name": "memory_mb", "max_regression_pct": 5.0}
                        ],
                    }

                def cleanup():
                    return None
                """
            ),
            encoding="utf-8",
        )
        manifest = project / "workload.json"
        manifest.write_text(
            json.dumps(
                {
                    "kind": "python",
                    "source": "adapter.py",
                    "objective": {
                        "primary_metric": {
                            "name": "p50_latency_ms",
                            "direction": "lower",
                        },
                        "min_effect_pct": 1.0,
                        "constraints": [
                            {"name": "memory_mb", "max_regression_pct": 5.0}
                        ],
                    },
                    "cases": [],
                }
            ),
            encoding="utf-8",
        )
        probe = project / "probe.py"
        probe.write_text(
            textwrap.dedent(
                """
                import json
                import os
                from pathlib import Path

                Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(json.dumps({
                    "schema_version": "cuda-workload-optimizer/probe-v1",
                    "probe_id": "timeline",
                    "kind": "timeline",
                    "status": "ok",
                    "metrics": {
                        "gpu_busy_pct": 42,
                        "cpu_busy_pct": 91,
                        "data_wait_pct": 45
                    },
                    "issues": [],
                    "artifacts": []
                }))
                """
            ),
            encoding="utf-8",
        )
        control = _control(root)
        control["budget"] = "fast"
        control["baseline_candidate"] = {"name": "baseline", "revision": "baseline"}
        control["mutation"]["project_paths"] = ["configs", "src"]
        control["probes"] = [
            {
                "id": "timeline",
                "kind": "timeline",
                "argv": [sys.executable, str(probe)],
                "timeout_seconds": 10,
            }
        ]
        del control["reviewer"]
        return control, root / "run", project

    def _change(self, revision: str = "optimized", **overrides) -> dict:
        change = _change_set()
        change["candidate"] = {"name": revision, "revision": revision}
        change["paths"] = ["configs/value.json"]
        change["commands"] = []
        change.update(overrides)
        return change

    def test_start_run_freezes_inputs_and_reaches_change_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)

            state = self.controller.start_run(control, run_dir)

            self.assertEqual(state["stage"], "change")
            self.assertEqual(state["next_action"], "register_change")
            self.assertEqual(
                state["completed_stages"], ["baseline", "probes", "diagnosis"]
            )
            self.assertEqual(len(state["control_digest"]), 64)
            self.assertEqual(len(state["workload_source_hash"]), 64)
            self.assertTrue((run_dir / "control_manifest.json").is_file())
            self.assertTrue((run_dir / "baseline" / "observation.json").is_file())
            self.assertTrue((run_dir / "diagnosis.json").is_file())
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            self.assertEqual(checkpoint, state)

    def test_confirmed_workload_win_is_promoted_and_keeps_project_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            self.controller.register_change(control, run_dir, self._change())
            config = project / "configs" / "value.json"
            config.write_text('{"workers": 8}\n', encoding="utf-8")

            decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "promoted")
            self.assertEqual(decision["primary_status"], "confirmed_win")
            self.assertEqual(config.read_text("utf-8"), '{"workers": 8}\n')
            state = self.controller.read_run_state(run_dir)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["next_action"], "done")
            self.assertTrue((run_dir / "evaluation.json").is_file())

    def test_loss_and_constraint_failure_restore_the_frozen_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for revision, expected_primary in (
                ("slow", "confirmed_loss"),
                ("constraint_bad", "confirmed_win"),
            ):
                with self.subTest(revision=revision):
                    case_root = root / revision
                    case_root.mkdir()
                    control, run_dir, project = self._workspace(case_root)
                    self.controller.start_run(control, run_dir)
                    self.controller.register_change(
                        control, run_dir, self._change(revision)
                    )
                    config = project / "configs" / "value.json"
                    original = config.read_text("utf-8")
                    config.write_text('{"workers": 99}\n', encoding="utf-8")

                    decision = self.controller.evaluate_change(run_dir)

                    self.assertEqual(decision["status"], "rejected")
                    self.assertEqual(decision["primary_status"], expected_primary)
                    self.assertEqual(config.read_text("utf-8"), original)

    def test_diff_outside_change_set_is_rejected_before_workload_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            self.controller.register_change(control, run_dir, self._change())
            outside = project / "src" / "unexpected.py"
            outside.write_text("unsafe = True\n", encoding="utf-8")

            with self.assertRaisesRegex(
                self.controller.ValidationError, "outside ChangeSet"
            ):
                self.controller.evaluate_change(run_dir)

            self.assertFalse(outside.exists())
            self.assertFalse((run_dir / "evaluation.json").exists())

    def test_correctness_command_failure_rejects_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            change = self._change(
                commands=[[sys.executable, "-c", "raise SystemExit(3)"]]
            )
            self.controller.register_change(control, run_dir, change)
            config = project / "configs" / "value.json"
            original = config.read_text("utf-8")
            config.write_text('{"workers": 12}\n', encoding="utf-8")

            decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "rejected")
            self.assertEqual(decision["reason"], "correctness_failed")
            self.assertEqual(config.read_text("utf-8"), original)

    def test_resume_does_not_repeat_completed_stages_or_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            first = self.controller.start_run(control, run_dir)
            diagnosis_mtime = (run_dir / "diagnosis.json").stat().st_mtime_ns
            second = self.controller.resume_run(run_dir)
            self.assertEqual(second, first)
            self.assertEqual(
                (run_dir / "diagnosis.json").stat().st_mtime_ns, diagnosis_mtime
            )

            self.controller.register_change(control, run_dir, self._change())
            (project / "configs" / "value.json").write_text(
                '{"workers": 8}\n', encoding="utf-8"
            )
            first_decision = self.controller.evaluate_change(run_dir)
            evaluation_mtime = (run_dir / "evaluation.json").stat().st_mtime_ns
            second_decision = self.controller.evaluate_change(run_dir)
            self.assertEqual(second_decision, first_decision)
            self.assertEqual(
                (run_dir / "evaluation.json").stat().st_mtime_ns,
                evaluation_mtime,
            )

    def test_cli_exposes_the_resumable_state_machine_commands(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("run", "status", "register-change", "evaluate", "resume"):
            self.assertIn(command, result.stdout)

    def test_isolated_environment_change_is_evaluated_without_host_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            environment = Path(control["mutation"]["environment_root"])
            environment.mkdir()
            lock = environment / "requirements.lock"
            lock.write_text("triton==3.3.0\n", encoding="utf-8")
            self.controller.start_run(control, run_dir)
            change = self._change(scope="isolated_environment")
            change["paths"] = ["requirements.lock"]
            self.controller.register_change(control, run_dir, change)
            lock.write_text("triton==3.4.0\n", encoding="utf-8")

            decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "promoted")
            self.assertEqual(lock.read_text("utf-8"), "triton==3.4.0\n")
            self.assertIn(
                "No host mutation was executed",
                (run_dir / "host_recommendations.md").read_text("utf-8"),
            )

    def test_deadline_is_enforced_before_change_registration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            state = self.controller.read_run_state(run_dir)
            state["deadline_epoch"] = time.time() - 1
            self.controller._write_state(run_dir, state)

            with self.assertRaisesRegex(self.controller.ValidationError, "deadline"):
                self.controller.register_change(control, run_dir, self._change())
            self.assertFalse((run_dir / "snapshot").exists())

    def test_workload_identity_drift_is_rejected_before_candidate_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            self.controller.register_change(control, run_dir, self._change())
            config = project / "configs" / "value.json"
            original = config.read_text("utf-8")
            config.write_text('{"workers": 8}\n', encoding="utf-8")
            adapter = project / "adapter.py"
            adapter.write_text(adapter.read_text("utf-8") + "\n# drift\n", "utf-8")

            with self.assertRaisesRegex(
                self.controller.ValidationError, "workload identity drifted"
            ):
                self.controller.evaluate_change(run_dir)

            self.assertEqual(config.read_text("utf-8"), original)
            self.assertFalse((run_dir / "evaluation.json").exists())

    def test_expired_budget_after_edit_rejects_and_restores_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            self.controller.register_change(control, run_dir, self._change())
            config = project / "configs" / "value.json"
            original = config.read_text("utf-8")
            config.write_text('{"workers": 8}\n', encoding="utf-8")
            state = self.controller.read_run_state(run_dir)
            state["deadline_epoch"] = time.time() - 1
            self.controller._write_state(run_dir, state)

            decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "rejected")
            self.assertEqual(decision["reason"], "budget_expired")
            self.assertEqual(config.read_text("utf-8"), original)

    def test_snapshot_rejects_symlinked_mutation_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            outside = root / "outside"
            outside.mkdir()
            shutil_target = project / "configs"
            for child in shutil_target.iterdir():
                child.unlink()
            shutil_target.rmdir()
            try:
                shutil_target.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(
                self.controller.ValidationError, "symlink|escapes project_root"
            ):
                self.controller.register_change(control, run_dir, self._change())


if __name__ == "__main__":
    unittest.main()
