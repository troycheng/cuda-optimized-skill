from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIAGNOSIS_FIXTURE = (
    ROOT / "tests" / "fixtures" / "active_diagnosis" / "emit_global_scan.py"
)
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


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_pid_gone(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.02)
    return not _pid_exists(pid)


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


def _enable_v2_readiness(
    control: dict,
    root: Path,
    *,
    status: str = "ready",
    max_age_seconds: int = 300,
    capability_ids: tuple[str, ...] = ("gpu-execute",),
) -> dict:
    project = Path(control["project_root"])
    environment = Path(control["mutation"]["environment_root"])
    environment.mkdir(parents=True, exist_ok=True)
    emitter = project / "readiness_probe.py"
    emitter.write_text(
        "import json, os, sys\n"
        "payload = {\n"
        " 'schema_version': 'cuda-workload-optimizer/readiness-probe-v1',\n"
        " 'requirement_id': sys.argv[1],\n"
        f" 'status': {status!r}, 'observations': {{}}, 'artifacts': []}}\n"
        "open(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'], 'w').write(json.dumps(payload))\n",
        encoding="utf-8",
    )
    contract_path = project / "readiness.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "cuda-workload-optimizer/readiness-contract-v1",
                "requested_claim": "workload",
                "budget": {"max_seconds": 30, "max_repairs": 0},
                "requirements": [
                    {
                        "id": capability_id,
                        "necessity": "required",
                        "control_scope": "project",
                        "phase": "foundation",
                        "kind": (
                            "gpu_execute"
                            if capability_id == "gpu-execute"
                            else "nsys_trace"
                            if capability_id == "nsys.timeline"
                            else "ncu_counters"
                            if capability_id == "ncu.counter_access"
                            else "benchmark_noise"
                        ),
                        "max_age_seconds": max_age_seconds,
                        "probe": {
                            "argv": [sys.executable, str(emitter), capability_id],
                            "timeout_seconds": 5,
                        },
                        "remediation": {"mode": "none"},
                    }
                    for capability_id in capability_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    control["schema_version"] = "cuda-workload-optimizer/control-v2"
    control["readiness_contract"] = str(contract_path)
    return control


def _enable_active_diagnosis(control: dict, root: Path) -> dict:
    project = Path(control["project_root"])
    adapter = project / "active_diagnosis_scan.py"
    shutil.copyfile(ACTIVE_DIAGNOSIS_FIXTURE, adapter)
    evidence_adapter = project / "collect_framework_evidence.py"
    evidence_adapter.write_text(
        "import json, os\n"
        "request = json.load(open(os.environ['CUDA_OPTIMIZER_EVIDENCE_REQUEST']))\n"
        "payload = {\n"
        " 'schema_version': 'cuda-optimizer/evidence-result-v1',\n"
        " 'request_signature': request['request_signature'],\n"
        " 'status': 'observed', 'outcome_id': 'gap-present',\n"
        " 'observations': {'launch_gap_us': 12.5}, 'artifacts': []}\n"
        "open(os.environ['CUDA_OPTIMIZER_EVIDENCE_OUTPUT'], 'w').write(json.dumps(payload))\n",
        encoding="utf-8",
    )
    contract_path = project / "active-diagnosis.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "cuda-optimizer/active-diagnosis-contract-v1",
                "global_scan_probe_id": "timeline",
                "adapter_path": str(adapter),
                "analysis_policy_sha256": "a" * 64,
                "source": {
                    "profiler": "nsys",
                    "profiler_version": "2026.3",
                    "export_schema": "sqlite-v1",
                    "adapter_id": "fixture-adapter",
                    "adapter_version": "1.0.0",
                    "adapter_sha256": hashlib.sha256(
                        adapter.read_bytes()
                    ).hexdigest(),
                },
                "actions": [
                    {
                        "action_id": "pytorch-operator-trace",
                        "adapter_path": str(evidence_adapter),
                        "adapter_sha256": hashlib.sha256(
                            evidence_adapter.read_bytes()
                        ).hexdigest(),
                        "argv": [sys.executable, str(evidence_adapter)],
                        "timeout_seconds": 5,
                    }
                ],
                "selection_policy": {
                    "schema_version": "cuda-optimizer/evidence-selection-policy-v1",
                    "max_cost": "high",
                    "max_perturbation": "high",
                    "max_risk": "low",
                    "remaining_profile_actions": 2,
                    "available_capability_ids": [
                        "ncu.counter_access",
                        "nsys.timeline",
                        "pytorch.profiler",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    control["analysis_contract"] = str(contract_path)
    control["probes"][0]["argv"] = [sys.executable, str(adapter)]
    readiness_path = Path(control.get("readiness_contract", ""))
    if readiness_path.is_file():
        readiness = json.loads(readiness_path.read_text("utf-8"))
        if not any(
            item["id"] == "pytorch.profiler"
            for item in readiness["requirements"]
        ):
            readiness["requirements"].append(
                {
                    "id": "pytorch.profiler",
                    "necessity": "required",
                    "control_scope": "project",
                    "phase": "foundation",
                    "kind": "benchmark_noise",
                    "max_age_seconds": 300,
                    "probe": {
                        "argv": [
                            sys.executable,
                            str(Path(control["project_root"]) / "readiness_probe.py"),
                            "pytorch.profiler",
                        ],
                        "timeout_seconds": 5,
                    },
                    "remediation": {"mode": "none"},
                }
            )
            readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    return control


def _change_set() -> dict:
    return {
        "schema_version": "cuda-workload-optimizer/change-v1",
        "id": "round-1-dataloader-workers",
        "hypothesis": "data wait dominates GPU idle time",
        "diagnosis_ids": ["cpu_data:data_wait"],
        "scope": "project",
        "candidate": {"name": "dataloader-workers-8", "revision": "worktree"},
        "paths": ["configs/serve.json"],
        "commands": [],
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

    def test_v2_requires_contained_readiness_contract_but_v1_remains_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control = _control(root)
            self.assertEqual(
                self.controller.validate_control_manifest(control)["schema_version"],
                "cuda-workload-optimizer/control-v1",
            )

            control["schema_version"] = "cuda-workload-optimizer/control-v2"
            with self.assertRaisesRegex(ValueError, "readiness_contract"):
                self.controller.validate_control_manifest(control)

            control["readiness_contract"] = str(root / "outside.json")
            with self.assertRaisesRegex(ValueError, "readiness_contract.*project_root"):
                self.controller.validate_control_manifest(control)

            control["readiness_contract"] = str(
                root / "project" / "readiness.json"
            )
            normalized = self.controller.validate_control_manifest(control)
            self.assertEqual(
                normalized["schema_version"],
                "cuda-workload-optimizer/control-v2",
            )

    def test_active_diagnosis_requires_explicit_contained_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control = _control(root)
            (root / "project").mkdir()
            control["analysis_contract"] = str(
                root / "project" / "active-diagnosis.json"
            )
            with self.assertRaisesRegex(ValueError, "control-v1.*analysis_contract"):
                self.controller.validate_control_manifest(control)

            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            normalized = self.controller.validate_control_manifest(control)
            self.assertEqual(normalized["analysis_contract"], control["analysis_contract"])

            control["analysis_contract"] = str(root / "outside.json")
            with self.assertRaisesRegex(ValueError, "analysis_contract.*project_root"):
                self.controller.validate_control_manifest(control)

    def test_cli_new_run_rejects_v1_but_validate_keeps_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control_path = root / "control.json"
            control_path.write_text(json.dumps(_control(root)), "utf-8")
            validate = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "validate",
                    "--control",
                    str(control_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validate.returncode, 0, validate.stderr)

            run = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "run",
                    "--control",
                    str(control_path),
                    "--run-dir",
                    str(root / "run"),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 2)
            self.assertIn("require control-v2", run.stderr)

    def test_reject_only_gate_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control = _control(root)
            control["evaluation_gate"] = "reject_only"
            normalized = self.controller.validate_control_manifest(control)
            self.assertEqual(normalized["evaluation_gate"], "reject_only")

            control["evaluation_gate"] = "rank_only"
            with self.assertRaisesRegex(
                self.controller.ValidationError, "evaluation_gate"
            ):
                self.controller.validate_control_manifest(control)

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
                "/etc/cuda-optimized-skill-isolated",
                "/usr/local/cuda-optimized-skill-isolated",
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

            executable_command = _change_set()
            executable_command["commands"] = [[sys.executable, "-c", "pass"]]
            with self.assertRaisesRegex(
                self.controller.ValidationError, "commands.*empty"
            ):
                self.controller.validate_change_set(executable_command, control)

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

    def test_successful_probe_cleans_background_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir = self._workspace(root)
            pid_file = root / "child.pid"
            script = self._probe_script(
                root / "project",
                f"""
                import subprocess
                child = subprocess.Popen([
                    sys.executable, "-c", "import time; time.sleep(30)"
                ])
                pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
                pathlib.Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(json.dumps({{
                    "schema_version": "cuda-workload-optimizer/probe-v1",
                    "probe_id": "timeline",
                    "kind": "timeline",
                    "status": "ok",
                    "metrics": {{"gpu_busy_pct": 70}},
                    "issues": [],
                    "artifacts": []
                }}))
                """,
            )
            normalized = self._configure(control, script)

            result = self.controller.run_probe(
                normalized["probes"][0], normalized, run_dir
            )

            self.assertEqual(result["status"], "ok")
            child_pid = int(pid_file.read_text("utf-8"))
            try:
                self.assertTrue(_wait_pid_gone(child_pid))
            finally:
                if _pid_exists(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

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
            (run_dir / "candidate.diff").write_text(
                '+{"token": "do-not-send"}\n'
                '+headers = {"Authorization": "Bearer do-not-send"}\n',
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
            request = json.loads(
                (run_dir / "review_request.json").read_text("utf-8")
            )
            self.assertIn("withheld", request["redacted_diff"])
            self.assertNotIn("do-not-send", json.dumps(request))


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
                    return {
                        "valid": (
                            isinstance(candidate, dict)
                            and candidate.get("revision") != "invalid"
                        )
                    }

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

    def _active_proposal(self, run_dir: Path) -> tuple[dict, dict]:
        active = run_dir / "active_diagnosis"
        epoch = json.loads((active / "epoch.json").read_text("utf-8"))
        context = json.loads((run_dir / "diagnosis_context.json").read_text("utf-8"))
        hypothesis = {
            "schema_version": "cuda-optimizer/hypothesis-set-v1",
            "set_id": "hypotheses-0001",
            "epoch_id": epoch["epoch_id"],
            "epoch_sha256": context["epoch_sha256"],
            "execution_map_sha256": context["execution_map_sha256"],
            "hypotheses": [
                {
                    "hypothesis_id": "h-framework-gap",
                    "kind": "mechanism",
                    "scope_node_ids": ["cpu-launch", "gpu-kernel"],
                    "statement": "CPU launch serialization delays the GPU kernel.",
                    "mechanism": "framework_launch_overhead",
                    "disposition": "active",
                    "confidence": "plausible",
                    "support_evidence_ids": ["ev-global-scan"],
                    "oppose_evidence_ids": [],
                    "missing_evidence_kinds": ["framework_trace"],
                    "falsification_question": "Does the launch gap remain after a framework trace?",
                },
                {
                    "hypothesis_id": "h-kernel-bound",
                    "kind": "mechanism",
                    "scope_node_ids": ["gpu-kernel"],
                    "statement": "Kernel execution dominates the critical GPU lane.",
                    "mechanism": "kernel_execution",
                    "disposition": "active",
                    "confidence": "inconclusive",
                    "support_evidence_ids": [],
                    "oppose_evidence_ids": [],
                    "missing_evidence_kinds": ["ncu_kernel"],
                    "falsification_question": "Does kernel evidence show no dominant stall?",
                },
            ],
            "relationships": [
                {
                    "relation": "exclusive",
                    "left": "h-framework-gap",
                    "right": "h-kernel-bound",
                }
            ],
        }
        hypothesis_result = self.controller._load_hypothesis_space_module().validate_hypothesis_set(
            hypothesis,
            epoch=epoch,
            execution_map=json.loads((active / "execution_map.json").read_text("utf-8")),
            evidence_catalog=json.loads((active / "evidence_catalog.json").read_text("utf-8")),
        )
        request = {
            "schema_version": "cuda-optimizer/evidence-request-set-v1",
            "request_set_id": "requests-0001",
            "epoch_id": epoch["epoch_id"],
            "epoch_sha256": context["epoch_sha256"],
            "hypothesis_set_sha256": hypothesis_result["hypothesis_set_sha256"],
            "requests": [
                {
                    "request_id": "req-framework",
                    "action_id": "pytorch-operator-trace",
                    "question": "Is launch serialization the cause rather than kernel execution?",
                    "target_hypothesis_ids": ["h-framework-gap", "h-kernel-bound"],
                    "exclusive_pairs": [
                        {"left": "h-framework-gap", "right": "h-kernel-bound"}
                    ],
                    "outcomes": [
                        {
                            "outcome_id": "gap-present",
                            "supports": ["h-framework-gap"],
                            "opposes": ["h-kernel-bound"],
                        },
                        {
                            "outcome_id": "kernel-dominant",
                            "supports": ["h-kernel-bound"],
                            "opposes": ["h-framework-gap"],
                        },
                    ],
                }
            ],
        }
        return hypothesis, request

    def test_v2_blocked_readiness_never_measures_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            _enable_v2_readiness(control, root, status="failed")
            blocked = {
                "schema_version": "cuda-workload-optimizer/readiness-report-v1",
                "status": "blocked",
                "can_start_diagnosis": False,
                "contract_digest": "a" * 64,
                "environment_identity_digest": "b" * 64,
            }

            with mock.patch.object(
                self.controller,
                "_run_readiness_gate",
                return_value=blocked,
                create=True,
            ) as readiness, mock.patch.object(
                self.controller, "_load_evaluate_module"
            ) as evaluate:
                state = self.controller.start_run(control, run_dir)

            self.assertEqual(state["stage"], "readiness")
            self.assertEqual(state["next_action"], "readiness_action")
            self.assertEqual(state["completed_stages"], [])
            readiness.assert_called_once()
            evaluate.assert_not_called()
            self.assertFalse((run_dir / "baseline" / "observation.json").exists())
            resumed = self.controller.resume_run(run_dir)
            self.assertEqual(resumed, state)
            readiness.assert_called_once()

    def test_v2_ready_readiness_precedes_baseline_and_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)

            state = self.controller.start_run(control, run_dir)

            self.assertEqual(state["next_action"], "register_change")
            self.assertEqual(
                state["completed_stages"],
                ["readiness", "baseline", "probes", "diagnosis"],
            )
            self.assertEqual(len(state["readiness_contract_digest"]), 64)
            self.assertEqual(len(state["readiness_report_digest"]), 64)
            self.assertTrue((run_dir / "readiness" / "report.json").is_file())
            self.assertTrue(
                (run_dir / "readiness" / "report.complete.json").is_file()
            )
            self.assertTrue((run_dir / "baseline" / "observation.json").is_file())

    def test_active_diagnosis_builds_hash_bound_context_and_waits_for_ai(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)

            state = self.controller.start_run(control, run_dir)

            self.assertEqual(state["stage"], "active_diagnosis")
            self.assertEqual(state["next_action"], "propose_hypotheses")
            self.assertEqual(
                state["completed_stages"],
                ["readiness", "baseline", "probes", "diagnosis", "diagnosis_context"],
            )
            context = json.loads((run_dir / "diagnosis_context.json").read_text("utf-8"))
            self.assertEqual(
                context["schema_version"],
                "cuda-optimizer/diagnosis-context-v1",
            )
            for field in (
                "epoch_sha256",
                "execution_map_sha256",
                "evidence_catalog_sha256",
                "action_catalog_sha256",
                "selection_policy_sha256",
            ):
                self.assertEqual(len(context[field]), 64)
            self.assertTrue((run_dir / "active_diagnosis" / "ledger" / "000001-context.json").is_file())

    def test_active_diagnosis_rejects_adapter_digest_mismatch_before_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            contract_path = Path(control["analysis_contract"])
            contract = json.loads(contract_path.read_text("utf-8"))
            contract["source"]["adapter_sha256"] = "0" * 64
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "adapter.*digest"):
                self.controller.start_run(control, run_dir)

            self.assertFalse((run_dir / "baseline" / "observation.json").exists())

    def test_active_diagnosis_resume_does_not_repeat_global_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            first = self.controller.start_run(control, run_dir)
            scan = run_dir / "active_diagnosis" / "global_scan.json"
            scan_mtime = scan.stat().st_mtime_ns
            Path(control["analysis_contract"]).unlink()

            second = self.controller.resume_run(run_dir)

            self.assertEqual(second, first)
            self.assertEqual(scan.stat().st_mtime_ns, scan_mtime)

    def test_active_diagnosis_proposal_is_replayed_and_chained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            self.controller.start_run(control, run_dir)
            hypothesis, request = self._active_proposal(run_dir)

            state = self.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )

            self.assertEqual(state["next_action"], "collect_evidence")
            selection = json.loads(
                (run_dir / "active_diagnosis" / "evidence_selection.json").read_text("utf-8")
            )
            self.assertEqual(selection["status"], "selected")
            self.assertEqual(selection["selected_request"]["request_id"], "req-framework")
            self.assertTrue(
                (run_dir / "active_diagnosis" / "ledger" / "000002-proposal.json").is_file()
            )

    def test_active_diagnosis_collects_evidence_and_returns_to_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            self.controller.start_run(control, run_dir)
            hypothesis, request = self._active_proposal(run_dir)
            self.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )

            state = self.controller.collect_active_diagnosis_evidence(control, run_dir)

            self.assertEqual(state["next_action"], "propose_hypotheses")
            catalog = json.loads(
                (run_dir / "active_diagnosis" / "evidence_catalog.json").read_text("utf-8")
            )
            self.assertEqual(len(catalog), 2)
            evidence_id = next(key for key in catalog if key != "ev-global-scan")
            self.assertEqual(catalog[evidence_id]["kind"], "framework_trace")
            history = json.loads(
                (run_dir / "active_diagnosis" / "request_history.json").read_text("utf-8")
            )
            self.assertEqual(history, [state["last_request_signature"]])
            policy = json.loads(
                (run_dir / "active_diagnosis" / "selection_policy.json").read_text("utf-8")
            )
            self.assertEqual(policy["remaining_profile_actions"], 1)

    def test_resume_collects_once_and_never_reexecutes_completed_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            contract = json.loads(Path(control["analysis_contract"]).read_text("utf-8"))
            adapter = Path(contract["actions"][0]["adapter_path"])
            counter = project / "evidence-count.txt"
            adapter.write_text(
                adapter.read_text("utf-8")
                + f"\nopen({str(counter)!r}, 'a').write('1\\n')\n",
                encoding="utf-8",
            )
            contract["actions"][0]["adapter_sha256"] = hashlib.sha256(
                adapter.read_bytes()
            ).hexdigest()
            Path(control["analysis_contract"]).write_text(json.dumps(contract), encoding="utf-8")
            self.controller.start_run(control, run_dir)
            hypothesis, request = self._active_proposal(run_dir)
            self.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )

            first = self.controller.resume_run(run_dir)
            second = self.controller.resume_run(run_dir)

            self.assertEqual(first["next_action"], "propose_hypotheses")
            self.assertEqual(second, first)
            self.assertEqual(counter.read_text("utf-8").splitlines(), ["1"])

    def test_resume_never_reexecutes_an_interrupted_evidence_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            self.controller.start_run(control, run_dir)
            hypothesis, request = self._active_proposal(run_dir)
            state = self.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            signature = state["selected_request_signature"]
            attempt = run_dir / "active_diagnosis" / "evidence" / signature
            attempt.mkdir(parents=True)
            (attempt / "intent.json").write_text("{}", encoding="utf-8")

            recovered = self.controller.resume_run(run_dir)

            self.assertEqual(recovered["next_action"], "manual_recovery")
            self.assertEqual(
                recovered["manual_recovery_reason"],
                "evidence_action_interrupted_not_reexecuted",
            )
            self.assertFalse((attempt / "execution.json").exists())

    def test_equivalent_request_history_survives_the_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            self.controller.start_run(control, run_dir)
            hypothesis, request = self._active_proposal(run_dir)
            self.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            self.controller.collect_active_diagnosis_evidence(control, run_dir)
            hypothesis, request = self._active_proposal(run_dir)
            request["request_set_id"] = "requests-0002"
            request["requests"][0]["request_id"] = "req-framework-renamed"

            state = self.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )

            self.assertEqual(state["next_action"], "evidence_gap")
            selection = json.loads(
                (run_dir / "active_diagnosis" / "evidence_selection.json").read_text("utf-8")
            )
            self.assertEqual(
                selection["rejections"][0]["reason"],
                "equivalent_request_already_attempted",
            )

    def test_readiness_report_overrides_claimed_available_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_active_diagnosis(control, root)
            _enable_v2_readiness(control, root, capability_ids=("gpu-execute",))
            self.controller.start_run(control, run_dir)
            policy = json.loads(
                (run_dir / "active_diagnosis" / "selection_policy.json").read_text("utf-8")
            )
            self.assertEqual(policy["available_capability_ids"], ["gpu-execute"])
            hypothesis, request = self._active_proposal(run_dir)
            state = self.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            self.assertEqual(state["next_action"], "evidence_gap")

    def test_active_diagnosis_rejects_stale_epoch_and_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            _enable_v2_readiness(control, root)
            _enable_active_diagnosis(control, root)
            self.controller.start_run(control, run_dir)
            hypothesis, request = self._active_proposal(run_dir)
            request["epoch_id"] = "epoch-stale"
            with self.assertRaisesRegex(ValueError, "epoch"):
                self.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )

            hypothesis, request = self._active_proposal(run_dir)
            (project / "configs" / "value.json").write_text(
                '{"workers": 99}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                self.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )

    def test_v2_project_drift_during_readiness_blocks_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            _enable_v2_readiness(control, root)
            emitter = project / "readiness_probe.py"
            emitter.write_text(
                emitter.read_text("utf-8")
                + f"\nopen({str(project / 'configs' / 'value.json')!r}, 'w').write('{{}}')\n",
                "utf-8",
            )

            with self.assertRaisesRegex(ValueError, "drifted during readiness"):
                self.controller.start_run(control, run_dir)
            self.assertFalse((run_dir / "baseline" / "observation.json").exists())

    def test_v2_workload_source_drift_during_readiness_blocks_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            _enable_v2_readiness(control, root)
            emitter = project / "readiness_probe.py"
            emitter.write_text(
                emitter.read_text("utf-8")
                + f"\nopen({str(project / 'adapter.py')!r}, 'a').write('\\n# drift\\n')\n",
                "utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "workload identity drifted during readiness"
            ):
                self.controller.start_run(control, run_dir)
            self.assertFalse((run_dir / "baseline" / "observation.json").exists())

    def test_v2_resume_does_not_rerun_fresh_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root)
            evaluator = mock.Mock()
            evaluator.measure_candidate.side_effect = RuntimeError("stop at baseline")
            with mock.patch.object(
                self.controller, "_load_evaluate_module", return_value=evaluator
            ), self.assertRaisesRegex(RuntimeError, "stop at baseline"):
                self.controller.start_run(control, run_dir)
            attempts = list((run_dir / "readiness" / "attempts").iterdir())

            with mock.patch.object(
                self.controller, "_load_evaluate_module", return_value=evaluator
            ), self.assertRaisesRegex(RuntimeError, "stop at baseline"):
                self.controller.resume_run(run_dir)

            self.assertEqual(
                sorted(path.name for path in attempts),
                sorted(
                    path.name
                    for path in (run_dir / "readiness" / "attempts").iterdir()
                ),
            )

    def test_v2_report_and_marker_tampering_block_resume_before_baseline(self) -> None:
        for target in ("report", "marker"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                control, run_dir, _project = self._workspace(root)
                _enable_v2_readiness(control, root)
                evaluator = mock.Mock()
                evaluator.measure_candidate.side_effect = RuntimeError(
                    "stop at baseline"
                )
                with mock.patch.object(
                    self.controller,
                    "_load_evaluate_module",
                    return_value=evaluator,
                ), self.assertRaisesRegex(RuntimeError, "stop at baseline"):
                    self.controller.start_run(control, run_dir)

                path = run_dir / "readiness" / (
                    "report.json" if target == "report" else "report.complete.json"
                )
                payload = json.loads(path.read_text("utf-8"))
                payload["tampered"] = True
                path.write_text(json.dumps(payload), "utf-8")
                with self.assertRaisesRegex(
                    ValueError, "readiness report|marker|digest"
                ):
                    self.controller.resume_run(run_dir)

    def test_v2_expired_readiness_is_refreshed_before_high_cost_probes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            _enable_v2_readiness(control, root, max_age_seconds=1)
            evaluator = mock.Mock()

            def slow_baseline(*args, **kwargs):
                time.sleep(1.05)
                return {"status": "measured"}

            evaluator.measure_candidate.side_effect = slow_baseline
            with mock.patch.object(
                self.controller, "_load_evaluate_module", return_value=evaluator
            ):
                state = self.controller.start_run(control, run_dir)

            self.assertEqual(state["next_action"], "register_change")
            attempts = list((run_dir / "readiness" / "attempts").iterdir())
            self.assertEqual(len(attempts), 2)

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

    def test_reject_only_positive_screen_is_rolled_back_and_never_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            control["evaluation_gate"] = "reject_only"
            self.controller.start_run(control, run_dir)
            self.controller.register_change(control, run_dir, self._change())
            config = project / "configs" / "value.json"
            original = config.read_text("utf-8")
            config.write_text('{"workers": 8}\n', encoding="utf-8")

            decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "rejected")
            self.assertEqual(
                decision["reason"], "reject_only_stage_cannot_promote"
            )
            self.assertEqual(decision["primary_status"], "confirmed_win")
            self.assertEqual(config.read_text("utf-8"), original)

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

    def test_workload_validation_failure_rejects_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            change = self._change(revision="invalid")
            self.controller.register_change(control, run_dir, change)
            config = project / "configs" / "value.json"
            original = config.read_text("utf-8")
            config.write_text('{"workers": 12}\n', encoding="utf-8")

            decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "rejected")
            self.assertEqual(decision["reason"], "workload_failed")
            self.assertEqual(config.read_text("utf-8"), original)

    def test_correctness_timeout_bounds_output_and_stops_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            pid_file = root / "child.pid"
            command = [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    import signal
                    import subprocess
                    import sys
                    import time
                    from pathlib import Path

                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    child = subprocess.Popen([
                        sys.executable,
                        "-c",
                        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
                    ])
                    Path({str(pid_file)!r}).write_text(str(child.pid))
                    print("x" * 1000000, flush=True)
                    time.sleep(30)
                    """
                ),
            ]
            change = self._change(commands=[command])

            started = time.monotonic()
            result = self.controller._run_correctness_commands(
                control, change, run_dir, timeout_seconds=0.2
            )

            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["commands"][0]["failure"], "TimeoutExpired")
            self.assertLessEqual(
                len(result["commands"][0]["stdout"].encode("utf-8")),
                64 * 1024 + len("...[truncated]"),
            )
            self.assertTrue(pid_file.exists())
            child_pid = int(pid_file.read_text("utf-8"))
            try:
                self.assertTrue(_wait_pid_gone(child_pid))
            finally:
                if _pid_exists(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    def test_successful_correctness_command_cleans_background_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            pid_file = root / "child.pid"
            command = [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    import subprocess
                    import sys
                    from pathlib import Path
                    child = subprocess.Popen([
                        sys.executable, "-c", "import time; time.sleep(30)"
                    ])
                    Path({str(pid_file)!r}).write_text(str(child.pid))
                    """
                ),
            ]

            result = self.controller._run_correctness_commands(
                control, self._change(commands=[command]), run_dir
            )

            self.assertEqual(result["status"], "passed")
            child_pid = int(pid_file.read_text("utf-8"))
            try:
                self.assertTrue(_wait_pid_gone(child_pid))
            finally:
                if _pid_exists(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

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

    def test_isolated_environment_must_exist_and_remain_frozen_from_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for case in ("created_late", "drifted"):
                with self.subTest(case=case):
                    case_root = root / case
                    case_root.mkdir()
                    control, run_dir, _project = self._workspace(case_root)
                    environment = Path(control["mutation"]["environment_root"])
                    if case == "drifted":
                        environment.mkdir()
                        (environment / "requirements.lock").write_text(
                            "triton==3.3.0\n", encoding="utf-8"
                        )
                    self.controller.start_run(control, run_dir)
                    if case == "created_late":
                        environment.mkdir()
                    (environment / "requirements.lock").write_text(
                        "triton==3.4.0\n", encoding="utf-8"
                    )
                    change = self._change(scope="isolated_environment")
                    change["paths"] = ["requirements.lock"]

                    with self.assertRaisesRegex(
                        self.controller.ValidationError,
                        "exist before baseline|drifted after baseline",
                    ):
                        self.controller.register_change(control, run_dir, change)

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

    def test_registered_change_set_and_before_identity_are_digest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for artifact in ("change_set.json", "rounds/round-1/before_identity.json"):
                with self.subTest(artifact=artifact):
                    case_root = root / artifact.replace("/", "-")
                    case_root.mkdir()
                    control, run_dir, project = self._workspace(case_root)
                    self.controller.start_run(control, run_dir)
                    self.controller.register_change(control, run_dir, self._change())
                    config = project / "configs" / "value.json"
                    original = config.read_text("utf-8")
                    config.write_text('{"workers": 8}\n', encoding="utf-8")
                    target = run_dir / artifact
                    payload = json.loads(target.read_text("utf-8"))
                    if artifact == "change_set.json":
                        payload["candidate"]["revision"] = "slow"
                    else:
                        payload["digest"] = "0" * 64
                    target.write_text(json.dumps(payload), encoding="utf-8")

                    decision = self.controller.evaluate_change(run_dir)

                    self.assertEqual(decision["status"], "rejected")
                    self.assertEqual(decision["reason"], "frozen_artifact_drift")
                    self.assertEqual(config.read_text("utf-8"), original)

    def test_change_registration_resumes_after_snapshot_commit_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            change = self._change()
            original_atomic = self.controller._atomic_json
            interrupted = False

            def interrupt_once(path, value):
                nonlocal interrupted
                if Path(path).name == "change_set.json" and not interrupted:
                    interrupted = True
                    raise OSError("simulated interruption")
                return original_atomic(path, value)

            with mock.patch.object(
                self.controller, "_atomic_json", side_effect=interrupt_once
            ):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    self.controller.register_change(control, run_dir, change)

            self.assertTrue((run_dir / "snapshot" / "project").is_dir())
            self.assertEqual(
                self.controller.read_run_state(run_dir)["next_action"],
                "register_change",
            )

            resumed = self.controller.register_change(control, run_dir, change)
            repeated = self.controller.register_change(control, run_dir, change)

            self.assertEqual(resumed["next_action"], "edit_then_evaluate")
            self.assertEqual(repeated, resumed)
            self.assertFalse((run_dir / "registration_pending.json").exists())

    def test_state_commit_recovers_interrupted_checkpoint_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            change = self._change()
            original_atomic = self.controller._atomic_json

            def interrupt_checkpoint(path, value):
                if (
                    Path(path).name == "checkpoint.json"
                    and value.get("next_action") == "edit_then_evaluate"
                ):
                    raise OSError("checkpoint write interrupted")
                return original_atomic(path, value)

            with mock.patch.object(
                self.controller, "_atomic_json", side_effect=interrupt_checkpoint
            ):
                with self.assertRaisesRegex(OSError, "checkpoint write interrupted"):
                    self.controller.register_change(control, run_dir, change)

            recovered = self.controller.read_run_state(run_dir)
            self.assertEqual(recovered["next_action"], "edit_then_evaluate")
            self.assertEqual(
                json.loads((run_dir / "checkpoint.json").read_text("utf-8")),
                recovered,
            )
            self.assertEqual(
                self.controller.register_change(control, run_dir, change), recovered
            )

    def test_frozen_control_drift_fails_and_checkpoint_mirror_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, _project = self._workspace(root)
            expected = self.controller.start_run(control, run_dir)
            checkpoint = run_dir / "checkpoint.json"
            damaged = copy.deepcopy(expected)
            damaged["next_action"] = "done"
            checkpoint.write_text(json.dumps(damaged), encoding="utf-8")

            self.assertEqual(self.controller.read_run_state(run_dir), expected)
            self.assertEqual(json.loads(checkpoint.read_text("utf-8")), expected)

            frozen = run_dir / "control_manifest.json"
            payload = json.loads(frozen.read_text("utf-8"))
            payload["budget"] = "thorough"
            frozen.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                self.controller.ValidationError, "frozen control"
            ):
                self.controller.resume_run(run_dir)

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

    def test_deadline_expiring_during_evaluation_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            self.controller.register_change(control, run_dir, self._change())
            config = project / "configs" / "value.json"
            original = config.read_text("utf-8")
            config.write_text('{"workers": 8}\n', encoding="utf-8")
            state = self.controller.read_run_state(run_dir)
            state["deadline_epoch"] = time.time() + 0.05
            self.controller._write_state(run_dir, state)
            evaluator = self.controller._load_evaluate_module()
            original_evaluate = evaluator.evaluate_pairs

            def delayed_evaluate(*args, **kwargs):
                time.sleep(0.1)
                return original_evaluate(*args, **kwargs)

            with mock.patch.object(evaluator, "evaluate_pairs", delayed_evaluate):
                decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "rejected")
            self.assertEqual(decision["reason"], "budget_expired")
            self.assertEqual(config.read_text("utf-8"), original)

    def test_rollback_failure_requires_manual_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            self.controller.register_change(
                control, run_dir, self._change(revision="slow")
            )
            (project / "configs" / "value.json").write_text(
                '{"workers": 8}\n', encoding="utf-8"
            )

            with mock.patch.object(
                self.controller, "_restore_snapshot", side_effect=OSError("disk full")
            ):
                decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "manual_recovery_required")
            self.assertFalse(decision["rolled_back"])
            state = self.controller.read_run_state(run_dir)
            self.assertEqual(state["status"], "manual_recovery_required")
            self.assertEqual(state["next_action"], "manual_recovery")

    def test_missing_snapshot_never_deletes_project_during_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, run_dir, project = self._workspace(root)
            self.controller.start_run(control, run_dir)
            self.controller.register_change(
                control, run_dir, self._change(revision="slow")
            )
            config = project / "configs" / "value.json"
            config.write_text('{"workers": 8}\n', encoding="utf-8")
            shutil.rmtree(run_dir / "snapshot" / "project")

            decision = self.controller.evaluate_change(run_dir)

            self.assertEqual(decision["status"], "manual_recovery_required")
            self.assertFalse(decision["rolled_back"])
            self.assertTrue(config.is_file())
            self.assertTrue((project / "src").is_dir())

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
