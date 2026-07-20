import importlib.util
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "readiness_probe.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cuda_readiness_probe", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _requirement(requirement_id: str, argv: list[str], timeout: float = 2) -> dict:
    return {
        "id": requirement_id,
        "necessity": "required",
        "control_scope": "project",
        "phase": "foundation",
        "kind": "gpu_execute",
        "max_age_seconds": 300,
        "probe": {"argv": argv, "timeout_seconds": timeout},
        "remediation": {"mode": "none"},
    }


def _emit_script(path: Path, payload: dict, *, stdout: str = "") -> None:
    path.write_text(
        "import json, os\n"
        f"print({stdout!r})\n"
        f"payload = {payload!r}\n"
        "with open(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'], 'w', encoding='utf-8') as f:\n"
        "    json.dump(payload, f)\n",
        "utf-8",
    )


def _valid_probe(requirement_id: str = "gpu-execute") -> dict:
    return {
        "schema_version": "cuda-workload-optimizer/readiness-probe-v1",
        "requirement_id": requirement_id,
        "status": "ready",
        "observations": {"device_count": 1, "sm_arch": "sm_120"},
        "artifacts": [],
    }


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_pid_gone(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.02)
    return not _pid_exists(pid)


class ReadinessProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_validate_probe_is_detached_and_rejects_open_or_mismatched_data(self) -> None:
        value = _valid_probe()
        validated = self.module.validate_probe(value, "gpu-execute")
        value["observations"]["device_count"] = 9
        self.assertEqual(validated["observations"]["device_count"], 1)

        for mutate, message in (
            (lambda item: item.__setitem__("unknown", True), "unknown"),
            (
                lambda item: item.__setitem__("requirement_id", "other"),
                "requirement_id",
            ),
            (lambda item: item.__setitem__("status", "passed"), "status"),
            (
                lambda item: item["observations"].__setitem__("value", float("nan")),
                "finite",
            ),
        ):
            with self.subTest(message=message):
                changed = _valid_probe()
                mutate(changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.module.validate_probe(changed, "gpu-execute")

    def test_valid_probe_publishes_execution_and_marker_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            script = project / "emit.py"
            _emit_script(script, _valid_probe(), stdout="probe complete")

            result = self.module.run_requirement(
                _requirement("gpu-execute", [sys.executable, str(script)]),
                run_dir=root / "run",
                project_root=project,
                environment_identity_digest="a" * 64,
                deadline_epoch=time.time() + 5,
            )

            probes = root / "run" / "readiness" / "probes"
            self.assertEqual(result["status"], "ready")
            self.assertTrue((probes / "gpu-execute.json").is_file())
            execution = json.loads(
                (probes / "gpu-execute.execution.json").read_text("utf-8")
            )
            marker = json.loads(
                (probes / "gpu-execute.complete.json").read_text("utf-8")
            )
            self.assertEqual(execution["environment_identity_digest"], "a" * 64)
            self.assertEqual(execution["uid"], os.getuid())
            self.assertEqual(execution["resolved_executable"], sys.executable)
            self.assertEqual(len(execution["executable_sha256"]), 64)
            self.assertIn("tool_version", execution)
            self.assertIn("visible_devices", execution)
            self.assertIn("gpu_identity", execution)
            self.assertIn("permission_state", execution)
            self.assertEqual(marker["requirement_id"], "gpu-execute")
            self.assertEqual(marker["probe_sha256"], self.module._sha256_file(
                probes / "gpu-execute.json"
            ))
            with self.assertRaisesRegex(FileExistsError, "complete|exists"):
                self.module.run_requirement(
                    _requirement("gpu-execute", [sys.executable, str(script)]),
                    run_dir=root / "run",
                    project_root=project,
                    environment_identity_digest="a" * 64,
                    deadline_epoch=time.time() + 5,
                )

    def test_exit_zero_without_probe_output_is_never_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            result = self.module.run_requirement(
                _requirement("empty", ["/usr/bin/true"]),
                run_dir=root / "run",
                project_root=project,
                environment_identity_digest="b" * 64,
                deadline_epoch=time.time() + 5,
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["observations"]["reason"], "probe_output_missing")

    def test_version_query_cannot_preseed_formal_probe_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            executable = project / "preseed.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "if '--version' in sys.argv:\n"
                "    output = os.environ.get('CUDA_OPTIMIZER_READINESS_OUTPUT')\n"
                "    if output:\n"
                f"        open(output, 'w').write(json.dumps({_valid_probe('preseed')!r}))\n"
                "    print('preseed 1.0')\n"
                "raise SystemExit(0)\n",
                "utf-8",
            )
            executable.chmod(0o755)
            result = self.module.run_requirement(
                _requirement("preseed", [str(executable)]),
                run_dir=root / "run",
                project_root=project,
                environment_identity_digest="3" * 64,
                deadline_epoch=time.time() + 5,
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["observations"]["reason"], "probe_output_missing")

    def test_version_query_uses_safe_environment_and_redacts_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            executable = project / "version-env.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "if '--version' in sys.argv:\n"
                "    print('version=' + os.environ.get('READINESS_TEST_SECRET', 'absent'))\n"
                "    raise SystemExit(0)\n"
                f"payload = {_valid_probe('version-env')!r}\n"
                "open(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'], 'w').write(json.dumps(payload))\n",
                "utf-8",
            )
            executable.chmod(0o755)
            old = os.environ.get("READINESS_TEST_SECRET")
            os.environ["READINESS_TEST_SECRET"] = "must-not-reach-version-query"
            try:
                self.module.run_requirement(
                    _requirement("version-env", [str(executable)]),
                    run_dir=root / "run",
                    project_root=project,
                    environment_identity_digest="7" * 64,
                    deadline_epoch=time.time() + 5,
                )
            finally:
                if old is None:
                    os.environ.pop("READINESS_TEST_SECRET", None)
                else:
                    os.environ["READINESS_TEST_SECRET"] = old
            execution_text = (
                root
                / "run"
                / "readiness"
                / "probes"
                / "version-env.execution.json"
            ).read_text("utf-8")

        self.assertNotIn("must-not-reach-version-query", execution_text)
        self.assertIn('"tool_version": "version=absent"', execution_text)

    def test_missing_command_nonzero_and_mismatched_output_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            cases = []
            cases.append(("missing", [str(project / "does-not-exist")], "unavailable"))

            nonzero = project / "nonzero.py"
            nonzero.write_text("raise SystemExit(7)\n", "utf-8")
            cases.append(("nonzero", [sys.executable, str(nonzero)], "failed"))

            mismatch = project / "mismatch.py"
            _emit_script(mismatch, _valid_probe("someone-else"))
            cases.append(("mismatch", [sys.executable, str(mismatch)], "failed"))

            for index, (name, argv, expected) in enumerate(cases):
                with self.subTest(name=name):
                    result = self.module.run_requirement(
                        _requirement(name, argv),
                        run_dir=root / f"run-{index}",
                        project_root=project,
                        environment_identity_digest="c" * 64,
                        deadline_epoch=time.time() + 5,
                    )
                    self.assertEqual(result["status"], expected)

    def test_timeout_kills_descendants_and_returns_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            child_pid_path = project / "child.pid"
            script = project / "hang.py"
            script.write_text(
                "import os, signal, subprocess, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
                f"open({str(child_pid_path)!r}, 'w').write(str(child.pid))\n"
                "time.sleep(60)\n",
                "utf-8",
            )
            result = self.module.run_requirement(
                _requirement("slow", [sys.executable, str(script)], timeout=0.2),
                run_dir=root / "run",
                project_root=project,
                environment_identity_digest="d" * 64,
                deadline_epoch=time.time() + 1,
            )
            child_pid = int(child_pid_path.read_text("utf-8"))
            self.assertEqual(result["status"], "unavailable")
            self.assertTrue(_wait_pid_gone(child_pid))

    def test_output_limit_duplicate_keys_and_parent_replacement_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()

            oversized = project / "oversized.py"
            oversized.write_text(
                "import os\n"
                "open(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'], 'wb').write(b'x' * (1024*1024+1))\n",
                "utf-8",
            )
            result = self.module.run_requirement(
                _requirement("oversized", [sys.executable, str(oversized)]),
                run_dir=root / "run-large",
                project_root=project,
                environment_identity_digest="e" * 64,
                deadline_epoch=time.time() + 5,
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["observations"]["reason"], "probe_output_too_large")

            duplicate = project / "duplicate.py"
            duplicate.write_text(
                "import os\n"
                "open(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'], 'w').write("
                "'{\"schema_version\":\"cuda-workload-optimizer/readiness-probe-v1\",'"
                "'\"requirement_id\":\"duplicate\",\"status\":\"ready\",'"
                "'\"status\":\"failed\",\"observations\":{},\"artifacts\":[]}')\n",
                "utf-8",
            )
            result = self.module.run_requirement(
                _requirement("duplicate", [sys.executable, str(duplicate)]),
                run_dir=root / "run-duplicate",
                project_root=project,
                environment_identity_digest="f" * 64,
                deadline_epoch=time.time() + 5,
            )
            self.assertEqual(result["status"], "failed")
            self.assertIn("duplicate", result["observations"]["reason"])

            replace = project / "replace.py"
            run_dir = root / "run-replace"
            probes = run_dir / "readiness" / "probes"
            displaced = run_dir / "readiness" / "probes-original"
            replace.write_text(
                "import json, os\n"
                "from pathlib import Path\n"
                "output = Path(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'])\n"
                f"output.parent.rename(Path({str(displaced)!r}))\n"
                "output.parent.mkdir()\n"
                f"output.write_text(json.dumps({_valid_probe('replace')!r}))\n",
                "utf-8",
            )
            with self.assertRaisesRegex(ValueError, "replaced|identity|unsafe"):
                self.module.run_requirement(
                    _requirement("replace", [sys.executable, str(replace)]),
                    run_dir=run_dir,
                    project_root=project,
                    environment_identity_digest="1" * 64,
                    deadline_epoch=time.time() + 5,
                )

    def test_hardlinked_probe_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            source = project / "outside.json"
            source.write_text(json.dumps(_valid_probe("hardlink")), "utf-8")
            script = project / "hardlink.py"
            script.write_text(
                "import os\n"
                f"os.link({str(source)!r}, os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'])\n",
                "utf-8",
            )

            result = self.module.run_requirement(
                _requirement("hardlink", [sys.executable, str(script)]),
                run_dir=root / "run",
                project_root=project,
                environment_identity_digest="4" * 64,
                deadline_epoch=time.time() + 5,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["observations"]["reason"], "probe_output_unsafe")

    def test_executable_identity_drift_cannot_publish_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            executable = project / "mutable-probe"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "if '--version' in sys.argv:\n"
                "    print('mutable-probe 1.0')\n"
                "    raise SystemExit(0)\n"
                f"payload = {_valid_probe('executable-drift')!r}\n"
                "pathlib.Path(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT']).write_text(json.dumps(payload))\n"
                "pathlib.Path(sys.argv[0]).write_text('#!/bin/sh\\nexit 0\\n')\n",
                "utf-8",
            )
            executable.chmod(0o755)

            result = self.module.run_requirement(
                _requirement("executable-drift", [str(executable)]),
                run_dir=root / "run",
                project_root=project,
                environment_identity_digest="5" * 64,
                deadline_epoch=time.time() + 5,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["observations"]["reason"], "executable_identity_changed"
        )

    def test_probe_script_identity_drift_cannot_publish_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            script = project / "mutable.py"
            script.write_text(
                "import json, os, pathlib\n"
                f"payload = {_valid_probe('script-drift')!r}\n"
                "pathlib.Path(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT']).write_text(json.dumps(payload))\n"
                "pathlib.Path(__file__).write_text('raise SystemExit(0)\\n')\n",
                "utf-8",
            )

            result = self.module.run_requirement(
                _requirement("script-drift", [sys.executable, str(script)]),
                run_dir=root / "run",
                project_root=project,
                environment_identity_digest="6" * 64,
                deadline_epoch=time.time() + 5,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["observations"]["reason"], "probe_input_identity_changed"
        )

    def test_logs_are_truncated_and_secrets_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            script = project / "logs.py"
            _emit_script(
                script,
                _valid_probe("logs"),
                stdout=(
                    "API_KEY=should-not-survive "
                    + "x" * (128 * 1024)
                    + "TAIL-SENTINEL"
                ),
            )
            result = self.module.run_requirement(
                _requirement("logs", [sys.executable, str(script)]),
                run_dir=root / "run",
                project_root=project,
                environment_identity_digest="2" * 64,
                deadline_epoch=time.time() + 5,
            )
            execution_path = (
                root
                / "run"
                / "readiness"
                / "probes"
                / "logs.execution.json"
            )
            execution_text = execution_path.read_text("utf-8")
            execution = json.loads(execution_text)
            self.assertEqual(result["status"], "ready")
            self.assertTrue(execution["logs_truncated"])
            self.assertNotIn("should-not-survive", execution_text)
            self.assertIn("[REDACTED]", execution["stdout"])
            self.assertIn("TAIL-SENTINEL", execution["stdout"])
            self.assertIn("truncated", execution["stdout"])
            self.assertLessEqual(
                len(execution["stdout"].encode("utf-8")),
                self.module.MAX_LOG_BYTES,
            )


if __name__ == "__main__":
    unittest.main()
