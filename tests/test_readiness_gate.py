import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cuda_{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _probe(requirement_id: str, status: str) -> dict:
    return {
        "schema_version": "cuda-workload-optimizer/readiness-probe-v1",
        "requirement_id": requirement_id,
        "status": status,
        "observations": {},
        "artifacts": [],
    }


class FakeProbeRunner:
    def __init__(self, statuses):
        self.statuses = {
            key: list(value) if isinstance(value, list) else [value]
            for key, value in statuses.items()
        }
        self.calls = []

    def __call__(self, requirement, **kwargs):
        requirement_id = requirement["id"]
        self.calls.append(requirement_id)
        values = self.statuses[requirement_id]
        status = values.pop(0) if len(values) > 1 else values[0]
        return _probe(requirement_id, status)


class FakeInstaller:
    def __init__(self, status="succeeded"):
        self.status = status
        self.calls = []

    def __call__(self, remediation, **kwargs):
        self.calls.append(remediation["authorization_id"])
        return {
            "schema_version": "cuda-workload-optimizer/readiness-install-v1",
            "status": self.status,
            "reason": None if self.status == "succeeded" else "pip_failed",
        }


class ReadinessGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = _load("readiness_gate")
        cls.install = _load("readiness_install")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.environment = self.root / "environment"
        self.project.mkdir()
        (self.environment / "bin").mkdir(parents=True)
        self.python = self.environment / "bin" / "python"
        self.python.write_text("#!/bin/sh\nexit 0\n", "utf-8")
        self.python.chmod(0o755)
        self.requirements = self.project / "requirements.lock"
        self.requirements.write_text("example==1 --hash=sha256:" + "a" * 64 + "\n", "utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def requirement(
        self,
        requirement_id: str,
        *,
        necessity="required",
        phase="foundation",
        scope="project",
        remediation=None,
        max_age_seconds=300,
    ):
        return {
            "id": requirement_id,
            "necessity": necessity,
            "control_scope": scope,
            "phase": phase,
            "kind": "gpu_execute" if phase == "foundation" else "workload_smoke",
            "max_age_seconds": max_age_seconds,
            "probe": {"argv": ["/usr/bin/true"], "timeout_seconds": 2},
            "remediation": remediation or {"mode": "none"},
        }

    def isolated_remediation(self):
        return {
            "mode": "isolated_pip",
            "authorization_id": "approved-env-install",
            "python": str(self.python),
            "requirements_file": str(self.requirements),
            "requirements_sha256": hashlib.sha256(
                self.requirements.read_bytes()
            ).hexdigest(),
            "timeout_seconds": 2,
        }

    def contract(self, requirements, *, max_seconds=30, max_repairs=1):
        return {
            "schema_version": "cuda-workload-optimizer/readiness-contract-v1",
            "requested_claim": "workload",
            "budget": {
                "max_seconds": max_seconds,
                "max_repairs": max_repairs,
            },
            "requirements": requirements,
        }

    def control(self, **identity_updates):
        identity = {
            "toolchain_digest": "a" * 64,
            "uid": os.getuid(),
            "container_identity": None,
            "gpu_identity": "GPU-0",
            "visible_devices": {"cuda": "0", "nvidia": "0"},
            "permission_state": "counters-denied",
        }
        identity.update(identity_updates)
        return {
            "project_root": str(self.project),
            "environment_root": str(self.environment),
            "environment_identity": identity,
        }

    def run_gate(
        self,
        contract,
        runner,
        installer=None,
        *,
        now=100.0,
        control=None,
        identity_provider=None,
    ):
        kwargs = {}
        if identity_provider is not None:
            kwargs["identity_provider"] = identity_provider
        return self.gate.run_gate(
            contract=contract,
            control=control or self.control(),
            run_dir=self.root / "run",
            probe_runner=runner,
            installer=installer or FakeInstaller("failed"),
            now=lambda: now,
            **kwargs,
        )

    def test_required_foundation_failure_skips_workload_phase(self) -> None:
        runner = FakeProbeRunner({"gpu": "failed", "workload": "ready"})
        report = self.run_gate(
            self.contract(
                [
                    self.requirement("gpu"),
                    self.requirement("workload", phase="workload"),
                ]
            ),
            runner,
        )

        self.assertFalse(report["can_start_diagnosis"])
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(runner.calls, ["gpu"])

    def test_every_declared_environment_identity_component_changes_digest(self) -> None:
        base = self.control()["environment_identity"]
        base_digest = self.gate.environment_identity_digest(base)
        variants = (
            {"toolchain_digest": "b" * 64},
            {"uid": base["uid"] + 1},
            {"container_identity": "container-b"},
            {"gpu_identity": "GPU-1"},
            {"visible_devices": {"cuda": "1", "nvidia": "1"}},
            {"permission_state": "counters-allowed"},
        )
        for update in variants:
            with self.subTest(update=update):
                changed = json.loads(json.dumps(base))
                changed.update(update)
                self.assertNotEqual(
                    self.gate.environment_identity_digest(changed), base_digest
                )

        open_identity = dict(base, surprise=True)
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.gate.environment_identity_digest(open_identity)

    def test_diagnostic_failure_degrades_but_still_runs_workload(self) -> None:
        runner = FakeProbeRunner({"nsys": "unavailable", "workload": "ready"})
        report = self.run_gate(
            self.contract(
                [
                    self.requirement("nsys", necessity="diagnostic"),
                    self.requirement("workload", phase="workload"),
                ]
            ),
            runner,
        )

        self.assertTrue(report["can_start_diagnosis"])
        self.assertEqual(report["status"], "degraded")
        self.assertEqual(runner.calls, ["nsys", "workload"])
        self.assertEqual(report["results"][0]["unsupported_capabilities"], ["gpu_execute"])

    def test_host_requirement_never_auto_repairs(self) -> None:
        runner = FakeProbeRunner({"counters": "failed"})
        installer = FakeInstaller()
        requirement = self.requirement(
            "counters",
            scope="host",
            remediation={
                "mode": "user_action",
                "message": "Enable the requested GPU counter permission.",
            },
        )
        updated_identity = self.control(toolchain_digest="b" * 64)[
            "environment_identity"
        ]
        report = self.run_gate(
            self.contract([requirement]),
            runner,
            installer,
            identity_provider=lambda: updated_identity,
        )

        self.assertEqual(report["status"], "user_action_required")
        self.assertEqual(installer.calls, [])
        self.assertIn("Enable the requested", report["next_actions"][0])

    def test_isolated_repair_retries_only_the_failed_requirement(self) -> None:
        runner = FakeProbeRunner({"python-deps": ["failed", "ready"]})
        installer = FakeInstaller()
        requirement = self.requirement(
            "python-deps",
            scope="isolated_environment",
            remediation=self.isolated_remediation(),
        )
        updated_identity = self.control(toolchain_digest="b" * 64)[
            "environment_identity"
        ]
        report = self.run_gate(
            self.contract([requirement]),
            runner,
            installer,
            identity_provider=lambda: updated_identity,
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(runner.calls, ["python-deps", "python-deps"])
        self.assertEqual(installer.calls, ["approved-env-install"])
        self.assertEqual(report["budget"]["repairs_used"], 1)
        self.assertGreaterEqual(report["budget"]["elapsed_seconds"], 0)

    def test_install_failure_and_failed_retry_are_blocked(self) -> None:
        requirement = self.requirement(
            "python-deps",
            scope="isolated_environment",
            remediation=self.isolated_remediation(),
        )
        for statuses, install_status, expected_calls in (
            (["failed"], "failed", 1),
            (["failed", "failed"], "succeeded", 2),
        ):
            with self.subTest(statuses=statuses, install_status=install_status):
                run_dir = self.root / f"run-{install_status}-{expected_calls}"
                runner = FakeProbeRunner({"python-deps": statuses})
                report = self.gate.run_gate(
                    contract=self.contract([requirement]),
                    control=self.control(),
                    run_dir=run_dir,
                    probe_runner=runner,
                    installer=FakeInstaller(install_status),
                    now=lambda: 100.0,
                    identity_provider=lambda: self.control(
                        toolchain_digest="b" * 64
                    )["environment_identity"],
                )
                self.assertEqual(report["status"], "blocked")
                self.assertEqual(len(runner.calls), expected_calls)

    def test_resume_reuses_fresh_evidence_but_expiry_or_identity_change_reruns(self) -> None:
        contract = self.contract(
            [
                self.requirement("short", max_age_seconds=1),
                self.requirement("long", max_age_seconds=100),
            ]
        )
        first = FakeProbeRunner({"short": "ready", "long": "ready"})
        self.run_gate(contract, first, now=100.0)
        self.assertEqual(first.calls, ["short", "long"])

        fresh = FakeProbeRunner({"short": "failed", "long": "failed"})
        report = self.run_gate(contract, fresh, now=100.5)
        self.assertEqual(fresh.calls, [])
        self.assertEqual(report["status"], "ready")

        expired = FakeProbeRunner({"short": "ready", "long": "failed"})
        report = self.run_gate(contract, expired, now=102.0)
        self.assertEqual(expired.calls, ["short"])
        self.assertEqual(report["status"], "ready")

        changed = FakeProbeRunner({"short": "ready", "long": "ready"})
        self.run_gate(
            contract,
            changed,
            now=102.5,
            control=self.control(gpu_identity="GPU-1"),
        )
        self.assertEqual(changed.calls, ["short", "long"])

    def test_resume_preserves_elapsed_time_and_does_not_reset_repair_budget(self) -> None:
        requirement = self.requirement(
            "python-deps",
            scope="isolated_environment",
            remediation=self.isolated_remediation(),
            max_age_seconds=1,
        )
        contract = self.contract([requirement], max_repairs=1)
        first = FakeProbeRunner({"python-deps": ["failed", "ready"]})
        updated_control = self.control(toolchain_digest="b" * 64)
        self.run_gate(
            contract,
            first,
            FakeInstaller(),
            now=100.0,
            identity_provider=lambda: updated_control["environment_identity"],
        )

        expired = FakeProbeRunner({"python-deps": "failed"})
        installer = FakeInstaller()
        report = self.run_gate(
            contract,
            expired,
            installer,
            now=102.0,
            control=updated_control,
        )

        self.assertEqual(expired.calls, ["python-deps"])
        self.assertEqual(installer.calls, [])
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["budget"]["repairs_used"], 1)
        self.assertGreaterEqual(report["budget"]["elapsed_seconds"], 2)

    def test_repair_refreshes_identity_and_restarts_all_foundation_probes(self) -> None:
        remediation = self.isolated_remediation()
        contract = self.contract(
            [
                self.requirement("gpu"),
                self.requirement(
                    "deps",
                    scope="isolated_environment",
                    remediation=remediation,
                ),
            ]
        )
        runner = FakeProbeRunner(
            {"gpu": ["ready", "ready"], "deps": ["failed", "ready"]}
        )
        updated_identity = self.control(toolchain_digest="b" * 64)[
            "environment_identity"
        ]
        report = self.run_gate(
            contract,
            runner,
            FakeInstaller(),
            identity_provider=lambda: updated_identity,
        )

        self.assertEqual(runner.calls, ["gpu", "deps", "gpu", "deps"])
        self.assertEqual(report["status"], "ready")
        self.assertEqual(
            report["environment_identity_digest"],
            self.gate.environment_identity_digest(updated_identity),
        )
        self.assertTrue(
            all(
                item["identity_digest"]
                == report["environment_identity_digest"]
                for item in report["results"]
            )
        )

    def test_successful_repair_without_refreshed_identity_fails_closed(self) -> None:
        requirement = self.requirement(
            "deps",
            scope="isolated_environment",
            remediation=self.isolated_remediation(),
        )
        runner = FakeProbeRunner({"deps": ["failed", "ready"]})
        report = self.run_gate(
            self.contract([requirement]),
            runner,
            FakeInstaller(),
        )

        self.assertEqual(runner.calls, ["deps"])
        self.assertEqual(report["status"], "blocked")

    def test_crash_during_install_cannot_reset_repair_or_wall_clock_budget(self) -> None:
        class CrashingInstaller:
            def __call__(self, remediation, **kwargs):
                raise RuntimeError("simulated crash")

        requirement = self.requirement(
            "deps",
            scope="isolated_environment",
            remediation=self.isolated_remediation(),
            max_age_seconds=1,
        )
        contract = self.contract([requirement], max_seconds=5, max_repairs=1)
        with self.assertRaisesRegex(RuntimeError, "simulated crash"):
            self.run_gate(
                contract,
                FakeProbeRunner({"deps": "failed"}),
                CrashingInstaller(),
                now=100.0,
            )

        installer = FakeInstaller()
        report = self.run_gate(
            contract,
            FakeProbeRunner({"deps": "failed"}),
            installer,
            now=106.0,
        )
        self.assertEqual(installer.calls, [])
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["budget"]["repairs_used"], 1)
        self.assertGreaterEqual(report["budget"]["elapsed_seconds"], 6)

    def test_report_or_completion_marker_tampering_fails_closed(self) -> None:
        contract = self.contract([self.requirement("gpu")])
        self.run_gate(contract, FakeProbeRunner({"gpu": "ready"}))
        report_path = self.root / "run" / "readiness" / "report.json"
        report_path.write_text("{}\n", "utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.run_gate(contract, FakeProbeRunner({"gpu": "ready"}), now=101.0)

        second_run = self.root / "second-run"
        self.gate.run_gate(
            contract=contract,
            control=self.control(),
            run_dir=second_run,
            probe_runner=FakeProbeRunner({"gpu": "ready"}),
            installer=FakeInstaller("failed"),
            now=lambda: 100.0,
        )
        marker_path = second_run / "readiness" / "report.complete.json"
        marker = json.loads(marker_path.read_text("utf-8"))
        marker["report_sha256"] = "0" * 64
        marker_path.write_text(json.dumps(marker), "utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.gate.run_gate(
                contract=contract,
                control=self.control(),
                run_dir=second_run,
                probe_runner=FakeProbeRunner({"gpu": "ready"}),
                installer=FakeInstaller("failed"),
                now=lambda: 101.0,
            )

    def test_real_runner_evidence_path_is_durable_and_relative(self) -> None:
        script = self.project / "ready.py"
        payload = _probe("gpu", "ready")
        script.write_text(
            "import json, os\n"
            f"payload = {payload!r}\n"
            "output = os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT']\n"
            "open(output, 'w').write(json.dumps(payload))\n",
            "utf-8",
        )
        requirement = self.requirement("gpu")
        requirement["probe"]["argv"] = [sys.executable, str(script)]
        report = self.gate.run_gate(
            contract=self.contract([requirement]),
            control=self.control(),
            run_dir=self.root / "real-run",
        )

        evidence_path = report["results"][0]["evidence_path"]
        self.assertFalse(Path(evidence_path).is_absolute())
        self.assertTrue((self.root / "real-run" / evidence_path).is_file())
        self.assertTrue(
            (
                self.root
                / "real-run"
                / "readiness"
                / "report.complete.json"
            ).is_file()
        )

    def test_installer_builds_only_hashed_isolated_pip_command(self) -> None:
        argv_path = self.environment / "argv.json"
        self.python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            f"pathlib.Path({str(argv_path)!r}).write_text(json.dumps(sys.argv[1:]))\n",
            "utf-8",
        )
        self.python.chmod(0o755)
        remediation = self.isolated_remediation()

        result = self.install.install_isolated_pip(
            remediation,
            project_root=self.project,
            environment_root=self.environment,
            run_dir=self.root / "install-run",
            deadline_epoch=time.time() + 20,
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            json.loads(argv_path.read_text("utf-8")),
            [
                "-I",
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(self.requirements),
            ],
        )

        self.requirements.write_text("drifted\n", "utf-8")
        argv_path.unlink()
        result = self.install.install_isolated_pip(
            remediation,
            project_root=self.project,
            environment_root=self.environment,
            run_dir=self.root / "install-drift",
            deadline_epoch=time.time() + 20,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "requirements_digest_mismatch")
        self.assertFalse(argv_path.exists())

    def test_installer_rejects_python_or_lockfile_identity_drift(self) -> None:
        remediation = self.isolated_remediation()
        self.python.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "pathlib.Path(sys.argv[0]).write_text('#!/bin/sh\\nexit 0\\n')\n",
            "utf-8",
        )
        self.python.chmod(0o755)
        result = self.install.install_isolated_pip(
            remediation,
            project_root=self.project,
            environment_root=self.environment,
            run_dir=self.root / "python-drift",
            deadline_epoch=time.time() + 20,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "python_identity_changed")

    def test_installer_supports_a_stable_venv_python_leaf_symlink(self) -> None:
        real_python = self.root / "system-python"
        real_python.write_text("#!/bin/sh\nexit 0\n", "utf-8")
        real_python.chmod(0o755)
        self.python.unlink()
        self.python.symlink_to(real_python)
        remediation = self.isolated_remediation()

        result = self.install.install_isolated_pip(
            remediation,
            project_root=self.project,
            environment_root=self.environment,
            run_dir=self.root / "symlink-python",
            deadline_epoch=time.time() + 20,
        )

        self.assertEqual(result["status"], "succeeded")

    def test_installer_soft_threshold_does_not_replace_the_readiness_deadline(self) -> None:
        self.python.write_text(
            "#!/usr/bin/env python3\nimport time\ntime.sleep(0.35)\n", "utf-8"
        )
        self.python.chmod(0o755)

        started = time.monotonic()
        result = self.install.install_isolated_pip(
            self.isolated_remediation(),
            project_root=self.project,
            environment_root=self.environment,
            run_dir=self.root / "maintenance-budget",
            deadline_epoch=time.time() + 2,
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertIsNone(result["reason"])
        self.assertTrue(result["soft_limit_exceeded"])
        self.assertEqual(result["stop_reason"], "completed")
        self.assertTrue(
            (
                self.root
                / "maintenance-budget"
                / "readiness"
                / "installs"
                / "approved-env-install.soft-checkpoint.json"
            ).is_file()
        )
        self.assertGreater(result["duration_seconds"], 0.2)
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
