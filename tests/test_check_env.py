from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CHECK_ENV_PATH = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "check_env.py"
IDENTITY_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "readiness_identity.py"
)


def _load_check_env():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_check_env", CHECK_ENV_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_identity():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_readiness_identity", IDENTITY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CheckEnvTests(unittest.TestCase):
    def test_generic_tool_inventory_never_claims_capability(self) -> None:
        check_env = _load_check_env()
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "nsys"
            executable.write_text("tool\n", "utf-8")
            executable.chmod(0o755)
            with mock.patch.object(
                check_env.shutil, "which", return_value=str(executable)
            ), mock.patch.object(
                check_env,
                "_run",
                return_value=(1, "", "version query failed"),
            ):
                result = check_env._detect_tool("nsys", ["--version"])

        self.assertTrue(result["available"])
        self.assertIsNone(result["usable"])
        self.assertEqual(result["version_query_returncode"], 1)
        self.assertEqual(len(result["sha256"]), 64)
        self.assertNotIn("ready", result)
        self.assertNotIn("can_profile", result)

    def test_collect_env_includes_all_diagnostic_build_and_binary_tools(self) -> None:
        check_env = _load_check_env()
        expected = {
            "nvcc",
            "ncu",
            "nvidia-smi",
            "nsys",
            "compute-sanitizer",
            "ptxas",
            "cuobjdump",
            "nvdisasm",
            "cmake",
            "ninja",
            "cc",
            "cxx",
        }
        with mock.patch.object(
            check_env, "_detect_tool", side_effect=lambda name, *args, **kwargs: {
                "available": False,
                "path": None,
                "realpath": None,
                "sha256": None,
                "version": None,
                "version_query_returncode": None,
                "usable": None,
            }
        ), mock.patch.object(check_env, "_detect_gpus", return_value=[]), mock.patch.object(
            check_env, "_detect_nvcc", return_value={}
        ), mock.patch.object(check_env, "_detect_ncu", return_value={}), mock.patch.object(
            check_env, "_detect_driver", return_value={}
        ), mock.patch.object(check_env, "_detect_cutlass", return_value={}), mock.patch.object(
            check_env, "_detect_python_libs", return_value={}
        ):
            result = check_env.collect_env()

        self.assertEqual(set(result["tools"]), expected)

    def test_driver_uses_supported_query_and_parses_max_cuda_version(self) -> None:
        check_env = _load_check_env()

        def fake_run(command, timeout=10):
            del timeout
            if "--query-gpu=driver_version" in command:
                return 0, "595.71.05\n595.71.05\n", ""
            if len(command) == 1:
                return (
                    0,
                    "| NVIDIA-SMI 595.71.05  Driver Version: 595.71.05  CUDA Version: 13.3 |\n",
                    "",
                )
            return 1, "", "Field 'cuda_version' is not a valid field to query."

        with mock.patch.object(
            check_env.shutil,
            "which",
            return_value="/usr/bin/nvidia-smi",
        ), mock.patch.object(check_env, "_run", side_effect=fake_run):
            driver = check_env._detect_driver()

        self.assertEqual(driver["driver_versions"], ["595.71.05"])
        self.assertEqual(driver["max_cuda_version"], "13.3")
        self.assertNotIn("raw", driver)

    def test_ncu_version_prefers_numeric_version_line(self) -> None:
        check_env = _load_check_env()
        version_output = (
            "NVIDIA (R) Nsight Compute Command Line Profiler\n"
            "Copyright (c) NVIDIA Corporation\n"
            "Version 2026.1.1.0 (build 12345678)\n"
        )
        with mock.patch.object(
            check_env.shutil,
            "which",
            return_value="/usr/local/bin/ncu",
        ), mock.patch.object(
            check_env, "_run", side_effect=[(0, version_output, ""), (0, "metric\n", "")]
        ):
            ncu = check_env._detect_ncu()

        self.assertEqual(ncu["version"], "2026.1.1.0")

    def test_identity_accepts_venv_python_symlink_and_changes_with_packages(self) -> None:
        identity = _load_identity()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment = root / "venv"
            (environment / "bin").mkdir(parents=True)
            real_python = root / "python-real"
            real_python.write_text("#!/bin/sh\nexit 0\n", "utf-8")
            real_python.chmod(0o755)
            (environment / "bin" / "python").symlink_to(real_python)
            (environment / "pyvenv.cfg").write_text("home = /python\n", "utf-8")
            packages = [["torch", "2.13.0"]]

            def fake_run(command, timeout=10):
                del timeout
                if "--version" in command:
                    return 0, "Python 3.13.5\n", ""
                return 0, json.dumps(packages), ""

            inventory = {
                "tools": {
                    "nsys": {
                        "available": True,
                        "path": "/tools/nsys",
                        "realpath": "/tools/nsys",
                        "sha256": "a" * 64,
                        "version": "2026.1",
                        "version_query_returncode": 0,
                        "usable": None,
                    }
                },
                "driver": {
                    "driver_versions": ["595.71.05"],
                    "max_cuda_version": "13.3",
                    "gpu_identities": [
                        {"uuid": "GPU-0", "pci_bus_id": "0000:01:00.0"}
                    ],
                },
            }
            first = identity.build_identity(
                environment_root=environment,
                inventory=inventory,
                run=fake_run,
            )
            packages.append(["triton", "3.5.0"])
            second = identity.build_identity(
                environment_root=environment,
                inventory=inventory,
                run=fake_run,
            )

        self.assertEqual(set(first), identity.IDENTITY_FIELDS)
        self.assertEqual(first["gpu_identity"], second["gpu_identity"])
        self.assertNotEqual(first["toolchain_digest"], second["toolchain_digest"])
        self.assertEqual(first["uid"], os.getuid())

    def test_identity_rejects_open_inventory_and_nonisolated_root(self) -> None:
        identity = _load_identity()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment = root / "env"
            environment.mkdir()
            with self.assertRaisesRegex(ValueError, "inventory"):
                identity.build_identity(
                    environment_root=environment,
                    inventory={"tools": {}, "driver": {}, "surprise": True},
                    run=lambda *args, **kwargs: (0, "[]", ""),
                )
            linked = root / "linked-env"
            linked.symlink_to(environment)
            with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                identity.build_identity(
                    environment_root=linked,
                    inventory={"tools": {}, "driver": {}},
                    run=lambda *args, **kwargs: (0, "[]", ""),
                )


if __name__ == "__main__":
    unittest.main()
