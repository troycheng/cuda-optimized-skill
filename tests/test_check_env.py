from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CHECK_ENV_PATH = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "check_env.py"


def _load_check_env():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_check_env", CHECK_ENV_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CheckEnvTests(unittest.TestCase):
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

        with mock.patch.object(check_env.shutil, "which", return_value="/usr/bin/nvidia-smi"), mock.patch.object(
            check_env, "_run", side_effect=fake_run
        ):
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
        with mock.patch.object(check_env.shutil, "which", return_value="/usr/local/bin/ncu"), mock.patch.object(
            check_env,
            "_run",
            side_effect=[(0, version_output, ""), (0, "metric\n", "")],
        ):
            ncu = check_env._detect_ncu()

        self.assertEqual(ncu["version"], "2026.1.1.0")


if __name__ == "__main__":
    unittest.main()
