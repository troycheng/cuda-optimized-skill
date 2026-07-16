from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
BENCHMARK = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "benchmark.py"
CHECK_ENV = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "check_env.py"
ARTIFACTS = Path(os.environ.get("CUDA_E2E_ARTIFACTS", "/tmp/cuda-sm120-acceptance"))


@dataclass(frozen=True)
class Case:
    name: str
    solution: str
    reference: str
    dims: tuple[str, ...]
    extra: tuple[str, ...] = ()


CASES = (
    Case(
        "triton_vector",
        "triton_vector.py",
        "triton_vector_ref.py",
        ("--N=1048576",),
    ),
    Case(
        "cuda_reduction",
        "cuda_reduction.cu",
        "cuda_reduction_ref.py",
        ("--N=1048576",),
        ("--ptr-size", "1048576"),
    ),
    Case(
        "cutlass_gemm",
        "cutlass_gemm.cu",
        "cutlass_gemm_ref.py",
        ("--M=512", "--N=512", "--K=512"),
        ("--ptr-size", "262144"),
    ),
)


def _run(command: list[str], json_path: Path) -> dict:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


@unittest.skipUnless(
    os.environ.get("CUDA_SM120_E2E") == "1",
    "set CUDA_SM120_E2E=1 on an SM120 CUDA host",
)
class Sm120AcceptanceTests(unittest.TestCase):
    def test_environment_and_fixture_matrix(self) -> None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        env_path = ARTIFACTS / "env.json"
        _run([sys.executable, str(CHECK_ENV), "--out", str(env_path)], env_path)
        env = json.loads(env_path.read_text(encoding="utf-8"))
        self.assertEqual(env["primary_sm_arch"], "sm_120")

        for case in CASES:
            with self.subTest(case=case.name):
                result_path = ARTIFACTS / case.name / "bench.json"
                command = [
                    sys.executable,
                    str(BENCHMARK),
                    str(FIXTURES / case.solution),
                    "--ref",
                    str(FIXTURES / case.reference),
                    "--warmup",
                    "3",
                    "--repeat",
                    "8",
                    "--json-out",
                    str(result_path),
                    *case.extra,
                    *case.dims,
                ]
                result = _run(command, result_path)
                self.assertTrue(result["correctness"]["passed"])
                self.assertIn("samples_ms", result["kernel"])
                self.assertGreater(len(result["kernel"]["samples_ms"]), 4)
                self.assertIn("median_ms", result["kernel"])
                self.assertIn("p95_ms", result["kernel"])
                self.assertIn("cv_pct", result["kernel"])


if __name__ == "__main__":
    unittest.main()
