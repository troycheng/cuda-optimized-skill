#!/usr/bin/env python3
"""CPU-only regression coverage for migrated three-skill capabilities."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "cuda-kernel-optimizer"
SCRIPTS = SKILL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_ncu_rep  # noqa: E402
import ablate  # noqa: E402
import common_options  # noqa: E402
import sass_check  # noqa: E402
import strategy_memory  # noqa: E402


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CoverageTests(unittest.TestCase):
    def test_shared_options_are_forwarded(self) -> None:
        state = {
            "ref_file": "/tmp/ref.py",
            "dims": {"N": 7, "M": 3},
            "ptr_size": 64,
            "benchmark_options": {
                "backend": "triton",
                "arch": "sm_120",
                "gpu": 2,
                "atol": 1e-5,
                "rtol": 2e-4,
                "seed": 9,
                "validation_seeds": "10,11",
                "nvcc_bin": "/opt/cuda/bin/nvcc",
            },
        }
        argv = common_options.benchmark_option_argv(state)
        joined = " ".join(argv)
        for expected in (
            "--backend triton", "--arch sm_120", "--gpu 2",
            "--atol 1e-05", "--rtol 0.0002", "--seed 9",
            "--validation-seeds 10,11", "--nvcc-bin /opt/cuda/bin/nvcc",
            "--ptr-size 64", "--M=3", "--N=7", "--ref /tmp/ref.py",
        ):
            self.assertIn(expected, joined)

    def test_strategy_memory_is_workload_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "memory.json")
            key = strategy_memory.scope_key(
                "cuda", "/x/a.cu", "/x/ref.py", {"N": 4}, "sm_120"
            )
            constraints = strategy_memory.record(
                path,
                key,
                {"backend": "cuda"},
                [
                    {"method_id": "good", "outcome": "positive"},
                    {"method_id": "bad", "outcome": "negative"},
                ],
                {"method_ids": ["good", "bad"], "outcome": "positive"},
            )
            self.assertEqual(constraints["preferred_method_ids"], ["good"])
            self.assertEqual(constraints["blocked_method_ids"], ["bad"])
            other = strategy_memory.load_constraints(path, "different")
            self.assertFalse(other["scope_seen"])

    def test_sass_uses_tri_state_for_triton_and_missing_binary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            state = run / "state.json"
            _write_json(state, {"run_dir": str(run)})

            iter1 = run / "iterv1"
            iter1.mkdir()
            (iter1 / "kernel.py").write_text("def run_kernel(): pass\n", encoding="utf-8")
            _write_json(iter1 / "methods.json", {"methods": [{"id": "triton.tile"}]})
            triton_result = sass_check.run(str(state), 1)
            self.assertEqual(triton_result["status"], "not_applicable")
            self.assertIsNone(triton_result["checks"][0]["verified"])

            iter2 = run / "iterv2"
            iter2.mkdir()
            (iter2 / "kernel.cu").write_text("// no compiled shared object\n", encoding="utf-8")
            _write_json(iter2 / "methods.json", {"methods": [{"id": "cuda.tile"}]})
            cuda_result = sass_check.run(str(state), 2)
            self.assertEqual(cuda_result["status"], "unknown")
            self.assertIsNone(cuda_result["checks"][0]["verified"])

    def test_missing_ablation_is_unknown_not_ineffective(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            iter1 = run / "iterv1"
            iter1.mkdir()
            state = run / "state.json"
            _write_json(state, {"run_dir": str(run), "noise_threshold_pct": 2.0})
            _write_json(iter1 / "bench.json", {
                "correctness": {"passed": True},
                "kernel": {"median_ms": 1.0, "average_ms": 1.1},
            })
            _write_json(iter1 / "methods.json", {"methods": [{"id": "method.missing"}]})
            result = ablate.run(str(state), 1)
            self.assertIsNone(result["attributions"][0]["contributed"])
            self.assertIsNone(result["attributions"][0]["attribution_ms"])

    def test_state_manifest_and_layered_winners(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = root / "kernel.cu"
            reference = root / "ref.py"
            baseline.write_text("// baseline\n", encoding="utf-8")
            reference.write_text("def reference(**kwargs): return kwargs\n", encoding="utf-8")
            env = root / "env-source.json"
            preflight = root / "preflight-source.json"
            memory = root / "strategy.json"
            _write_json(env, {"selected_gpu": {"index": 1, "sm_arch": "sm_120"}})
            _write_json(preflight, {
                "ok": True,
                "baseline": {"path": str(baseline), "backend": "cuda_or_cutlass"},
                "ref": {"path": str(reference)},
                "warnings": [],
                "errors": [],
            })

            init = _run(
                str(SCRIPTS / "state.py"), "init",
                "--baseline", str(baseline),
                "--ref", str(reference),
                "--iterations", "2",
                "--dims", '{"N":4}',
                "--env", str(env),
                "--preflight", str(preflight),
                "--benchmark-options", '{"backend":"cuda","arch":"sm_120","gpu":1}',
                "--ncu-options", '{"ncu_bin":"/opt/ncu","launch_count":1}',
                "--strategy-memory", str(memory),
            )
            info = json.loads(init.stdout)
            run = Path(info["run_dir"])
            state_path = Path(info["state"])
            self.assertTrue((run / "env.json").is_file())
            self.assertTrue((run / "preflight.json").is_file())
            self.assertTrue((run / "preflight.md").is_file())
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["benchmark_options"]["gpu"], 1)
            self.assertEqual(manifest["artifacts"]["preflight_markdown"], str(run / "preflight.md"))

            baseline_bench = run / "baseline" / "bench.json"
            _write_json(baseline_bench, {
                "correctness": {"passed": True},
                "kernel": {"median_ms": 2.0, "average_ms": 2.2},
            })
            _run(
                str(SCRIPTS / "state.py"), "set-baseline-metric",
                "--state", str(state_path), "--bench", str(baseline_bench),
            )

            iter1 = run / "iterv1"
            kernel1 = iter1 / "kernel.cu"
            kernel1.write_text("// candidate 1\n", encoding="utf-8")
            bench1 = iter1 / "bench.json"
            methods1 = iter1 / "methods.json"
            attribution1 = iter1 / "attribution.json"
            sass1 = iter1 / "sass_check.json"
            report1 = iter1 / "kernel.ncu-rep"
            status1 = iter1 / "kernel_profile_status.json"
            report1.write_bytes(b"fake-report")
            _write_json(bench1, {
                "correctness": {"passed": True},
                "kernel": {"median_ms": 1.0, "average_ms": 1.1},
            })
            _write_json(methods1, {"methods": [{"id": "method.good", "axis": "compute"}]})
            _write_json(attribution1, {"attributions": [{
                "method_id": "method.good", "contributed": True, "attribution_ms": 0.2,
            }]})
            _write_json(sass1, {"status": "verified", "checks": [{
                "method_id": "method.good", "verified": True,
                "verification_status": "verified",
            }]})
            _write_json(status1, {
                "profile_status": "success", "analysis_status": "success",
                "ncu_rep": str(report1),
            })
            _run(
                str(SCRIPTS / "state.py"), "update",
                "--state", str(state_path), "--iter", "1",
                "--kernel", str(kernel1), "--bench", str(bench1),
                "--methods-json", str(methods1), "--attribution", str(attribution1),
                "--sass-check", str(sass1), "--ncu-status", str(status1),
                "--skip-validation",
            )

            iter2 = run / "iterv2"
            kernel2 = iter2 / "kernel.cu"
            kernel2.write_text("// candidate 2\n", encoding="utf-8")
            bench2 = iter2 / "bench.json"
            methods2 = iter2 / "methods.json"
            _write_json(bench2, {
                "correctness": {"passed": True},
                "kernel": {"median_ms": 0.8, "average_ms": 0.9},
            })
            _write_json(methods2, {"methods": [{"id": "method.unverified", "axis": "memory"}]})
            _run(
                str(SCRIPTS / "state.py"), "update",
                "--state", str(state_path), "--iter", "2",
                "--kernel", str(kernel2), "--bench", str(bench2),
                "--methods-json", str(methods2), "--skip-validation",
            )

            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state_data["best_metric_ms"], 0.8)
            self.assertEqual(state_data["best_profiled_metric_ms"], 1.0)
            self.assertEqual(state_data["best_profiled_file"], str(kernel1))
            self.assertEqual(state_data["effective_methods"][0]["id"], "method.good")
            self.assertEqual(state_data["unverified_methods"][0]["id"], "method.unverified")
            self.assertIn(
                "method.good",
                state_data["strategy_memory"]["constraints"]["preferred_method_ids"],
            )
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["winner_state"]["benchmark_winner"]["metric_ms"], 0.8)
            self.assertEqual(manifest["winner_state"]["fully_profiled_winner"]["metric_ms"], 1.0)

    def test_existing_ncu_report_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = root / "sample.ncu-rep"
            report.write_bytes(b"fake")
            source = root / "kernel.cu"
            source.write_text('__global__ void solve_kernel() {}\n', encoding="utf-8")
            os.utime(report, (1, 1))
            fake_ncu = root / "ncu"
            fake_ncu.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "if '--csv' in sys.argv:\n"
                " print('\\\"Kernel Name\\\",\\\"Metric Name\\\",\\\"Metric Unit\\\",\\\"Metric Value\\\"')\n"
                " print('\\\"solve\\\",\\\"sm__throughput.avg.pct_of_peak_sustained_elapsed\\\",\\\"%\\\",\\\"20\\\"')\n"
                " print('\\\"solve\\\",\\\"dram__throughput.avg.pct_of_peak_sustained_elapsed\\\",\\\"%\\\",\\\"90\\\"')\n"
                " print('\\\"solve\\\",\\\"smsp__warp_issue_stalled_barrier_per_inst_issued\\\",\\\"%\\\",\\\"30\\\"')\n"
                "else:\n"
                " print('fake ncu import')\n",
                encoding="utf-8",
            )
            fake_ncu.chmod(fake_ncu.stat().st_mode | stat.S_IEXEC)
            out = root / "analysis"
            result = analyze_ncu_rep.analyze(
                str(report), str(out), str(fake_ncu), 3, str(source)
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["backend"], "cuda")
            self.assertEqual(result["kernel_names"], ["solve"])
            self.assertEqual(
                result["freshness_assessment"],
                "possible_stale_source_newer_than_report",
            )
            self.assertEqual(result["metric_count"], 3)
            self.assertEqual(result["primary_axis"], "memory")
            self.assertTrue((out / "analysis.json").is_file())
            self.assertTrue((out / "analysis.md").is_file())

    def test_orchestrator_exposes_migrated_options(self) -> None:
        result = _run(str(SCRIPTS / "orchestrate.py"), "setup", "--help")
        for option in (
            "--backend", "--arch", "--gpu", "--atol", "--rtol", "--seed",
            "--validation-seeds", "--nvcc-bin", "--ncu-bin", "--strategy-memory",
        ):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
