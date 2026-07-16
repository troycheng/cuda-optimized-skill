from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SANITIZE_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "sanitize.py"
)
POLICY_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "references"
    / "sanitizer_policy.json"
)
ORCHESTRATE_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "orchestrate.py"
)


def _load_sanitize():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_sanitize_test", SANITIZE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_orchestrate():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_orchestrate_sanitize_test", ORCHESTRATE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SanitizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sanitize = _load_sanitize()
        self.policy = {
            "schema_version": 1,
            "tools": ["memcheck", "racecheck", "initcheck", "synccheck"],
            "methods": {
                "latency.async_pipeline": [
                    "memcheck",
                    "racecheck",
                    "synccheck",
                ],
                "memory.vectorized_access": ["memcheck", "initcheck"],
            },
        }

    def test_targeted_async_method_selects_memcheck_racecheck_synccheck(self):
        tools = self.sanitize.select_tools(
            method_ids=["latency.async_pipeline"],
            mode="targeted",
            policy=self.policy,
        )
        self.assertEqual(tools, ["memcheck", "racecheck", "synccheck"])

    def test_targeted_union_is_deduplicated_in_canonical_order(self):
        tools = self.sanitize.select_tools(
            method_ids=[
                "memory.vectorized_access",
                "latency.async_pipeline",
                "memory.vectorized_access",
            ],
            mode="targeted",
            policy=self.policy,
        )
        self.assertEqual(
            tools, ["memcheck", "racecheck", "initcheck", "synccheck"]
        )

    def test_targeted_unmatched_method_runs_no_tools(self):
        self.assertEqual(
            self.sanitize.select_tools(
                ["compute.launch_config"], mode="targeted", policy=self.policy
            ),
            [],
        )

    def test_full_always_runs_all_tools(self):
        self.assertEqual(
            self.sanitize.select_tools([], mode="full", policy=self.policy),
            ["memcheck", "racecheck", "initcheck", "synccheck"],
        )

    def test_repository_policy_is_valid_and_covers_async_pipeline(self):
        policy = self.sanitize.load_policy(POLICY_PATH)
        registry = json.loads(
            (POLICY_PATH.parent / "method_registry.json").read_text(encoding="utf-8")
        )["methods"]
        self.assertEqual(policy["schema_version"], 1)
        self.assertEqual(set(policy["methods"]) - set(registry), set())
        self.assertEqual(
            self.sanitize.select_tools(
                ["latency.async_pipeline"], mode="targeted", policy=policy
            ),
            ["memcheck", "racecheck", "synccheck"],
        )

    def test_missing_compute_sanitizer_is_explicit_not_passed(self):
        result = self.sanitize.run_tools(
            executable=None,
            tools=["memcheck"],
            command=[sys.executable, "bench.py"],
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["passed"])
        self.assertEqual(result["coverage"], "degraded")
        self.assertEqual(len(result["results"]), 1)
        evidence = result["results"][0]
        self.assertEqual(evidence["status"], "unavailable")
        self.assertIsNone(evidence["returncode"])
        self.assertEqual(evidence["stdout"], "")
        self.assertIn("unavailable", evidence["stderr"])
        self.assertEqual(
            evidence["command"][1:5],
            ["--tool", "memcheck", "--error-exitcode", "86"],
        )

    def _fake_sanitizer(self, root: Path, *, returncode: int) -> Path:
        executable = root / "compute-sanitizer"
        executable.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import sys
                print(json.dumps(sys.argv[1:]))
                print("fake sanitizer stderr", file=sys.stderr)
                raise SystemExit({returncode})
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    def test_each_tool_records_command_returncode_stdout_stderr_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = self._fake_sanitizer(Path(tmp), returncode=0)
            result = self.sanitize.run_tools(
                executable=str(executable),
                tools=["memcheck", "synccheck"],
                command=[sys.executable, "bench.py", "--profile-only"],
            )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(result["coverage"], "complete")
        self.assertEqual(len(result["results"]), 2)
        for tool, evidence in zip(
            ["memcheck", "synccheck"], result["results"]
        ):
            self.assertEqual(evidence["tool"], tool)
            self.assertEqual(evidence["returncode"], 0)
            self.assertEqual(evidence["status"], "passed")
            self.assertIn("fake sanitizer stderr", evidence["stderr"])
            self.assertEqual(
                evidence["command"],
                [
                    str(executable),
                    "--tool",
                    tool,
                    "--error-exitcode",
                    "86",
                    sys.executable,
                    "bench.py",
                    "--profile-only",
                ],
            )

    def test_error_exitcode_86_rejects_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = self._fake_sanitizer(Path(tmp), returncode=86)
            result = self.sanitize.run_tools(
                executable=str(executable),
                tools=["racecheck"],
                command=[sys.executable, "bench.py"],
            )

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertEqual(result["coverage"], "complete")
        self.assertEqual(result["results"][0]["returncode"], 86)
        self.assertEqual(result["results"][0]["status"], "failed")

    def test_no_targeted_tools_is_explicit_not_applicable(self):
        result = self.sanitize.run_tools(
            executable=None, tools=[], command=[sys.executable, "bench.py"]
        )
        self.assertEqual(result["status"], "not_applicable")
        self.assertTrue(result["passed"])
        self.assertEqual(result["coverage"], "not_applicable")
        self.assertEqual(result["results"], [])

    def test_cli_writes_artifact_and_returns_86_for_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = self._fake_sanitizer(root, returncode=86)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(self.policy), encoding="utf-8")
            methods_path = root / "methods.json"
            methods_path.write_text(
                json.dumps({"methods": [{"id": "latency.async_pipeline"}]}),
                encoding="utf-8",
            )
            output = root / "sanitizer.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SANITIZE_PATH),
                    "--mode",
                    "targeted",
                    "--policy",
                    str(policy_path),
                    "--methods-json",
                    str(methods_path),
                    "--compute-sanitizer",
                    str(executable),
                    "--out",
                    str(output),
                    "--",
                    sys.executable,
                    "bench.py",
                ],
                capture_output=True,
                text=True,
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 86)
        self.assertEqual(artifact["status"], "failed")
        self.assertEqual(
            [item["tool"] for item in artifact["results"]],
            ["memcheck", "racecheck", "synccheck"],
        )

    def test_candidate_binding_records_exact_regular_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "kernel.py"
            candidate.write_text("# candidate\n", encoding="utf-8")
            result = self.sanitize.bind_candidate(
                {
                    "status": "passed",
                    "passed": True,
                    "coverage": "complete",
                    "results": [],
                },
                candidate_file=candidate,
                input_hash="f" * 64,
            )

        self.assertEqual(result["candidate_file"], str(candidate.resolve()))
        self.assertRegex(result["candidate_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(result["input_hash"], "f" * 64)


class OrchestratorSanitizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrate = _load_orchestrate()

    def _candidate(self, root: Path, name: str, branch_index: int) -> dict:
        path = root / "branches" / name / "kernel.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
        return {
            "status": "confirmed_win",
            "kernel": str(path),
            "branch_index": branch_index,
            "statistics": {"estimate_pct": float(10 - branch_index)},
        }

    def test_quick_targeted_runs_only_policy_selected_tools_for_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = root / "methods.json"
            methods.write_text(
                json.dumps({"methods": [{"id": "latency.async_pipeline"}]}),
                encoding="utf-8",
            )
            candidate = self._candidate(root, "b1", 1)
            state = {
                "input_hash": "a" * 64,
                "dims": {"N": 64},
                "ptr_size": 128,
            }

            def runner(command, **_kwargs):
                output = Path(command[command.index("--out") + 1])
                candidate_file = command[command.index("--candidate-file") + 1]
                payload = {
                    "status": "passed",
                    "passed": True,
                    "coverage": "complete",
                    "mode": "targeted",
                    "method_ids": ["latency.async_pipeline"],
                    "selected_tools": ["memcheck", "racecheck", "synccheck"],
                    "candidate_file": str(Path(candidate_file).resolve()),
                    "candidate_sha256": self.orchestrate.sha256_file(candidate_file),
                    "input_hash": "a" * 64,
                    "results": [],
                }
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload), encoding="utf-8")
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(self.orchestrate, "_run", side_effect=runner) as run:
                artifact = self.orchestrate._run_sanitizer_gate(
                    state=state,
                    policy=self.orchestrate.resolve_budget("quick"),
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=lambda: 123.0,
                )

        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--mode") + 1], "targeted")
        self.assertEqual(run.call_args.kwargs["hard_timeout"], 123.0)
        benchmark = command[command.index("--") + 1 :]
        self.assertEqual(benchmark[0:2], [sys.executable, "benchmark.py"])
        self.assertIn("--profile-only", benchmark)
        self.assertIn("--N=64", benchmark)
        self.assertEqual(artifact["status"], "passed")
        self.assertEqual(artifact["coverage"], "complete")

    def test_thorough_full_runs_every_outer_finalist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = root / "methods.json"
            methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
            candidates = [
                self._candidate(root, "b1", 1),
                self._candidate(root, "b2", 2),
                self._candidate(root, "b3", 3),
            ]

            def runner(command, **_kwargs):
                output = Path(command[command.index("--out") + 1])
                candidate_file = command[command.index("--candidate-file") + 1]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "passed": True,
                            "coverage": "complete",
                            "mode": "full",
                            "method_ids": [],
                            "selected_tools": [
                                "memcheck",
                                "racecheck",
                                "initcheck",
                                "synccheck",
                            ],
                            "candidate_file": str(Path(candidate_file).resolve()),
                            "candidate_sha256": self.orchestrate.sha256_file(
                                candidate_file
                            ),
                            "input_hash": "b" * 64,
                            "results": [],
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(self.orchestrate, "_run", side_effect=runner) as run:
                artifact = self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "b" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("thorough"),
                    candidates=candidates,
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=lambda: 100.0,
                )

        self.assertEqual(run.call_count, 3)
        self.assertEqual(
            [
                call.args[0][call.args[0].index("--mode") + 1]
                for call in run.call_args_list
            ],
            ["full", "full", "full"],
        )
        self.assertEqual(len(artifact["candidates"]), 3)

    def test_failed_candidate_is_rejected_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate(root, "b1", 1)
            aggregate = self.orchestrate._aggregate_sanitizer_results(
                state={"input_hash": "c" * 64},
                mode="targeted",
                candidates=[candidate],
                reports=[
                    {
                        "status": "failed",
                        "passed": False,
                        "coverage": "complete",
                        "candidate_file": self.orchestrate._candidate_file(candidate),
                        "candidate_sha256": self.orchestrate.sha256_file(
                            self.orchestrate._candidate_file(candidate)
                        ),
                        "input_hash": "c" * 64,
                    }
                ],
            )
            eligible, rejected = self.orchestrate._sanitizer_candidate_outcomes(
                aggregate, [candidate], mode="kernel-only"
            )

        self.assertEqual(eligible, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["status"], "rejected_correctness")
        self.assertEqual(aggregate["status"], "rejected_correctness")

    def test_unavailable_evidence_continues_with_degraded_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate(root, "b1", 1)
            candidate_file = self.orchestrate._candidate_file(candidate)
            aggregate = self.orchestrate._aggregate_sanitizer_results(
                state={"input_hash": "d" * 64},
                mode="targeted",
                candidates=[candidate],
                reports=[
                    {
                        "status": "unavailable",
                        "passed": False,
                        "coverage": "degraded",
                        "candidate_file": candidate_file,
                        "candidate_sha256": self.orchestrate.sha256_file(
                            candidate_file
                        ),
                        "input_hash": "d" * 64,
                    }
                ],
            )
            eligible, rejected = self.orchestrate._sanitizer_candidate_outcomes(
                aggregate, [candidate], mode="kernel-only"
            )

        self.assertEqual(len(eligible), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(aggregate["status"], "unavailable")
        self.assertEqual(aggregate["coverage"], "degraded")

    def test_gate_rejects_methods_symlink_before_reusing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real-methods.json"
            real.write_text(json.dumps({"methods": []}), encoding="utf-8")
            alias = root / "methods.json"
            alias.symlink_to(real)
            candidate = self._candidate(root, "b1", 1)

            with self.assertRaisesRegex(ValueError, "methods.json.*symlink"):
                self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "e" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("quick"),
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=alias,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )

    def test_gate_rejects_report_from_different_method_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = root / "methods.json"
            methods.write_text(
                json.dumps({"methods": [{"id": "latency.async_pipeline"}]}),
                encoding="utf-8",
            )
            candidate = self._candidate(root, "b1", 1)

            def runner(command, **_kwargs):
                output = Path(command[command.index("--out") + 1])
                candidate_file = command[command.index("--candidate-file") + 1]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "passed": True,
                            "coverage": "complete",
                            "mode": "targeted",
                            "method_ids": ["memory.vectorized_access"],
                            "selected_tools": ["memcheck", "initcheck"],
                            "candidate_file": str(Path(candidate_file).resolve()),
                            "candidate_sha256": self.orchestrate.sha256_file(
                                candidate_file
                            ),
                            "input_hash": "f" * 64,
                            "results": [],
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(self.orchestrate, "_run", side_effect=runner):
                with self.assertRaisesRegex(ValueError, "method_ids drifted"):
                    self.orchestrate._run_sanitizer_gate(
                        state={
                            "input_hash": "f" * 64,
                            "dims": {},
                            "ptr_size": 0,
                        },
                        policy=self.orchestrate.resolve_budget("quick"),
                        candidates=[candidate],
                        iter_dir=root,
                        methods_json=methods,
                        benchmark="benchmark.py",
                        hard_timeout=10.0,
                    )


if __name__ == "__main__":
    unittest.main()
