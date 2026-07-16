from __future__ import annotations

import importlib.util
import hashlib
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tests.test_orchestrate as orchestrate_tests


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


def _sanitizer_report(
    *,
    candidate_file,
    input_hash: str,
    mode: str,
    method_ids,
    selected_tools,
    benchmark_command,
    status: str = "passed",
) -> dict:
    candidate = Path(candidate_file).resolve()
    results = []
    for index, tool in enumerate(selected_tools):
        if status == "unavailable":
            returncode = None
            tool_status = "unavailable"
        elif status == "timed_out" and index == 0:
            returncode = None
            tool_status = "timed_out"
        elif status == "failed" and index == 0:
            returncode = 86
            tool_status = "failed"
        else:
            returncode = 0
            tool_status = "passed"
        results.append(
            {
                "tool": tool,
                "command": [
                    "compute-sanitizer",
                    "--tool",
                    tool,
                    "--error-exitcode",
                    "86",
                    *benchmark_command,
                ],
                "returncode": returncode,
                "stdout": f"{tool} stdout",
                "stderr": f"{tool} stderr",
                "status": tool_status,
            }
        )
    coverage = {
        "passed": "complete",
        "failed": "complete",
        "unavailable": "degraded",
        "timed_out": "incomplete",
        "not_applicable": "not_applicable",
    }[status]
    return {
        "status": status,
        "passed": status in {"passed", "not_applicable"},
        "coverage": coverage,
        "mode": mode,
        "method_ids": list(method_ids),
        "selected_tools": list(selected_tools),
        "candidate_file": str(candidate),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "input_hash": input_hash,
        "results": results,
    }


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
        self.assertIn(
            "racecheck",
            self.sanitize.select_tools(
                ["latency.reduce_sync_count"], mode="targeted", policy=policy
            ),
        )

    def test_policy_rejects_method_ids_that_collide_after_strip(self):
        policy = {
            "schema_version": 1,
            "tools": ["memcheck", "racecheck", "initcheck", "synccheck"],
            "methods": {
                " latency.async_pipeline ": ["memcheck"],
                "latency.async_pipeline": ["racecheck"],
            },
        }
        with self.assertRaisesRegex(ValueError, "collision"):
            self.sanitize.validate_policy(policy)

    def test_policy_path_must_not_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            policy.write_text(json.dumps(self.policy), encoding="utf-8")
            alias = root / "policy-alias.json"
            alias.symlink_to(policy)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.sanitize.load_policy(alias)

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

    def test_each_tool_has_a_finite_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "compute-sanitizer"
            executable.write_text(
                f"#!{sys.executable}\nimport time\ntime.sleep(60)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            result = self.sanitize.run_tools(
                executable=str(executable),
                tools=["memcheck"],
                command=[sys.executable, "bench.py"],
                timeout_seconds=0.05,
            )
        self.assertEqual(result["status"], "timed_out")
        self.assertFalse(result["passed"])
        self.assertEqual(result["coverage"], "incomplete")
        self.assertEqual(result["results"][0]["status"], "timed_out")

    def test_timeout_seconds_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.sanitize.run_tools(
                executable=None,
                tools=["memcheck"],
                command=[sys.executable, "bench.py"],
                timeout_seconds=float("inf"),
            )

    def test_inner_runner_kills_residual_group_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child_pid_path = root / "child.pid"
            executable = root / "compute-sanitizer"
            executable.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import signal
                    import subprocess
                    import sys
                    from pathlib import Path
                    child = subprocess.Popen(
                        [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    Path({str(child_pid_path)!r}).write_text(str(child.pid))
                    raise SystemExit(0)
                    """
                ),
                encoding="utf-8",
            )
            executable.chmod(0o755)
            child_pid = None
            try:
                result = self.sanitize.run_tools(
                    executable=str(executable),
                    tools=["memcheck"],
                    command=[sys.executable, "bench.py"],
                    timeout_seconds=1.0,
                )
                child_pid = int(child_pid_path.read_text("utf-8"))
                self.assertEqual(result["status"], "passed")
                self.assertTrue(_wait_pid_gone(child_pid))
            finally:
                if child_pid is not None and _pid_exists(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    def test_outer_orchestrator_timeout_forwards_kill_to_inner_tool_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            leader_pid_path = root / "leader.pid"
            child_pid_path = root / "child.pid"
            executable = root / "compute-sanitizer"
            executable.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import os
                    import signal
                    import subprocess
                    import sys
                    import time
                    from pathlib import Path
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    Path({str(leader_pid_path)!r}).write_text(str(os.getpid()))
                    child = subprocess.Popen(
                        [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    Path({str(child_pid_path)!r}).write_text(str(child.pid))
                    time.sleep(60)
                    """
                ),
                encoding="utf-8",
            )
            executable.chmod(0o755)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(self.policy), encoding="utf-8")
            methods_path = root / "methods.json"
            methods_path.write_text(
                json.dumps({"methods": [{"id": "latency.async_pipeline"}]}),
                encoding="utf-8",
            )
            candidate = root / "kernel.py"
            candidate.write_text("# candidate\n", encoding="utf-8")
            output = root / "sanitizer.json"
            orchestrate = _load_orchestrate()
            leader_pid = child_pid = None
            try:
                completed = orchestrate._run(
                    [
                        sys.executable,
                        str(SANITIZE_PATH),
                        "--mode", "targeted",
                        "--policy", str(policy_path),
                        "--methods-json", str(methods_path),
                        "--compute-sanitizer", str(executable),
                        "--candidate-file", str(candidate),
                        "--input-hash", "a" * 64,
                        "--out", str(output),
                        "--timeout", "30",
                        "--", sys.executable, "bench.py",
                    ],
                    capture_output=True,
                    hard_timeout=1.0,
                )
                self.assertTrue(completed.timed_out)
                leader_pid = int(leader_pid_path.read_text("utf-8"))
                child_pid = int(child_pid_path.read_text("utf-8"))
                self.assertTrue(_wait_pid_gone(leader_pid))
                self.assertTrue(_wait_pid_gone(child_pid))
            finally:
                for pid in (leader_pid, child_pid):
                    if pid is not None and _pid_exists(pid):
                        os.kill(pid, signal.SIGKILL)

    def test_cli_timeout_default_is_positive_and_zero_is_rejected(self):
        parser = self.sanitize.build_parser()
        args = parser.parse_args(
            ["--mode", "targeted", "--methods-json", "m.json", "--out", "o.json"]
        )
        self.assertGreater(args.timeout, 0.0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--mode", "targeted", "--methods-json", "m.json",
                    "--out", "o.json", "--timeout", "0",
                ]
            )

    def test_result_contract_rejects_incomplete_or_contradictory_tool_evidence(self):
        command = [sys.executable, "bench.py"]
        valid = {
            "status": "passed",
            "passed": True,
            "coverage": "complete",
            "executable": "compute-sanitizer",
            "results": [
                {
                    "tool": "memcheck",
                    "command": [
                        "compute-sanitizer",
                        "--tool",
                        "memcheck",
                        "--error-exitcode",
                        "86",
                        *command,
                    ],
                    "returncode": 0,
                    "stdout": "stdout",
                    "stderr": "stderr",
                    "status": "passed",
                }
            ],
        }
        self.assertEqual(
            self.sanitize.validate_result(
                valid, selected_tools=["memcheck"], command=command
            )["status"],
            "passed",
        )
        invalid = []
        missing_result = json.loads(json.dumps(valid))
        missing_result["results"] = []
        invalid.append(missing_result)
        wrong_tool = json.loads(json.dumps(valid))
        wrong_tool["results"][0]["tool"] = "racecheck"
        invalid.append(wrong_tool)
        missing_exitcode = json.loads(json.dumps(valid))
        missing_exitcode["results"][0]["command"] = [
            "compute-sanitizer",
            "--tool",
            "memcheck",
            *command,
        ]
        invalid.append(missing_exitcode)
        contradictory = json.loads(json.dumps(valid))
        contradictory["passed"] = False
        invalid.append(contradictory)
        missing_stream = json.loads(json.dumps(valid))
        del missing_stream["results"][0]["stderr"]
        invalid.append(missing_stream)
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.sanitize.validate_result(
                    payload, selected_tools=["memcheck"], command=command
                )

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
            output_mode = os.stat(output).st_mode & 0o777

        self.assertEqual(completed.returncode, 86)
        self.assertEqual(artifact["status"], "failed")
        self.assertEqual(
            [item["tool"] for item in artifact["results"]],
            ["memcheck", "racecheck", "synccheck"],
        )
        self.assertNotIn("fake sanitizer stderr", completed.stdout)
        self.assertEqual(output_mode, 0o600)

    def test_atomic_writer_fsyncs_file_and_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            self.sanitize.os, "fsync", wraps=os.fsync
        ) as fsync:
            self.sanitize._atomic_write_json(Path(tmp) / "out.json", {"ok": True})
        self.assertGreaterEqual(fsync.call_count, 2)

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

    def _completed_full_gate(self, root: Path, input_hash: str):
        methods = root / "methods.json"
        methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
        candidate = self._candidate(root, "b1", 1)
        state = {"input_hash": input_hash, "dims": {}, "ptr_size": 0}
        policy = self.orchestrate.resolve_budget("thorough")

        def runner(command, **_kwargs):
            child_output = Path(command[command.index("--out") + 1])
            candidate_file = command[command.index("--candidate-file") + 1]
            child_output.parent.mkdir(parents=True, exist_ok=True)
            child_output.write_text(
                json.dumps(
                    _sanitizer_report(
                        candidate_file=candidate_file,
                        input_hash=input_hash,
                        mode="full",
                        method_ids=[],
                        selected_tools=[
                            "memcheck", "racecheck", "initcheck", "synccheck"
                        ],
                        benchmark_command=command[command.index("--") + 1 :],
                    )
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(
                returncode=0, stdout="", stderr="", timed_out=False
            )

        with mock.patch.object(self.orchestrate, "_run", side_effect=runner):
            aggregate = self.orchestrate._run_sanitizer_gate(
                state=state,
                policy=policy,
                candidates=[candidate],
                iter_dir=root,
                methods_json=methods,
                benchmark="benchmark.py",
                hard_timeout=10.0,
            )
        return state, policy, methods, candidate, aggregate

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
                payload = _sanitizer_report(
                    candidate_file=candidate_file,
                    input_hash="a" * 64,
                    mode="targeted",
                    method_ids=["latency.async_pipeline"],
                    selected_tools=["memcheck", "racecheck", "synccheck"],
                    benchmark_command=command[command.index("--") + 1 :],
                )
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
            expected_methods_hash = self.orchestrate.sha256_file(methods)
            expected_policy_hash = self.orchestrate.sha256_file(POLICY_PATH)

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
        self.assertEqual(artifact["methods_sha256"], expected_methods_hash)
        self.assertEqual(artifact["policy_sha256"], expected_policy_hash)
        self.assertEqual(artifact["method_ids"], ["latency.async_pipeline"])
        self.assertEqual(
            artifact["selected_tools"], ["memcheck", "racecheck", "synccheck"]
        )

    def test_aggregate_loader_rejects_methods_drift_and_derived_status_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            methods = root / "methods.json"
            methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
            candidate = self._candidate(root, "b1", 1)
            state = {"input_hash": "9" * 64, "dims": {}, "ptr_size": 0}

            def runner(command, **_kwargs):
                output = Path(command[command.index("--out") + 1])
                candidate_file = command[command.index("--candidate-file") + 1]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        _sanitizer_report(
                            candidate_file=candidate_file,
                            input_hash=state["input_hash"],
                            mode="full",
                            method_ids=[],
                            selected_tools=[
                                "memcheck", "racecheck", "initcheck", "synccheck"
                            ],
                            benchmark_command=command[command.index("--") + 1 :],
                        )
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            policy = self.orchestrate.resolve_budget("thorough")
            with mock.patch.object(self.orchestrate, "_run", side_effect=runner):
                self.orchestrate._run_sanitizer_gate(
                    state=state,
                    policy=policy,
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            methods.write_text(
                json.dumps({"methods": [{"id": "latency.async_pipeline"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "methods_sha256"):
                self.orchestrate._load_sanitizer_aggregate(
                    root,
                    state=state,
                    policy=policy,
                    methods_json=methods,
                    candidates=[candidate],
                )
            methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
            aggregate_path = root / "sanitizer.json"
            aggregate = json.loads(aggregate_path.read_text("utf-8"))
            aggregate["status"] = "rejected_correctness"
            aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "derived status"):
                self.orchestrate._load_sanitizer_aggregate(
                    root,
                    state=state,
                    policy=policy,
                    methods_json=methods,
                    candidates=[candidate],
                )

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
                        _sanitizer_report(
                            candidate_file=candidate_file,
                            input_hash="b" * 64,
                            mode="full",
                            method_ids=[],
                            selected_tools=[
                                "memcheck",
                                "racecheck",
                                "initcheck",
                                "synccheck",
                            ],
                            benchmark_command=command[command.index("--") + 1 :],
                        )
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

    def test_full_candidates_with_identical_bytes_keep_distinct_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = root / "methods.json"
            methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
            first = self._candidate(root, "b1", 1)
            second = self._candidate(root, "b2", 2)
            Path(second["kernel"]).write_bytes(Path(first["kernel"]).read_bytes())

            def runner(command, **_kwargs):
                output = Path(command[command.index("--out") + 1])
                candidate_file = command[command.index("--candidate-file") + 1]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        _sanitizer_report(
                            candidate_file=candidate_file,
                            input_hash="1" * 64,
                            mode="full",
                            method_ids=[],
                            selected_tools=[
                                "memcheck",
                                "racecheck",
                                "initcheck",
                                "synccheck",
                            ],
                            benchmark_command=command[command.index("--") + 1 :],
                        )
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(self.orchestrate, "_run", side_effect=runner) as run:
                aggregate = self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "1" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("thorough"),
                    candidates=[first, second],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            eligible, rejected = self.orchestrate._sanitizer_candidate_outcomes(
                aggregate, [first, second], mode="full"
            )
            artifact_hashes_match = all(
                item["artifact_sha256"]
                == hashlib.sha256(Path(item["artifact"]).read_bytes()).hexdigest()
                for item in aggregate["candidates"]
            )

        self.assertEqual(run.call_count, 2)
        artifacts = [item["artifact"] for item in aggregate["candidates"]]
        self.assertEqual(len(set(artifacts)), 2)
        self.assertTrue(artifact_hashes_match)
        self.assertEqual(len(eligible), 2)
        self.assertEqual(rejected, [])

    def test_aggregate_loader_rejects_missing_or_tampered_raw_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            methods = root / "methods.json"
            methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
            candidate = self._candidate(root, "b1", 1)
            state = {"input_hash": "8" * 64, "dims": {}, "ptr_size": 0}
            policy = self.orchestrate.resolve_budget("thorough")

            def runner(command, **_kwargs):
                output = Path(command[command.index("--out") + 1])
                candidate_file = command[command.index("--candidate-file") + 1]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        _sanitizer_report(
                            candidate_file=candidate_file,
                            input_hash=state["input_hash"],
                            mode="full",
                            method_ids=[],
                            selected_tools=[
                                "memcheck", "racecheck", "initcheck", "synccheck"
                            ],
                            benchmark_command=command[command.index("--") + 1 :],
                        )
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(self.orchestrate, "_run", side_effect=runner):
                aggregate = self.orchestrate._run_sanitizer_gate(
                    state=state,
                    policy=policy,
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            artifact = Path(aggregate["candidates"][0]["artifact"])
            original = artifact.read_bytes()
            artifact.unlink()
            with self.assertRaisesRegex(ValueError, "artifact"):
                self.orchestrate._load_sanitizer_aggregate(
                    root,
                    state=state,
                    policy=policy,
                    methods_json=methods,
                    candidates=[candidate],
                )
            artifact.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "artifact"):
                self.orchestrate._load_sanitizer_aggregate(
                    root,
                    state=state,
                    policy=policy,
                    methods_json=methods,
                    candidates=[candidate],
                )

    def test_failed_candidate_is_rejected_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate(root, "b1", 1)
            candidate_file = self.orchestrate._candidate_file(candidate)
            aggregate = self.orchestrate._aggregate_sanitizer_results(
                state={"input_hash": "c" * 64},
                mode="targeted",
                candidates=[candidate],
                reports=[
                    _sanitizer_report(
                        candidate_file=candidate_file,
                        input_hash="c" * 64,
                        mode="targeted",
                        method_ids=["latency.reduce_sync_count"],
                        selected_tools=["racecheck"],
                        benchmark_command=[sys.executable, "benchmark.py"],
                        status="failed",
                    )
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
                    _sanitizer_report(
                        candidate_file=candidate_file,
                        input_hash="d" * 64,
                        mode="targeted",
                        method_ids=["latency.reduce_sync_count"],
                        selected_tools=["racecheck"],
                        benchmark_command=[sys.executable, "benchmark.py"],
                        status="unavailable",
                    )
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
                        _sanitizer_report(
                            candidate_file=candidate_file,
                            input_hash="f" * 64,
                            mode="targeted",
                            method_ids=["memory.vectorized_access"],
                            selected_tools=["memcheck", "initcheck"],
                            benchmark_command=command[command.index("--") + 1 :],
                        )
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

    def test_mixed_failed_and_eligible_finalists_are_partial_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._candidate(root, "b1", 1)
            second = self._candidate(root, "b2", 2)
            reports = [
                _sanitizer_report(
                    candidate_file=first["kernel"],
                    input_hash="2" * 64,
                    mode="full",
                    method_ids=[],
                    selected_tools=[
                        "memcheck",
                        "racecheck",
                        "initcheck",
                        "synccheck",
                    ],
                    benchmark_command=[sys.executable, "benchmark.py"],
                    status="failed",
                ),
                _sanitizer_report(
                    candidate_file=second["kernel"],
                    input_hash="2" * 64,
                    mode="full",
                    method_ids=[],
                    selected_tools=[
                        "memcheck",
                        "racecheck",
                        "initcheck",
                        "synccheck",
                    ],
                    benchmark_command=[sys.executable, "benchmark.py"],
                    status="passed",
                ),
            ]
            aggregate = self.orchestrate._aggregate_sanitizer_results(
                state={"input_hash": "2" * 64},
                mode="full",
                candidates=[first, second],
                reports=reports,
            )
            eligible, rejected = self.orchestrate._sanitizer_candidate_outcomes(
                aggregate, [first, second], mode="full"
            )

        self.assertEqual(aggregate["status"], "partial_rejection")
        self.assertTrue(aggregate["passed"])
        self.assertEqual(len(eligible), 1)
        self.assertEqual(len(rejected), 1)

    def test_sanitizer_timeout_resume_reuses_completed_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = root / "methods.json"
            methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
            candidates = [
                self._candidate(root, "b1", 1),
                self._candidate(root, "b2", 2),
            ]
            first_path = self.orchestrate._candidate_file(candidates[0])
            second_path = self.orchestrate._candidate_file(candidates[1])
            calls = []
            child_outputs = []
            should_timeout = {str(Path(candidates[1]["kernel"]).resolve()): True}

            def runner(command, **_kwargs):
                candidate_file = command[command.index("--candidate-file") + 1]
                calls.append(candidate_file)
                if should_timeout.get(candidate_file):
                    should_timeout[candidate_file] = False
                    output = Path(command[command.index("--out") + 1])
                    output.write_text(
                        json.dumps(
                            _sanitizer_report(
                                candidate_file=candidate_file,
                                input_hash="3" * 64,
                                mode="full",
                                method_ids=[],
                                selected_tools=[
                                    "memcheck", "racecheck", "initcheck", "synccheck"
                                ],
                                benchmark_command=command[command.index("--") + 1 :],
                                status="timed_out",
                            )
                        ),
                        encoding="utf-8",
                    )
                    return SimpleNamespace(
                        returncode=124, stdout="", stderr="", timed_out=False
                    )
                output = Path(command[command.index("--out") + 1])
                child_outputs.append(output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        _sanitizer_report(
                            candidate_file=candidate_file,
                            input_hash="3" * 64,
                            mode="full",
                            method_ids=[],
                            selected_tools=[
                                "memcheck",
                                "racecheck",
                                "initcheck",
                                "synccheck",
                            ],
                            benchmark_command=command[command.index("--") + 1 :],
                        )
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(self.orchestrate, "_run", side_effect=runner):
                interrupted = self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "3" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("thorough"),
                    candidates=candidates,
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
                completed = self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "3" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("thorough"),
                    candidates=candidates,
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )

        first_path = str(Path(candidates[0]["kernel"]).resolve())
        second_path = str(Path(candidates[1]["kernel"]).resolve())
        self.assertTrue(interrupted["timed_out"])
        self.assertEqual(completed["status"], "passed")
        self.assertEqual(calls.count(first_path), 1)
        self.assertEqual(calls.count(second_path), 2)
        self.assertTrue(
            all(path.name.endswith(".unbound.json") for path in child_outputs)
        )

    def test_resume_discards_valid_raw_artifact_missing_parent_binding_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            methods = root / "methods.json"
            methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
            candidates = [
                self._candidate(root, "b1", 1),
                self._candidate(root, "b2", 2),
            ]
            first_path = self.orchestrate._candidate_file(candidates[0])
            second_path = self.orchestrate._candidate_file(candidates[1])
            calls = []
            interrupt_second = {"value": True}

            def runner(command, **_kwargs):
                candidate_file = command[command.index("--candidate-file") + 1]
                calls.append(candidate_file)
                output = Path(command[command.index("--out") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        _sanitizer_report(
                            candidate_file=candidate_file,
                            input_hash="5" * 64,
                            mode="full",
                            method_ids=[],
                            selected_tools=[
                                "memcheck", "racecheck", "initcheck", "synccheck"
                            ],
                            benchmark_command=command[command.index("--") + 1 :],
                        )
                    ),
                    encoding="utf-8",
                )
                if candidate_file == self.orchestrate._candidate_file(
                    candidates[1]
                ) and interrupt_second["value"]:
                    interrupt_second["value"] = False
                    return SimpleNamespace(
                        returncode=124, stdout="", stderr="", timed_out=True
                    )
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(self.orchestrate, "_run", side_effect=runner):
                interrupted = self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "5" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("thorough"),
                    candidates=candidates,
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
                completed = self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "5" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("thorough"),
                    candidates=candidates,
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )

        self.assertTrue(interrupted["timed_out"])
        self.assertEqual(completed["status"], "passed")
        self.assertEqual(calls.count(first_path), 1)
        self.assertEqual(calls.count(second_path), 2)

    def test_resume_rejects_raw_artifact_with_wrong_parent_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            methods = root / "methods.json"
            methods.write_text(json.dumps({"methods": []}), encoding="utf-8")
            candidate = self._candidate(root, "b1", 1)

            def runner(command, **_kwargs):
                output = Path(command[command.index("--out") + 1])
                candidate_file = command[command.index("--candidate-file") + 1]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        _sanitizer_report(
                            candidate_file=candidate_file,
                            input_hash="6" * 64,
                            mode="full",
                            method_ids=[],
                            selected_tools=[
                                "memcheck", "racecheck", "initcheck", "synccheck"
                            ],
                            benchmark_command=command[command.index("--") + 1 :],
                        )
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(self.orchestrate, "_run", side_effect=runner):
                aggregate = self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "6" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("thorough"),
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            raw = Path(aggregate["candidates"][0]["artifact"])
            payload = json.loads(raw.read_text("utf-8"))
            payload["methods_sha256"] = "0" * 64
            raw.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "methods_sha256 drifted"):
                self.orchestrate._run_sanitizer_gate(
                    state={"input_hash": "6" * 64, "dims": {}, "ptr_size": 0},
                    policy=self.orchestrate.resolve_budget("thorough"),
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )

    def test_final_missing_both_parent_bindings_is_rejected_without_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            state, policy, methods, candidate, aggregate = self._completed_full_gate(
                root, "7" * 64
            )
            final = Path(aggregate["candidates"][0]["artifact"])
            payload = json.loads(final.read_text("utf-8"))
            payload.pop("methods_sha256")
            payload.pop("policy_sha256")
            final.write_text(json.dumps(payload), encoding="utf-8")
            runner = mock.Mock()
            with mock.patch.object(
                self.orchestrate, "_run", runner
            ), self.assertRaisesRegex(ValueError, "parent binding"):
                self.orchestrate._run_sanitizer_gate(
                    state=state,
                    policy=policy,
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            runner.assert_not_called()

    def test_final_is_authoritative_when_unbound_also_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            state, policy, methods, candidate, aggregate = self._completed_full_gate(
                root, "8" * 64
            )
            final = Path(aggregate["candidates"][0]["artifact"])
            unbound = final.with_name(f"{final.stem}.unbound.json")
            unbound.write_text("{}", encoding="utf-8")
            runner = mock.Mock()
            with mock.patch.object(self.orchestrate, "_run", runner):
                reused = self.orchestrate._run_sanitizer_gate(
                    state=state,
                    policy=policy,
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            self.assertEqual(reused["status"], "passed")
            runner.assert_not_called()
            self.assertFalse(unbound.exists())
            unbound.write_text("{}", encoding="utf-8")
            payload = json.loads(final.read_text("utf-8"))
            payload["policy_sha256"] = "0" * 64
            final.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                self.orchestrate, "_run", runner
            ), self.assertRaisesRegex(ValueError, "policy_sha256 drifted"):
                self.orchestrate._run_sanitizer_gate(
                    state=state,
                    policy=policy,
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            runner.assert_not_called()
            self.assertTrue(final.is_file())
            self.assertTrue(unbound.is_file())

    def test_unbound_symlink_and_non_regular_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            state, policy, methods, candidate, aggregate = self._completed_full_gate(
                root, "9" * 64
            )
            final = Path(aggregate["candidates"][0]["artifact"])
            unbound = final.with_name(f"{final.stem}.unbound.json")
            final.unlink()
            target = root / "outside.json"
            target.write_text("{}", encoding="utf-8")
            unbound.symlink_to(target)
            runner = mock.Mock()
            with mock.patch.object(
                self.orchestrate, "_run", runner
            ), self.assertRaisesRegex(ValueError, "unbound.*symlink"):
                self.orchestrate._run_sanitizer_gate(
                    state=state,
                    policy=policy,
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            unbound.unlink()
            unbound.mkdir()
            with mock.patch.object(
                self.orchestrate, "_run", runner
            ), self.assertRaisesRegex(ValueError, "unbound.*regular"):
                self.orchestrate._run_sanitizer_gate(
                    state=state,
                    policy=policy,
                    candidates=[candidate],
                    iter_dir=root,
                    methods_json=methods,
                    benchmark="benchmark.py",
                    hard_timeout=10.0,
                )
            runner.assert_not_called()

    def test_profile_binding_requires_safe_unchanged_ncu_top(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            kernel = root / "kernel.py"
            kernel.write_text("# kernel\n", encoding="utf-8")
            state = {"input_hash": "4" * 64}
            top = root / "ncu_top.json"
            top.write_text(
                json.dumps({"profiled_file": str(kernel), "axes": {}}),
                encoding="utf-8",
            )
            self.orchestrate._write_candidate_profile_binding(
                root,
                state=state,
                candidate_id="1",
                kernel=str(kernel),
                returncode=0,
            )
            self.assertTrue(
                self.orchestrate._profile_binding_matches(
                    root,
                    state=state,
                    candidate_id="1",
                    kernel=str(kernel),
                )
            )
            original = top.read_text("utf-8")
            top.write_text(original + " ", encoding="utf-8")
            self.assertFalse(
                self.orchestrate._profile_binding_matches(
                    root,
                    state=state,
                    candidate_id="1",
                    kernel=str(kernel),
                )
            )
            top.unlink()
            self.assertFalse(
                self.orchestrate._profile_binding_matches(
                    root,
                    state=state,
                    candidate_id="1",
                    kernel=str(kernel),
                )
            )


class SanitizerLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = orchestrate_tests.LifecycleIntegrationTests()
        fixture.setUp()
        self.fixture = fixture
        self.orchestrate = fixture.orchestrate

    def _run_case(self, root: Path, *, sanitizer_status: str):
        run_dir, state_path = self.fixture._setup(root)
        self.fixture._write_winner_artifacts(run_dir)
        iter_dir = run_dir / "iterv1"
        (iter_dir / "methods.json").write_text(
            json.dumps(
                {"methods": [{"id": "latency.reduce_sync_count", "name": "sync"}]}
            ),
            encoding="utf-8",
        )

        def gate(*, state, policy, candidates, **_kwargs):
            candidate = candidates[0]
            candidate_file = self.orchestrate._candidate_file(candidate)
            report = _sanitizer_report(
                candidate_file=candidate_file,
                input_hash=state["input_hash"],
                mode=policy.sanitizer_mode,
                method_ids=["latency.reduce_sync_count"],
                selected_tools=["racecheck"],
                benchmark_command=[sys.executable, "benchmark.py"],
                status=sanitizer_status,
            )
            report.update(
                methods_sha256=self.orchestrate.sha256_file(iter_dir / "methods.json"),
                policy_sha256=self.orchestrate.sha256_file(POLICY_PATH),
            )
            raw = iter_dir / "sanitizer" / "fixture.json"
            self.orchestrate.atomic_write_json(raw, report)
            report["artifact"] = str(raw.resolve())
            aggregate = self.orchestrate._aggregate_sanitizer_results(
                state=state,
                mode=policy.sanitizer_mode,
                candidates=candidates,
                reports=[report],
            )
            aggregate.update(
                methods_sha256=self.orchestrate.sha256_file(iter_dir / "methods.json"),
                policy_sha256=self.orchestrate.sha256_file(POLICY_PATH),
                method_ids=["latency.reduce_sync_count"],
                selected_tools=["racecheck"],
            )
            self.orchestrate.atomic_write_json(iter_dir / "sanitizer.json", aggregate)
            return aggregate

        def runner(command, **_kwargs):
            if Path(command[1]).name == "profile_ncu.py":
                kernel = next(iter(iter_dir.glob("kernel.*")))
                self.orchestrate.atomic_write_json(
                    iter_dir / "ncu_top.json",
                    {"profiled_file": str(kernel.resolve()), "axes": {}},
                )
            if Path(command[1]).name == "state.py":
                return subprocess.run(command, capture_output=True, text=True)
            return SimpleNamespace(
                returncode=0, stdout="", stderr="", timed_out=False
            )

        evaluator = mock.Mock(
            side_effect=AssertionError("workload evaluator must not run")
        )
        with mock.patch.object(
            self.orchestrate, "_run_sanitizer_gate", side_effect=gate
        ), mock.patch.object(
            self.orchestrate, "_run", side_effect=runner
        ), mock.patch.object(
            self.orchestrate, "evaluate_outer_candidate", evaluator
        ), contextlib.redirect_stdout(io.StringIO()):
            self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))
        return run_dir, state_path, evaluator

    def test_all_sanitizer_failed_skips_workload_and_persists_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path, evaluator = self._run_case(
                Path(tmp), sanitizer_status="failed"
            )
            state = json.loads(state_path.read_text("utf-8"))
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))
            workload_result = json.loads(
                (run_dir / "iterv1" / "workload_result.json").read_text("utf-8")
            )

        evaluator.assert_not_called()
        self.assertEqual(workload_result["decision"]["status"], "rejected_correctness")
        self.assertEqual(checkpoint["candidate_status"], "rejected_correctness")
        self.assertEqual(state["history"][-1]["status"], "rejected_correctness")

    def test_rc124_timeout_checkpoint_resumes_at_sanitizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path = self.fixture._setup(Path(tmp))
            self.fixture._write_winner_artifacts(run_dir)
            iter_dir = run_dir / "iterv1"
            profile_calls = []

            def runner(command, **_kwargs):
                script = Path(command[1]).name
                if script == "profile_ncu.py":
                    kernel = next(iter(iter_dir.glob("kernel.*")))
                    profile_calls.append(str(kernel))
                    self.orchestrate.atomic_write_json(
                        iter_dir / "ncu_top.json",
                        {"profiled_file": str(kernel.resolve()), "axes": {}},
                    )
                if script == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), mock.patch.object(
                self.orchestrate,
                "_run_sanitizer_gate",
                return_value={"timed_out": True, "candidate_id": "1"},
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))
            interrupted = json.loads(
                (run_dir / "checkpoint.json").read_text("utf-8")
            )
            self.assertEqual(interrupted["stage"], "candidate_sanitizer")
            self.assertEqual(interrupted["status"], "budget_exhausted")
            self.assertEqual(
                self.orchestrate.resume(
                    interrupted,
                    input_hash=json.loads(state_path.read_text("utf-8"))["input_hash"],
                )["next_stage"],
                "candidate_sanitizer",
            )
            with mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))
            completed = json.loads(
                (run_dir / "checkpoint.json").read_text("utf-8")
            )

        self.assertEqual(len(profile_calls), 1)
        self.assertEqual(completed["stage"], "decision")

    def test_unavailable_completes_with_persisted_degraded_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, state_path = self.fixture._setup(Path(tmp))
            self.fixture._write_winner_artifacts(run_dir)
            iter_dir = run_dir / "iterv1"
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": [{"id": "latency.reduce_sync_count"}]}),
                encoding="utf-8",
            )

            def gate(*, state, policy, candidates, **_kwargs):
                candidate = candidates[0]
                candidate_file = self.orchestrate._candidate_file(candidate)
                report = _sanitizer_report(
                    candidate_file=candidate_file,
                    input_hash=state["input_hash"],
                    mode=policy.sanitizer_mode,
                    method_ids=["latency.reduce_sync_count"],
                    selected_tools=["racecheck"],
                    benchmark_command=[sys.executable, "benchmark.py"],
                    status="unavailable",
                )
                report.update(
                    methods_sha256=self.orchestrate.sha256_file(
                        iter_dir / "methods.json"
                    ),
                    policy_sha256=self.orchestrate.sha256_file(POLICY_PATH),
                )
                raw = iter_dir / "sanitizer" / "fixture.json"
                self.orchestrate.atomic_write_json(raw, report)
                report["artifact"] = str(raw.resolve())
                aggregate = self.orchestrate._aggregate_sanitizer_results(
                    state=state,
                    mode=policy.sanitizer_mode,
                    candidates=candidates,
                    reports=[report],
                )
                aggregate.update(
                    methods_sha256=self.orchestrate.sha256_file(
                        iter_dir / "methods.json"
                    ),
                    policy_sha256=self.orchestrate.sha256_file(POLICY_PATH),
                    method_ids=["latency.reduce_sync_count"],
                    selected_tools=["racecheck"],
                )
                self.orchestrate.atomic_write_json(iter_dir / "sanitizer.json", aggregate)
                return aggregate

            def runner(command, **_kwargs):
                if Path(command[1]).name == "profile_ncu.py":
                    kernel = next(iter(iter_dir.glob("kernel.*")))
                    self.orchestrate.atomic_write_json(
                        iter_dir / "ncu_top.json",
                        {"profiled_file": str(kernel.resolve()), "axes": {}},
                    )
                if Path(command[1]).name == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(
                self.orchestrate, "_run_sanitizer_gate", side_effect=gate
            ), mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))
            state = json.loads(state_path.read_text("utf-8"))
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text("utf-8"))

        self.assertTrue(state["sanitizer_coverage_degraded"])
        self.assertEqual(state["sanitizer_coverage"], "degraded")
        self.assertEqual(
            checkpoint["stage_evidence"]["candidate_sanitizer"]["coverage"],
            "degraded",
        )

    def test_sanitizer_replacement_reprofiles_and_rebinds_ncu_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _state_path = self.fixture._setup(Path(tmp))
            self.fixture._write_winner_artifacts(run_dir)
            iter_dir = run_dir / "iterv1"
            branch_dir = iter_dir / "branches"
            first_file = branch_dir / "b1" / "kernel.py"
            second_file = branch_dir / "b2" / "kernel.py"
            first_file.parent.mkdir(parents=True)
            second_file.parent.mkdir(parents=True)
            first_file.write_text("# sanitizer rejects first\n", encoding="utf-8")
            second_file.write_text("# sanitizer accepts second\n", encoding="utf-8")
            expected_profiled_hashes = [
                self.orchestrate.sha256_file(first_file),
                self.orchestrate.sha256_file(second_file),
            ]
            (iter_dir / "methods.json").write_text(
                json.dumps({"methods": [{"id": "latency.reduce_sync_count"}]}),
                encoding="utf-8",
            )
            statistics = self.fixture._statistics()
            first = {
                "status": "confirmed_win",
                "kernel": str(first_file),
                "branch_index": 1,
                "statistics": statistics,
            }
            second = {
                "status": "confirmed_win",
                "kernel": str(second_file),
                "branch_index": 2,
                "statistics": {**statistics, "estimate_pct": 2.0},
            }

            def select(_payload, **_kwargs):
                return first, [first, second]

            def gate(*, state, policy, candidates, **_kwargs):
                reports = []
                for index, candidate in enumerate(candidates):
                    reports.append(
                        _sanitizer_report(
                            candidate_file=self.orchestrate._candidate_file(candidate),
                            input_hash=state["input_hash"],
                            mode=policy.sanitizer_mode,
                            method_ids=["latency.reduce_sync_count"],
                            selected_tools=["racecheck"],
                            benchmark_command=[sys.executable, "benchmark.py"],
                            status="failed" if index == 0 else "passed",
                        )
                    )
                aggregate = self.orchestrate._aggregate_sanitizer_results(
                    state=state,
                    mode=policy.sanitizer_mode,
                    candidates=candidates,
                    reports=reports,
                )
                self.orchestrate.atomic_write_json(
                    iter_dir / "sanitizer.json", aggregate
                )
                return aggregate

            profiled_hashes = []

            def runner(command, **_kwargs):
                script = Path(command[1]).name
                if script == "profile_ncu.py":
                    kernel = next(iter(iter_dir.glob("kernel.*")))
                    profiled_hashes.append(self.orchestrate.sha256_file(kernel))
                    self.orchestrate.atomic_write_json(
                        iter_dir / "ncu_top.json",
                        {"profiled_file": str(kernel.resolve()), "axes": {}},
                    )
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr="", timed_out=False
                    )
                if script == "sass_check.py":
                    raise RuntimeError("stop after sanitizer reprofile")
                if script == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(
                self.orchestrate, "_select_lifecycle_candidate", side_effect=select
            ), mock.patch.object(
                self.orchestrate, "_run_sanitizer_gate", side_effect=gate
            ), mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), self.assertRaisesRegex(RuntimeError, "stop after sanitizer reprofile"):
                self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))

            binding = json.loads(
                (iter_dir / "candidate_profile_binding.json").read_text("utf-8")
            )
            selection = json.loads(
                (iter_dir / "selected_candidate.json").read_text("utf-8")
            )

        self.assertEqual(
            profiled_hashes,
            expected_profiled_hashes,
        )
        self.assertEqual(selection["candidate_id"], "2")
        self.assertEqual(binding["candidate_id"], "2")
        self.assertEqual(binding["candidate_sha256"], profiled_hashes[-1])

    def test_candidate_sanitizer_resume_reprofiles_when_ncu_top_was_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _state_path = self.fixture._setup(Path(tmp))
            self.fixture._write_winner_artifacts(run_dir)
            iter_dir = run_dir / "iterv1"
            profile_calls = []
            stop_once = {"sass": True}

            def runner(command, **_kwargs):
                script = Path(command[1]).name
                if script == "profile_ncu.py":
                    kernel = next(iter(iter_dir.glob("kernel.*")))
                    profile_calls.append(self.orchestrate.sha256_file(kernel))
                    self.orchestrate.atomic_write_json(
                        iter_dir / "ncu_top.json",
                        {"profiled_file": str(kernel.resolve()), "axes": {}},
                    )
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr="", timed_out=False
                    )
                if script == "sass_check.py" and stop_once["sass"]:
                    stop_once["sass"] = False
                    raise RuntimeError("stop in candidate sanitizer")
                if script == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), self.assertRaisesRegex(RuntimeError, "stop in candidate sanitizer"):
                self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))
            (iter_dir / "ncu_top.json").unlink()
            with mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), contextlib.redirect_stdout(io.StringIO()):
                self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))

            binding = json.loads(
                (iter_dir / "candidate_profile_binding.json").read_text("utf-8")
            )
            final_top_hash = self.orchestrate.sha256_file(iter_dir / "ncu_top.json")

        self.assertEqual(len(profile_calls), 2)
        self.assertEqual(binding["ncu_top_sha256"], final_top_hash)

    def test_workload_stage_rejects_deleted_ncu_top_without_running_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _state_path = self.fixture._setup(Path(tmp))
            self.fixture._write_winner_artifacts(run_dir)
            iter_dir = run_dir / "iterv1"

            def runner(command, **_kwargs):
                script = Path(command[1]).name
                if script == "profile_ncu.py":
                    kernel = next(iter(iter_dir.glob("kernel.*")))
                    self.orchestrate.atomic_write_json(
                        iter_dir / "ncu_top.json",
                        {"profiled_file": str(kernel.resolve()), "axes": {}},
                    )
                if script == "state.py":
                    return subprocess.run(command, capture_output=True, text=True)
                return SimpleNamespace(
                    returncode=0, stdout="", stderr="", timed_out=False
                )

            with mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), mock.patch.object(
                self.orchestrate,
                "_load_sanitizer_aggregate",
                side_effect=RuntimeError("stop before workload load"),
            ), self.assertRaisesRegex(RuntimeError, "stop before workload load"):
                self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))
            (iter_dir / "ncu_top.json").unlink()
            workload = mock.Mock(side_effect=AssertionError("workload must not run"))
            with mock.patch.object(
                self.orchestrate, "_run", side_effect=runner
            ), mock.patch.object(
                self.orchestrate, "evaluate_outer_candidate", workload
            ), self.assertRaisesRegex(ValueError, "profile evidence"):
                self.orchestrate.cmd_close_iter(self.fixture._close_args(run_dir))

        workload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
