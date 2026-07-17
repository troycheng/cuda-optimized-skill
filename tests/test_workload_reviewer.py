from __future__ import annotations

import importlib.util
import json
import os
import signal
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
    / "workload_reviewer.py"
)


def _load_reviewer():
    module_name = "cuda_optimizer_workload_reviewer_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workload reviewer: {SCRIPT_PATH}")
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


def _request(reviewer) -> dict:
    return reviewer.build_review_request(
        diagnosis={"primary_category": "cpu_data", "confidence": "medium"},
        change_set={"id": "round-1", "scope": "project"},
        redacted_diff="workers: 4 -> 8",
        experiment={"blocks": 5, "primary_metric": "p50_latency_ms"},
        artifact_hashes={"diagnosis.json": "a" * 64},
    )


def _response(request_digest: str, **overrides) -> dict:
    value = {
        "schema_version": "cuda-workload-optimizer/review-v1",
        "request_digest": request_digest,
        "verdict": "support",
        "concerns": [],
        "suggested_experiments": ["repeat with a fixed request trace"],
    }
    value.update(overrides)
    return value


class ReviewerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviewer = _load_reviewer()

    def test_request_is_canonical_digest_bound_and_detached(self) -> None:
        diagnosis = {"primary_category": "kernel"}
        request = self.reviewer.build_review_request(
            diagnosis=diagnosis,
            change_set={"id": "round-1"},
            redacted_diff="diff --git a/a b/a",
            experiment={"blocks": 5},
            artifact_hashes={"diagnosis.json": "b" * 64},
        )
        self.assertEqual(len(request["request_digest"]), 64)
        self.assertEqual(
            request["request_digest"], self.reviewer.request_digest(request)
        )
        diagnosis["primary_category"] = "changed"
        self.assertEqual(request["diagnosis"]["primary_category"], "kernel")

    def test_response_accepts_only_advisory_verdicts_and_matching_digest(self) -> None:
        request = _request(self.reviewer)
        for verdict in ("support", "challenge", "insufficient"):
            value = _response(request["request_digest"], verdict=verdict)
            self.assertEqual(
                self.reviewer.validate_review_response(value, request)["verdict"],
                verdict,
            )
        for verdict in ("execute", "promote", "approve_and_run"):
            with self.subTest(verdict=verdict), self.assertRaises(
                self.reviewer.ReviewerError
            ):
                self.reviewer.validate_review_response(
                    _response(request["request_digest"], verdict=verdict), request
                )

    def test_response_rejects_unknown_keys_callbacks_and_mismatched_digest(self) -> None:
        request = _request(self.reviewer)
        unknown = _response(request["request_digest"])
        unknown["command"] = ["python3", "change.py"]
        with self.assertRaisesRegex(self.reviewer.ReviewerError, "unknown"):
            self.reviewer.validate_review_response(unknown, request)

        mismatch = _response("0" * 64)
        with self.assertRaisesRegex(self.reviewer.ReviewerError, "digest"):
            self.reviewer.validate_review_response(mismatch, request)

    def test_response_concerns_are_closed_and_bounded(self) -> None:
        request = _request(self.reviewer)
        concern = {
            "severity": "medium",
            "category": "experiment",
            "message": "check warmup drift",
        }
        value = _response(request["request_digest"], concerns=[concern])
        self.assertEqual(
            self.reviewer.validate_review_response(value, request)["concerns"],
            [concern],
        )
        invalid = _response(
            request["request_digest"], concerns=[{**concern, "callback": "run"}]
        )
        with self.assertRaisesRegex(self.reviewer.ReviewerError, "unknown"):
            self.reviewer.validate_review_response(invalid, request)


class ReviewerProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviewer = _load_reviewer()

    def _script(self, root: Path, body: str) -> Path:
        path = root / "reviewer.py"
        path.write_text(
            "import json, os, pathlib, sys, time\n" + textwrap.dedent(body),
            encoding="utf-8",
        )
        return path

    def test_cli_receives_json_stdin_in_empty_cwd_and_returns_advice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            script = self._script(
                root,
                """
                request = json.load(sys.stdin)
                assert list(pathlib.Path.cwd().iterdir()) == []
                print(json.dumps({
                    "schema_version": "cuda-workload-optimizer/review-v1",
                    "request_digest": request["request_digest"],
                    "verdict": "challenge",
                    "concerns": [{
                        "severity": "medium",
                        "category": "experiment",
                        "message": "add a warmup stability check"
                    }],
                    "suggested_experiments": ["repeat after warmup"]
                }))
                """,
            )
            request = _request(self.reviewer)
            artifact = self.reviewer.run_reviewer(
                {"argv": [sys.executable, str(script)], "timeout_seconds": 5},
                request,
                root / "run",
            )

            self.assertEqual(artifact["status"], "completed")
            self.assertEqual(artifact["response"]["verdict"], "challenge")
            self.assertEqual(artifact["request_digest"], request["request_digest"])
            stored = json.loads((root / "run" / "review.json").read_text("utf-8"))
            self.assertEqual(stored, artifact)
            self.assertEqual(len(artifact["execution"]["stdin_sha256"]), 64)

    def test_malformed_duplicate_and_oversized_stdout_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cases = {
                "malformed": 'print("not-json")',
                "duplicate": (
                    'request=json.load(sys.stdin); print(\'{"schema_version":"cuda-workload-optimizer/review-v1",'
                    '"request_digest":"%s","request_digest":"%s","verdict":"support",'
                    '"concerns":[],"suggested_experiments":[]}\' % '
                    '(request["request_digest"], request["request_digest"]))'
                ),
                "oversized": 'print("x" * 4096)',
            }
            for name, body in cases.items():
                with self.subTest(name=name):
                    case_root = root / name
                    case_root.mkdir()
                    script = self._script(case_root, body)
                    artifact = self.reviewer.run_reviewer(
                        {"argv": [sys.executable, str(script)], "timeout_seconds": 5},
                        _request(self.reviewer),
                        case_root / "run",
                        output_limit_bytes=512,
                    )
                    self.assertEqual(artifact["status"], "unavailable")
                    self.assertIsNone(artifact["response"])

    def test_timeout_and_missing_cli_degrade_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            script = self._script(root, "time.sleep(30)")
            cases = (
                {"argv": [sys.executable, str(script)], "timeout_seconds": 1},
                {"argv": [str(root / "missing-cli")], "timeout_seconds": 1},
            )
            for config in cases:
                with self.subTest(config=config):
                    started = time.monotonic()
                    artifact = self.reviewer.run_reviewer(
                        config, _request(self.reviewer), root / ("run-" + str(time.time_ns()))
                    )
                    self.assertLess(time.monotonic() - started, 5)
                    self.assertEqual(artifact["status"], "unavailable")
                    self.assertEqual(artifact["response"], None)

    def test_timeout_kills_term_ignoring_reviewer_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            pid_file = root / "child.pid"
            script = self._script(
                root,
                f"""
                import signal
                import subprocess
                child = subprocess.Popen([
                    sys.executable,
                    "-c",
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
                ])
                pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
                time.sleep(30)
                """,
            )

            artifact = self.reviewer.run_reviewer(
                {"argv": [sys.executable, str(script)], "timeout_seconds": 1},
                _request(self.reviewer),
                root / "run",
            )

            self.assertEqual(artifact["status"], "unavailable")
            self.assertTrue(pid_file.exists())
            child_pid = int(pid_file.read_text("utf-8"))
            try:
                self.assertTrue(_wait_pid_gone(child_pid))
            finally:
                if _pid_exists(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    def test_successful_reviewer_cleans_background_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            pid_file = root / "child.pid"
            script = self._script(
                root,
                f"""
                import subprocess
                request = json.load(sys.stdin)
                child = subprocess.Popen([
                    sys.executable, "-c", "import time; time.sleep(30)"
                ])
                pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
                print(json.dumps({{
                    "schema_version": "cuda-workload-optimizer/review-v1",
                    "request_digest": request["request_digest"],
                    "verdict": "support",
                    "concerns": [],
                    "suggested_experiments": []
                }}))
                """,
            )

            artifact = self.reviewer.run_reviewer(
                {"argv": [sys.executable, str(script)], "timeout_seconds": 5},
                _request(self.reviewer),
                root / "run",
            )

            self.assertEqual(artifact["status"], "completed")
            child_pid = int(pid_file.read_text("utf-8"))
            try:
                self.assertTrue(_wait_pid_gone(child_pid))
            finally:
                if _pid_exists(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    def test_stderr_is_bounded_and_secret_values_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            script = self._script(
                root,
                """
                request = json.load(sys.stdin)
                print("API_TOKEN=" + os.environ.get("API_TOKEN", "missing"), file=sys.stderr)
                print("z" * 4096, file=sys.stderr)
                print(json.dumps({
                    "schema_version": "cuda-workload-optimizer/review-v1",
                    "request_digest": request["request_digest"],
                    "verdict": "support",
                    "concerns": [],
                    "suggested_experiments": []
                }))
                """,
            )
            original = os.environ.get("API_TOKEN")
            os.environ["API_TOKEN"] = "never-store-this"
            try:
                artifact = self.reviewer.run_reviewer(
                    {"argv": [sys.executable, str(script)], "timeout_seconds": 5},
                    _request(self.reviewer),
                    root / "run",
                    output_limit_bytes=512,
                )
            finally:
                if original is None:
                    os.environ.pop("API_TOKEN", None)
                else:
                    os.environ["API_TOKEN"] = original
            self.assertEqual(artifact["status"], "completed")
            self.assertTrue(artifact["execution"]["stderr_truncated"])
            self.assertNotIn("never-store-this", json.dumps(artifact))


if __name__ == "__main__":
    unittest.main()
