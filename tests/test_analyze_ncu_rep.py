from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
SCRIPT = SCRIPTS / "analyze_ncu_rep.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_ncu_rep_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalyzeNcuRepInputTests(unittest.TestCase):
    def test_strict_positive_numbers_reject_bools_zero_and_non_finite(self) -> None:
        module = _load()
        for value in (0, -1, True, "1", 1.2):
            with self.assertRaises((TypeError, ValueError)):
                module._strict_positive_int(value)
        self.assertEqual(module._strict_positive_int(2), 2)
        for value in (0, -1, True, float("nan"), float("inf"), "1"):
            with self.assertRaises((TypeError, ValueError)):
                module._strict_positive_float(value)
        self.assertEqual(module._strict_positive_float(1.5), 1.5)

    def test_capture_regular_file_rejects_missing_directory_fifo_and_symlinks(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regular = root / "regular.rep"
            regular.write_bytes(b"report")
            directory = root / "directory"
            directory.mkdir()
            fifo = root / "report.fifo"
            os.mkfifo(fifo)
            leaf_link = root / "leaf-link.rep"
            leaf_link.symlink_to(regular)
            parent = root / "real-parent"
            parent.mkdir()
            (parent / "child.rep").write_bytes(b"child")
            parent_link = root / "parent-link"
            parent_link.symlink_to(parent, target_is_directory=True)
            for path in (root / "missing.rep", directory, fifo, leaf_link, parent_link / "child.rep"):
                with self.assertRaises(ValueError):
                    module.capture_regular_file(path)
            captured = module.capture_regular_file(regular)
            self.assertEqual(captured["size"], 6)
            self.assertEqual(captured["sha256"], __import__("hashlib").sha256(b"report").hexdigest())
            self.assertEqual(captured["path"], os.path.abspath(regular))

    def test_validate_output_directory_does_not_follow_non_directories_or_symlinks(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "file"
            file_path.write_text("x", encoding="utf-8")
            directory = root / "dir"
            directory.mkdir()
            link = root / "dir-link"
            link.symlink_to(directory, target_is_directory=True)
            self.assertEqual(module.validate_output_directory(directory), os.path.abspath(directory))
            for path in (file_path, link, root / "missing"):
                with self.assertRaises(ValueError):
                    module.validate_output_directory(path)

    def test_resolve_executable_records_requested_and_final_regular_executable(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "ncu-real"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            link = root / "ncu-link"
            link.symlink_to(executable)
            result = module.resolve_executable(str(link))
            self.assertEqual(result["requested"], str(link))
            self.assertEqual(result["resolved"], os.path.realpath(link))
            executable.chmod(0o600)
            with self.assertRaises(ValueError):
                module.resolve_executable(str(executable))

    def test_cli_rejects_unsafe_inputs_before_running_ncu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.rep"
            source = root / "source.py"
            output = root / "output"
            report.write_bytes(b"rep")
            source.write_text("x = 1\n", encoding="utf-8")
            output.mkdir()
            fake_ncu = root / "ncu"
            fake_ncu.write_text("#!/bin/sh\ntouch \"$MARKER\"\n", encoding="utf-8")
            fake_ncu.chmod(0o700)
            marker = root / "ran"
            base = [sys.executable, str(SCRIPT), "--report", str(report), "--source", str(source), "--output", str(output), "--ncu-bin", str(fake_ncu)]
            for extra in (("--ncu-num", "0"), ("--ncu-num", "-1"), ("--ncu-num", "true"), ("--timeout", "0"), ("--timeout", "-1"), ("--timeout", "nan"), ("--timeout", "inf"), ("--timeout", "true")):
                completed = subprocess.run(base + list(extra), env={**os.environ, "MARKER": str(marker)}, capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0, (extra, completed.stderr))
                self.assertFalse(marker.exists(), extra)


class AnalyzeNcuRepBoundedRunTests(unittest.TestCase):
    def test_bounded_run_captures_invalid_utf8_and_truncates_while_draining(self) -> None:
        module = _load()
        code = "import sys; sys.stdout.buffer.write(b'abcdef\\xff'); sys.stderr.buffer.write(b'123456')"
        result = module._run_bounded([sys.executable, "-c", code], timeout=2, output_limit=4)
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["returncode"], 0)
        self.assertIsInstance(result["stdout"], str)
        self.assertIn("abcd", result["stdout"])
        self.assertEqual(result["stderr"], "1234")

    def test_bounded_run_terminates_process_group_after_timeout(self) -> None:
        module = _load()
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)"
        started = time.monotonic()
        result = module._run_bounded([sys.executable, "-c", code], timeout=0.2, output_limit=1024)
        self.assertTrue(result["timed_out"])
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNotNone(result["returncode"])

    def test_bounded_run_rejects_non_list_argv(self) -> None:
        module = _load()
        with self.assertRaises((TypeError, ValueError)):
            module._run_bounded("echo nope", timeout=1, output_limit=10)


if __name__ == "__main__":
    unittest.main()
