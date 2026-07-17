from __future__ import annotations

import hashlib
import importlib.util
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
SCRIPT = SCRIPTS / "analyze_ncu_rep.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_ncu_rep_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


class AnalyzeNcuRepInputTests(unittest.TestCase):
    def test_parser_uses_approved_positional_and_optional_interface(self) -> None:
        module = _load()
        args = module.build_parser().parse_args(["report.rep", "--out-dir", "out"])
        self.assertEqual(args.report, "report.rep")
        self.assertIsNone(args.source)
        self.assertEqual(args.out_dir, "out")
        self.assertEqual(args.ncu_num, 5)
        self.assertEqual(args.timeout, 120.0)

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

    def test_capture_regular_file_has_field_specific_errors_and_no_symlink_following(self) -> None:
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
            for field in ("REPORT", "SOURCE"):
                for path in (root / "missing.rep", directory, fifo, leaf_link, parent_link / "child.rep"):
                    with self.assertRaisesRegex(ValueError, field):
                        module.capture_regular_file(path, field)
            captured = module.capture_regular_file(regular, "REPORT")
            self.assertEqual(captured["size"], 6)
            self.assertEqual(captured["sha256"], hashlib.sha256(b"report").hexdigest())
            self.assertEqual(captured["path"], os.path.abspath(regular))

    def test_validate_output_directory_creates_missing_leaf_and_rejects_files_and_symlinks(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "file"
            file_path.write_text("x", encoding="utf-8")
            directory = root / "dir"
            directory.mkdir()
            link = root / "dir-link"
            link.symlink_to(directory, target_is_directory=True)
            parent_link = root / "parent-link"
            parent_link.symlink_to(directory, target_is_directory=True)
            missing = root / "created"
            self.assertEqual(module.validate_output_directory(missing), os.path.abspath(missing))
            self.assertTrue(missing.is_dir())
            for path in (file_path, link, parent_link / "child"):
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

    def test_cli_validates_report_source_output_and_numbers_before_ncu(self) -> None:
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
            base = [sys.executable, str(SCRIPT), str(report), "--source", str(source), "--out-dir", str(output), "--ncu-bin", str(fake_ncu)]
            directory = root / "directory"
            directory.mkdir()
            fifo = root / "input.fifo"
            os.mkfifo(fifo)
            leaf_link = root / "leaf-link"
            leaf_link.symlink_to(report)
            real_parent = root / "real-parent"
            real_parent.mkdir()
            parent_child = real_parent / "child"
            parent_child.write_text("x", encoding="utf-8")
            parent_link = root / "parent-link"
            parent_link.symlink_to(real_parent, target_is_directory=True)
            unsafe_inputs = [root / "missing", directory, fifo, leaf_link, parent_link / "child"]
            output_file = root / "output-file"
            output_file.write_text("x", encoding="utf-8")
            output_link = root / "output-link"
            output_link.symlink_to(output, target_is_directory=True)
            invalid = []
            for candidate in unsafe_inputs:
                invalid.append(([str(candidate), "--out-dir", str(output)], "REPORT"))
                invalid.append(([str(report), "--source", str(candidate), "--out-dir", str(output)], "SOURCE"))
            for candidate in (output_file, output_link):
                invalid.append(([str(report), "--out-dir", str(candidate)], "output"))
            for value in ("0", "-1", "true"):
                invalid.append((base[2:] + ["--ncu-num", value], "error:"))
            for value in ("0", "-1", "nan", "inf", "true"):
                invalid.append((base[2:] + ["--timeout", value], "error:"))
            for pieces, expected in invalid:
                command = [sys.executable, str(SCRIPT)] + pieces
                completed = subprocess.run(command, env={**os.environ, "MARKER": str(marker)}, capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0, (pieces, completed.stderr))
                self.assertIn(expected, completed.stderr)
                self.assertFalse(marker.exists(), pieces)


class AnalyzeNcuRepBoundedRunTests(unittest.TestCase):
    def test_bounded_run_drains_more_than_one_mib_but_only_retains_limit(self) -> None:
        module = _load()
        code = "import sys; sys.stdout.buffer.write(b'x' * (1024 * 1024 + 1))"
        result = module._run_bounded([sys.executable, "-c", code], timeout=2, output_limit=4)
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["stdout"], "xxxx")

    def test_bounded_run_captures_invalid_utf8(self) -> None:
        module = _load()
        result = module._run_bounded([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff')"], timeout=2, output_limit=4)
        self.assertIn("�", result["stdout"])

    def test_bounded_run_kills_term_ignoring_parent_and_child(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            pids = Path(tmp) / "pids"
            code = (
                "import os,signal,subprocess,sys,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "open(sys.argv[1],'w').write(str(os.getpid())+'\\n'); "
                "subprocess.Popen([sys.executable,'-c',\"import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open(sys.argv[1],'a').write(str(os.getpid())+'\\\\n'); time.sleep(30)\",sys.argv[1]]); time.sleep(30)"
            )
            result = module._run_bounded([sys.executable, "-c", code, str(pids)], timeout=0.2, output_limit=1024)
            self.assertTrue(result["timed_out"])
            recorded = [int(value) for value in pids.read_text().splitlines()]
            self.assertEqual(len(recorded), 2)
            for _ in range(50):
                if all(_gone(pid) for pid in recorded):
                    break
                time.sleep(0.02)
            self.assertTrue(all(_gone(pid) for pid in recorded), recorded)

    def test_bounded_run_cleans_descendant_when_leader_exits(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c',\"import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(30)\",sys.argv[1]])"
            result = module._run_bounded([sys.executable, "-c", code, str(pid_file)], timeout=0.2, output_limit=1024)
            self.assertTrue(result["timed_out"])
            pid = int(pid_file.read_text())
            for _ in range(50):
                if _gone(pid):
                    break
                time.sleep(0.02)
            self.assertTrue(_gone(pid), pid)

    def test_bounded_run_times_out_when_leader_exits_and_descendant_closes_stdio(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c',\"import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); os.close(1); os.close(2); time.sleep(30)\",sys.argv[1]])"
            result = module._run_bounded([sys.executable, "-c", code, str(pid_file)], timeout=0.2, output_limit=1024)
            self.assertTrue(result["timed_out"])
            pid = int(pid_file.read_text())
            for _ in range(50):
                if _gone(pid):
                    break
                time.sleep(0.02)
            self.assertTrue(_gone(pid), pid)

    def test_bounded_run_cleans_process_if_reader_start_fails(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pid"
            real_popen = module.subprocess.Popen

            def launch(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                for _ in range(100):
                    if pid_file.exists():
                        return process
                    time.sleep(0.01)
                self.fail("child did not record its PID")

            code = "import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(30)"
            with mock.patch.object(module.subprocess, "Popen", side_effect=launch), mock.patch.object(module.threading.Thread, "start", side_effect=RuntimeError("reader start")):
                with self.assertRaisesRegex(RuntimeError, "reader start"):
                    module._run_bounded([sys.executable, "-c", code, str(pid_file)], timeout=2, output_limit=1024)
            pid = int(pid_file.read_text())
            for _ in range(50):
                if _gone(pid):
                    break
                time.sleep(0.02)
            self.assertTrue(_gone(pid), pid)

    def test_bounded_run_rejects_non_list_argv(self) -> None:
        module = _load()
        with self.assertRaises((TypeError, ValueError)):
            module._run_bounded("echo nope", timeout=1, output_limit=10)


if __name__ == "__main__":
    unittest.main()
