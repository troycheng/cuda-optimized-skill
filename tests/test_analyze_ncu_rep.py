from __future__ import annotations

import hashlib
import importlib.util
import json
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


def _fake_ncu(root: Path) -> Path:
    executable = root / "fake-ncu"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["NCU_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv) + "\\n")

mode = os.environ.get("NCU_MODE", "success")
args = sys.argv[1:]
if args == ["--version"]:
    print("NVIDIA Nsight Compute 2026.1")
    raise SystemExit(0)
if mode == "all-fail":
    print("import failed", file=sys.stderr)
    raise SystemExit(7)
page = args[-1] if len(args) >= 2 and args[-2] == "--page" else ""
if mode == "summary-only" and page != "summary":
    print("unavailable", file=sys.stderr)
    raise SystemExit(8)
if mode == "raw-fail" and page == "raw":
    print("raw unavailable", file=sys.stderr)
    raise SystemExit(9)
if page == "summary":
    print("summary evidence")
elif page == "details":
    print("details evidence")
elif page == "raw":
    print('"Kernel Name","Metric Name","Metric Unit","Metric Value"')
    print('"kernel-a","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","75"')
else:
    raise SystemExit(10)
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def _run_cli(module, root: Path, *, mode: str = "success", source: Path | None = None):
    report = root / "input.ncu-rep"
    if not report.exists():
        report.write_bytes(b"captured-report")
    output = root / "analysis"
    ncu = _fake_ncu(root)
    calls = root / "calls.jsonl"
    argv = [str(report), "--out-dir", str(output), "--ncu-bin", str(ncu)]
    if source is not None:
        argv.extend(["--source", str(source)])
    with mock.patch.dict(os.environ, {"NCU_CALLS": str(calls), "NCU_MODE": mode}):
        return module.main(argv), report, output, ncu, calls


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
    def test_bounded_run_latches_group_disappearance_before_reader_completion(self) -> None:
        module = _load()

        class Process:
            pid = 43210
            stdout = None
            stderr = None
            returncode = 0

            def poll(self):
                return 0

            def wait(self):
                return 0

            def kill(self):
                self.returncode = -signal.SIGKILL

        class Reader:
            ident = 1

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def is_alive(self):
                return True

            def join(self, timeout=None):
                pass

        calls = []

        def killpg(pgid, signum):
            calls.append((pgid, signum))
            raise ProcessLookupError()

        with mock.patch.object(module.subprocess, "Popen", return_value=Process()), mock.patch.object(module.threading, "Thread", Reader), mock.patch.object(module.os, "killpg", side_effect=killpg):
            result = module._run_bounded(["ignored"], timeout=0.02, output_limit=4)
        self.assertTrue(result["timed_out"])
        self.assertEqual(calls, [(43210, 0)])

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

    def test_bounded_run_cleans_process_if_reader_construction_fails(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pid"
            real_popen = module.subprocess.Popen
            launched = []

            def launch(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                launched.append(process)
                for _ in range(100):
                    if pid_file.exists():
                        return process
                    time.sleep(0.01)
                self.fail("child did not record its PID")

            code = "import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(30)"
            try:
                with mock.patch.object(module.subprocess, "Popen", side_effect=launch), mock.patch.object(module.threading, "Thread", side_effect=RuntimeError("reader construction")):
                    with self.assertRaisesRegex(RuntimeError, "reader construction"):
                        module._run_bounded([sys.executable, "-c", code, str(pid_file)], timeout=2, output_limit=1024)
                pid = int(pid_file.read_text())
                for _ in range(50):
                    if _gone(pid):
                        break
                    time.sleep(0.02)
                gone_before_cleanup = _gone(pid)
                streams_closed_before_cleanup = launched[0].stdout.closed and launched[0].stderr.closed
            finally:
                if launched:
                    process = launched[0]
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    for stream in (process.stdout, process.stderr):
                        if stream is not None and not stream.closed:
                            stream.close()
            self.assertTrue(gone_before_cleanup, pid)
            self.assertTrue(streams_closed_before_cleanup)

    def test_bounded_run_rejects_non_list_argv(self) -> None:
        module = _load()
        with self.assertRaises((TypeError, ValueError)):
            module._run_bounded("echo nope", timeout=1, output_limit=10)


class AnalyzeNcuRepImportTests(unittest.TestCase):
    def test_invalid_report_or_source_removes_stale_marker_before_ncu(self) -> None:
        module = _load()
        for invalid_field in ("REPORT", "SOURCE"):
            with self.subTest(field=invalid_field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report = root / "input.ncu-rep"
                report.write_bytes(b"report")
                source = root / "source.py"
                source.write_text("kernel", encoding="utf-8")
                output = root / "analysis"
                output.mkdir()
                marker = output / "analysis.json"
                marker.write_text('{"old": true}', encoding="utf-8")
                ncu = _fake_ncu(root)
                argv = [str(report), "--out-dir", str(output), "--ncu-bin", str(ncu)]
                if invalid_field == "REPORT":
                    report.unlink()
                else:
                    source_link = root / "source-link.py"
                    source_link.symlink_to(source)
                    argv.extend(["--source", str(source_link)])
                with mock.patch.object(module, "_run_bounded") as run:
                    returncode = module.main(argv)
                self.assertEqual(returncode, 1)
                self.assertFalse(marker.exists())
                run.assert_not_called()

    def test_unsafe_output_is_not_touched_before_validation(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "input.ncu-rep"
            report.write_bytes(b"report")
            real_output = root / "real-output"
            real_output.mkdir()
            marker = real_output / "analysis.json"
            marker.write_text('{"old": true}', encoding="utf-8")
            output_link = root / "output-link"
            output_link.symlink_to(real_output, target_is_directory=True)
            ncu = _fake_ncu(root)
            with mock.patch.object(module, "_run_bounded") as run:
                returncode = module.main(
                    [str(report), "--out-dir", str(output_link), "--ncu-bin", str(ncu)]
                )
            self.assertEqual(returncode, 1)
            self.assertTrue(marker.exists())
            run.assert_not_called()

    def test_imports_report_with_exact_resolved_ncu_argv_order(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            returncode, report, _output, ncu, calls_path = _run_cli(module, root)
            calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
        resolved = os.path.realpath(ncu)
        captured_report = os.path.abspath(report)
        self.assertEqual(returncode, 0)
        self.assertEqual(
            calls,
            [
                [resolved, "--version"],
                [resolved, "--import", captured_report, "--page", "summary"],
                [resolved, "--import", captured_report, "--page", "details"],
                [resolved, "--import", captured_report, "--csv", "--page", "raw"],
            ],
        )

    def test_csv_analysis_reuses_profile_parser_aggregator_and_ranker(self) -> None:
        module = _load()
        profile = module.profile_ncu
        long_csv = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"kernel-a","dram__bytes.sum","byte","1,024"\n'
            '"kernel-b","dram__bytes.sum","byte","2,048"\n'
            '"kernel-a","sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_active","%","25"\n'
            '"kernel-a","smsp__warp_issue_stalled_wait_per_inst_issued","cycle","bad"\n'
        )
        expected_rows = profile._parse_ncu_csv(long_csv)
        expected_agg = profile._aggregate_across_kernels(expected_rows)
        expected_rankings = profile._rank_by_axis(expected_agg, 3)
        with mock.patch.object(profile, "_parse_ncu_csv", wraps=profile._parse_ncu_csv) as parse, mock.patch.object(
            profile, "_aggregate_across_kernels", wraps=profile._aggregate_across_kernels
        ) as aggregate, mock.patch.object(profile, "_rank_by_axis", wraps=profile._rank_by_axis) as rank:
            result = module._analyze_csv(long_csv, 3)
        parse.assert_called_once_with(long_csv)
        aggregate.assert_called_once_with(expected_rows)
        rank.assert_called_once_with(expected_agg, 3)
        self.assertEqual(result["rankings"], expected_rankings)
        self.assertEqual(result["kernels"], ["kernel-a", "kernel-b"])
        self.assertEqual(result["metric_count"], len(expected_agg))
        self.assertEqual(result["primary_axis"]["quality"], "heuristic")

    def test_wide_csv_and_unclassified_csv_use_profile_behavior(self) -> None:
        module = _load()
        wide = (
            '"Kernel Name","gpu__time_duration.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed"\n'
            '"short","10","20"\n'
            '"longest","30","80"\n'
        )
        wide_result = module._analyze_csv(wide, 5)
        self.assertEqual(wide_result["kernels"], ["longest"])
        self.assertEqual(wide_result["metric_count"], 1)
        unknown = module._analyze_csv(
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"k","unclassified.metric","","10"\n',
            5,
        )
        self.assertEqual(unknown["primary_axis"], {"axis": "unknown", "quality": "heuristic"})

    def test_successful_unclassified_raw_csv_is_preserved_with_unknown_axis(self) -> None:
        module = _load()
        raw = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"kernel","unclassified.metric","","10"\n'
        )
        result = {"timed_out": False, "truncated": False, "returncode": 0, "stdout": "", "stderr": ""}
        responses = [
            {**result, "stdout": "NCU 1"},
            {**result, "stdout": "summary"},
            {**result, "stdout": "details"},
            {**result, "stdout": raw},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "input.ncu-rep"
            report.write_bytes(b"report")
            output = root / "analysis"
            ncu = _fake_ncu(root)
            with mock.patch.object(module, "_run_bounded", side_effect=responses):
                returncode = module.main([str(report), "--out-dir", str(output), "--ncu-bin", str(ncu)])
            payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(returncode, 0)
            self.assertEqual((output / "raw.csv").read_text(encoding="utf-8"), raw)
            self.assertTrue(payload["artifacts"]["raw.csv"]["available"])
            self.assertEqual(payload["primary_axis"], {"axis": "unknown", "quality": "heuristic"})

    def test_partial_and_hard_exit_codes_and_fixed_missing_outputs(self) -> None:
        module = _load()
        for mode, expected in (("summary-only", 2), ("raw-fail", 2), ("all-fail", 1)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                returncode, _report, output, _ncu, _calls = _run_cli(module, root, mode=mode)
                self.assertEqual(returncode, expected)
                if expected == 1:
                    self.assertFalse((output / "analysis.json").exists())
                    continue
                analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
                self.assertEqual(analysis["status"], "partial")
                for name in (
                    "summary.txt",
                    "summary.stderr.txt",
                    "details.txt",
                    "details.stderr.txt",
                    "raw.csv",
                    "analysis.md",
                ):
                    self.assertTrue((output / name).is_file(), name)
                if mode == "summary-only":
                    self.assertEqual((output / "details.txt").read_bytes(), b"")
                    self.assertEqual((output / "raw.csv").read_bytes(), b"")
                    self.assertFalse(analysis["artifacts"]["details.txt"]["available"])
                    self.assertFalse(analysis["artifacts"]["raw.csv"]["available"])

    def test_any_truncated_command_makes_interpretable_analysis_partial(self) -> None:
        module = _load()
        raw = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"kernel","dram__bytes.sum","byte","10"\n'
        )
        outputs = ("NCU 1", "summary", "details", raw)
        names = ("version", "summary", "details", "raw")
        for truncated_index, command_name in enumerate(names):
            with self.subTest(command=command_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report = root / "input.ncu-rep"
                report.write_bytes(b"report")
                output = root / "analysis"
                ncu = _fake_ncu(root)
                responses = [
                    {
                        "timed_out": False,
                        "truncated": index == truncated_index,
                        "returncode": 0,
                        "stdout": stdout,
                        "stderr": "",
                    }
                    for index, stdout in enumerate(outputs)
                ]
                with mock.patch.object(module, "_run_bounded", side_effect=responses):
                    returncode = module.main(
                        [str(report), "--out-dir", str(output), "--ncu-bin", str(ncu)]
                    )
                payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
                self.assertEqual(returncode, 2)
                self.assertEqual(payload["status"], "partial")
                self.assertTrue(payload["commands"][command_name]["truncated"])

    def test_every_command_record_contains_its_bounded_stderr(self) -> None:
        module = _load()
        raw = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"kernel","dram__bytes.sum","byte","10"\n'
        )
        outputs = ("NCU 1", "summary", "details", raw)
        names = ("version", "summary", "details", "raw")
        responses = [
            {
                "timed_out": False,
                "truncated": False,
                "returncode": 0,
                "stdout": stdout,
                "stderr": f"{name} bounded stderr",
            }
            for name, stdout in zip(names, outputs)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "input.ncu-rep"
            report.write_bytes(b"report")
            output = root / "analysis"
            ncu = _fake_ncu(root)
            with mock.patch.object(module, "_run_bounded", side_effect=responses):
                returncode = module.main(
                    [str(report), "--out-dir", str(output), "--ncu-bin", str(ncu)]
                )
            payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(returncode, 0)
        for name in names:
            self.assertEqual(payload["commands"][name]["stderr"], f"{name} bounded stderr")

    def test_report_and_source_same_metadata_drift_are_hard_failures(self) -> None:
        module = _load()
        for drift_target in ("report", "source"):
            with self.subTest(drift_target=drift_target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "kernel.py"
                source.write_bytes(b"source-A")
                output = root / "analysis"
                output.mkdir()
                (output / "analysis.json").write_text('{"old": true}', encoding="utf-8")
                report = root / "input.ncu-rep"
                report.write_bytes(b"report-A")
                ncu = _fake_ncu(root)
                calls_path = root / "calls.jsonl"
                real_run = module._run_bounded
                calls = 0

                def mutate_after_summary(argv, timeout, output_limit):
                    nonlocal calls
                    result = real_run(argv, timeout, output_limit)
                    calls += 1
                    if calls == 2:
                        target = report if drift_target == "report" else source
                        before = target.stat()
                        replacement = b"report-B" if drift_target == "report" else b"source-B"
                        target.write_bytes(replacement)
                        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
                    return result

                argv = [str(report), "--source", str(source), "--out-dir", str(output), "--ncu-bin", str(ncu)]
                with mock.patch.dict(os.environ, {"NCU_CALLS": str(calls_path), "NCU_MODE": "success"}), mock.patch.object(
                    module, "_run_bounded", side_effect=mutate_after_summary
                ):
                    returncode = module.main(argv)
                self.assertEqual(returncode, 1)
                self.assertFalse((output / "analysis.json").exists())

    def test_timeout_is_hard_failure_and_removes_stale_marker(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "input.ncu-rep"
            report.write_bytes(b"report")
            output = root / "analysis"
            output.mkdir()
            marker = output / "analysis.json"
            marker.write_text('{"old": true}', encoding="utf-8")
            ncu = _fake_ncu(root)
            normal = {"timed_out": False, "truncated": False, "returncode": 0, "stdout": "version", "stderr": ""}
            timeout = {**normal, "timed_out": True, "returncode": -signal.SIGKILL}
            with mock.patch.object(module, "_run_bounded", side_effect=[normal, timeout]):
                returncode = module.main([str(report), "--out-dir", str(output), "--ncu-bin", str(ncu)])
            self.assertEqual(returncode, 1)
            self.assertFalse(marker.exists())

    def test_hostile_markdown_is_escaped_and_bundle_is_published_marker_last(self) -> None:
        module = _load()
        hostile = "![x](file:///tmp/leak)|\n# heading<>"
        raw = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            f'"{hostile}","dram__bytes.sum","byte","1"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "input.ncu-rep"
            report.write_bytes(b"report")
            output = root / "analysis"
            output.mkdir()
            (output / "analysis.json").write_text('{"old": true}', encoding="utf-8")
            ncu = _fake_ncu(root)
            result = {"timed_out": False, "truncated": False, "returncode": 0, "stdout": "", "stderr": ""}
            responses = [
                {**result, "stdout": "NCU 1"},
                {**result, "stdout": "summary"},
                {**result, "stdout": "details"},
                {**result, "stdout": raw},
            ]
            events = []
            real_remove = module.artifact_store.remove_regular_file
            real_publish = module.artifact_store.publish_regular_bundle
            real_json = module.artifact_store.atomic_write_json
            with mock.patch.object(module, "_run_bounded", side_effect=responses), mock.patch.object(
                module.artifact_store,
                "remove_regular_file",
                side_effect=lambda *a, **kw: (events.append("remove"), real_remove(*a, **kw))[1],
            ), mock.patch.object(
                module.artifact_store,
                "publish_regular_bundle",
                side_effect=lambda *a, **kw: (events.append("bundle"), real_publish(*a, **kw))[1],
            ), mock.patch.object(
                module.artifact_store,
                "atomic_write_json",
                side_effect=lambda *a, **kw: (events.append("marker"), real_json(*a, **kw))[1],
            ):
                returncode = module.main([str(report), "--out-dir", str(output), "--ncu-bin", str(ncu)])
            self.assertEqual(returncode, 0)
            self.assertEqual(events, ["remove", "bundle", "marker"])
            markdown = (output / "analysis.md").read_text(encoding="utf-8")
            for unsafe in ("![x]", "](file:", "|\n# heading", "<>"):
                self.assertNotIn(unsafe, markdown)
            payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "cuda-kernel-optimizer/ncu-analysis-v1")
            self.assertEqual(payload["counter_access"], "not_probed")
            self.assertNotIn("analysis.json", payload["artifacts"])
            self.assertEqual(len(payload["limits"]), 3)
            for name, info in payload["artifacts"].items():
                content = (output / name).read_bytes()
                self.assertEqual(info["sha256"], hashlib.sha256(content).hexdigest())
                self.assertEqual(info["size"], len(content))

    def test_supporting_publish_failure_after_one_write_leaves_no_completion_marker(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "input.ncu-rep"
            report.write_bytes(b"report")
            output = root / "analysis"
            output.mkdir()
            marker = output / "analysis.json"
            marker.write_text('{"old": true}', encoding="utf-8")
            summary = output / "summary.txt"
            summary.write_text("old summary", encoding="utf-8")
            ncu = _fake_ncu(root)
            result = {"timed_out": False, "truncated": False, "returncode": 0, "stdout": "ok", "stderr": ""}
            real_write = module.artifact_store._atomic_write_leaf
            writes = 0

            def fail_on_second_write(*args, **kwargs):
                nonlocal writes
                writes += 1
                if writes == 2:
                    raise OSError("disk full")
                return real_write(*args, **kwargs)

            with mock.patch.object(
                module, "_run_bounded", side_effect=[result, result, result, result]
            ), mock.patch.object(
                module.artifact_store, "_atomic_write_leaf", side_effect=fail_on_second_write
            ):
                returncode = module.main([str(report), "--out-dir", str(output), "--ncu-bin", str(ncu)])
            self.assertEqual(returncode, 1)
            self.assertEqual(writes, 2)
            self.assertEqual(summary.read_text(encoding="utf-8"), "ok")
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
