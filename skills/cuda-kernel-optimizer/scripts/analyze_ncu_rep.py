#!/usr/bin/env python3
"""Analyze an existing NCU report without launching a target kernel."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import artifact_store  # noqa: E402
import profile_ncu  # noqa: E402


SCHEMA_VERSION = "cuda-kernel-optimizer/ncu-analysis-v1"
OUTPUT_LIMIT = 1024 * 1024
SUPPORTING_FILES = (
    "summary.txt",
    "summary.stderr.txt",
    "details.txt",
    "details.stderr.txt",
    "raw.csv",
    "analysis.md",
)
LIMITS = [
    "Importing an existing report does not prove current performance-counter permission.",
    "The captured source identity does not prove that the report was produced from that source.",
    "Standalone report analysis does not prove an end-to-end performance benefit.",
]


def _strict_positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("value must be a positive integer")
    return value


def _strict_positive_float(value: Any) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise TypeError("value must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("value must be a positive finite number")
    return result


def _physical_absolute(path: os.PathLike[str] | str) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def capture_regular_file(path: os.PathLike[str] | str, field: str) -> dict[str, Any]:
    """Capture regular bytes without resolving or following symlinks."""
    physical_path = _physical_absolute(path)
    try:
        mode = os.lstat(physical_path).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError("path is not a regular file")
        payload = artifact_store.read_regular_bytes(physical_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"{field}: unsafe regular file: {physical_path}") from error
    return {
        "path": physical_path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_output_directory(path: os.PathLike[str] | str) -> str:
    physical_path = _physical_absolute(path)
    target = Path(physical_path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(target.anchor, flags)
    try:
        for index, component in enumerate(target.parts[1:]):
            component_flags = flags if index == 0 else flags | nofollow
            try:
                child_fd = os.open(component, component_flags, dir_fd=directory_fd)
            except FileNotFoundError:
                os.mkdir(component, 0o755, dir_fd=directory_fd)
                child_fd = os.open(component, component_flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(f"output path is unsafe or not a directory: {physical_path}") from error
            os.close(directory_fd)
            directory_fd = child_fd
    except BaseException:
        os.close(directory_fd)
        raise
    os.close(directory_fd)
    return physical_path


def resolve_executable(requested: os.PathLike[str] | str) -> dict[str, str]:
    requested_text = os.fspath(requested)
    candidate = requested_text if os.path.dirname(requested_text) else shutil.which(requested_text)
    if not candidate:
        raise ValueError(f"executable was not found: {requested_text}")
    resolved = os.path.realpath(candidate)
    try:
        mode = os.stat(resolved, follow_symlinks=False).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"executable was not found: {requested_text}") from error
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise ValueError(f"executable is not a regular executable: {requested_text}")
    return {"requested": requested_text, "resolved": resolved}


def _run_bounded(argv: list[str], timeout: float, output_limit: int) -> dict[str, Any]:
    if type(argv) is not list or not argv or any(type(item) is not str for item in argv):
        raise TypeError("argv must be a non-empty list of strings")
    timeout = _strict_positive_float(timeout)
    output_limit = _strict_positive_int(output_limit)
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = [False]

    def drain(stream, name: str) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                available = output_limit - len(captured[name])
                if available > 0:
                    captured[name].extend(chunk[:available])
                if len(chunk) > available:
                    truncated[0] = True
        finally:
            stream.close()

    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    readers = []
    group_gone = False

    def signal_group(signum: int) -> None:
        nonlocal group_gone
        if group_gone:
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            group_gone = True
        except PermissionError:
            # If the OS no longer permits addressing the session, still reap
            # the direct child rather than leaking it during error cleanup.
            process.kill()
            group_gone = True

    def group_exists() -> bool:
        nonlocal group_gone
        if group_gone:
            return False
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            group_gone = True
            return False
        except PermissionError:
            process.kill()
            group_gone = True
            return False
        return True

    def stop_group() -> None:
        signal_group(signal.SIGTERM)
        grace_deadline = time.monotonic() + 0.2
        while time.monotonic() < grace_deadline and group_exists():
            time.sleep(0.01)
        if group_exists():
            signal_group(signal.SIGKILL)

    timed_out = False
    try:
        readers.extend(
            [
                threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
                threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True),
            ]
        )
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                process.wait()
                group_still_exists = group_exists()
                if not any(reader.is_alive() for reader in readers) and not group_still_exists:
                    break
            time.sleep(0.01)
        else:
            timed_out = True
        if timed_out:
            stop_group()
        process.wait()
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=1)
    except BaseException:
        stop_group()
        process.wait()
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        raise
    return {
        "timed_out": timed_out,
        "truncated": truncated[0],
        "returncode": process.returncode,
        "stdout": bytes(captured["stdout"]).decode("utf-8", errors="replace"),
        "stderr": bytes(captured["stderr"]).decode("utf-8", errors="replace"),
    }


def _positive_int_argument(value: str) -> int:
    try:
        return _strict_positive_int(int(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _positive_float_argument(value: str) -> float:
    try:
        return _strict_positive_float(float(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _command_result(result: dict[str, Any], *, available: bool) -> dict[str, Any]:
    return {
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "truncated": result["truncated"],
        "stderr": result["stderr"],
        "available": available,
    }


def _analyze_csv(csv_text: str, ncu_num: int) -> dict[str, Any]:
    """Normalize and rank imported CSV through the optimizer's shared rubric."""
    rows = profile_ncu._parse_ncu_csv(csv_text)
    aggregate = profile_ncu._aggregate_across_kernels(rows)
    rankings = profile_ncu._rank_by_axis(aggregate, ncu_num)
    kernels = sorted(
        {
            row.get("Kernel Name", "").strip()
            for row in rows
            if row.get("Kernel Name", "").strip()
        }
    )

    best_axis = "unknown"
    best_score = float("-inf")
    for axis in ("compute", "memory", "latency"):
        for metric in rankings[axis]:
            value = float(metric["value"])
            if not math.isfinite(value):
                continue
            score = value if metric["higher_is_worse"] else 100.0 - value
            if score > best_score:
                best_score = score
                best_axis = axis
    return {
        "metric_count": len(aggregate),
        "kernels": kernels,
        "rankings": rankings,
        "primary_axis": {"axis": best_axis, "quality": "heuristic"},
    }


def _escape_markdown(value: Any) -> str:
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    escaped = []
    for character in text:
        if character in r"\\`*_{}[]()#+-.!|<>":
            escaped.append("\\")
        escaped.append(character)
    return "".join(escaped)


def _render_markdown(payload: dict[str, Any]) -> bytes:
    lines = [
        "# NCU report analysis",
        "",
        f"Status: {_escape_markdown(payload['status'])}",
        f"Report: {_escape_markdown(payload['report']['path'])}",
        f"Report SHA-256: {_escape_markdown(payload['report']['sha256'])}",
        f"Primary axis: {_escape_markdown(payload['primary_axis']['axis'])} (heuristic)",
        "",
        "## Kernels",
        "",
    ]
    if payload["kernels"]:
        lines.extend(f"* {_escape_markdown(kernel)}" for kernel in payload["kernels"])
    else:
        lines.append("No classified kernel names were available.")
    lines.extend(["", "## Ranked metrics", ""])
    for axis in ("compute", "memory", "latency"):
        lines.append(f"### {axis.capitalize()}")
        lines.append("")
        metrics = payload["rankings"][axis]
        if not metrics:
            lines.append("No classified metrics were available.")
        for metric in metrics:
            unit = metric.get("unit") or ""
            lines.append(
                "* "
                + _escape_markdown(metric["name"])
                + ": "
                + _escape_markdown(metric["value"])
                + (" " + _escape_markdown(unit) if unit else "")
            )
        lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"* {_escape_markdown(limit)}" for limit in payload["limits"])
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _same_identity(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return (
        before["path"] == after["path"]
        and before["size"] == after["size"]
        and before["sha256"] == after["sha256"]
    )


def _run_analysis(args: argparse.Namespace) -> int:
    output = validate_output_directory(args.out_dir)
    marker = os.path.join(output, "analysis.json")
    artifact_store.remove_regular_file(marker, missing_ok=True)

    report = capture_regular_file(args.report, "REPORT")
    source = capture_regular_file(args.source, "SOURCE") if args.source is not None else None
    ncu = resolve_executable(args.ncu_bin)
    executable = ncu["resolved"]
    report_path = report["path"]
    commands: dict[str, dict[str, Any]] = {}

    version_result = _run_bounded(
        [executable, "--version"], args.timeout, OUTPUT_LIMIT
    )
    if version_result["timed_out"]:
        raise TimeoutError("NCU version query timed out")
    version_available = version_result["returncode"] == 0 and bool(
        version_result["stdout"].strip()
    )
    commands["version"] = _command_result(
        version_result, available=version_available
    )

    command_specs = (
        ("summary", [executable, "--import", report_path, "--page", "summary"]),
        ("details", [executable, "--import", report_path, "--page", "details"]),
        ("raw", [executable, "--import", report_path, "--csv", "--page", "raw"]),
    )
    results: dict[str, dict[str, Any]] = {}
    for name, argv in command_specs:
        result = _run_bounded(argv, args.timeout, OUTPUT_LIMIT)
        if result["timed_out"]:
            raise TimeoutError(f"NCU {name} import timed out")
        results[name] = result

    report_after = capture_regular_file(report_path, "REPORT")
    if not _same_identity(report, report_after):
        raise ValueError("REPORT identity changed during import")
    if source is not None:
        source_after = capture_regular_file(source["path"], "SOURCE")
        if not _same_identity(source, source_after):
            raise ValueError("SOURCE identity changed during import")

    available = {
        name: result["returncode"] == 0 and bool(result["stdout"])
        for name, result in results.items()
    }
    csv_analysis = (
        _analyze_csv(results["raw"]["stdout"], args.ncu_num)
        if available["raw"]
        else _analyze_csv("", args.ncu_num)
    )
    available["raw"] = available["raw"] and csv_analysis["metric_count"] > 0
    for name, result in results.items():
        commands[name] = _command_result(result, available=available[name])

    interpretable = available["summary"] or available["details"] or available["raw"]
    if not interpretable:
        raise RuntimeError("all NCU report imports failed or were uninterpretable")
    complete = (
        version_available
        and all(results[name]["returncode"] == 0 for name in results)
        and all(available.values())
        and not version_result["truncated"]
        and not any(result["truncated"] for result in results.values())
    )
    status = "success" if complete else "partial"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "counter_access": "not_probed",
        "report": report,
        "source": source,
        "ncu": {
            "requested": ncu["requested"],
            "resolved": executable,
            "version": version_result["stdout"].strip() if version_available else None,
        },
        "commands": commands,
        **csv_analysis,
        "limits": list(LIMITS),
        "artifacts": {},
    }
    writes = {
        "summary.txt": (
            results["summary"]["stdout"].encode("utf-8") if available["summary"] else b""
        ),
        "summary.stderr.txt": results["summary"]["stderr"].encode("utf-8"),
        "details.txt": (
            results["details"]["stdout"].encode("utf-8") if available["details"] else b""
        ),
        "details.stderr.txt": results["details"]["stderr"].encode("utf-8"),
        "raw.csv": results["raw"]["stdout"].encode("utf-8") if available["raw"] else b"",
    }
    writes["analysis.md"] = _render_markdown(payload)
    hashes = artifact_store.publish_regular_bundle(output, writes)
    availability = {
        "summary.txt": available["summary"],
        "summary.stderr.txt": bool(results["summary"]["stderr"]),
        "details.txt": available["details"],
        "details.stderr.txt": bool(results["details"]["stderr"]),
        "raw.csv": available["raw"],
        "analysis.md": True,
    }
    payload["artifacts"] = {
        name: {
            "sha256": hashes[name],
            "size": len(writes[name]),
            "available": availability[name],
        }
        for name in SUPPORTING_FILES
    }
    artifact_store.atomic_write_json(marker, payload)
    return 0 if status == "success" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", metavar="REPORT")
    parser.add_argument("--source", metavar="SOURCE")
    parser.add_argument("--out-dir", required=True, metavar="OUTPUT")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--ncu-num", type=_positive_int_argument, default=5)
    parser.add_argument("--timeout", type=_positive_float_argument, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run_analysis(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
