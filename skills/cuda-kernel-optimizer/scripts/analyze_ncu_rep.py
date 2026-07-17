#!/usr/bin/env python3
"""Validate NCU report inputs and provide bounded subprocess execution."""

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
    capture_regular_file(args.report, "REPORT")
    if args.source is not None:
        capture_regular_file(args.source, "SOURCE")
    validate_output_directory(args.out_dir)
    resolve_executable(args.ncu_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
