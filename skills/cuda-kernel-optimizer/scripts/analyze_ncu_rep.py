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


def capture_regular_file(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Capture regular bytes without resolving or following symlinks."""
    physical_path = _physical_absolute(path)
    try:
        mode = os.lstat(physical_path).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"artifact file does not exist: {physical_path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"artifact path is not a regular file: {physical_path}")
    payload = artifact_store.read_regular_bytes(physical_path)
    return {
        "path": physical_path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _assert_no_symlink_path(path: str) -> None:
    target = Path(path)
    current = Path(target.anchor)
    for index, component in enumerate(target.parts[1:]):
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as error:
            raise ValueError(f"path does not exist: {target}") from error
        # macOS compatibility aliases (/var and /tmp) may be symlinks directly
        # below the filesystem root; descendants remain strictly no-follow.
        if index and stat.S_ISLNK(mode):
            raise ValueError(f"path contains a symlink: {target}")


def validate_output_directory(path: os.PathLike[str] | str) -> str:
    physical_path = _physical_absolute(path)
    _assert_no_symlink_path(physical_path)
    if not stat.S_ISDIR(os.lstat(physical_path).st_mode):
        raise ValueError(f"output path is not a directory: {physical_path}")
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
    readers = [
        threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None and not any(reader.is_alive() for reader in readers):
                break
            time.sleep(0.01)
        else:
            timed_out = True
        if not timed_out and process.poll() is not None and any(reader.is_alive() for reader in readers):
            timed_out = True
        if timed_out:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            grace_deadline = time.monotonic() + 0.2
            while time.monotonic() < grace_deadline and any(reader.is_alive() for reader in readers):
                time.sleep(0.01)
            if any(reader.is_alive() for reader in readers):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.wait()
        for reader in readers:
            reader.join(timeout=1)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        for reader in readers:
            reader.join(timeout=1)
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
    parser.add_argument("--report", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--ncu-num", type=_positive_int_argument, default=1)
    parser.add_argument("--timeout", type=_positive_float_argument, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    capture_regular_file(args.report)
    capture_regular_file(args.source)
    validate_output_directory(args.output)
    resolve_executable(args.ncu_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
