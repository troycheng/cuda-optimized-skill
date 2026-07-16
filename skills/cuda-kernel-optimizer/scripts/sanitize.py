#!/usr/bin/env python3
"""Run an explicit Compute Sanitizer correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
ERROR_EXITCODE = 86
DEFAULT_POLICY = (
    Path(__file__).resolve().parent.parent / "references" / "sanitizer_policy.json"
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_ACTIVE_TOOL_PROCESS = None
_ACTIVE_TOOL_PGID = None


def _string_list(value, field: str, *, allow_empty: bool) -> list[str]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field} must be a sequence of strings")
    try:
        values = list(value)
    except TypeError as error:
        raise ValueError(f"{field} must be a sequence of strings") from error
    if not allow_empty and not values:
        raise ValueError(f"{field} must not be empty")
    for index, item in enumerate(values):
        if type(item) is not str or not item.strip():
            raise ValueError(f"{field}[{index}] must be a non-empty string")
    return [item.strip() for item in values]


def validate_policy(policy) -> dict:
    if not isinstance(policy, Mapping):
        raise ValueError("sanitizer policy must be a mapping")
    if set(policy) != {"schema_version", "tools", "methods"}:
        raise ValueError(
            "sanitizer policy must contain only schema_version, tools, and methods"
        )
    if policy.get("schema_version") != 1:
        raise ValueError("sanitizer policy schema_version must be 1")
    tools = _string_list(policy.get("tools"), "policy.tools", allow_empty=False)
    if tools != list(TOOLS):
        raise ValueError(
            "policy.tools must be memcheck, racecheck, initcheck, synccheck"
        )
    methods = policy.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("policy.methods must be a mapping")
    normalized_methods = {}
    for method_id, selected in methods.items():
        if type(method_id) is not str or not method_id.strip():
            raise ValueError("policy method ids must be non-empty strings")
        method_tools = _string_list(
            selected, f"policy.methods.{method_id}", allow_empty=False
        )
        if len(set(method_tools)) != len(method_tools):
            raise ValueError(f"policy.methods.{method_id} contains duplicate tools")
        unknown = sorted(set(method_tools) - set(TOOLS))
        if unknown:
            raise ValueError(
                f"policy.methods.{method_id} contains unknown tool: {unknown[0]}"
            )
        normalized_id = method_id.strip()
        if normalized_id in normalized_methods:
            raise ValueError(
                f"policy method id collision after strip: {normalized_id}"
            )
        normalized_methods[normalized_id] = method_tools
    return {
        "schema_version": 1,
        "tools": list(TOOLS),
        "methods": normalized_methods,
    }


def load_policy(path=DEFAULT_POLICY) -> dict:
    policy_path = Path(path).expanduser()
    if policy_path.is_symlink():
        raise ValueError("sanitizer policy must not be a symlink")
    try:
        info = policy_path.lstat()
    except OSError as error:
        raise ValueError(f"sanitizer policy is unreadable: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("sanitizer policy must be a regular file")
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"sanitizer policy is unreadable: {error}") from error
    return validate_policy(payload)


def select_tools(method_ids, *, mode: str, policy) -> list[str]:
    normalized = validate_policy(policy)
    ids = _string_list(method_ids, "method_ids", allow_empty=True)
    if mode == "full":
        return list(normalized["tools"])
    if mode != "targeted":
        raise ValueError("sanitizer mode must be targeted or full")
    selected = {
        tool
        for method_id in ids
        for tool in normalized["methods"].get(method_id, [])
    }
    return [tool for tool in normalized["tools"] if tool in selected]


def _tool_command(executable: str | None, tool: str, command: list[str]) -> list[str]:
    return [
        executable or "compute-sanitizer",
        "--tool",
        tool,
        "--error-exitcode",
        str(ERROR_EXITCODE),
        *command,
    ]


def validate_result(result, *, selected_tools, command=None) -> dict:
    if not isinstance(result, Mapping):
        raise ValueError("sanitizer result must be a mapping")
    tools = _string_list(selected_tools, "selected_tools", allow_empty=True)
    if len(set(tools)) != len(tools):
        raise ValueError("selected_tools contains duplicates")
    benchmark_command = (
        [] if command is None else _string_list(command, "command", allow_empty=False)
    )
    records = result.get("results")
    if not isinstance(records, list) or len(records) != len(tools):
        raise ValueError("sanitizer results must align with selected_tools")
    statuses = []
    required = {"tool", "command", "returncode", "stdout", "stderr", "status"}
    for index, (tool, record) in enumerate(zip(tools, records)):
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError(f"sanitizer results[{index}] fields are invalid")
        if record.get("tool") != tool:
            raise ValueError(f"sanitizer results[{index}].tool is out of order")
        tool_command = record.get("command")
        if not isinstance(tool_command, list) or len(tool_command) < 5:
            raise ValueError(f"sanitizer results[{index}].command is invalid")
        if tool_command[1:5] != [
            "--tool",
            tool,
            "--error-exitcode",
            str(ERROR_EXITCODE),
        ]:
            raise ValueError(
                f"sanitizer results[{index}].command lacks the required gate flags"
            )
        if benchmark_command and tool_command[5:] != benchmark_command:
            raise ValueError(
                f"sanitizer results[{index}].command benchmark drifted"
            )
        if type(record.get("stdout")) is not str or type(record.get("stderr")) is not str:
            raise ValueError(
                f"sanitizer results[{index}] stdout/stderr must be strings"
            )
        tool_status = record.get("status")
        returncode = record.get("returncode")
        if tool_status == "passed":
            if returncode != 0 or isinstance(returncode, bool):
                raise ValueError(
                    f"sanitizer results[{index}] passed requires returncode 0"
                )
        elif tool_status == "failed":
            if (
                isinstance(returncode, bool)
                or not isinstance(returncode, int)
                or returncode == 0
            ):
                raise ValueError(
                    f"sanitizer results[{index}] failed requires nonzero returncode"
                )
        elif tool_status in {"unavailable", "timed_out"}:
            if returncode is not None:
                raise ValueError(
                    f"sanitizer results[{index}] {tool_status} requires null returncode"
                )
        else:
            raise ValueError(f"sanitizer results[{index}].status is invalid")
        statuses.append(tool_status)

    status_set = set(statuses)
    if not tools:
        expected = ("not_applicable", True, "not_applicable")
    elif "timed_out" in status_set:
        expected = ("timed_out", False, "incomplete")
    elif "failed" in status_set:
        expected = (
            "failed",
            False,
            "degraded" if "unavailable" in status_set else "complete",
        )
    elif "unavailable" in status_set:
        expected = ("unavailable", False, "degraded")
    else:
        expected = ("passed", True, "complete")
    actual = (result.get("status"), result.get("passed"), result.get("coverage"))
    if actual != expected:
        raise ValueError("sanitizer top-level status/passed/coverage is contradictory")
    return dict(result)


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_gone(process_group_id: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.01)
    return not _process_group_exists(process_group_id)


def _terminate_tool_group(process_group_id: int, grace_seconds: float = 0.2) -> None:
    if not _process_group_exists(process_group_id):
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_process_group_gone(process_group_id, grace_seconds):
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    if not _wait_process_group_gone(process_group_id, 1.0):
        raise RuntimeError("compute-sanitizer process group survived SIGKILL")


def _terminate_active_tool(process, process_group_id: int):
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        stdout, stderr = process.communicate(timeout=0.2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    _terminate_tool_group(process_group_id, grace_seconds=0.05)
    return stdout or "", stderr or ""


def _forward_termination(signum, _frame) -> None:
    process = _ACTIVE_TOOL_PROCESS
    process_group_id = _ACTIVE_TOOL_PGID
    if process_group_id is not None and process is not None:
        _terminate_active_tool(process, process_group_id)
    raise SystemExit(128 + signum)


def _install_signal_forwarding() -> None:
    signal.signal(signal.SIGTERM, _forward_termination)
    signal.signal(signal.SIGINT, _forward_termination)


def _run_tool_with_timeout(command: list[str], timeout_seconds: float):
    global _ACTIVE_TOOL_PROCESS, _ACTIVE_TOOL_PGID
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    _ACTIVE_TOOL_PROCESS = process
    _ACTIVE_TOOL_PGID = process.pid
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            _terminate_tool_group(process.pid)
            return process.returncode, stdout or "", stderr or "", False
        except subprocess.TimeoutExpired:
            stdout, stderr = _terminate_active_tool(process, process.pid)
            return None, stdout or "", stderr or "", True
    finally:
        _ACTIVE_TOOL_PROCESS = None
        _ACTIVE_TOOL_PGID = None


def run_tools(*, executable, tools, command, timeout_seconds: float = 300.0) -> dict:
    selected_tools = _string_list(tools, "tools", allow_empty=True)
    unknown = sorted(set(selected_tools) - set(TOOLS))
    if unknown:
        raise ValueError(f"unknown sanitizer tool: {unknown[0]}")
    if len(set(selected_tools)) != len(selected_tools):
        raise ValueError("tools contains duplicates")
    benchmark_command = _string_list(command, "command", allow_empty=False)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not float(timeout_seconds) > 0.0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    timeout_seconds = float(timeout_seconds)
    if not selected_tools:
        return validate_result({
            "status": "not_applicable",
            "passed": True,
            "coverage": "not_applicable",
            "executable": executable,
            "results": [],
        }, selected_tools=[], command=None)

    results = []
    unavailable = executable is None
    for tool in selected_tools:
        tool_command = _tool_command(executable, tool, benchmark_command)
        if unavailable:
            results.append(
                {
                    "tool": tool,
                    "command": tool_command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "compute-sanitizer unavailable",
                    "status": "unavailable",
                }
            )
            continue
        try:
            returncode, stdout, stderr, timed_out = _run_tool_with_timeout(
                tool_command, timeout_seconds
            )
        except OSError as error:
            unavailable = True
            results.append(
                {
                    "tool": tool,
                    "command": tool_command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": f"compute-sanitizer unavailable: {error}",
                    "status": "unavailable",
                }
            )
            continue
        results.append(
            {
                "tool": tool,
                "command": tool_command,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "status": (
                    "timed_out"
                    if timed_out
                    else "passed" if returncode == 0 else "failed"
                ),
            }
        )

    statuses = {item["status"] for item in results}
    if "timed_out" in statuses:
        status = "timed_out"
        passed = False
        coverage = "incomplete"
    elif "failed" in statuses:
        status = "failed"
        passed = False
        coverage = "degraded" if "unavailable" in statuses else "complete"
    elif "unavailable" in statuses:
        status = "unavailable"
        passed = False
        coverage = "degraded"
    else:
        status = "passed"
        passed = True
        coverage = "complete"
    result = {
        "status": status,
        "passed": passed,
        "coverage": coverage,
        "executable": executable,
        "results": results,
    }
    return validate_result(
        result, selected_tools=selected_tools, command=benchmark_command
    )


def bind_candidate(result, *, candidate_file, input_hash: str) -> dict:
    if not isinstance(result, Mapping):
        raise ValueError("sanitizer result must be a mapping")
    if type(input_hash) is not str or not _SHA256.fullmatch(input_hash):
        raise ValueError("input_hash must be 64 hexadecimal characters")
    candidate = Path(candidate_file).expanduser()
    if candidate.is_symlink():
        raise ValueError("sanitizer candidate must not be a symlink")
    try:
        info = candidate.lstat()
    except OSError as error:
        raise ValueError(f"sanitizer candidate is missing: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("sanitizer candidate must be a regular file")
    resolved = candidate.resolve(strict=True)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    bound = dict(result)
    bound.update(
        {
            "candidate_file": str(resolved),
            "candidate_sha256": digest,
            "input_hash": input_hash.lower(),
        }
    )
    return bound


def _load_method_ids(path) -> list[str]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"methods.json is unreadable: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("methods"), list):
        raise ValueError("methods.json must contain a top-level methods list")
    method_ids = []
    for index, method in enumerate(payload["methods"]):
        if not isinstance(method, Mapping):
            raise ValueError(f"methods[{index}] must be a mapping")
        method_id = method.get("id")
        if type(method_id) is not str or not method_id.strip():
            raise ValueError(f"methods[{index}].id must be a non-empty string")
        method_ids.append(method_id.strip())
    return method_ids


def _atomic_write_json(path, payload) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_descriptor = os.open(str(destination.parent), flags)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a positive number") from error
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return timeout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Compute Sanitizer tools selected by optimization method"
    )
    parser.add_argument("--mode", required=True, choices=["targeted", "full"])
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--methods-json", required=True)
    parser.add_argument("--compute-sanitizer", default="")
    parser.add_argument("--candidate-file")
    parser.add_argument("--input-hash")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=_positive_timeout, default=300.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _install_signal_forwarding()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a benchmark command is required after --")
    try:
        policy = load_policy(args.policy)
        method_ids = _load_method_ids(args.methods_json)
        tools = select_tools(method_ids, mode=args.mode, policy=policy)
        requested = args.compute_sanitizer.strip()
        executable = shutil.which(requested or "compute-sanitizer")
        result = run_tools(
            executable=executable,
            tools=tools,
            command=command,
            timeout_seconds=args.timeout,
        )
    except ValueError as error:
        parser.error(str(error))
    result.update(
        {
            "mode": args.mode,
            "method_ids": method_ids,
            "selected_tools": tools,
        }
    )
    if (args.candidate_file is None) != (args.input_hash is None):
        parser.error("--candidate-file and --input-hash must be provided together")
    if args.candidate_file is not None:
        try:
            result = bind_candidate(
                result,
                candidate_file=args.candidate_file,
                input_hash=args.input_hash,
            )
        except ValueError as error:
            parser.error(str(error))
    _atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "coverage": result["coverage"],
                "selected_tools": result["selected_tools"],
                "artifact": str(Path(args.out).expanduser().resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if result["status"] == "failed":
        return ERROR_EXITCODE
    return 124 if result["status"] == "timed_out" else 0


if __name__ == "__main__":
    raise SystemExit(main())
