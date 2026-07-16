#!/usr/bin/env python3
"""Run an explicit Compute Sanitizer correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
ERROR_EXITCODE = 86
DEFAULT_POLICY = (
    Path(__file__).resolve().parent.parent / "references" / "sanitizer_policy.json"
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


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
        normalized_methods[method_id.strip()] = method_tools
    return {
        "schema_version": 1,
        "tools": list(TOOLS),
        "methods": normalized_methods,
    }


def load_policy(path=DEFAULT_POLICY) -> dict:
    policy_path = Path(path).expanduser()
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


def run_tools(*, executable, tools, command) -> dict:
    selected_tools = _string_list(tools, "tools", allow_empty=True)
    unknown = sorted(set(selected_tools) - set(TOOLS))
    if unknown:
        raise ValueError(f"unknown sanitizer tool: {unknown[0]}")
    if len(set(selected_tools)) != len(selected_tools):
        raise ValueError("tools contains duplicates")
    benchmark_command = _string_list(command, "command", allow_empty=False)
    if not selected_tools:
        return {
            "status": "not_applicable",
            "passed": True,
            "coverage": "not_applicable",
            "executable": executable,
            "results": [],
        }

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
            completed = subprocess.run(
                tool_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
                "returncode": completed.returncode,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
                "status": "passed" if completed.returncode == 0 else "failed",
            }
        )

    statuses = {item["status"] for item in results}
    if "failed" in statuses:
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
    return {
        "status": status,
        "passed": passed,
        "coverage": coverage,
        "executable": executable,
        "results": results,
    }


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
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


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
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return ERROR_EXITCODE if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
