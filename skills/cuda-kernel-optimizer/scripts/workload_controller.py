#!/usr/bin/env python3
"""Strict contracts and orchestration entry point for workload optimization."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CONTROL_SCHEMA = "cuda-workload-optimizer/control-v1"
CHANGE_SCHEMA = "cuda-workload-optimizer/change-v1"
_BUDGETS = {"fast", "balanced", "thorough"}
_PROBE_KINDS = {
    "environment",
    "timeline",
    "framework",
    "cpu_data",
    "transfer",
    "communication",
    "io",
    "custom",
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|cookie|credential|password|secret|token)($|[_-])",
    re.IGNORECASE,
)


class ValidationError(ValueError):
    """Raised when a workload-controller contract is not closed and safe."""


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: os.PathLike[str] | str) -> dict:
    """Load one JSON object while rejecting duplicate keys and non-finite numbers."""
    source = Path(path)
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError(f"cannot read JSON file {source}: {error}") from error
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda token: (_raise_invalid_number(token)),
        )
    except ValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValidationError(f"invalid JSON in {source}: {error}") from error
    if type(value) is not dict:
        raise ValidationError(f"JSON root must be an object: {source}")
    return value


def _raise_invalid_number(token: str):
    raise ValidationError(f"JSON number must be finite: {token}")


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{field} must be an object")
    return value


def _closed(value: dict, allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValidationError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _required(value: dict, required: set[str], field: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValidationError(f"{field} is missing required fields: {', '.join(missing)}")


def _string(value: Any, field: str, *, max_length: int = 4096) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    if len(value) > max_length:
        raise ValidationError(f"{field} exceeds {max_length} characters")
    return value


def _identifier(value: Any, field: str) -> str:
    text = _string(value, field, max_length=128)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValidationError(f"{field} must be a safe identifier")
    return text


def _timeout(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or not 1 <= number <= 3600:
        raise ValidationError(f"{field} must be between 1 and 3600 seconds")
    return number


def _argv(value: Any, field: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{field} argv must be a non-empty array")
    result = []
    for index, item in enumerate(value):
        result.append(_string(item, f"{field} argv[{index}]"))
    return result


def _absolute(value: Any, field: str) -> Path:
    text = _string(value, field)
    expanded = Path(os.path.expanduser(text))
    if not expanded.is_absolute():
        raise ValidationError(f"{field} must be an absolute path")
    return expanded.resolve(strict=False)


def _relative(value: Any, field: str) -> Path:
    text = _string(value, field)
    path = Path(text)
    if path.is_absolute() or text in {".", ".."} or ".." in path.parts:
        raise ValidationError(f"{field} must be a contained relative path")
    normalized = Path(os.path.normpath(text))
    if str(normalized) in {"", ".", ".."} or ".." in normalized.parts:
        raise ValidationError(f"{field} must be a contained relative path")
    return normalized


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _json_copy(value: Any, field: str, *, reject_sensitive: bool = False) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return copy.deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{field} numbers must be finite")
        return value
    if type(value) is list:
        return [
            _json_copy(item, f"{field}[{index}]", reject_sensitive=reject_sensitive)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValidationError(f"{field} keys must be non-empty strings")
            if reject_sensitive and _SENSITIVE_KEY.search(key):
                raise ValidationError(f"{field} must not contain credentials: {key}")
            result[key] = _json_copy(
                item, f"{field}.{key}", reject_sensitive=reject_sensitive
            )
        return result
    raise ValidationError(f"{field} must contain JSON-compatible values")


def _string_list(value: Any, field: str, *, identifiers: bool = False) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{field} must be a non-empty array")
    result = []
    for index, item in enumerate(value):
        if identifiers:
            result.append(_identifier(item, f"{field}[{index}]"))
        else:
            result.append(_string(item, f"{field}[{index}]"))
    if len(set(result)) != len(result):
        raise ValidationError(f"{field} must not contain duplicates")
    return result


def validate_control_manifest(value: Mapping[str, Any], source_path=None) -> dict:
    """Validate and detach the closed v2.4 controller manifest."""
    control = _object(value, "control")
    allowed = {
        "schema_version",
        "project_root",
        "workload_manifest",
        "baseline_candidate",
        "budget",
        "mutation",
        "probes",
        "reviewer",
    }
    required = allowed - {"reviewer"}
    _closed(control, allowed, "control")
    _required(control, required, "control")
    if control["schema_version"] != CONTROL_SCHEMA:
        raise ValidationError(f"schema_version must be {CONTROL_SCHEMA}")

    project_root = _absolute(control["project_root"], "project_root")
    workload_manifest = _absolute(
        control["workload_manifest"], "workload_manifest"
    )
    if not _is_within(workload_manifest, project_root):
        raise ValidationError("workload_manifest must be inside project_root")
    baseline = _object(control["baseline_candidate"], "baseline_candidate")
    if not baseline:
        raise ValidationError("baseline_candidate must not be empty")
    _json_copy(baseline, "baseline_candidate", reject_sensitive=True)
    if control["budget"] not in _BUDGETS:
        raise ValidationError("budget must be fast, balanced, or thorough")

    mutation = _object(control["mutation"], "mutation")
    mutation_fields = {"project_paths", "environment_root", "host_policy"}
    _closed(mutation, mutation_fields, "mutation")
    _required(mutation, mutation_fields, "mutation")
    project_paths = _string_list(mutation["project_paths"], "project_paths")
    normalized_roots = []
    for index, item in enumerate(project_paths):
        relative = _relative(item, f"project_paths[{index}]")
        candidate = (project_root / relative).resolve(strict=False)
        if not _is_within(candidate, project_root):
            raise ValidationError(f"project_paths[{index}] escapes project_root")
        normalized_roots.append(relative)
    environment_root = _absolute(mutation["environment_root"], "environment_root")
    if _is_within(environment_root, project_root):
        raise ValidationError("environment_root must be outside project_root")
    if mutation["host_policy"] != "recommend_only":
        raise ValidationError("host_policy must be recommend_only")

    probes = control["probes"]
    if type(probes) is not list or not probes:
        raise ValidationError("probes must be a non-empty array")
    probe_ids = set()
    for index, item in enumerate(probes):
        probe = _object(item, f"probes[{index}]")
        fields = {"id", "kind", "argv", "timeout_seconds"}
        _closed(probe, fields, f"probes[{index}]")
        _required(probe, fields, f"probes[{index}]")
        probe_id = _identifier(probe["id"], f"probes[{index}].id")
        if probe_id in probe_ids:
            raise ValidationError("probe ids must be unique")
        probe_ids.add(probe_id)
        if probe["kind"] not in _PROBE_KINDS:
            raise ValidationError(f"probes[{index}].kind is unsupported")
        _argv(probe["argv"], f"probes[{index}]")
        _timeout(probe["timeout_seconds"], f"probes[{index}].timeout_seconds")

    reviewer = control.get("reviewer")
    if reviewer is not None:
        reviewer = _object(reviewer, "reviewer")
        fields = {"argv", "timeout_seconds"}
        _closed(reviewer, fields, "reviewer")
        _required(reviewer, fields, "reviewer")
        _argv(reviewer["argv"], "reviewer")
        _timeout(reviewer["timeout_seconds"], "reviewer.timeout_seconds")

    return _json_copy(control, "control", reject_sensitive=True)


def validate_change_set(value: Mapping[str, Any], control: Mapping[str, Any]) -> dict:
    """Validate a bounded project or isolated-environment ChangeSet."""
    change = _object(value, "change_set")
    fields = {
        "schema_version",
        "id",
        "hypothesis",
        "diagnosis_ids",
        "scope",
        "candidate",
        "paths",
        "commands",
        "rollback",
        "expected_metrics",
    }
    _closed(change, fields, "change_set")
    _required(change, fields, "change_set")
    if change["schema_version"] != CHANGE_SCHEMA:
        raise ValidationError(f"change_set.schema_version must be {CHANGE_SCHEMA}")
    _identifier(change["id"], "change_set.id")
    _string(change["hypothesis"], "change_set.hypothesis")
    _string_list(change["diagnosis_ids"], "change_set.diagnosis_ids")
    if change["scope"] not in {"project", "isolated_environment"}:
        raise ValidationError("change_set.scope must be project or isolated_environment")
    candidate = _object(change["candidate"], "change_set.candidate")
    if not candidate:
        raise ValidationError("change_set.candidate must not be empty")
    _json_copy(candidate, "change_set.candidate", reject_sensitive=True)

    paths = _string_list(change["paths"], "change_set.paths")
    relative_paths = [
        _relative(item, f"change_set.paths[{index}]")
        for index, item in enumerate(paths)
    ]
    if change["scope"] == "project":
        allowed_roots = [
            _relative(item, "control.mutation.project_paths")
            for item in control["mutation"]["project_paths"]
        ]
        for index, path in enumerate(relative_paths):
            if not any(path == root or _is_within(path, root) for root in allowed_roots):
                raise ValidationError(
                    f"change_set.paths[{index}] is outside declared project_paths"
                )

    commands = change["commands"]
    if type(commands) is not list:
        raise ValidationError("change_set.commands must be an array of argv arrays")
    for index, command in enumerate(commands):
        if type(command) is not list:
            raise ValidationError("change_set.commands must contain argv arrays")
        _argv(command, f"change_set.commands[{index}]")
    if change["rollback"] != "restore_frozen_snapshot":
        raise ValidationError("change_set.rollback must be restore_frozen_snapshot")
    _string_list(change["expected_metrics"], "change_set.expected_metrics")
    return _json_copy(change, "change_set", reject_sensitive=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and run bounded GPU workload optimization rounds."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate controller JSON")
    validate.add_argument("--control", required=True)
    validate.add_argument("--change-set")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            control = validate_control_manifest(
                load_json_object(args.control), args.control
            )
            if args.change_set:
                validate_change_set(load_json_object(args.change_set), control)
            print(json.dumps({"status": "valid"}, sort_keys=True))
            return 0
    except ValidationError as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
