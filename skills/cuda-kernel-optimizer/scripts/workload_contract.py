#!/usr/bin/env python3
"""Freeze and verify the immutable identity of one optimization workload."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DRAFT_SCHEMA = "cuda-optimizer/workload-contract-draft-v1"
FROZEN_SCHEMA = "cuda-optimizer/workload-contract-v1"
_CLAIMS = {"kernel", "workload", "serving"}
_DIRECTIONS = {"lower", "higher"}
_BUDGETS = {"quick", "balanced", "thorough"}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _load_artifact_store():
    path = Path(__file__).with_name("artifact_store.py")
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_artifact_store_contract", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ARTIFACT_STORE = _load_artifact_store()


class ValidationError(ValueError):
    """Raised when a workload contract is open, unsafe, or inconsistent."""


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_number(token: str):
    raise ValidationError(f"JSON number must be finite: {token}")


def load_json_strict(path: str | os.PathLike) -> dict:
    """Read a regular JSON object without following symlinks."""
    try:
        raw = _ARTIFACT_STORE.read_regular_bytes(path)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_invalid_number,
        )
    except ValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValidationError(f"invalid or unsafe JSON file {path}: {error}") from error
    if type(value) is not dict:
        raise ValidationError("contract JSON root must be an object")
    return value


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{field} must be an object")
    return value


def _closed(value: Mapping, fields: set[str], name: str) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValidationError(f"{name} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValidationError(f"{name} is missing required fields: {', '.join(missing)}")


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


def _finite(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be a finite number")
    if number < minimum:
        raise ValidationError(f"{field} must be at least {minimum}")
    return number


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{field} must be a positive integer")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{field} must be a non-empty array")
    result = [_string(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValidationError(f"{field} must not contain duplicates")
    return result


def _absolute(value: Any, field: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(_string(value, field))))
    if not Path(_string(value, field)).expanduser().is_absolute():
        raise ValidationError(f"{field} must be an absolute path")
    return path


def _relative(value: Any, field: str) -> Path:
    text = _string(value, field)
    path = Path(text)
    if path.is_absolute() or text in {".", ".."} or ".." in path.parts:
        raise ValidationError(f"{field} must be a contained relative path")
    normalized = Path(os.path.normpath(text))
    if str(normalized) in {"", ".", ".."} or ".." in normalized.parts:
        raise ValidationError(f"{field} must be a contained relative path")
    return normalized


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _copy_json(value: Any, field: str = "contract") -> Any:
    if value is None or type(value) in {bool, str, int}:
        return copy.deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{field} numbers must be finite")
        return value
    if type(value) is list:
        return [_copy_json(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValidationError(f"{field} keys must be non-empty strings")
            result[key] = _copy_json(item, f"{field}.{key}")
        return result
    raise ValidationError(f"{field} must contain JSON-compatible values")


def _validate_common(value: Mapping[str, Any], *, frozen: bool) -> dict:
    contract = _object(value, "contract")
    fields = {
        "schema_version",
        "run_id",
        "requested_claim",
        "project_root",
        "artifacts",
        "workload",
        "objective",
        "budget",
        "mutation",
        "evidence",
    }
    if frozen:
        fields.add("contract_sha256")
    _closed(contract, fields, "contract")
    expected_schema = FROZEN_SCHEMA if frozen else DRAFT_SCHEMA
    if contract["schema_version"] != expected_schema:
        raise ValidationError(f"schema_version must be {expected_schema}")
    _identifier(contract["run_id"], "run_id")
    if contract["requested_claim"] not in _CLAIMS:
        raise ValidationError("requested_claim must be kernel, workload, or serving")

    project_root = _absolute(contract["project_root"], "project_root")
    if not project_root.is_dir():
        raise ValidationError("project_root must be an existing directory")

    artifacts = contract["artifacts"]
    if type(artifacts) is not list or not artifacts:
        raise ValidationError("artifacts must be a non-empty array")
    roles = set()
    paths = set()
    artifact_fields = {"role", "path", "sha256", "size_bytes"} if frozen else {"role", "path"}
    for index, item in enumerate(artifacts):
        artifact = _object(item, f"artifacts[{index}]")
        _closed(artifact, artifact_fields, f"artifacts[{index}]")
        role = _identifier(artifact["role"], f"artifacts[{index}].role")
        relative = _relative(artifact["path"], f"artifacts[{index}].path")
        if role in roles:
            raise ValidationError("artifact roles must be unique")
        if str(relative) in paths:
            raise ValidationError("artifact paths must be unique")
        roles.add(role)
        paths.add(str(relative))
        candidate = Path(os.path.abspath(project_root / relative))
        if not _inside(candidate, project_root):
            raise ValidationError(f"artifacts[{index}].path escapes project_root")
        if frozen:
            if type(artifact["sha256"]) is not str or _SHA256.fullmatch(artifact["sha256"]) is None:
                raise ValidationError(f"artifacts[{index}].sha256 must be lowercase SHA-256")
            _positive_integer(artifact["size_bytes"], f"artifacts[{index}].size_bytes")

    workload = _object(contract["workload"], "workload")
    _closed(workload, {"argv", "input_distribution", "representative_cases"}, "workload")
    _string_list(workload["argv"], "workload.argv")
    _string(workload["input_distribution"], "workload.input_distribution")
    _string_list(workload["representative_cases"], "workload.representative_cases")

    objective = _object(contract["objective"], "objective")
    _closed(
        objective,
        {"metric", "unit", "direction", "aggregation", "minimum_practical_effect_pct", "constraints"},
        "objective",
    )
    _identifier(objective["metric"], "objective.metric")
    _string(objective["unit"], "objective.unit", max_length=64)
    if objective["direction"] not in _DIRECTIONS:
        raise ValidationError("objective.direction must be lower or higher")
    _identifier(objective["aggregation"], "objective.aggregation")
    _finite(objective["minimum_practical_effect_pct"], "objective.minimum_practical_effect_pct")
    _string_list(objective["constraints"], "objective.constraints")

    budget = _object(contract["budget"], "budget")
    _closed(budget, {"preset", "max_seconds", "max_candidates"}, "budget")
    if budget["preset"] not in _BUDGETS:
        raise ValidationError("budget.preset must be quick, balanced, or thorough")
    _finite(budget["max_seconds"], "budget.max_seconds", minimum=1.0)
    _positive_integer(budget["max_candidates"], "budget.max_candidates")

    mutation = _object(contract["mutation"], "mutation")
    _closed(mutation, {"project_paths", "environment_root", "host_policy"}, "mutation")
    mutation_paths = _string_list(mutation["project_paths"], "mutation.project_paths")
    normalized = []
    for index, item in enumerate(mutation_paths):
        relative = _relative(item, f"mutation.project_paths[{index}]")
        candidate = Path(os.path.abspath(project_root / relative))
        if not _inside(candidate, project_root):
            raise ValidationError(f"mutation.project_paths[{index}] escapes project_root")
        normalized.append(candidate)
    for index, path in enumerate(normalized):
        for other in normalized[index + 1 :]:
            if path == other or _inside(path, other) or _inside(other, path):
                raise ValidationError("mutation.project_paths must not overlap")
    environment_root = _absolute(mutation["environment_root"], "mutation.environment_root")
    if environment_root == project_root or _inside(environment_root, project_root) or _inside(project_root, environment_root):
        raise ValidationError("mutation.environment_root must be isolated from project_root")
    if mutation["host_policy"] != "recommend_only":
        raise ValidationError("mutation.host_policy must be recommend_only")

    evidence = _object(contract["evidence"], "evidence")
    _closed(evidence, {"max_age_seconds"}, "evidence")
    _finite(evidence["max_age_seconds"], "evidence.max_age_seconds", minimum=1.0)

    if frozen:
        digest = contract["contract_sha256"]
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValidationError("contract_sha256 must be lowercase SHA-256")
    return _copy_json(contract)


def validate_draft(value: Mapping[str, Any]) -> dict:
    """Validate and detach a draft contract without reading bound artifacts."""
    return _validate_common(value, frozen=False)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def freeze_contract(value: Mapping[str, Any], out_path: str | os.PathLike) -> dict:
    """Bind regular artifact bytes and create one immutable contract file."""
    draft = validate_draft(value)
    project_root = Path(draft["project_root"])
    frozen = _copy_json(draft)
    frozen["schema_version"] = FROZEN_SCHEMA
    bindings = []
    for artifact in draft["artifacts"]:
        path = project_root / artifact["path"]
        try:
            raw = _ARTIFACT_STORE.read_regular_bytes(path)
        except ValueError as error:
            raise ValidationError(f"artifact is missing, a symlink, or unsafe: {path}") from error
        if not raw:
            raise ValidationError(f"artifact must not be empty: {path}")
        bindings.append(
            {
                "role": artifact["role"],
                "path": artifact["path"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    frozen["artifacts"] = bindings
    frozen["contract_sha256"] = _canonical_digest(frozen)
    _validate_common(frozen, frozen=True)
    try:
        _ARTIFACT_STORE.create_regular_json(out_path, frozen)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot create workload contract: {error}") from error
    return frozen


def verify_frozen_contract(path: str | os.PathLike) -> dict:
    """Rehash the contract and every bound artifact, failing on any drift."""
    frozen = _validate_common(load_json_strict(path), frozen=True)
    expected_digest = frozen["contract_sha256"]
    digest_input = _copy_json(frozen)
    digest_input.pop("contract_sha256")
    # freeze_contract computes the digest before adding contract_sha256.
    actual_digest = _canonical_digest(digest_input)
    if actual_digest != expected_digest:
        raise ValidationError("contract identity changed: contract_sha256 mismatch")
    project_root = Path(frozen["project_root"])
    for artifact in frozen["artifacts"]:
        source = project_root / artifact["path"]
        try:
            raw = _ARTIFACT_STORE.read_regular_bytes(source)
        except ValueError as error:
            raise ValidationError(f"artifact identity changed or is unsafe: {source}") from error
        if len(raw) != artifact["size_bytes"] or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            raise ValidationError(f"artifact identity changed: {artifact['role']} sha256 mismatch")
    return frozen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze a draft contract")
    freeze.add_argument("--input", required=True)
    freeze.add_argument("--out", required=True)
    verify = subparsers.add_parser("verify", help="verify a frozen contract")
    verify.add_argument("--input", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        result = freeze_contract(load_json_strict(args.input), args.out)
    else:
        result = verify_frozen_contract(args.input)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

