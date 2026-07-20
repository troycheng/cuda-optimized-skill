#!/usr/bin/env python3
"""Build the stable, read-only environment identity used by readiness v1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Callable, Mapping


IDENTITY_FIELDS = {
    "toolchain_digest",
    "uid",
    "container_identity",
    "gpu_identity",
    "visible_devices",
    "permission_state",
}
_TOOL_FIELDS = {
    "available",
    "path",
    "realpath",
    "sha256",
    "version",
    "version_query_returncode",
    "usable",
}
_DRIVER_FIELDS = {
    "available",
    "path",
    "driver_versions",
    "max_cuda_version",
    "gpu_identities",
}


def _load_sibling(name: str, module_name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_STORE = _load_sibling("artifact_store.py", "cuda_readiness_identity_store")
_CONTRACT = _load_sibling(
    "readiness_contract.py", "cuda_readiness_identity_contract"
)


class ValidationError(ValueError):
    """Raised when inventory or an identity input is unsafe or open."""


def _strict_copy(value: Any, field: str) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{field} numbers must be finite")
        return value
    if type(value) is list:
        return [
            _strict_copy(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValidationError(f"{field} keys must be non-empty strings")
            result[key] = _strict_copy(item, f"{field}.{key}")
        return result
    raise ValidationError(f"{field} must contain strict JSON values")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_inventory(value: Mapping[str, Any]) -> dict:
    if type(value) is not dict or set(value) != {"tools", "driver"}:
        raise ValidationError("inventory must contain exactly tools and driver")
    tools = value["tools"]
    driver = value["driver"]
    if type(tools) is not dict or type(driver) is not dict:
        raise ValidationError("inventory tools and driver must be objects")
    clean_tools = {}
    for name, item in sorted(tools.items()):
        if type(name) is not str or not name or type(item) is not dict:
            raise ValidationError("inventory.tools entries must be named objects")
        unknown = set(item) - _TOOL_FIELDS
        if unknown:
            raise ValidationError(
                f"inventory.tools.{name} contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        clean_tools[name] = _strict_copy(item, f"inventory.tools.{name}")
    unknown_driver = set(driver) - _DRIVER_FIELDS
    if unknown_driver:
        raise ValidationError(
            "inventory.driver contains unknown fields: "
            + ", ".join(sorted(unknown_driver))
        )
    return {
        "tools": clean_tools,
        "driver": _strict_copy(driver, "inventory.driver"),
    }


def _python_path(environment: Path) -> Path | None:
    candidates = (
        environment / "bin" / "python",
        environment / "bin" / "python3",
        environment / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            return candidate
    return None


def _sha256_optional(path: Path) -> str | None:
    try:
        return _STORE.sha256_file(path)
    except (OSError, ValueError):
        return None


def _distribution_records(environment: Path) -> list[dict]:
    records = []
    patterns = (
        "lib/python*/site-packages/*.dist-info/RECORD",
        "Lib/site-packages/*.dist-info/RECORD",
    )
    for pattern in patterns:
        for path in sorted(environment.glob(pattern)):
            digest = _sha256_optional(path)
            if digest is not None:
                records.append(
                    {
                        "path": path.relative_to(environment).as_posix(),
                        "sha256": digest,
                    }
                )
    return records


def _python_inventory(
    environment: Path, run: Callable[[list[str], int], tuple[int, str, str]]
) -> dict:
    python = _python_path(environment)
    if python is None:
        return {
            "path": None,
            "realpath": None,
            "symlink_target": None,
            "sha256": None,
            "version": None,
            "distributions": [],
            "distribution_records": _distribution_records(environment),
            "pyvenv_cfg_sha256": _sha256_optional(environment / "pyvenv.cfg"),
        }
    metadata = python.lstat()
    realpath = Path(os.path.realpath(python))
    if not realpath.is_file():
        raise ValidationError("isolated Python resolves to a non-regular file")
    version_rc, version_out, version_err = run(
        [str(python), "--version"], timeout=10
    )
    version_text = (version_out + "\n" + version_err).strip()
    version = version_text.splitlines()[0][:512] if version_text else None
    query = (
        "import importlib.metadata as m,json;"
        "print(json.dumps(sorted([[d.metadata.get('Name') or d.name,d.version] "
        "for d in m.distributions()])))"
    )
    dist_rc, dist_out, _dist_err = run(
        [str(python), "-I", "-c", query], timeout=20
    )
    distributions = []
    if dist_rc == 0:
        try:
            value = json.loads(dist_out)
        except json.JSONDecodeError as error:
            raise ValidationError(
                f"isolated Python distribution inventory is invalid: {error}"
            ) from error
        if type(value) is not list:
            raise ValidationError("isolated Python distributions must be an array")
        for index, item in enumerate(value):
            if (
                type(item) is not list
                or len(item) != 2
                or any(type(part) is not str or not part for part in item)
            ):
                raise ValidationError(
                    f"isolated Python distribution {index} is invalid"
                )
        distributions = value
    return {
        "path": str(python),
        "realpath": str(realpath),
        "symlink_target": (
            os.readlink(python) if stat.S_ISLNK(metadata.st_mode) else None
        ),
        "sha256": _sha256_optional(realpath),
        "version": version,
        "version_query_returncode": version_rc,
        "distribution_query_returncode": dist_rc,
        "distributions": distributions,
        "distribution_records": _distribution_records(environment),
        "pyvenv_cfg_sha256": _sha256_optional(environment / "pyvenv.cfg"),
    }


def build_identity(
    *,
    environment_root: Path,
    inventory: Mapping[str, Any],
    run: Callable = None,
) -> dict:
    """Build the exact readiness-gate identity without claiming capability."""
    environment = _CONTRACT._safe_root(
        environment_root, "environment_root"
    )
    clean_inventory = _validate_inventory(inventory)
    if run is None:
        check_env = _load_sibling(
            "check_env.py", "cuda_readiness_identity_check_env"
        )
        run = check_env._run
    python = _python_inventory(environment, run)
    driver = clean_inventory["driver"]
    stable_payload = {
        "tools": clean_inventory["tools"],
        "driver_versions": driver.get("driver_versions", []),
        "max_cuda_version": driver.get("max_cuda_version"),
        "python": python,
    }
    gpu_identities = driver.get("gpu_identities", [])
    configured_gpu = os.environ.get("CUDA_OPTIMIZER_GPU_IDENTITY")
    gpu_identity = configured_gpu
    if gpu_identity is None and gpu_identities:
        gpu_identity = hashlib.sha256(
            _canonical_bytes(gpu_identities)
        ).hexdigest()
    return {
        "toolchain_digest": hashlib.sha256(
            _canonical_bytes(stable_payload)
        ).hexdigest(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "container_identity": os.environ.get("CUDA_OPTIMIZER_CONTAINER_ID"),
        "gpu_identity": gpu_identity,
        "visible_devices": {
            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
        "permission_state": os.environ.get(
            "CUDA_OPTIMIZER_COUNTER_PERMISSION"
        ),
    }
