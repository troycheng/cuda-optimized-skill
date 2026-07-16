#!/usr/bin/env python3
"""Best-effort GPU telemetry for timing-block stability checks."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from collections.abc import Mapping
from numbers import Real


_QUERY_FIELDS = (
    "temperature.gpu",
    "clocks.sm",
    "power.draw",
    "memory.used",
    "utilization.gpu",
)
_RESULT_FIELDS = (
    "temperature_c",
    "sm_clock_mhz",
    "power_w",
    "memory_used_mb",
    "gpu_utilization_pct",
)


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite real number")
    return parsed


def _nonnegative_finite(value, name: str) -> float:
    parsed = _finite_real(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _physical_metric(value, field: str, name: str) -> float:
    parsed = _finite_real(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    if field == "gpu_utilization_pct" and parsed > 100:
        raise ValueError(f"{name} must be at most 100")
    return parsed


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def read_gpu_telemetry(*, timeout: float = 2.0) -> dict:
    """Read the first visible GPU with one ``nvidia-smi`` CSV query.

    Collection is deliberately best effort: command and parsing failures are
    returned as unavailable records so they do not abort a benchmark.
    """
    timeout_value = _finite_real(timeout, "timeout")
    if timeout_value <= 0:
        raise ValueError("timeout must be positive")

    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_value,
        )
    except subprocess.TimeoutExpired:
        return _unavailable("nvidia_smi_timeout")
    except OSError:
        return _unavailable("nvidia_smi_unavailable")

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().replace("\n", " ")
        reason = f"nvidia_smi_exit_{completed.returncode}"
        if detail:
            reason = f"{reason}: {detail}"
        return _unavailable(reason)

    try:
        rows = [
            row
            for row in csv.reader((completed.stdout or "").splitlines())
            if row and any(cell.strip() for cell in row)
        ]
        if not rows or len(rows[0]) != len(_RESULT_FIELDS):
            raise ValueError("expected five CSV fields")
        values = [
            _physical_metric(float(cell.strip()), field, f"nvidia_smi.{field}")
            for field, cell in zip(_RESULT_FIELDS, rows[0])
        ]
    except (TypeError, ValueError):
        return _unavailable("nvidia_smi_parse_error")

    return {
        "available": True,
        **dict(zip(_RESULT_FIELDS, values)),
    }


def _validated_reading(payload, name: str) -> tuple[dict, bool, str | None]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    reading = dict(payload)
    has_metrics = any(field in reading for field in _RESULT_FIELDS)
    if "available" not in reading:
        available = has_metrics
        unavailable_reason = None if has_metrics else f"{name}_no_metrics"
    else:
        available = reading["available"]
        if type(available) is not bool:
            raise ValueError(f"{name}.available must be a bool")
        unavailable_reason = f"{name}_unavailable" if not available else None

    for field in _RESULT_FIELDS:
        if field in reading:
            _physical_metric(reading[field], field, f"{name}.{field}")
    return reading, available, unavailable_reason


def validate_block(
    before,
    after,
    max_temperature_delta_c: float = 5,
    max_clock_delta_pct: float = 5,
) -> dict:
    """Validate environment stability across one paired timing block.

    Missing telemetry produces an explicit unknown result, not an invalid one.
    The caller's mappings are never modified.
    """
    temperature_limit = _nonnegative_finite(
        max_temperature_delta_c, "max_temperature_delta_c"
    )
    clock_limit = _nonnegative_finite(
        max_clock_delta_pct, "max_clock_delta_pct"
    )
    before_copy, before_available, before_unknown = _validated_reading(
        before, "before"
    )
    after_copy, after_available, after_unknown = _validated_reading(
        after, "after"
    )

    invalid_reasons: list[str] = []
    unknown_reasons: list[str] = []
    temperature_delta = None
    clock_delta_pct = None
    clock_delta_capped = False

    if not before_available:
        unknown_reasons.append(before_unknown or "before_unavailable")
    if not after_available:
        unknown_reasons.append(after_unknown or "after_unavailable")

    if before_available and after_available:
        before_temperature = before_copy.get("temperature_c")
        after_temperature = after_copy.get("temperature_c")
        if before_temperature is None or after_temperature is None:
            unknown_reasons.append("temperature_delta_unknown")
        else:
            temperature_delta = abs(
                float(after_temperature) - float(before_temperature)
            )
            if not math.isfinite(temperature_delta):
                raise ValueError("temperature_delta_c must be finite")
            if temperature_delta > temperature_limit:
                invalid_reasons.append("temperature_delta")

        before_clock = before_copy.get("sm_clock_mhz")
        after_clock = after_copy.get("sm_clock_mhz")
        if (
            before_clock is None
            or after_clock is None
            or before_clock == 0
            or after_clock == 0
        ):
            unknown_reasons.append("clock_delta_unknown")
        else:
            clock_delta_pct = (
                abs(float(after_clock) - float(before_clock))
                / float(before_clock)
                * 100.0
            )
            if not math.isfinite(clock_delta_pct):
                clock_delta_pct = sys.float_info.max
                clock_delta_capped = True
                invalid_reasons.append("clock_delta")
            elif clock_delta_pct > clock_limit:
                invalid_reasons.append("clock_delta")

    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "telemetry_status": "unknown" if unknown_reasons else "available",
        "unknown_reasons": unknown_reasons,
        "temperature_delta_c": temperature_delta,
        "clock_delta_pct": clock_delta_pct,
        "clock_delta_capped": clock_delta_capped,
    }
