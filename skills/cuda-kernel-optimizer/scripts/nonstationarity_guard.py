#!/usr/bin/env python3
"""Read-only V2.8 comparability guard for nonstationary serving evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path, PureWindowsPath

import artifact_store


_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ORDERS = {"AB", "BA"}
_ROLES = {"baseline", "candidate"}
_PHASES = {"burn_in", "timed"}
_MODES = {"time_windows", "count_windows"}
_RECOMMENDATIONS = {
    "unsupported_measurement_mode": "use_fixed_duration_time_windows",
    "duration_out_of_bounds": "keep_duration_within_frozen_bounds",
    "phase_shift": "increase_predeclared_burn_in_or_fix_state",
    "state_pair_mismatch": "isolate_state_or_narrow_the_experiment",
    "unusable_observation": "recollect_complete_predeclared_blocks",
    "insufficient_complete_blocks": "collect_more_predeclared_blocks",
}


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_bytes(raw: bytes, field: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def load_json_strict(path: Path | str) -> dict:
    return load_json_bytes(artifact_store.read_regular_bytes(path), str(path))


def _closed(value: object, *, keys: set[str], field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    unknown = set(value) - keys
    missing = keys - set(value)
    if unknown:
        raise ValueError(f"{field} has unknown keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{field} is missing keys: {sorted(missing)}")
    return value


def _string(value: object, field: str, *, safe_id: bool = False) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if safe_id and _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a stable identifier")
    return value


def _finite(value: object, field: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _artifact_ref(value: object, field: str) -> dict:
    reference = _closed(value, keys={"path", "sha256"}, field=field)
    raw_path = _string(reference["path"], f"{field}.path")
    path = Path(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or raw_path in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part in {"", ".", ".."} for part in windows_path.parts)
    ):
        raise ValueError(f"{field}.path must be a safe relative artifact path")
    return {"path": raw_path, "sha256": _sha256(reference["sha256"], f"{field}.sha256")}


def _validate_anchor(value: object) -> dict:
    anchor = _closed(
        value,
        keys={"schema_version", "design_sha256", "design_artifact"},
        field="anchor",
    )
    if anchor["schema_version"] != 1:
        raise ValueError("anchor.schema_version must be 1")
    return {
        "schema_version": 1,
        "design_sha256": _sha256(anchor["design_sha256"], "anchor.design_sha256"),
        "design_artifact": _artifact_ref(anchor["design_artifact"], "anchor.design_artifact"),
    }


def canonical_digest(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _metric(value: object, field: str) -> dict:
    metric = _closed(value, keys={"name", "unit", "direction"}, field=field)
    direction = _string(metric["direction"], f"{field}.direction")
    if direction not in {"lower", "higher"}:
        raise ValueError(f"{field}.direction is not supported")
    return {
        "name": _string(metric["name"], f"{field}.name", safe_id=True),
        "unit": _string(metric["unit"], f"{field}.unit", safe_id=True),
        "direction": direction,
    }


def _optional_tolerance(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field, minimum=0)


def _validate_design(value: object) -> dict:
    design = _closed(
        value,
        keys={
            "schema_version", "metric", "measurement", "assignment_method",
            "blocks", "state_dimensions",
        },
        field="design",
    )
    if design["schema_version"] != 1:
        raise ValueError("design.schema_version must be 1")
    measurement = _closed(
        design["measurement"],
        keys={
            "mode", "minimum_duration_ms", "maximum_duration_ms",
            "burn_in_rows_per_segment", "minimum_complete_blocks",
        },
        field="design.measurement",
    )
    mode = _string(measurement["mode"], "design.measurement.mode")
    if mode not in _MODES:
        raise ValueError("design.measurement.mode is not supported")
    minimum_duration = _finite(
        measurement["minimum_duration_ms"], "design.measurement.minimum_duration_ms", minimum=0
    )
    maximum_duration = _finite(
        measurement["maximum_duration_ms"], "design.measurement.maximum_duration_ms", minimum=0
    )
    if minimum_duration <= 0 or maximum_duration < minimum_duration:
        raise ValueError("design measurement duration bounds are invalid")
    burn_in_rows = _integer(
        measurement["burn_in_rows_per_segment"],
        "design.measurement.burn_in_rows_per_segment",
        minimum=1,
    )
    minimum_blocks = _integer(
        measurement["minimum_complete_blocks"],
        "design.measurement.minimum_complete_blocks",
        minimum=4,
    )
    assignment_method = _string(design["assignment_method"], "design.assignment_method")
    if assignment_method != "site_randomized_balanced":
        raise ValueError("design assignment must be site_randomized_balanced")
    blocks_value = design["blocks"]
    if not isinstance(blocks_value, list) or len(blocks_value) < 4:
        raise ValueError("design.blocks must contain at least four blocks")
    blocks = []
    for index, value_item in enumerate(blocks_value):
        item = _closed(value_item, keys={"block_id", "order"}, field=f"design.blocks[{index}]")
        order = _string(item["order"], f"design.blocks[{index}].order")
        if order not in _ORDERS:
            raise ValueError("design block order must be AB or BA")
        blocks.append({
            "block_id": _string(item["block_id"], f"design.blocks[{index}].block_id", safe_id=True),
            "order": order,
        })
    block_ids = [item["block_id"] for item in blocks]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("design block ids must be unique")
    if abs(sum(item["order"] == "AB" for item in blocks) - sum(item["order"] == "BA" for item in blocks)) > 1:
        raise ValueError("design assignment must remain balanced between AB and BA")
    if minimum_blocks > len(blocks):
        raise ValueError("minimum complete blocks cannot exceed planned blocks")
    dimensions_value = design["state_dimensions"]
    if not isinstance(dimensions_value, list) or not dimensions_value:
        raise ValueError("design.state_dimensions must be a non-empty array")
    dimensions = []
    tolerance_keys = {
        "pair_max_absolute", "pair_max_percent", "phase_max_absolute", "phase_max_percent"
    }
    for index, value_item in enumerate(dimensions_value):
        field = f"design.state_dimensions[{index}]"
        item = _closed(
            value_item,
            keys={"name", "unit", "epsilon", *tolerance_keys},
            field=field,
        )
        result = {
            "name": _string(item["name"], f"{field}.name", safe_id=True),
            "unit": _string(item["unit"], f"{field}.unit", safe_id=True),
            "epsilon": _finite(item["epsilon"], f"{field}.epsilon", minimum=0),
        }
        if result["epsilon"] <= 0:
            raise ValueError(f"{field}.epsilon must be positive")
        for key in tolerance_keys:
            result[key] = _optional_tolerance(item[key], f"{field}.{key}")
        if result["pair_max_absolute"] is None and result["pair_max_percent"] is None:
            raise ValueError(f"{field} must declare a pair tolerance")
        if result["phase_max_absolute"] is None and result["phase_max_percent"] is None:
            raise ValueError(f"{field} must declare a phase tolerance")
        dimensions.append(result)
    names = [item["name"] for item in dimensions]
    if len(names) != len(set(names)):
        raise ValueError("design state dimension names must be unique")
    return {
        "schema_version": 1,
        "metric": _metric(design["metric"], "design.metric"),
        "measurement": {
            "mode": mode,
            "minimum_duration_ms": minimum_duration,
            "maximum_duration_ms": maximum_duration,
            "burn_in_rows_per_segment": burn_in_rows,
            "minimum_complete_blocks": minimum_blocks,
        },
        "assignment_method": assignment_method,
        "blocks": blocks,
        "state_dimensions": dimensions,
    }


def _validate_series(value: object, frozen: dict) -> dict:
    series = _closed(
        value,
        keys={
            "schema_version", "design_sha256", "source_artifact", "metric",
            "measurement_mode", "observations",
        },
        field="series",
    )
    if series["schema_version"] != 1:
        raise ValueError("series.schema_version must be 1")
    if _sha256(series["design_sha256"], "series.design_sha256") != canonical_digest(frozen):
        raise ValueError("series is bound to another design")
    metric = _metric(series["metric"], "series.metric")
    if metric != frozen["metric"]:
        raise ValueError("series metric drifted from the frozen design")
    measurement_mode = _string(series["measurement_mode"], "series.measurement_mode")
    if measurement_mode not in _MODES:
        raise ValueError("series.measurement_mode is not supported")
    if measurement_mode != frozen["measurement"]["mode"]:
        raise ValueError("series measurement mode drifted from the frozen design")
    observations_value = series["observations"]
    if not isinstance(observations_value, list):
        raise ValueError("series.observations must be an array")
    state_names = {item["name"] for item in frozen["state_dimensions"]}
    observations = []
    for index, value_item in enumerate(observations_value):
        field = f"series.observations[{index}]"
        item = _closed(
            value_item,
            keys={
                "sequence_index", "block_id", "segment_index", "role", "phase",
                "duration_ms", "metric_value", "usable", "states",
            },
            field=field,
        )
        role = _string(item["role"], f"{field}.role")
        phase = _string(item["phase"], f"{field}.phase")
        if role not in _ROLES or phase not in _PHASES:
            raise ValueError(f"{field} role or phase is not supported")
        if not isinstance(item["states"], dict) or set(item["states"]) != state_names:
            raise ValueError(f"{field} state dimensions do not match the frozen design")
        states = {
            name: _finite(item["states"][name], f"{field}.states.{name}")
            for name in sorted(state_names)
        }
        metric_value = item["metric_value"]
        if phase == "timed":
            metric_value = _finite(metric_value, f"{field}.metric_value")
        elif metric_value is not None:
            raise ValueError(f"{field}.metric_value must be null during burn-in")
        if type(item["usable"]) is not bool:
            raise ValueError(f"{field}.usable must be boolean")
        observations.append({
            "sequence_index": _integer(item["sequence_index"], f"{field}.sequence_index"),
            "block_id": _string(item["block_id"], f"{field}.block_id", safe_id=True),
            "segment_index": _integer(item["segment_index"], f"{field}.segment_index"),
            "role": role,
            "phase": phase,
            "duration_ms": _finite(item["duration_ms"], f"{field}.duration_ms", minimum=0),
            "metric_value": metric_value,
            "usable": item["usable"],
            "states": states,
        })
    expected = []
    sequence_index = 0
    burn_rows = frozen["measurement"]["burn_in_rows_per_segment"]
    for block in frozen["blocks"]:
        roles = ("baseline", "candidate") if block["order"] == "AB" else ("candidate", "baseline")
        for segment_index, role in enumerate(roles):
            for _ in range(burn_rows):
                expected.append((sequence_index, block["block_id"], segment_index, role, "burn_in"))
                sequence_index += 1
            expected.append((sequence_index, block["block_id"], segment_index, role, "timed"))
            sequence_index += 1
    actual = [
        (row["sequence_index"], row["block_id"], row["segment_index"], row["role"], row["phase"])
        for row in observations
    ]
    if actual != expected:
        raise ValueError("series observations do not match the frozen chronological plan")
    return {
        "schema_version": 1,
        "design_sha256": series["design_sha256"],
        "source_artifact": _artifact_ref(series["source_artifact"], "series.source_artifact"),
        "metric": metric,
        "measurement_mode": measurement_mode,
        "observations": observations,
    }


def _within_tolerance(
    reference: float,
    observed: float,
    *,
    maximum_absolute: float | None,
    maximum_percent: float | None,
    epsilon: float,
) -> bool:
    absolute = abs(observed - reference)
    percent = 100.0 * absolute / max(abs(reference), epsilon)
    return (maximum_absolute is None or absolute <= maximum_absolute) and (
        maximum_percent is None or percent <= maximum_percent
    )


def evaluate(
    design: object,
    series: object,
    *,
    expected_design_sha256: str,
    anchor_sha256: str,
) -> dict:
    frozen = _validate_design(design)
    expected_design = _sha256(expected_design_sha256, "expected_design_sha256")
    frozen_anchor = _sha256(anchor_sha256, "anchor_sha256")
    if canonical_digest(frozen) != expected_design:
        raise ValueError("design does not match the frozen design anchor")
    observed = _validate_series(series, frozen)
    reasons: set[str] = set()
    failed_blocks = []
    passed_blocks = 0
    if frozen["measurement"]["mode"] != "time_windows":
        reasons.add("unsupported_measurement_mode")
    block_rows = {
        block["block_id"]: [
            row for row in observed["observations"] if row["block_id"] == block["block_id"]
        ]
        for block in frozen["blocks"]
    }
    for block in frozen["blocks"]:
        block_reasons: set[str] = set()
        rows = block_rows[block["block_id"]]
        if any(not row["usable"] for row in rows):
            block_reasons.add("unusable_observation")
        if any(
            row["duration_ms"] < frozen["measurement"]["minimum_duration_ms"]
            or row["duration_ms"] > frozen["measurement"]["maximum_duration_ms"]
            for row in rows
        ):
            block_reasons.add("duration_out_of_bounds")
        timed_by_role = {
            role: next(row for row in rows if row["role"] == role and row["phase"] == "timed")
            for role in sorted(_ROLES)
        }
        for dimension in frozen["state_dimensions"]:
            name = dimension["name"]
            for role in sorted(_ROLES):
                burn_rows = [row for row in rows if row["role"] == role and row["phase"] == "burn_in"]
                if not _within_tolerance(
                    burn_rows[-1]["states"][name],
                    timed_by_role[role]["states"][name],
                    maximum_absolute=dimension["phase_max_absolute"],
                    maximum_percent=dimension["phase_max_percent"],
                    epsilon=dimension["epsilon"],
                ):
                    block_reasons.add("phase_shift")
            if not _within_tolerance(
                timed_by_role["baseline"]["states"][name],
                timed_by_role["candidate"]["states"][name],
                maximum_absolute=dimension["pair_max_absolute"],
                maximum_percent=dimension["pair_max_percent"],
                epsilon=dimension["epsilon"],
            ):
                block_reasons.add("state_pair_mismatch")
        if block_reasons:
            reasons.update(block_reasons)
            failed_blocks.append({
                "block_id": block["block_id"],
                "reasons": sorted(block_reasons),
            })
        elif frozen["measurement"]["mode"] == "time_windows":
            passed_blocks += 1
    if passed_blocks < frozen["measurement"]["minimum_complete_blocks"]:
        reasons.add("insufficient_complete_blocks")
    status = "comparable_paired_state" if not reasons else "inconclusive_nonstationary"
    recommendations = sorted({_RECOMMENDATIONS[reason] for reason in reasons})
    return {
        "schema_version": 1,
        "status": status,
        "performance_gain_claimed": False,
        "anchor_sha256": frozen_anchor,
        "design_sha256": canonical_digest(frozen),
        "series_sha256": canonical_digest(observed),
        "source_sha256": observed["source_artifact"]["sha256"],
        "total_blocks": len(frozen["blocks"]),
        "complete_blocks": passed_blocks,
        "reasons": sorted(reasons),
        "failed_blocks": failed_blocks,
        "next_experiment": {
            "action": (
                "proceed_to_performance_gate"
                if status == "comparable_paired_state"
                else "redesign_before_measurement"
            ),
            "recommendations": recommendations,
            "host_policy": "recommend_only",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check predeclared serving state comparability without running a target."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="freeze one design in a create-once anchor")
    init.add_argument("--design", required=True)
    init.add_argument("--anchor", required=True)
    check = commands.add_parser("check", help="evaluate one normalized chronological series")
    check.add_argument("--anchor", required=True)
    check.add_argument("--series", required=True)
    return parser


def _freeze_anchor(design_path: Path | str, anchor_path: Path | str) -> dict:
    design_file = Path(design_path)
    anchor_file = Path(anchor_path)
    if anchor_file.name != "nonstationarity-anchor.json":
        raise ValueError("anchor must be named nonstationarity-anchor.json")
    if os.path.abspath(design_file.parent) != os.path.abspath(anchor_file.parent):
        raise ValueError("design and anchor must share one directory")
    raw = artifact_store.read_regular_bytes(design_file)
    frozen = _validate_design(load_json_bytes(raw, str(design_file)))
    return {
        "schema_version": 1,
        "design_sha256": canonical_digest(frozen),
        "design_artifact": {
            "path": design_file.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }


def _load_anchored_design(anchor_path: Path | str) -> tuple[dict, dict]:
    anchor_file = Path(anchor_path)
    if anchor_file.name != "nonstationarity-anchor.json":
        raise ValueError("anchor must be named nonstationarity-anchor.json")
    anchor = _validate_anchor(load_json_strict(anchor_file))
    reference = anchor["design_artifact"]
    design_file = anchor_file.parent / reference["path"]
    raw = artifact_store.read_regular_bytes(design_file)
    design = load_json_bytes(raw, str(design_file))
    frozen = _validate_design(design)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("frozen design artifact digest does not match the create-once anchor")
    if canonical_digest(frozen) != anchor["design_sha256"]:
        raise ValueError("frozen design semantic digest does not match the create-once anchor")
    return design, anchor


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            result = _freeze_anchor(args.design, args.anchor)
            artifact_store.create_regular_json(args.anchor, result)
        else:
            design_value, anchor = _load_anchored_design(args.anchor)
            series_value = load_json_strict(args.series)
            frozen = _validate_design(design_value)
            observed = _validate_series(series_value, frozen)
            source = observed["source_artifact"]
            raw_source = artifact_store.read_regular_bytes(Path(args.series).parent / source["path"])
            if hashlib.sha256(raw_source).hexdigest() != source["sha256"]:
                raise ValueError("series source artifact digest does not match its bound file")
            result = evaluate(
                design_value,
                series_value,
                expected_design_sha256=anchor["design_sha256"],
                anchor_sha256=canonical_digest(anchor),
            )
    except (FileExistsError, OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
