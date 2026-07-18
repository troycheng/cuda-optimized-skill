#!/usr/bin/env python3
"""Read-only direction admission and append-only stop ledger for V2.7."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath

import artifact_store


CLAIM_LAYERS = {"kernel", "runtime", "workload", "serving"}
BOTTLENECK_CLASSES = {
    "kernel",
    "framework",
    "cpu_data",
    "transfer",
    "communication",
    "io",
    "environment",
}
METRIC_DIRECTIONS = {"lower", "higher"}
METRIC_KINDS = {"additive_time", "throughput", "composite"}
REQUESTS = {"admit", "close", "reopen"}
REOPEN_REASONS = {"new_measurement_window", "new_target_identity"}
ACTIONS = {
    "admit_direction",
    "switch_to_higher_impact",
    "close_direction",
    "direction_closed",
    "unrankable",
}
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DECISION_NAME = re.compile(r"decision-([0-9]{4})\.json\Z")


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


def _finite(value: object, field: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _optional_positive(value: object, field: str) -> float | None:
    if value is None:
        return None
    result = _finite(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive when present")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field)


def _optional_nonnegative(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field, minimum=0)


def _canonical_digest(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_objective(value: object) -> dict:
    objective = _closed(
        value,
        keys={
            "claim_layer",
            "metric_name",
            "metric_unit",
            "metric_direction",
            "metric_kind",
            "minimum_effect_absolute",
            "minimum_effect_percent",
        },
        field="portfolio.objective",
    )
    claim_layer = _string(objective["claim_layer"], "objective.claim_layer")
    if claim_layer not in CLAIM_LAYERS:
        raise ValueError("objective.claim_layer is not in the closed taxonomy")
    metric_direction = _string(
        objective["metric_direction"], "objective.metric_direction"
    )
    if metric_direction not in METRIC_DIRECTIONS:
        raise ValueError("objective.metric_direction is not supported")
    metric_kind = _string(objective["metric_kind"], "objective.metric_kind")
    if metric_kind not in METRIC_KINDS:
        raise ValueError("objective.metric_kind is not supported")
    absolute = _optional_positive(
        objective["minimum_effect_absolute"], "objective.minimum_effect_absolute"
    )
    percent = _optional_positive(
        objective["minimum_effect_percent"], "objective.minimum_effect_percent"
    )
    if absolute is None and percent is None:
        raise ValueError("objective must freeze at least one minimum effect")
    return {
        "claim_layer": claim_layer,
        "metric_name": _string(objective["metric_name"], "objective.metric_name", safe_id=True),
        "metric_unit": _string(objective["metric_unit"], "objective.metric_unit", safe_id=True),
        "metric_direction": metric_direction,
        "metric_kind": metric_kind,
        "minimum_effect_absolute": absolute,
        "minimum_effect_percent": percent,
    }


def _validate_direction(value: object, index: int) -> dict:
    field = f"portfolio.directions[{index}]"
    direction = _closed(
        value,
        keys={
            "id",
            "claim_layer",
            "bottleneck_class",
            "target_artifact",
            "component_artifact",
            "component_id",
            "metric_name",
            "metric_unit",
            "metric_direction",
            "metric_kind",
            "total_metric",
            "component_metric",
            "evidence_artifact",
        },
        field=field,
    )
    claim_layer = _string(direction["claim_layer"], f"{field}.claim_layer")
    if claim_layer not in CLAIM_LAYERS:
        raise ValueError(f"{field}.claim_layer is not in the closed taxonomy")
    bottleneck = _string(direction["bottleneck_class"], f"{field}.bottleneck_class")
    if bottleneck not in BOTTLENECK_CLASSES:
        raise ValueError(f"{field}.bottleneck_class is not in the closed taxonomy")
    metric_direction = _string(
        direction["metric_direction"], f"{field}.metric_direction"
    )
    if metric_direction not in METRIC_DIRECTIONS:
        raise ValueError(f"{field}.metric_direction is not supported")
    metric_kind = _string(direction["metric_kind"], f"{field}.metric_kind")
    if metric_kind not in METRIC_KINDS:
        raise ValueError(f"{field}.metric_kind is not supported")
    total = _finite(direction["total_metric"], f"{field}.total_metric")
    component = _finite(direction["component_metric"], f"{field}.component_metric", minimum=0)
    if total <= 0:
        raise ValueError(f"{field}.total_metric must be positive")
    if component > total:
        raise ValueError(f"{field}.component_metric cannot exceed total_metric")
    result = {
        "id": _string(direction["id"], f"{field}.id", safe_id=True),
        "claim_layer": claim_layer,
        "bottleneck_class": bottleneck,
        "target_artifact": _artifact_ref(direction["target_artifact"], f"{field}.target_artifact"),
        "component_artifact": _artifact_ref(
            direction["component_artifact"], f"{field}.component_artifact"
        ),
        "component_id": _string(
            direction["component_id"], f"{field}.component_id", safe_id=True
        ),
        "metric_name": _string(direction["metric_name"], f"{field}.metric_name", safe_id=True),
        "metric_unit": _string(direction["metric_unit"], f"{field}.metric_unit", safe_id=True),
        "metric_direction": metric_direction,
        "metric_kind": metric_kind,
        "total_metric": total,
        "component_metric": component,
        "evidence_artifact": _artifact_ref(
            direction["evidence_artifact"], f"{field}.evidence_artifact"
        ),
    }
    family_fields = {
        key: result[key]
        for key in (
            "claim_layer",
            "bottleneck_class",
            "metric_name",
            "metric_unit",
            "metric_direction",
            "metric_kind",
        )
    }
    family_fields["component_artifact_sha256"] = result["component_artifact"]["sha256"]
    result["direction_family_key"] = _canonical_digest(family_fields)
    result["direction_key"] = _canonical_digest(
        {**family_fields, "target_identity_sha256": result["target_artifact"]["sha256"]}
    )
    return result


def _validate_portfolio(value: object) -> dict:
    portfolio = _closed(
        value,
        keys={
            "schema_version",
            "objective",
            "environment_artifact",
            "measurement_window_artifact",
            "directions",
        },
        field="portfolio",
    )
    if portfolio["schema_version"] != 1:
        raise ValueError("portfolio.schema_version must be 1")
    directions_value = portfolio["directions"]
    if not isinstance(directions_value, list) or not directions_value:
        raise ValueError("portfolio.directions must be a non-empty array")
    directions = [_validate_direction(item, index) for index, item in enumerate(directions_value)]
    ids = [item["id"] for item in directions]
    families = [item["direction_family_key"] for item in directions]
    if len(ids) != len(set(ids)):
        raise ValueError("portfolio direction ids must be unique")
    if len(families) != len(set(families)):
        raise ValueError("portfolio direction families must be unique")
    objective = _validate_objective(portfolio["objective"])
    comparable_totals = {
        item["total_metric"] for item in directions if _comparable(item, objective)
    }
    if len(comparable_totals) > 1:
        raise ValueError("comparable directions must use the same total_metric")
    return {
        "schema_version": 1,
        "objective": objective,
        "environment_artifact": _artifact_ref(
            portfolio["environment_artifact"], "portfolio.environment_artifact"
        ),
        "measurement_window_artifact": _artifact_ref(
            portfolio["measurement_window_artifact"], "portfolio.measurement_window_artifact"
        ),
        "directions": directions,
    }


def verify_portfolio_artifacts(portfolio: object, portfolio_path: Path | str) -> dict:
    """Rehash every portfolio artifact relative to the no-follow portfolio path."""
    validated = _validate_portfolio(portfolio)
    base = Path(portfolio_path).parent
    references = [
        ("environment_artifact", validated["environment_artifact"], None),
        ("measurement_window_artifact", validated["measurement_window_artifact"], None),
    ]
    for direction in validated["directions"]:
        references.extend(
            (
                (f"{direction['id']}.target_artifact", direction["target_artifact"], None),
                (f"{direction['id']}.component_artifact", direction["component_artifact"], None),
                (f"{direction['id']}.evidence_artifact", direction["evidence_artifact"], direction),
            )
        )
    for field, reference, direction in references:
        raw = artifact_store.read_regular_bytes(base / reference["path"])
        actual = hashlib.sha256(raw).hexdigest()
        if actual != reference["sha256"]:
            raise ValueError(f"{field} artifact digest does not match its bound file")
        if direction is not None:
            evidence = _closed(
                load_json_bytes(raw, field),
                keys={
                    "schema_version", "source_artifact", "component_artifact_sha256",
                    "target_artifact_sha256", "measurement_window_sha256",
                    "claim_layer", "bottleneck_class", "metric_name", "metric_unit",
                    "metric_direction", "metric_kind", "total_metric", "component_metric",
                },
                field=field,
            )
            expected = {
                "schema_version": 1,
                "component_artifact_sha256": direction["component_artifact"]["sha256"],
                "target_artifact_sha256": direction["target_artifact"]["sha256"],
                "measurement_window_sha256": validated["measurement_window_artifact"]["sha256"],
                "claim_layer": direction["claim_layer"],
                "bottleneck_class": direction["bottleneck_class"],
                "metric_name": direction["metric_name"],
                "metric_unit": direction["metric_unit"],
                "metric_direction": direction["metric_direction"],
                "metric_kind": direction["metric_kind"],
                "total_metric": direction["total_metric"],
                "component_metric": direction["component_metric"],
            }
            for key, expected_value in expected.items():
                if evidence[key] != expected_value or type(evidence[key]) is bool:
                    raise ValueError(f"{field} evidence field {key} does not match the portfolio")
            source = _artifact_ref(evidence["source_artifact"], f"{field}.source_artifact")
            source_raw = artifact_store.read_regular_bytes(base / source["path"])
            if hashlib.sha256(source_raw).hexdigest() != source["sha256"]:
                raise ValueError(f"{field} source artifact digest does not match its bound file")
    return validated


def freeze_lineage(portfolio: object) -> dict:
    validated = _validate_portfolio(portfolio)
    return {
        "schema_version": 1,
        "objective": copy.deepcopy(validated["objective"]),
        "environment_sha256": validated["environment_artifact"]["sha256"],
        "direction_families": sorted(
            (
                {
                    "direction_family_key": item["direction_family_key"],
                    "baseline_total_metric": item["total_metric"],
                }
                for item in validated["directions"]
            ),
            key=lambda item: item["direction_family_key"],
        ),
        "initial_portfolio_sha256": _canonical_digest(validated),
    }


def _validate_lineage(value: object) -> dict:
    lineage = _closed(
        value,
        keys={
            "schema_version",
            "objective",
            "environment_sha256",
            "direction_families",
            "initial_portfolio_sha256",
        },
        field="lineage",
    )
    if lineage["schema_version"] != 1:
        raise ValueError("lineage.schema_version must be 1")
    families = lineage["direction_families"]
    if not isinstance(families, list) or not families:
        raise ValueError("lineage.direction_families must be a non-empty array")
    validated_families = []
    for index, item in enumerate(families):
        family = _closed(
            item,
            keys={"direction_family_key", "baseline_total_metric"},
            field=f"lineage.direction_families[{index}]",
        )
        total = _finite(
            family["baseline_total_metric"],
            f"lineage.direction_families[{index}].baseline_total_metric",
        )
        if total <= 0:
            raise ValueError("lineage baseline_total_metric must be positive")
        validated_families.append(
            {
                "direction_family_key": _sha256(
                    family["direction_family_key"],
                    f"lineage.direction_families[{index}].direction_family_key",
                ),
                "baseline_total_metric": total,
            }
        )
    keys = [item["direction_family_key"] for item in validated_families]
    if keys != sorted(set(keys)):
        raise ValueError("lineage.direction_families must be sorted and unique")
    return {
        "schema_version": 1,
        "objective": _validate_objective(lineage["objective"]),
        "environment_sha256": _sha256(
            lineage["environment_sha256"], "lineage.environment_sha256"
        ),
        "direction_families": validated_families,
        "initial_portfolio_sha256": _sha256(
            lineage["initial_portfolio_sha256"], "lineage.initial_portfolio_sha256"
        ),
    }


def _family_entry(lineage: Mapping, family_key: str) -> dict:
    matches = [
        item for item in lineage["direction_families"]
        if item["direction_family_key"] == family_key
    ]
    if len(matches) != 1:
        raise ValueError("direction family is not frozen in the lineage")
    return matches[0]


def _effect_reachable(
    direction: Mapping, objective: Mapping, baseline_total_metric: float
) -> tuple[float, float, bool]:
    absolute = float(direction["component_metric"])
    if absolute > baseline_total_metric:
        raise ValueError("component_metric cannot exceed its frozen baseline total")
    percent = 100.0 * absolute / baseline_total_metric
    thresholds = (
        objective["minimum_effect_absolute"],
        objective["minimum_effect_percent"],
    )
    reachable = (thresholds[0] is None or absolute >= thresholds[0]) and (
        thresholds[1] is None or percent >= thresholds[1]
    )
    return absolute, percent, reachable


def _comparable(direction: Mapping, objective: Mapping) -> bool:
    return (
        objective["claim_layer"] == direction["claim_layer"]
        and objective["metric_name"] == direction["metric_name"]
        and objective["metric_unit"] == direction["metric_unit"]
        and objective["metric_direction"] == direction["metric_direction"] == "lower"
        and objective["metric_kind"] == direction["metric_kind"] == "additive_time"
    )


def decide_direction(
    portfolio: object,
    lineage: object,
    direction_id: str,
    *,
    previous: Mapping | None = None,
    family_history: list[Mapping] | None = None,
    latest_by_family: Mapping[str, Mapping] | None = None,
    closed_decision_sha256: str | None = None,
    request: str = "admit",
) -> dict:
    validated = _validate_portfolio(portfolio)
    frozen = _validate_lineage(lineage)
    direction_id = _string(direction_id, "direction_id", safe_id=True)
    if request not in REQUESTS:
        raise ValueError("request must be admit, close, or reopen")
    if validated["objective"] != frozen["objective"]:
        raise ValueError("portfolio objective drifted from the frozen lineage")
    if validated["environment_artifact"]["sha256"] != frozen["environment_sha256"]:
        raise ValueError("portfolio environment drifted from the frozen lineage")
    current_families = sorted(item["direction_family_key"] for item in validated["directions"])
    frozen_families = [item["direction_family_key"] for item in frozen["direction_families"]]
    if current_families != frozen_families:
        raise ValueError("portfolio must preserve frozen direction family set")
    matches = [item for item in validated["directions"] if item["id"] == direction_id]
    if len(matches) != 1:
        raise ValueError("direction_id is not a unique portfolio direction")
    selected = matches[0]
    previous_value = dict(previous) if previous is not None else None
    history_values = [dict(item) for item in (family_history or [])]
    latest_values = {key: dict(value) for key, value in (latest_by_family or {}).items()}
    if previous_value is not None and previous_value.get("direction_family_key") != selected["direction_family_key"]:
        raise ValueError("previous decision belongs to a different direction family")
    if any(item.get("direction_family_key") != selected["direction_family_key"] for item in history_values):
        raise ValueError("family history contains another direction family")

    base = {
        "schema_version": 1,
        "direction_id": selected["id"],
        "direction_family_key": selected["direction_family_key"],
        "direction_key": selected["direction_key"],
        "claim_layer": selected["claim_layer"],
        "measurement_window_sha256": validated["measurement_window_artifact"]["sha256"],
        "evidence_sha256": selected["evidence_artifact"]["sha256"],
        "portfolio_sha256": _canonical_digest(validated),
        "performance_gain_claimed": False,
        "reopen_reason": None,
        "closed_decision_sha256": None,
    }
    comparable = _comparable(selected, frozen["objective"])
    absolute = percent = None
    reachable = False
    if comparable:
        family = _family_entry(frozen, selected["direction_family_key"])
        absolute, percent, reachable = _effect_reachable(
            selected, frozen["objective"], family["baseline_total_metric"]
        )

    if request == "reopen":
        if previous_value is None or previous_value.get("state") != "closed":
            raise ValueError("reopen requires the latest closed family decision")
        if not comparable or not reachable:
            raise ValueError("reopen evidence does not meet the frozen admission floor")
        used_evidence = {item.get("evidence_sha256") for item in history_values}
        if selected["evidence_artifact"]["sha256"] in used_evidence or (
            not history_values
            and previous_value.get("evidence_sha256") == selected["evidence_artifact"]["sha256"]
        ):
            raise ValueError("reopen requires new evidence not used earlier in the family chain")
        closed_sha = _sha256(closed_decision_sha256, "closed_decision_sha256")
        target_changed = previous_value.get("direction_key") != selected["direction_key"]
        window_changed = (
            previous_value.get("measurement_window_sha256")
            != validated["measurement_window_artifact"]["sha256"]
        )
        if not target_changed and not window_changed:
            raise ValueError("reopen requires a new measurement window or target identity")
        base["reopen_reason"] = (
            "new_target_identity" if target_changed else "new_measurement_window"
        )
        base["closed_decision_sha256"] = closed_sha
        minimum_absolute = frozen["objective"]["minimum_effect_absolute"] or 0.0
        minimum_percent = frozen["objective"]["minimum_effect_percent"] or 0.0
        previous_absolute = previous_value.get("upper_bound_absolute")
        previous_percent = previous_value.get("upper_bound_percent")
        if (
            type(previous_absolute) not in (int, float)
            or type(previous_percent) not in (int, float)
            or absolute < float(previous_absolute) + minimum_absolute
            or percent < float(previous_percent) + minimum_percent
        ):
            raise ValueError("reopen requires a material upper-bound increase over the closure")
    elif previous_value is not None and previous_value.get("state") == "closed":
        return {
            **base,
            "state": "closed",
            "action": "direction_closed",
            "transition": "blocked",
            "reason": "latest family decision is closed; provide qualified reopen evidence",
            "upper_bound_absolute": absolute,
            "upper_bound_percent": percent,
            "recommended_direction_id": None,
            "admitted": False,
        }

    if request == "close":
        action = "close_direction"
        state = "closed"
        reason = "direction explicitly closed and bound to this evidence snapshot"
        recommended = None
        admitted = False
    elif not comparable:
        action = "unrankable"
        state = "open"
        reason = "automatic ranking is limited to same-layer additive lower-is-better time"
        recommended = None
        admitted = False
    elif not reachable:
        action = "close_direction"
        state = "closed"
        reason = "full-elimination upper bound misses the frozen minimum effect"
        recommended = None
        admitted = False
    else:
        closed_families = {
            key for key, value in latest_values.items() if value.get("state") == "closed"
        }
        if request == "reopen":
            closed_families.discard(selected["direction_family_key"])
        candidates = [
            item
            for item in validated["directions"]
            if _comparable(item, frozen["objective"])
            and item["direction_family_key"] not in closed_families
        ]
        higher = [
            item for item in candidates
            if item["component_metric"] > selected["component_metric"]
        ]
        if higher:
            leader = max(
                higher,
                key=lambda item: (item["component_metric"], item["direction_key"]),
            )
            action = "switch_to_higher_impact"
            state = "open"
            reason = "another same-layer direction has a larger full-elimination upper bound"
            recommended = leader["id"]
            admitted = False
        else:
            action = "admit_direction"
            state = "open"
            reason = "direction meets the frozen floor and has the largest comparable upper bound"
            recommended = selected["id"]
            admitted = True
    return {
        **base,
        "state": state,
        "action": action,
        "transition": "reopen" if request == "reopen" else request,
        "reason": reason,
        "upper_bound_absolute": absolute,
        "upper_bound_percent": percent,
        "recommended_direction_id": recommended,
        "admitted": admitted,
    }


def _decision_files(run_dir: Path) -> list[Path]:
    directory = run_dir / "direction-decisions"
    if directory.is_symlink():
        raise ValueError("direction decision directory must not be a symlink")
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError("direction decision path must be a directory")
    files = []
    for entry in os.scandir(directory):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError("direction decision directory contains an unsafe entry")
        match = _DECISION_NAME.fullmatch(entry.name)
        if match is None:
            raise ValueError("direction decision directory contains an unknown file")
        files.append((int(match.group(1)), directory / entry.name))
    files.sort()
    expected = list(range(1, len(files) + 1))
    if [index for index, _ in files] != expected:
        raise ValueError("direction decision ledger has an index gap")
    return [path for _, path in files]


def _load_ledger(run_dir: Path, lineage: Mapping) -> tuple[list[dict], list[str]]:
    lineage_sha = _canonical_digest(lineage)
    records = []
    hashes = []
    prior_sha = None
    for index, path in enumerate(_decision_files(run_dir), start=1):
        raw = artifact_store.read_regular_bytes(path)
        file_sha = hashlib.sha256(raw).hexdigest()
        record = _validate_decision_record(load_json_bytes(raw, str(path)), lineage)
        if record.get("decision_index") != index:
            raise ValueError("direction decision index does not match its canonical filename")
        if record.get("lineage_sha256") != lineage_sha:
            raise ValueError("direction decision is bound to another lineage")
        if record.get("previous_decision_sha256") != prior_sha:
            raise ValueError("direction decision hash chain is broken")
        _validate_reopen_history(record, records, hashes, lineage)
        records.append(record)
        hashes.append(file_sha)
        prior_sha = file_sha
    return records, hashes


def _validate_reopen_history(
    record: Mapping, records: list[dict], hashes: list[str], lineage: Mapping
) -> None:
    if record["transition"] != "reopen":
        return
    family_indices = [
        index for index, prior in enumerate(records)
        if prior["direction_family_key"] == record["direction_family_key"]
    ]
    if not family_indices:
        raise ValueError("reopen has no earlier family decision")
    closed_index = family_indices[-1]
    closed = records[closed_index]
    if closed["state"] != "closed" or record["closed_decision_sha256"] != hashes[closed_index]:
        raise ValueError("reopen must reference the latest closed family decision")
    target_changed = record["direction_key"] != closed["direction_key"]
    window_changed = record["measurement_window_sha256"] != closed["measurement_window_sha256"]
    expected_reason = "new_target_identity" if target_changed else "new_measurement_window"
    if record["reopen_reason"] != expected_reason or (not target_changed and not window_changed):
        raise ValueError("reopen reason does not match the target or measurement-window change")
    used_evidence = {records[index]["evidence_sha256"] for index in family_indices}
    if record["evidence_sha256"] in used_evidence:
        raise ValueError("reopen evidence was used earlier in the family chain")
    objective = lineage["objective"]
    minimum_absolute = objective["minimum_effect_absolute"] or 0.0
    minimum_percent = objective["minimum_effect_percent"] or 0.0
    if (
        record["upper_bound_absolute"] is None
        or record["upper_bound_percent"] is None
        or closed["upper_bound_absolute"] is None
        or closed["upper_bound_percent"] is None
        or record["upper_bound_absolute"] < closed["upper_bound_absolute"] + minimum_absolute
        or record["upper_bound_percent"] < closed["upper_bound_percent"] + minimum_percent
    ):
        raise ValueError("reopen does not satisfy the frozen material-increase rule")


def _validate_decision_record(value: object, lineage: Mapping) -> dict:
    record = _closed(
        value,
        keys={
            "schema_version",
            "direction_id",
            "direction_family_key",
            "direction_key",
            "claim_layer",
            "measurement_window_sha256",
            "evidence_sha256",
            "portfolio_sha256",
            "performance_gain_claimed",
            "admitted",
            "state",
            "action",
            "transition",
            "reason",
            "upper_bound_absolute",
            "upper_bound_percent",
            "recommended_direction_id",
            "decision_index",
            "lineage_sha256",
            "previous_decision_sha256",
            "reopen_reason",
            "closed_decision_sha256",
        },
        field="direction decision",
    )
    if record["schema_version"] != 1:
        raise ValueError("direction decision schema_version must be 1")
    family_key = _sha256(record["direction_family_key"], "decision.direction_family_key")
    _family_entry(lineage, family_key)
    action = _string(record["action"], "decision.action")
    if action not in ACTIONS:
        raise ValueError("decision.action is not supported")
    state = _string(record["state"], "decision.state")
    if state not in {"open", "closed"}:
        raise ValueError("decision.state is not supported")
    transition = _string(record["transition"], "decision.transition")
    if transition not in REQUESTS | {"blocked"}:
        raise ValueError("decision.transition is not supported")
    reopen_reason = record["reopen_reason"]
    closed_decision_sha = _optional_sha256(
        record["closed_decision_sha256"], "decision.closed_decision_sha256"
    )
    if transition == "reopen":
        if reopen_reason not in REOPEN_REASONS or closed_decision_sha is None:
            raise ValueError("reopen decision requires its reason and closed decision digest")
    elif reopen_reason is not None or closed_decision_sha is not None:
        raise ValueError("non-reopen decision cannot carry reopen metadata")
    admitted = record["admitted"]
    if type(admitted) is not bool:
        raise ValueError("decision.admitted must be boolean")
    if admitted != (action == "admit_direction"):
        raise ValueError("decision.admitted conflicts with its action")
    if (state == "closed") != (action in {"close_direction", "direction_closed"}):
        raise ValueError("decision.state conflicts with its action")
    if transition == "reopen" and action not in {"admit_direction", "switch_to_higher_impact"}:
        raise ValueError("reopen decision must return to an open admitted or switched state")
    claim_layer = _string(record["claim_layer"], "decision.claim_layer")
    if claim_layer not in CLAIM_LAYERS:
        raise ValueError("decision.claim_layer is not supported")
    recommended = record["recommended_direction_id"]
    if recommended is not None:
        recommended = _string(recommended, "decision.recommended_direction_id", safe_id=True)
    if action in {"admit_direction", "switch_to_higher_impact"} and recommended is None:
        raise ValueError("decision action requires a recommended direction")
    if record["performance_gain_claimed"] is not False:
        raise ValueError("direction decision cannot claim a performance gain")
    percent = _optional_nonnegative(record["upper_bound_percent"], "decision.upper_bound_percent")
    if percent is not None and percent > 100:
        raise ValueError("decision.upper_bound_percent cannot exceed 100")
    return {
        **record,
        "direction_id": _string(record["direction_id"], "decision.direction_id", safe_id=True),
        "direction_family_key": family_key,
        "direction_key": _sha256(record["direction_key"], "decision.direction_key"),
        "claim_layer": claim_layer,
        "measurement_window_sha256": _sha256(record["measurement_window_sha256"], "decision.measurement_window_sha256"),
        "evidence_sha256": _sha256(record["evidence_sha256"], "decision.evidence_sha256"),
        "portfolio_sha256": _sha256(record["portfolio_sha256"], "decision.portfolio_sha256"),
        "admitted": admitted,
        "state": state,
        "action": action,
        "transition": transition,
        "reason": _string(record["reason"], "decision.reason"),
        "upper_bound_absolute": _optional_nonnegative(record["upper_bound_absolute"], "decision.upper_bound_absolute"),
        "upper_bound_percent": percent,
        "recommended_direction_id": recommended,
        "decision_index": _integer(record["decision_index"], "decision.decision_index", minimum=1),
        "lineage_sha256": _sha256(record["lineage_sha256"], "decision.lineage_sha256"),
        "previous_decision_sha256": _optional_sha256(record["previous_decision_sha256"], "decision.previous_decision_sha256"),
        "reopen_reason": reopen_reason,
        "closed_decision_sha256": closed_decision_sha,
    }


def _latest_family(records: list[dict], family_key: str) -> dict | None:
    for record in reversed(records):
        if record.get("direction_family_key") == family_key:
            return record
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admit or stop optimization directions without running the target."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="freeze a direction lineage")
    init.add_argument("--portfolio", required=True)
    init.add_argument("--run-dir", required=True)
    check = commands.add_parser("check", help="append one direction decision")
    check.add_argument("--portfolio", required=True)
    check.add_argument("--run-dir", required=True)
    check.add_argument("--direction-id", required=True)
    check.add_argument("--request", choices=sorted(REQUESTS), default="admit")
    check.add_argument("--expected-tail-sha256")
    status = commands.add_parser("status", help="validate and summarize the ledger")
    status.add_argument("--run-dir", required=True)
    status.add_argument("--expected-tail-sha256")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_dir = Path(args.run_dir)
        lineage_path = run_dir / "direction-lineage.json"
        if args.command == "init":
            portfolio_raw = load_json_strict(args.portfolio)
            verify_portfolio_artifacts(portfolio_raw, args.portfolio)
            lineage = freeze_lineage(portfolio_raw)
            artifact_store.create_regular_json(lineage_path, lineage)
            result = lineage
        else:
            lineage = _validate_lineage(load_json_strict(lineage_path))
            records, decision_hashes = _load_ledger(run_dir, lineage)
            current_tail = decision_hashes[-1] if decision_hashes else None
            expected_tail_argument = getattr(args, "expected_tail_sha256", None)
            if expected_tail_argument is not None:
                expected_tail = _sha256(expected_tail_argument, "expected_tail_sha256")
                if current_tail != expected_tail:
                    raise ValueError("expected tail SHA-256 does not match the ledger")
            if args.command == "status":
                result = {
                    "schema_version": 1,
                    "lineage_sha256": _canonical_digest(lineage),
                    "decision_count": len(records),
                    "latest_action": records[-1]["action"] if records else None,
                    "ledger_tail_sha256": current_tail,
                }
            else:
                if records and args.expected_tail_sha256 is None:
                    raise ValueError("expected tail SHA-256 is required before appending")
                snapshot_raw = load_json_strict(args.portfolio)
                verify_portfolio_artifacts(snapshot_raw, args.portfolio)
                snapshot = _validate_portfolio(snapshot_raw)
                selected = next(
                    (item for item in snapshot["directions"] if item["id"] == args.direction_id),
                    None,
                )
                if selected is None:
                    raise ValueError("direction_id is not present in the portfolio")
                previous_family = _latest_family(records, selected["direction_family_key"])
                family_history = [
                    record for record in records
                    if record["direction_family_key"] == selected["direction_family_key"]
                ]
                latest_by_family = {}
                for record in records:
                    latest_by_family[record["direction_family_key"]] = record
                previous_family_sha = None
                if previous_family is not None:
                    previous_family_index = max(
                        index for index, record in enumerate(records)
                        if record["direction_family_key"] == selected["direction_family_key"]
                    )
                    previous_family_sha = decision_hashes[previous_family_index]
                decision = decide_direction(
                    snapshot_raw,
                    lineage,
                    args.direction_id,
                    previous=previous_family,
                    family_history=family_history,
                    latest_by_family=latest_by_family,
                    closed_decision_sha256=previous_family_sha,
                    request=args.request,
                )
                index = len(records) + 1
                decision.update(
                    {
                        "decision_index": index,
                        "lineage_sha256": _canonical_digest(lineage),
                        "previous_decision_sha256": current_tail,
                    }
                )
                output = run_dir / "direction-decisions" / f"decision-{index:04d}.json"
                artifact_store.create_regular_json(output, decision)
                result = decision
    except (FileExistsError, OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
