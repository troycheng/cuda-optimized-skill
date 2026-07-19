#!/usr/bin/env python3
"""Calibrate measurement stability from a verified workload contract."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import math
import re
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CALIBRATION_SCHEMA = "cuda-optimizer/stability-calibration-v1"
AUDIT_SCHEMA = "cuda-optimizer/stability-audit-v1"
MDE_METHOD = "paired_log_ratio_normal_approximation"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CALIBRATION_REASONS = {
    "hard_guardrail_failed",
    "insufficient_valid_pairs",
    "noise_exceeds_minimum_practical_effect",
    "mde_exceeds_minimum_practical_effect",
}
_AUDIT_REASONS = _CALIBRATION_REASONS | {
    "baseline_shift_exceeds_calibrated_noise",
}
_CALIBRATION_FIELDS = {
    "schema_version",
    "contract_sha256",
    "environment_sha256",
    "source_sha256",
    "recorded_at",
    "confidence",
    "power",
    "bootstrap_samples",
    "min_valid_pairs",
    "seed",
    "audit_every_candidates",
    "minimum_practical_effect_pct",
    "valid_pairs",
    "invalid_pairs",
    "baseline_median",
    "noise_median_pct",
    "noise_ci_low_pct",
    "noise_ci_high_pct",
    "mde_method",
    "minimum_detectable_effect_pct",
    "decision_threshold_pct",
    "hard_guardrails_passed",
    "environment_state",
    "measurable",
    "reasons",
    "calibration_sha256",
    "controller_attestation",
}
_AUDIT_FIELDS = {
    "schema_version",
    "contract_sha256",
    "environment_sha256",
    "anchor_calibration_sha256",
    "replay_calibration_sha256",
    "source_sha256",
    "recorded_at",
    "baseline_shift_pct",
    "calibrated_noise_bound_pct",
    "environment_state",
    "measurable",
    "reasons",
    "audit_sha256",
    "controller_attestation",
}


class ValidationError(ValueError):
    pass


def _sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"cuda_stability_{name}", path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load {name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PAIRED_STATS = _sibling("paired_stats")
_WORKLOAD_CONTRACT = _sibling("workload_contract")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("stability artifact must contain finite JSON values") from exc


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValidationError(f"{label} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValidationError(f"{label} must be at least {minimum}")
    return number


def _positive_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _key(value: Any) -> bytes:
    if not isinstance(value, bytes) or len(value) < 32:
        raise ValidationError("controller_seal_key must contain at least 32 bytes")
    return value


def _attest(value: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(_key(key), _canonical_bytes(value), hashlib.sha256).hexdigest()


def _policy(contract_path: str | Path) -> tuple[dict, dict]:
    try:
        contract = _WORKLOAD_CONTRACT.verify_frozen_contract(contract_path)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"verified workload contract required: {exc}") from exc
    stability = contract["stability"]
    return contract, {
        "contract_sha256": contract["contract_sha256"],
        "minimum_practical_effect_pct": float(
            contract["objective"]["minimum_practical_effect_pct"]
        ),
        "confidence": float(stability["confidence"]),
        "power": float(stability["power"]),
        "bootstrap_samples": stability["bootstrap_samples"],
        "min_valid_pairs": stability["min_valid_pairs"],
        "seed": stability["seed"],
        "audit_every_candidates": stability["audit_every_candidates"],
    }


def _blocks(value: Any) -> tuple[list[float], list[float], list[float], int]:
    if type(value) is not list or not value:
        raise ValidationError("blocks must be a non-empty array")
    seen = set()
    noise = []
    baseline_values = []
    signed_log_ratios = []
    invalid = 0
    for index, block in enumerate(value):
        if type(block) is not dict or set(block) != {"pair_id", "first", "second", "valid"}:
            raise ValidationError(f"blocks[{index}] must be closed")
        pair_id = block["pair_id"]
        if type(pair_id) is not str or _IDENTIFIER.fullmatch(pair_id) is None:
            raise ValidationError(f"blocks[{index}].pair_id is invalid")
        if pair_id in seen:
            raise ValidationError("block pair_id values must be unique")
        seen.add(pair_id)
        if type(block["valid"]) is not bool:
            raise ValidationError(f"blocks[{index}].valid must be boolean")
        first = _finite(block["first"], f"blocks[{index}].first")
        second = _finite(block["second"], f"blocks[{index}].second")
        if first <= 0.0 or second <= 0.0:
            raise ValidationError(f"blocks[{index}] measurements must be positive")
        if not block["valid"]:
            invalid += 1
            continue
        baseline_values.extend((first, second))
        midpoint = (first + second) / 2.0
        noise.append(abs(second - first) / midpoint * 100.0)
        signed_log_ratios.append(math.log(second / first) * 100.0)
    return noise, baseline_values, signed_log_ratios, invalid


def _minimum_detectable_effect(
    signed_log_ratios: list[float], *, confidence: float, power: float
) -> float | None:
    """Return a transparent normal approximation over paired log ratios.

    This is an environment-calibration diagnostic, not the final candidate
    acceptance test. Candidate wins still require paired workload evidence.
    """
    if len(signed_log_ratios) < 2:
        return None
    standard_deviation = statistics.stdev(signed_log_ratios)
    normal = statistics.NormalDist()
    z_confidence = normal.inv_cdf(0.5 + confidence / 2.0)
    z_power = normal.inv_cdf(power)
    log_effect_pct = (
        (z_confidence + z_power)
        * standard_deviation
        / math.sqrt(len(signed_log_ratios))
    )
    try:
        effect = math.expm1(log_effect_pct / 100.0) * 100.0
    except OverflowError as exc:
        raise ValidationError("paired measurements are too unstable to calibrate") from exc
    if not math.isfinite(effect):
        raise ValidationError("minimum detectable effect is not finite")
    return effect


def _state(
    *,
    hard_guardrails_passed: bool,
    valid_pairs: int,
    min_valid_pairs: int,
    noise_ci_high_pct: float | None,
    minimum_detectable_effect_pct: float | None,
    minimum_practical_effect_pct: float,
) -> tuple[str, list[str]]:
    if not hard_guardrails_passed:
        return "red", ["hard_guardrail_failed"]
    if valid_pairs < min_valid_pairs:
        return "yellow", ["insufficient_valid_pairs"]
    reasons = []
    if noise_ci_high_pct is not None and noise_ci_high_pct > minimum_practical_effect_pct:
        reasons.append("noise_exceeds_minimum_practical_effect")
    if (
        minimum_detectable_effect_pct is None
        or minimum_detectable_effect_pct > minimum_practical_effect_pct
    ):
        reasons.append("mde_exceeds_minimum_practical_effect")
    return ("yellow", reasons) if reasons else ("green", [])


def calibrate(
    *,
    contract_path: str | Path,
    blocks: list[Mapping[str, Any]],
    hard_guardrails_passed: bool,
    environment_sha256: str,
    source_sha256: str,
    recorded_at: float,
    controller_seal_key: bytes,
) -> dict:
    """Create a Controller-attested calibration bound to a frozen contract."""
    _contract, policy = _policy(contract_path)
    noise, baseline_values, log_ratios, invalid = _blocks(blocks)
    if type(hard_guardrails_passed) is not bool:
        raise ValidationError("hard_guardrails_passed must be boolean")
    timestamp = _finite(recorded_at, "recorded_at", minimum=0.0)
    if noise:
        ci_low, ci_high = _PAIRED_STATS.bootstrap_median_ci(
            noise,
            confidence=policy["confidence"],
            samples=policy["bootstrap_samples"],
            seed=policy["seed"],
        )
        noise_median = statistics.median(noise)
    else:
        ci_low = ci_high = noise_median = None
    mde = _minimum_detectable_effect(
        log_ratios,
        confidence=policy["confidence"],
        power=policy["power"],
    )
    state, reasons = _state(
        hard_guardrails_passed=hard_guardrails_passed,
        valid_pairs=len(noise),
        min_valid_pairs=policy["min_valid_pairs"],
        noise_ci_high_pct=ci_high,
        minimum_detectable_effect_pct=mde,
        minimum_practical_effect_pct=policy["minimum_practical_effect_pct"],
    )
    decision_threshold = (
        None
        if mde is None
        else max(policy["minimum_practical_effect_pct"], mde)
    )
    result = {
        "schema_version": CALIBRATION_SCHEMA,
        "contract_sha256": policy["contract_sha256"],
        "environment_sha256": _sha(environment_sha256, "environment_sha256"),
        "source_sha256": _sha(source_sha256, "source_sha256"),
        "recorded_at": timestamp,
        "confidence": policy["confidence"],
        "power": policy["power"],
        "bootstrap_samples": policy["bootstrap_samples"],
        "min_valid_pairs": policy["min_valid_pairs"],
        "seed": policy["seed"],
        "audit_every_candidates": policy["audit_every_candidates"],
        "minimum_practical_effect_pct": policy["minimum_practical_effect_pct"],
        "valid_pairs": len(noise),
        "invalid_pairs": invalid,
        "baseline_median": (
            statistics.median(baseline_values) if baseline_values else None
        ),
        "noise_median_pct": noise_median,
        "noise_ci_low_pct": ci_low,
        "noise_ci_high_pct": ci_high,
        "mde_method": MDE_METHOD,
        "minimum_detectable_effect_pct": mde,
        "decision_threshold_pct": decision_threshold,
        "hard_guardrails_passed": hard_guardrails_passed,
        "environment_state": state,
        "measurable": state == "green",
        "reasons": reasons,
    }
    result["calibration_sha256"] = _canonical_digest(result)
    result["controller_attestation"] = _attest(result, controller_seal_key)
    return result


def _validate_reason_list(value: Any, allowed: set[str], label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValidationError(f"{label} must be an array of strings")
    if len(value) != len(set(value)) or not set(value) <= allowed:
        raise ValidationError(f"{label} contains duplicate or unsupported values")
    return value


def validate_calibration(
    value: Mapping[str, Any],
    *,
    contract_path: str | Path,
    controller_seal_key: bytes,
) -> dict:
    if type(value) is not dict or set(value) != _CALIBRATION_FIELDS:
        raise ValidationError("calibration must be a closed artifact")
    attestation = _sha(value["controller_attestation"], "controller_attestation")
    attested = dict(value)
    attested.pop("controller_attestation")
    if not hmac.compare_digest(attestation, _attest(attested, controller_seal_key)):
        raise ValidationError("Controller calibration attestation changed")
    expected = _sha(value["calibration_sha256"], "calibration_sha256")
    unsigned = dict(attested)
    unsigned.pop("calibration_sha256")
    if _canonical_digest(unsigned) != expected:
        raise ValidationError("calibration digest changed")
    if value["schema_version"] != CALIBRATION_SCHEMA:
        raise ValidationError("unsupported calibration schema")

    _contract, policy = _policy(contract_path)
    expected_policy = {
        "contract_sha256": policy["contract_sha256"],
        "confidence": policy["confidence"],
        "power": policy["power"],
        "bootstrap_samples": policy["bootstrap_samples"],
        "min_valid_pairs": policy["min_valid_pairs"],
        "seed": policy["seed"],
        "audit_every_candidates": policy["audit_every_candidates"],
        "minimum_practical_effect_pct": policy["minimum_practical_effect_pct"],
    }
    if any(value[field] != expected_value for field, expected_value in expected_policy.items()):
        raise ValidationError("calibration contract or stability policy changed")
    for field in ("contract_sha256", "environment_sha256", "source_sha256"):
        _sha(value[field], field)
    _finite(value["recorded_at"], "recorded_at", minimum=0.0)
    _positive_int(value["bootstrap_samples"], "bootstrap_samples", minimum=1000)
    minimum_pairs = _positive_int(value["min_valid_pairs"], "min_valid_pairs", minimum=4)
    _positive_int(value["audit_every_candidates"], "audit_every_candidates")
    if type(value["seed"]) is not int:
        raise ValidationError("calibration seed must be an integer")
    if type(value["valid_pairs"]) is not int or value["valid_pairs"] < 0:
        raise ValidationError("valid_pairs must be a non-negative integer")
    valid_pairs = value["valid_pairs"]
    if valid_pairs == 0:
        if value["baseline_median"] is not None:
            raise ValidationError("zero valid pairs must not report a baseline median")
    elif _finite(value["baseline_median"], "baseline_median") <= 0.0:
        raise ValidationError("baseline_median must be positive")
    if type(value["invalid_pairs"]) is not int or value["invalid_pairs"] < 0:
        raise ValidationError("invalid_pairs must be a non-negative integer")
    _finite(
        value["minimum_practical_effect_pct"],
        "minimum_practical_effect_pct",
        minimum=0.0,
    )
    noise_fields = ("noise_median_pct", "noise_ci_low_pct", "noise_ci_high_pct")
    if valid_pairs == 0:
        if any(value[field] is not None for field in noise_fields):
            raise ValidationError("zero valid pairs must not report noise statistics")
    else:
        for field in noise_fields:
            _finite(value[field], field, minimum=0.0)
        if value["noise_ci_low_pct"] > value["noise_ci_high_pct"]:
            raise ValidationError("calibration noise CI is reversed")
    if value["mde_method"] != MDE_METHOD:
        raise ValidationError("calibration MDE method changed")
    mde = value["minimum_detectable_effect_pct"]
    threshold = value["decision_threshold_pct"]
    if mde is None:
        if threshold is not None or valid_pairs >= 2:
            raise ValidationError("calibration detectable effect is inconsistent")
    else:
        mde = _finite(mde, "minimum_detectable_effect_pct", minimum=0.0)
        expected_threshold = max(value["minimum_practical_effect_pct"], mde)
        if threshold != expected_threshold:
            raise ValidationError("calibration decision threshold changed")
    if type(value["hard_guardrails_passed"]) is not bool:
        raise ValidationError("calibration hard guardrail status is invalid")
    expected_state, expected_reasons = _state(
        hard_guardrails_passed=value["hard_guardrails_passed"],
        valid_pairs=valid_pairs,
        min_valid_pairs=minimum_pairs,
        noise_ci_high_pct=value["noise_ci_high_pct"],
        minimum_detectable_effect_pct=mde,
        minimum_practical_effect_pct=value["minimum_practical_effect_pct"],
    )
    _validate_reason_list(value["reasons"], _CALIBRATION_REASONS, "calibration reasons")
    if value["environment_state"] != expected_state or value["reasons"] != expected_reasons:
        raise ValidationError("calibration state or reasons changed")
    if value["measurable"] is not (expected_state == "green"):
        raise ValidationError("calibration measurable flag changed")
    return json.loads(_canonical_bytes(value))


def audit(
    anchor: Mapping[str, Any],
    *,
    contract_path: str | Path,
    blocks: list[Mapping[str, Any]],
    hard_guardrails_passed: bool,
    environment_sha256: str,
    source_sha256: str,
    recorded_at: float,
    controller_seal_key: bytes,
) -> dict:
    clean_anchor = validate_calibration(
        anchor,
        contract_path=contract_path,
        controller_seal_key=controller_seal_key,
    )
    if clean_anchor["environment_state"] != "green" or not clean_anchor["measurable"]:
        raise ValidationError("periodic audit requires a green measurable calibration")
    timestamp = _finite(recorded_at, "recorded_at", minimum=0.0)
    if timestamp < clean_anchor["recorded_at"]:
        raise ValidationError("audit recorded time precedes calibration time")
    current_environment = _sha(environment_sha256, "environment_sha256")
    current_source = _sha(source_sha256, "source_sha256")
    if current_environment != clean_anchor["environment_sha256"]:
        raise ValidationError("audit environment identity changed")
    if current_source != clean_anchor["source_sha256"]:
        raise ValidationError("audit source identity changed")
    replay = calibrate(
        contract_path=contract_path,
        blocks=blocks,
        hard_guardrails_passed=hard_guardrails_passed,
        environment_sha256=current_environment,
        source_sha256=current_source,
        recorded_at=timestamp,
        controller_seal_key=controller_seal_key,
    )
    baseline = clean_anchor["baseline_median"]
    shift = abs(replay["baseline_median"] - baseline) / baseline * 100.0
    state = replay["environment_state"]
    reasons = list(replay["reasons"])
    if state != "red" and shift > clean_anchor["noise_ci_high_pct"]:
        state = "yellow"
        if "baseline_shift_exceeds_calibrated_noise" not in reasons:
            reasons.append("baseline_shift_exceeds_calibrated_noise")
    result = {
        "schema_version": AUDIT_SCHEMA,
        "contract_sha256": clean_anchor["contract_sha256"],
        "environment_sha256": clean_anchor["environment_sha256"],
        "anchor_calibration_sha256": clean_anchor["calibration_sha256"],
        "replay_calibration_sha256": replay["calibration_sha256"],
        "source_sha256": replay["source_sha256"],
        "recorded_at": timestamp,
        "baseline_shift_pct": shift,
        "calibrated_noise_bound_pct": clean_anchor["noise_ci_high_pct"],
        "environment_state": state,
        "measurable": state == "green",
        "reasons": reasons,
    }
    result["audit_sha256"] = _canonical_digest(result)
    result["controller_attestation"] = _attest(result, controller_seal_key)
    return result


def validate_audit(value: Mapping[str, Any], *, controller_seal_key: bytes) -> dict:
    if type(value) is not dict or set(value) != _AUDIT_FIELDS:
        raise ValidationError("stability audit must be a closed artifact")
    attestation = _sha(value["controller_attestation"], "controller_attestation")
    attested = dict(value)
    attested.pop("controller_attestation")
    if not hmac.compare_digest(attestation, _attest(attested, controller_seal_key)):
        raise ValidationError("Controller audit attestation changed")
    expected = _sha(value["audit_sha256"], "audit_sha256")
    unsigned = dict(attested)
    unsigned.pop("audit_sha256")
    if _canonical_digest(unsigned) != expected:
        raise ValidationError("stability audit digest changed")
    if value["schema_version"] != AUDIT_SCHEMA:
        raise ValidationError("unsupported stability audit schema")
    for field in (
        "contract_sha256",
        "environment_sha256",
        "anchor_calibration_sha256",
        "replay_calibration_sha256",
        "source_sha256",
    ):
        _sha(value[field], field)
    for field in ("recorded_at", "baseline_shift_pct", "calibrated_noise_bound_pct"):
        _finite(value[field], field, minimum=0.0)
    if value["environment_state"] not in {"green", "yellow", "red"}:
        raise ValidationError("stability audit environment state is invalid")
    if value["measurable"] is not (value["environment_state"] == "green"):
        raise ValidationError("stability audit measurable flag changed")
    _validate_reason_list(value["reasons"], _AUDIT_REASONS, "stability audit reasons")
    return json.loads(_canonical_bytes(value))
