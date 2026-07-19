#!/usr/bin/env python3
"""Validate a single-variable GPU software-stack comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


SCHEMA = "cuda-version-audit/v1"
ROOT_FIELDS = {
    "schema_version",
    "baseline",
    "candidate",
    "fresh_build_per_stack",
    "fresh_timing_cache_per_stack",
    "self_repeat_stable_per_stack",
    "reuse_engine_across_stacks",
    "correctness",
    "timing_started",
    "measurement_evidence_ids",
    "invalid_evidence_ids",
}
SIDE_FIELDS = {"frozen", "stack", "derived"}
FROZEN_FIELDS = {
    "source_sha256",
    "onnx_sha256",
    "build_recipe_sha256",
    "request_corpus_sha256",
    "correctness_contract_sha256",
    "benchmark_design_sha256",
    "model_config_sha256",
    "custom_backend_sha256",
    "gpu_uuid",
    "driver_version",
    "clock_policy",
}
STACK_FIELDS = {"image_digest", "triton_version", "tensorrt_version", "cuda_version"}
DERIVED_FIELDS = {"plugin_sha256", "engine_sha256", "timing_cache_sha256"}
CORRECTNESS_FIELDS = {"passed", "evidence_id"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
EVIDENCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class InputError(ValueError):
    pass


def _load_artifact_store():
    path = Path(__file__).with_name("artifact_store.py")
    spec = importlib.util.spec_from_file_location("cuda_version_audit_store", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STORE = _load_artifact_store()


def _pairs_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise InputError("duplicate JSON key: %s" % key)
        value[key] = item
    return value


def _invalid_number(token):
    raise InputError("JSON number must be finite: %s" % token)


def load_json(path):
    try:
        raw = STORE.read_regular_bytes(path)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_invalid_number,
        )
    except InputError:
        raise
    except (UnicodeError, json.JSONDecodeError, OSError, ValueError) as error:
        raise InputError("invalid or unsafe input: %s" % error) from error
    if type(value) is not dict:
        raise InputError("input root must be an object")
    return value


def _closed(value, fields, name, reasons):
    if type(value) is not dict:
        reasons.append("invalid_mapping:%s" % name)
        return {}
    for field in sorted(fields - set(value)):
        reasons.append("missing_field:%s.%s" % (name, field))
    for field in sorted(set(value) - fields):
        reasons.append("unexpected_field:%s.%s" % (name, field))
    return value


def _string(value, name, reasons, *, pattern=None):
    if type(value) is not str or not value or len(value) > 256:
        reasons.append("invalid_value:%s" % name)
        return None
    if pattern is not None and pattern.fullmatch(value) is None:
        reasons.append("invalid_value:%s" % name)
        return None
    return value


def _literal_bool(value, name, reasons):
    if type(value) is not bool:
        reasons.append("invalid_boolean:%s" % name)
        return None
    return value


def _id_list(value, name, reasons):
    if type(value) is not list:
        reasons.append("invalid_%s" % name)
        return []
    result = []
    for index, item in enumerate(value):
        if _string(item, "%s[%d]" % (name, index), reasons, pattern=EVIDENCE_ID):
            result.append(item)
    if len(result) != len(set(result)):
        reasons.append("duplicate_%s" % name)
    return result


def validate(payload):
    reasons = []
    root = _closed(payload, ROOT_FIELDS, "root", reasons)
    if root.get("schema_version") != SCHEMA:
        reasons.append("invalid_schema_version")

    stacks = {}
    for role in ("baseline", "candidate"):
        side = _closed(root.get(role), SIDE_FIELDS, role, reasons)
        frozen = _closed(side.get("frozen"), FROZEN_FIELDS, "%s.frozen" % role, reasons)
        stack = _closed(side.get("stack"), STACK_FIELDS, "%s.stack" % role, reasons)
        derived = _closed(side.get("derived"), DERIVED_FIELDS, "%s.derived" % role, reasons)
        for field, value in frozen.items():
            pattern = HEX64 if field.endswith("_sha256") else None
            _string(value, "%s.frozen.%s" % (role, field), reasons, pattern=pattern)
        for field, value in stack.items():
            _string(value, "%s.stack.%s" % (role, field), reasons)
        for field, value in derived.items():
            _string(value, "%s.derived.%s" % (role, field), reasons, pattern=HEX64)
        stacks[role] = (frozen, stack)

    if all(role in stacks for role in ("baseline", "candidate")):
        base_frozen, base_stack = stacks["baseline"]
        candidate_frozen, candidate_stack = stacks["candidate"]
        for field in sorted(FROZEN_FIELDS):
            if base_frozen.get(field) != candidate_frozen.get(field):
                reasons.append("frozen_mismatch:%s" % field)
        if base_stack == candidate_stack:
            reasons.append("software_stack_did_not_change")

    fresh_build = _literal_bool(root.get("fresh_build_per_stack"), "fresh_build_per_stack", reasons)
    fresh_cache = _literal_bool(root.get("fresh_timing_cache_per_stack"), "fresh_timing_cache_per_stack", reasons)
    stable = _literal_bool(root.get("self_repeat_stable_per_stack"), "self_repeat_stable_per_stack", reasons)
    reused = _literal_bool(root.get("reuse_engine_across_stacks"), "reuse_engine_across_stacks", reasons)
    timing_started = _literal_bool(root.get("timing_started"), "timing_started", reasons)
    if fresh_build is not True:
        reasons.append("plugin_or_engine_not_fresh_per_stack")
    if fresh_cache is not True:
        reasons.append("timing_cache_not_fresh_per_stack")
    if reused is not False:
        reasons.append("engine_reused_across_stacks")

    correctness = _closed(root.get("correctness"), CORRECTNESS_FIELDS, "correctness", reasons)
    correctness_passed = _literal_bool(correctness.get("passed"), "correctness.passed", reasons)
    correctness_id = _string(
        correctness.get("evidence_id"),
        "correctness.evidence_id",
        reasons,
        pattern=EVIDENCE_ID,
    )
    measurements = _id_list(root.get("measurement_evidence_ids"), "measurement_evidence_ids", reasons)
    invalid = _id_list(root.get("invalid_evidence_ids"), "invalid_evidence_ids", reasons)
    referenced = set(measurements)
    if correctness_id:
        referenced.add(correctness_id)
    for evidence_id in sorted(referenced & set(invalid)):
        reasons.append("invalid_evidence_referenced:%s" % evidence_id)

    if timing_started is True and correctness_passed is not True:
        reasons.append("timing_started_before_correctness_passed")
    if timing_started is True and stable is not True:
        reasons.append("timing_started_before_self_repeat_stability")
    if timing_started is False and measurements:
        reasons.append("measurement_evidence_without_timing")

    return {
        "schema_version": SCHEMA,
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "frozen_fields_checked": sorted(FROZEN_FIELDS),
        "allowed_stack_fields": sorted(STACK_FIELDS),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        payload = load_json(args.input)
        report = validate(payload)
        STORE.atomic_write_json(args.out, report)
    except (InputError, OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
