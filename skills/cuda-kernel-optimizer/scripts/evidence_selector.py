#!/usr/bin/env python3
"""Select the next discriminating evidence action without model-owned scores."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


REQUEST_SCHEMA = "cuda-optimizer/evidence-request-set-v1"
CATALOG_SCHEMA = "cuda-optimizer/evidence-action-catalog-v1"
POLICY_SCHEMA = "cuda-optimizer/evidence-selection-policy-v1"
SELECTION_SCHEMA = "cuda-optimizer/evidence-selection-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3}


class ValidationError(ValueError):
    """Raised when a request or Controller policy is not replayable."""


def _closed(value: Any, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise ValidationError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValidationError(f"{label} contains unknown fields: {sorted(unknown)}")
    return value


def _text(value: Any, label: str, *, maximum: int = 1024) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise ValidationError(f"{label} exceeds {maximum} characters")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label, maximum=128)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValidationError(f"{label} must be a safe identifier")
    return text


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _ids(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if type(value) is not list or (not value and not allow_empty):
        raise ValidationError(f"{label} must be {'an array' if allow_empty else 'a non-empty array'}")
    result = [_identifier(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValidationError(f"{label} must not contain duplicates")
    return sorted(result)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_catalog(value: Mapping[str, Any]) -> tuple[dict, dict[str, dict]]:
    root = _closed(value, {"schema_version", "catalog_id", "actions"}, "action_catalog")
    if root["schema_version"] != CATALOG_SCHEMA:
        raise ValidationError(f"action_catalog.schema_version must be {CATALOG_SCHEMA}")
    _identifier(root["catalog_id"], "action_catalog.catalog_id")
    if type(root["actions"]) is not list or not root["actions"]:
        raise ValidationError("action_catalog.actions must be non-empty")
    actions = []
    by_id = {}
    for index, raw in enumerate(root["actions"]):
        item = _closed(
            raw,
            {
                "action_id",
                "evidence_kind",
                "required_capability_ids",
                "cost",
                "perturbation",
                "risk",
                "control_scope",
                "repeatable",
            },
            f"action_catalog.actions[{index}]",
        )
        action_id = _identifier(item["action_id"], f"actions[{index}].action_id")
        if action_id in by_id:
            raise ValidationError("action catalog ids must be unique")
        _identifier(item["evidence_kind"], f"actions[{index}].evidence_kind")
        capabilities = _ids(
            item["required_capability_ids"],
            f"actions[{index}].required_capability_ids",
            allow_empty=True,
        )
        for field in ("cost", "perturbation", "risk"):
            if item[field] not in _LEVELS:
                raise ValidationError(f"actions[{index}].{field} is unsupported")
        if item["control_scope"] not in {"read_only", "project_copy"}:
            raise ValidationError("evidence action control_scope is unsupported")
        if (
            item["control_scope"] == "project_copy"
            and item["evidence_kind"] != "direction_experiment"
        ):
            raise ValidationError(
                "project_copy is reserved for a direction experiment"
            )
        if type(item["repeatable"]) is not bool:
            raise ValidationError(f"actions[{index}].repeatable must be a boolean")
        normalized = {**copy.deepcopy(dict(item)), "required_capability_ids": capabilities}
        actions.append(normalized)
        by_id[action_id] = normalized
    actions.sort(key=lambda item: item["action_id"])
    return {
        "schema_version": CATALOG_SCHEMA,
        "catalog_id": root["catalog_id"],
        "actions": actions,
    }, by_id


def _validate_policy(value: Mapping[str, Any]) -> dict:
    policy = _closed(
        value,
        {
            "schema_version",
            "max_cost",
            "max_perturbation",
            "max_risk",
            "remaining_profile_actions",
            "available_capability_ids",
        },
        "selection_policy",
    )
    if policy["schema_version"] != POLICY_SCHEMA:
        raise ValidationError(f"selection_policy.schema_version must be {POLICY_SCHEMA}")
    for field in ("max_cost", "max_perturbation", "max_risk"):
        if policy[field] not in _LEVELS:
            raise ValidationError(f"selection_policy.{field} is unsupported")
    remaining = policy["remaining_profile_actions"]
    if type(remaining) is not int or remaining < 0:
        raise ValidationError("remaining_profile_actions must be non-negative")
    return {
        **copy.deepcopy(dict(policy)),
        "available_capability_ids": _ids(
            policy["available_capability_ids"],
            "selection_policy.available_capability_ids",
            allow_empty=True,
        ),
    }


def _trusted_hypotheses(value: Mapping[str, Any]) -> tuple[dict, str]:
    result = _closed(
        value,
        {"hypothesis_set", "hypothesis_set_sha256", "active_hypothesis_ids"},
        "hypothesis_result",
    )
    hypothesis_set = result["hypothesis_set"]
    if type(hypothesis_set) is not dict:
        raise ValidationError("hypothesis_result.hypothesis_set must be an object")
    digest = _sha(result["hypothesis_set_sha256"], "hypothesis_set_sha256")
    if _canonical_digest(hypothesis_set) != digest:
        raise ValidationError("hypothesis result digest does not match content")
    active = _ids(result["active_hypothesis_ids"], "active_hypothesis_ids", allow_empty=True)
    actual_active = sorted(
        item["hypothesis_id"]
        for item in hypothesis_set.get("hypotheses", [])
        if item.get("disposition") == "active"
    )
    if active != actual_active:
        raise ValidationError("hypothesis result active set does not match content")
    return hypothesis_set, digest


def _request_signature(epoch_id: str, action: Mapping[str, Any], request: Mapping[str, Any]) -> str:
    return _canonical_digest(
        {
            "epoch_id": epoch_id,
            "evidence_kind": action["evidence_kind"],
            "target_hypothesis_ids": request["target_hypothesis_ids"],
            "exclusive_pairs": request["exclusive_pairs"],
        }
    )


def select_evidence_request(
    value: Mapping[str, Any],
    *,
    epoch: Mapping[str, Any],
    hypothesis_result: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any],
    action_catalog: Mapping[str, Any],
    policy: Mapping[str, Any],
    request_history: Sequence[str],
    completed_action_ids: Sequence[str] = (),
) -> dict:
    """Validate AI requests, replay Controller costs, and choose deterministically."""
    if type(epoch) is not dict:
        raise ValidationError("epoch must be a Controller object")
    epoch_id = _identifier(epoch.get("epoch_id"), "epoch.epoch_id")
    epoch_sha = _sha(value.get("epoch_sha256"), "request_set.epoch_sha256")
    # Epoch artifacts are already validated by the preceding stage; the digest
    # still prevents a proposal from being replayed against a different epoch.
    if _canonical_digest(epoch) != epoch_sha:
        raise ValidationError("request_set epoch digest does not match Controller")
    hypotheses, hypothesis_digest = _trusted_hypotheses(hypothesis_result)
    catalog, actions = _validate_catalog(action_catalog)
    controller_policy = _validate_policy(policy)
    if not isinstance(evidence_catalog, Mapping):
        raise ValidationError("evidence_catalog must be an object")
    evidence_kinds = {
        evidence_id: item.get("kind") for evidence_id, item in evidence_catalog.items()
    }
    for evidence_id, kind in evidence_kinds.items():
        _identifier(evidence_id, "evidence_catalog id")
        _identifier(kind, "evidence_catalog kind")
    history = {_sha(item, "request_history signature") for item in request_history}
    completed_actions = set(
        _ids(list(completed_action_ids), "completed_action_ids", allow_empty=True)
    )
    if not completed_actions.issubset(actions):
        raise ValidationError("completed_action_ids contains an unknown action")

    root = _closed(
        value,
        {
            "schema_version",
            "request_set_id",
            "epoch_id",
            "epoch_sha256",
            "hypothesis_set_sha256",
            "requests",
        },
        "request_set",
    )
    if root["schema_version"] != REQUEST_SCHEMA:
        raise ValidationError(f"request_set.schema_version must be {REQUEST_SCHEMA}")
    _identifier(root["request_set_id"], "request_set.request_set_id")
    if root["epoch_id"] != epoch_id:
        raise ValidationError("request_set epoch_id does not match Controller")
    if root["hypothesis_set_sha256"] != hypothesis_digest:
        raise ValidationError("request_set hypothesis digest does not match Controller")
    if type(root["requests"]) is not list or not 1 <= len(root["requests"]) <= 32:
        raise ValidationError("request_set.requests must contain 1 to 32 entries")

    active = {
        item["hypothesis_id"]: item
        for item in hypotheses["hypotheses"]
        if item["disposition"] == "active"
    }
    exclusive = {
        (item["left"], item["right"])
        for item in hypotheses["relationships"]
        if item["relation"] == "exclusive"
    }
    if active and all(item["confidence"] == "direction_supported" for item in active.values()):
        gap_reason = "hypotheses_sufficiently_supported"
        return _selection(
            "sufficient", epoch_id, hypothesis_digest, catalog, controller_policy,
            None, [], [], gap_reason
        )
    if controller_policy["remaining_profile_actions"] == 0:
        return _selection(
            "evidence_gap", epoch_id, hypothesis_digest, catalog, controller_policy,
            None, [], [], "profile_budget_exhausted"
        )

    normalized_requests = []
    seen_ids = set()
    for index, raw in enumerate(root["requests"]):
        item = _closed(
            raw,
            {
                "request_id",
                "action_id",
                "question",
                "target_hypothesis_ids",
                "exclusive_pairs",
                "outcomes",
            },
            f"requests[{index}]",
        )
        request_id = _identifier(item["request_id"], f"requests[{index}].request_id")
        if request_id in seen_ids:
            raise ValidationError("request ids must be unique")
        seen_ids.add(request_id)
        action_id = _identifier(item["action_id"], f"requests[{index}].action_id")
        if action_id not in actions:
            raise ValidationError(f"request cites unknown action {action_id}")
        _text(item["question"], f"requests[{index}].question")
        targets = _ids(item["target_hypothesis_ids"], f"requests[{index}].target_hypothesis_ids")
        if not set(targets).issubset(active):
            raise ValidationError("request target is not an active hypothesis")
        pairs = item["exclusive_pairs"]
        if type(pairs) is not list:
            raise ValidationError("request exclusive_pairs must be an array")
        normalized_pairs = []
        for pair_index, raw_pair in enumerate(pairs):
            pair = _closed(raw_pair, {"left", "right"}, f"exclusive_pairs[{pair_index}]")
            left = _identifier(pair["left"], "exclusive pair left")
            right = _identifier(pair["right"], "exclusive pair right")
            if left >= right:
                raise ValidationError("exclusive pair must use canonical order")
            if left not in targets or right not in targets:
                raise ValidationError("exclusive pair members must both be request targets")
            if (left, right) not in exclusive:
                raise ValidationError("request pair is not an admitted exclusive relationship")
            normalized_pairs.append({"left": left, "right": right})
        normalized_pairs.sort(key=lambda pair: (pair["left"], pair["right"]))
        if len({(p["left"], p["right"]) for p in normalized_pairs}) != len(normalized_pairs):
            raise ValidationError("request exclusive pairs must be unique")

        outcomes = item["outcomes"]
        if type(outcomes) is not list or not 2 <= len(outcomes) <= 8:
            raise ValidationError("request outcomes must contain 2 to 8 alternatives that change hypotheses")
        normalized_outcomes = []
        effect_signatures = set()
        changed = set()
        outcome_ids = set()
        for outcome_index, raw_outcome in enumerate(outcomes):
            outcome = _closed(
                raw_outcome,
                {"outcome_id", "supports", "opposes"},
                f"outcomes[{outcome_index}]",
            )
            outcome_id = _identifier(outcome["outcome_id"], "outcome_id")
            if outcome_id in outcome_ids:
                raise ValidationError("outcome ids must be unique")
            outcome_ids.add(outcome_id)
            supports = _ids(outcome["supports"], "outcome.supports", allow_empty=True)
            opposes = _ids(outcome["opposes"], "outcome.opposes", allow_empty=True)
            if set(supports) & set(opposes):
                raise ValidationError("outcome cannot support and oppose the same hypothesis")
            if not (set(supports) | set(opposes)).issubset(targets):
                raise ValidationError("outcome effects must stay within request targets")
            changed.update(supports)
            changed.update(opposes)
            effect_signatures.add((tuple(supports), tuple(opposes)))
            normalized_outcomes.append(
                {"outcome_id": outcome_id, "supports": supports, "opposes": opposes}
            )
        if not changed or len(effect_signatures) < 2:
            raise ValidationError("request outcomes do not change competing hypotheses")
        for pair in normalized_pairs:
            left = pair["left"]
            right = pair["right"]
            favors_left = any(
                left in outcome["supports"] and right in outcome["opposes"]
                for outcome in normalized_outcomes
            )
            favors_right = any(
                right in outcome["supports"] and left in outcome["opposes"]
                for outcome in normalized_outcomes
            )
            if not (favors_left and favors_right):
                raise ValidationError(
                    "exclusive pair outcomes must discriminate in both directions"
                )
        opposed = {
            hypothesis_id
            for outcome in normalized_outcomes
            for hypothesis_id in outcome["opposes"]
        }
        if not set(targets).issubset(opposed):
            raise ValidationError(
                "every target hypothesis must be falsifiable by an outcome"
            )
        normalized_outcomes.sort(key=lambda item: item["outcome_id"])
        normalized_requests.append(
            {
                "request_id": request_id,
                "action_id": action_id,
                "question": item["question"],
                "target_hypothesis_ids": targets,
                "exclusive_pairs": normalized_pairs,
                "outcomes": normalized_outcomes,
            }
        )

    candidates = []
    rejections = []
    missing_capabilities = set()
    available = set(controller_policy["available_capability_ids"])
    for request in normalized_requests:
        action = actions[request["action_id"]]
        missing = set(action["required_capability_ids"]) - available
        reason = None
        if missing:
            reason = "required_capability_unavailable"
            missing_capabilities.update(missing)
        else:
            for field, policy_field in (
                ("cost", "max_cost"),
                ("perturbation", "max_perturbation"),
                ("risk", "max_risk"),
            ):
                if _LEVELS[action[field]] > _LEVELS[controller_policy[policy_field]]:
                    reason = f"{field}_exceeds_policy"
                    break
        signature = _request_signature(epoch_id, action, request)
        if reason is None and not action["repeatable"] and action["action_id"] in completed_actions:
            reason = "action_is_not_repeatable"
        if reason is None and signature in history:
            reason = "equivalent_request_already_attempted"
        if reason is not None:
            rejections.append({"request_id": request["request_id"], "reason": reason})
            continue
        falsifiable = {
            hypothesis_id
            for outcome in request["outcomes"]
            for hypothesis_id in outcome["opposes"]
        }
        independent_gain = 0
        for hypothesis_id in request["target_hypothesis_ids"]:
            existing = {
                evidence_kinds.get(evidence_id)
                for evidence_id in active[hypothesis_id]["support_evidence_ids"]
            }
            if action["evidence_kind"] not in existing:
                independent_gain += 1
        rank = (
            -len(request["exclusive_pairs"]),
            -len(falsifiable),
            -independent_gain,
            _LEVELS[action["perturbation"]],
            _LEVELS[action["risk"]],
            _LEVELS[action["cost"]],
            request["request_id"],
        )
        candidates.append((rank, request, action, signature, len(falsifiable), independent_gain))
    rejections.sort(key=lambda item: item["request_id"])
    if not candidates:
        return _selection(
            "evidence_gap", epoch_id, hypothesis_digest, catalog, controller_policy,
            None, rejections, sorted(missing_capabilities), "no_admissible_discriminator"
        )
    candidates.sort(key=lambda item: item[0])
    _, request, action, signature, falsifiable_count, independent_gain = candidates[0]
    selected = {
        **copy.deepcopy(request),
        "controller_action": copy.deepcopy(action),
        "request_signature": signature,
        "discrimination": {
            "exclusive_pair_count": len(request["exclusive_pairs"]),
            "falsifiable_hypothesis_count": falsifiable_count,
            "independent_evidence_gain": independent_gain,
        },
    }
    return _selection(
        "selected", epoch_id, hypothesis_digest, catalog, controller_policy,
        selected, rejections, sorted(missing_capabilities), None
    )


def _selection(
    status: str,
    epoch_id: str,
    hypothesis_digest: str,
    catalog: Mapping[str, Any],
    policy: Mapping[str, Any],
    selected: Mapping[str, Any] | None,
    rejections: list[dict],
    missing_capabilities: list[str],
    gap_reason: str | None,
) -> dict:
    return {
        "schema_version": SELECTION_SCHEMA,
        "status": status,
        "epoch_id": epoch_id,
        "hypothesis_set_sha256": hypothesis_digest,
        "action_catalog_sha256": _canonical_digest(catalog),
        "selection_policy_sha256": _canonical_digest(policy),
        "selected_request": None if selected is None else copy.deepcopy(dict(selected)),
        "rejections": copy.deepcopy(rejections),
        "missing_capability_ids": list(missing_capabilities),
        "gap_reason": gap_reason,
    }
