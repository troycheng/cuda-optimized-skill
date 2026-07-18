#!/usr/bin/env python3
"""Return a small, target-compatible set of optimization knowledge cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REFERENCE_DIR = Path(__file__).resolve().parent.parent / "references"
REGISTRY_PATH = REFERENCE_DIR / "method_registry.json"
WORKLOAD_PATH = REFERENCE_DIR / "workload_methods.json"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _kernel_cards(
    registry: Dict[str, Any],
    arch: str,
    axis: Optional[str],
    bottleneck: Optional[str],
    observed_metrics: Dict[str, float],
) -> List[Dict[str, Any]]:
    feature_map = registry.get("arch_feature_map", {})
    if arch not in feature_map:
        raise ValueError(
            "Unknown architecture %s; exact capability data is required." % arch
        )
    available = set(feature_map[arch])
    needle = (bottleneck or "").lower()
    cards = []
    for method_id, method in registry["methods"].items():
        if axis and method.get("axis") != axis:
            continue
        required = set(method.get("required_features", []))
        if not required.issubset(available):
            continue
        searchable = " ".join(
            [method_id, str(method.get("name", "")), str(method.get("trigger_metric", ""))]
        ).lower()
        if needle and needle not in searchable:
            continue
        metric = method.get("trigger_metric")
        direction = method.get("trigger_direction")
        threshold = method.get("trigger_bad")
        observed = observed_metrics.get(metric) if metric else None
        applicability = "unverified"
        if observed is not None and threshold is not None:
            triggered = (direction == "low_is_bad" and observed <= threshold) or (
                direction == "high_is_bad" and observed >= threshold
            )
            applicability = (
                "observed_bad_trigger" if triggered else "observed_not_triggered"
            )
        cards.append(
            {
                "id": method_id,
                "layer": "kernel",
                "axis": method["axis"],
                "priority": method["priority"],
                "name": method["name"],
                "trigger_metric": method.get("trigger_metric"),
                "trigger_direction": method.get("trigger_direction"),
                "trigger_bad": threshold,
                "observed_value": observed,
                "applicability": applicability,
                "required_features": method.get("required_features", []),
                "reference_impl": method.get("reference_impl"),
                "evidence_required": "correctness plus paired target measurement",
            }
        )
    applicability_rank = {
        "observed_bad_trigger": 0,
        "unverified": 1,
        "observed_not_triggered": 2,
    }
    return sorted(
        cards,
        key=lambda item: (
            applicability_rank[item["applicability"]],
            item["priority"],
            item["id"],
        ),
    )


def _workload_cards(bottleneck: Optional[str]) -> List[Dict[str, Any]]:
    cards = _load(WORKLOAD_PATH)["methods"]
    if bottleneck:
        needle = bottleneck.lower()
        cards = [
            item
            for item in cards
            if needle
            in " ".join(
                [item["id"], item["bottleneck"], item["name"], " ".join(item["signals"])]
            ).lower()
        ]
    return sorted(cards, key=lambda item: (item["priority"], item["id"]))


def query(
    arch: str,
    layer: str = "kernel",
    axis: Optional[str] = None,
    bottleneck: Optional[str] = None,
    limit: int = 5,
    observed_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if limit < 1 or limit > 20:
        raise ValueError("limit must be between 1 and 20")
    registry = _load(REGISTRY_PATH)
    if layer == "kernel":
        methods = _kernel_cards(
            registry, arch, axis, bottleneck, observed_metrics or {}
        )
    elif layer == "workload":
        if arch not in registry.get("arch_feature_map", {}):
            raise ValueError(
                "Unknown architecture %s; exact capability data is required." % arch
            )
        methods = _workload_cards(bottleneck)
    else:
        raise ValueError("layer must be kernel or workload")
    return {
        "schema_version": "cuda-optimizer/knowledge-query-v1",
        "arch": arch,
        "layer": layer,
        "filters": {"axis": axis, "bottleneck": bottleneck},
        "observed_metrics": observed_metrics or {},
        "methods": methods[:limit],
        "truncated": len(methods) > limit,
        "promotion_authority": "local_correctness_and_measurement_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query a bounded set of offline GPU optimization knowledge cards."
    )
    parser.add_argument("--arch", required=True, help="Exact architecture, for example sm_120")
    parser.add_argument("--layer", choices=("kernel", "workload"), default="kernel")
    parser.add_argument("--axis", choices=("compute", "memory", "latency"))
    parser.add_argument("--bottleneck")
    parser.add_argument(
        "--metrics", help="Optional JSON object mapping profiler metric names to values"
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    try:
        observed_metrics = _load(Path(args.metrics)) if args.metrics else {}
        if not isinstance(observed_metrics, dict) or not all(
            isinstance(value, (int, float)) for value in observed_metrics.values()
        ):
            parser.error("--metrics must contain a JSON object of numeric values")
        result = query(
            arch=args.arch,
            layer=args.layer,
            axis=args.axis,
            bottleneck=args.bottleneck,
            limit=args.limit,
            observed_metrics=observed_metrics,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
