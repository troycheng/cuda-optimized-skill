#!/usr/bin/env python3
"""Persistent, workload-scoped optimization strategy memory."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path


DEFAULT_PATH = os.path.expanduser(
    "~/.codex/state/cuda-kernel-optimizer/global_strategy_memory.json"
)


def _read(path: str, default: dict) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else default
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def _write(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _token(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value or "").strip("_").lower()
    return value[:64] or "none"


def scope_key(
    backend: str,
    baseline: str,
    ref: str,
    dims: dict,
    arch: str,
) -> str:
    dims_hash = hashlib.sha256(
        json.dumps(dims or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return "__".join([
        _token(backend or "auto"),
        _token(Path(baseline).stem),
        _token(Path(ref).stem),
        _token(arch or "auto_arch"),
        dims_hash,
    ])


def load_constraints(path: str, key: str) -> dict:
    root = _read(path, {"version": 2, "scopes": {}})
    scope = (root.get("scopes") or {}).get(key) or {}
    methods = scope.get("methods") or {}
    preferred = sorted(
        mid for mid, item in methods.items() if item.get("last_outcome") == "positive"
    )
    blocked = sorted(
        mid for mid, item in methods.items()
        if item.get("last_outcome") in {"negative", "rejected"}
    )
    return {
        "preferred_method_ids": preferred,
        "blocked_method_ids": blocked,
        "scope_seen": bool(scope),
    }


def record(
    path: str,
    key: str,
    meta: dict,
    method_outcomes: list[dict],
    bundle_outcome: dict,
) -> dict:
    root = _read(path, {"version": 2, "updated_at": "", "scopes": {}})
    root.setdefault("version", 2)
    scopes = root.setdefault("scopes", {})
    scope = scopes.setdefault(key, {
        "meta": meta,
        "methods": {},
        "bundles": {},
    })
    scope["meta"] = meta

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    method_store = scope.setdefault("methods", {})
    for outcome in method_outcomes:
        mid = str(outcome.get("method_id") or "").strip()
        if not mid or outcome.get("outcome") not in {"positive", "negative", "rejected"}:
            continue
        old = method_store.get(mid) or {}
        method_store[mid] = {
            "last_outcome": outcome["outcome"],
            "last_reason": outcome.get("reason"),
            "last_evidence": outcome.get("evidence") or {},
            "count": int(old.get("count", 0)) + 1,
            "first_seen": old.get("first_seen") or now,
            "last_seen": now,
        }

    method_ids = sorted(set(bundle_outcome.get("method_ids") or []))
    if method_ids:
        fingerprint = hashlib.sha256("\0".join(method_ids).encode("utf-8")).hexdigest()[:16]
        bundles = scope.setdefault("bundles", {})
        old = bundles.get(fingerprint) or {}
        bundles[fingerprint] = {
            "method_ids": method_ids,
            "last_outcome": bundle_outcome.get("outcome", "unknown"),
            "last_evidence": bundle_outcome.get("evidence") or {},
            "count": int(old.get("count", 0)) + 1,
            "first_seen": old.get("first_seen") or now,
            "last_seen": now,
        }

    root["updated_at"] = now
    _write(path, root)
    return load_constraints(path, key)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect workload-scoped strategy memory")
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--scope", default="")
    args = p.parse_args()
    payload = _read(args.path, {"version": 2, "scopes": {}})
    if args.scope:
        payload = (payload.get("scopes") or {}).get(args.scope) or {}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
