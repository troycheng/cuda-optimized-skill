#!/usr/bin/env python3
"""Verify that claimed optimization methods are actually present in compiled SASS.

Uses cuobjdump --dump-sass on the compiled .so to grep for expected instruction
patterns defined in references/sass_signatures.json.

Writes iterv{i}/sass_check.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compiler_evidence  # noqa: E402


_DEFAULT_SIGNATURES = Path(__file__).resolve().parent.parent / "references" / "sass_signatures.json"


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_so_file(kernel_path: str) -> str | None:
    """Find the compiled .so corresponding to a .cu kernel."""
    base = os.path.splitext(kernel_path)[0]
    for ext in (".so", ".dll"):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def _dump_sass(so_path: str) -> str:
    """Run cuobjdump --dump-sass and return output."""
    cuobjdump = "cuobjdump"
    try:
        r = subprocess.run(
            [cuobjdump, "--dump-sass", so_path],
            capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
            timeout=30,
        )
        if r.returncode != 0:
            detail = (r.stderr or r.stdout or f"exit code {r.returncode}").strip()
            return f"ERROR: cuobjdump failed: {detail}"
        sass_text = r.stdout or ""
        if not sass_text.strip():
            return "ERROR: cuobjdump produced no SASS output"
        return sass_text
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return f"ERROR: {e}"


def check_method_sass(
    method_id: str,
    sass_text: str,
    signatures: dict,
) -> dict:
    """Check if a method's expected SASS patterns appear in the disassembly."""
    result = {
        "method_id": method_id,
        "verified": False,
        "status": "unavailable",
        "patterns_checked": [],
        "patterns_found": [],
        "patterns_missing": [],
    }

    methods = signatures.get("methods", {})
    if not isinstance(methods, dict) or method_id not in methods:
        result["note"] = "method_signature_unavailable"
        return result
    method_sigs = methods.get(method_id)
    if not isinstance(method_sigs, dict):
        result["note"] = "method_signature_invalid"
        return result
    patterns = method_sigs.get("sass_patterns", [])
    require_any = method_sigs.get("require_any", True)

    if not patterns:
        result["status"] = "not_applicable"
        result["note"] = "no_patterns_defined"
        return result
    if not isinstance(patterns, list) or not all(
        type(pattern) is str and pattern for pattern in patterns
    ) or type(require_any) is not bool:
        result["note"] = "method_signature_invalid"
        return result

    result["patterns_checked"] = patterns

    try:
        for pattern in patterns:
            if re.search(pattern, sass_text, re.IGNORECASE):
                result["patterns_found"].append(pattern)
            else:
                result["patterns_missing"].append(pattern)
    except re.error as error:
        result["note"] = f"invalid_sass_pattern: {error}"
        return result

    if require_any:
        result["verified"] = len(result["patterns_found"]) > 0
    else:
        # require_all
        result["verified"] = len(result["patterns_missing"]) == 0
    result["status"] = "passed" if result["verified"] else "failed"

    return result


def _status_checks(methods_list: list, status: str, note: str) -> list:
    return [
        {
            "method_id": method.get("id", "unknown"),
            "verified": False,
            "status": status,
            "patterns_checked": [],
            "patterns_found": [],
            "patterns_missing": [],
            "note": note,
        }
        for method in methods_list
    ]


def _checks_status(checks: list) -> str:
    statuses = [check.get("status") for check in checks]
    if "failed" in statuses:
        return "failed"
    required = [status for status in statuses if status != "not_applicable"]
    if required and all(status == "passed" for status in required):
        return "passed"
    if statuses and not required:
        return "not_applicable"
    return "unavailable"


def _binary_matches_record(identity: dict, record: dict) -> bool:
    return (
        record.get("status") == "available"
        and record.get("path") == str(identity["path"])
        and record.get("sha256") == identity["sha256"]
        and record.get("size_bytes") == identity["size_bytes"]
    )


def _write_and_return(iter_dir: str, result: dict) -> dict:
    _write_result(iter_dir, result)
    return result


def run(state_path: str, iteration: int, signatures_path: str = None) -> dict:
    state = _load_json(state_path)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")

    # Load method choices
    methods_path = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_path):
        sys.exit(f"methods.json not found at {methods_path}")
    methods_data = _load_json(methods_path)
    methods_list = methods_data.get("methods", [])

    # Load SASS signatures
    sig_path = signatures_path or str(_DEFAULT_SIGNATURES)
    if os.path.isfile(sig_path):
        signatures = _load_json(sig_path)
    else:
        signatures = {"methods": {}}

    # Find compiled kernel
    kernel_path = None
    for ext in (".cu", ".py"):
        candidate = os.path.join(iter_dir, f"kernel{ext}")
        if os.path.isfile(candidate):
            kernel_path = candidate
            break

    if not kernel_path:
        return {"error": "no_kernel_found", "checks": []}

    # For Triton kernels, SASS check is not directly applicable
    if kernel_path.endswith(".py"):
        checks = _status_checks(
            methods_list, "not_applicable", "triton_kernel_sass_not_applicable"
        )
        return _write_and_return(iter_dir, {
            "kernel": kernel_path,
            "backend": "triton",
            "status": "not_applicable",
            "binary_sha256": None,
            "checks": checks,
        })

    # CUDA/CUTLASS: find .so and dump SASS
    so_path = _find_so_file(kernel_path)
    evidence_dir = Path(iter_dir) / "compiler_evidence"
    if not so_path:
        checks = _status_checks(methods_list, "unavailable", "binary_not_found")
        return _write_and_return(iter_dir, {
            "error": "so_not_found",
            "kernel": kernel_path,
            "backend": "cuda",
            "status": "unavailable",
            "binary_sha256": None,
            "checks": checks,
        })

    try:
        manifest = compiler_evidence.load_manifest(evidence_dir)
    except (OSError, ValueError) as error:
        checks = _status_checks(
            methods_list, "unavailable", f"compiler_evidence_unavailable: {error}"
        )
        return _write_and_return(iter_dir, {
            "kernel": kernel_path,
            "so": so_path,
            "backend": "cuda",
            "status": "unavailable",
            "binary_sha256": None,
            "checks": checks,
        })

    before_dump = compiler_evidence.artifact_identity(so_path)
    if before_dump is None or not _binary_matches_record(
        before_dump, manifest["binary"]
    ):
        checks = _status_checks(
            methods_list, "unavailable", "binary_evidence_mismatch"
        )
        return _write_and_return(iter_dir, {
            "kernel": kernel_path,
            "so": so_path,
            "backend": "cuda",
            "status": "unavailable",
            "binary_sha256": None,
            "checks": checks,
        })

    sass_text = _dump_sass(so_path)

    if sass_text.startswith("ERROR:"):
        compiler_evidence.update_manifest(
            evidence_dir,
            discovered={"sass": None},
        )
        checks = _status_checks(
            methods_list, "unavailable", f"cuobjdump_unavailable: {sass_text}"
        )
        return _write_and_return(iter_dir, {
            "kernel": kernel_path,
            "so": so_path,
            "backend": "cuda",
            "status": "unavailable",
            "binary_sha256": before_dump["sha256"],
            "sass_error": sass_text,
            "checks": checks,
        })

    after_dump = compiler_evidence.artifact_identity(so_path)
    if after_dump != before_dump:
        try:
            compiler_evidence.update_manifest(
                evidence_dir, discovered={"binary": None, "sass": None}
            )
        except (OSError, ValueError):
            pass
        checks = _status_checks(
            methods_list, "unavailable", "binary_changed_during_sass_dump"
        )
        return _write_and_return(iter_dir, {
            "kernel": kernel_path,
            "so": so_path,
            "backend": "cuda",
            "status": "unavailable",
            "binary_sha256": None,
            "checks": checks,
        })

    sass_path = evidence_dir / "sass.txt"
    try:
        compiler_evidence.atomic_write_text(sass_path, sass_text)
        manifest = compiler_evidence.update_manifest(
            evidence_dir,
            binary=so_path,
            discovered={"sass": sass_path},
            binary_sha256=before_dump["sha256"],
        )
    except (OSError, ValueError) as error:
        checks = _status_checks(
            methods_list, "unavailable", f"sass_evidence_write_failed: {error}"
        )
        return _write_and_return(iter_dir, {
            "kernel": kernel_path,
            "so": so_path,
            "backend": "cuda",
            "status": "unavailable",
            "binary_sha256": None,
            "checks": checks,
        })

    # Check each method
    checks = []
    for m in methods_list:
        mid = m.get("id", "unknown")
        check = check_method_sass(mid, sass_text, signatures)
        checks.append(check)

    result = {
        "kernel": kernel_path,
        "so": so_path,
        "backend": "cuda",
        "status": _checks_status(checks),
        "binary_sha256": manifest["binary_sha256"],
        "sass_lines": len(sass_text.splitlines()),
        "checks": checks,
    }

    _write_result(iter_dir, result)
    print(json.dumps(result, indent=2))
    return result


def _write_result(iter_dir: str, result: dict):
    out_path = Path(iter_dir) / "sass_check.json"
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    compiler_evidence.atomic_write_text(out_path, payload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--signatures", default=None)
    args = p.parse_args()
    run(args.state, args.iter, args.signatures)


if __name__ == "__main__":
    main()
