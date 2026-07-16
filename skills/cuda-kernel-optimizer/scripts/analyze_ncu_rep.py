#!/usr/bin/env python3
"""Analyze an existing Nsight Compute report without requiring a run state."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from profile_ncu import (
    _aggregate_across_kernels,
    _import_metrics_csv,
    _parse_ncu_csv,
    _rank_by_axis,
)


def _run_import(ncu_bin: str, report: str, args: list[str]) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            [ncu_bin, "--import", report, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except OSError as e:
        return -1, "", str(e)


def _axis_score(items: list[dict]) -> float | None:
    scores = []
    for item in items:
        value = item.get("value")
        if value is None:
            continue
        value = max(0.0, min(100.0, float(value)))
        scores.append(value if item.get("higher_is_worse") else 100.0 - value)
    return max(scores) if scores else None


def _source_identity(source: str, report: str) -> tuple[str, str, list[str]]:
    if not source or not os.path.isfile(source):
        return "unknown", "unverified_no_source", []
    text = Path(source).read_text(encoding="utf-8", errors="ignore")
    suffix = Path(source).suffix.lower()
    if suffix == ".py":
        backend = "triton"
        hints: list[str] = []
    else:
        stripped = re.sub(r"//.*", "", text)
        stripped = re.sub(r"/\*[\s\S]*?\*/", "", stripped)
        backend = (
            "cutlass"
            if re.search(r"#\s*include\s*<\s*(cutlass|cute)/|\b(cutlass|cute)::", stripped)
            else "cuda"
        )
        hints = re.findall(r"__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    freshness = (
        "possible_stale_source_newer_than_report"
        if os.path.getmtime(source) > os.path.getmtime(report)
        else "unverified_mtime_only"
    )
    return backend, freshness, hints


def _render_markdown(result: dict) -> str:
    lines = [
        "# Nsight Compute Report Analysis",
        "",
        f"- Report: `{result['report']}`",
        f"- Status: `{result['status']}`",
        f"- Backend: `{result['backend']}`",
        f"- Kernel names: `{result.get('kernel_names') or []}`",
        f"- Freshness: `{result['freshness_assessment']}`",
        f"- Analysis quality: `{result['analysis_quality']}`",
        f"- Primary axis: `{result.get('primary_axis') or 'unknown'}`",
        "",
    ]
    for axis in ("compute", "memory", "latency"):
        lines += [f"## {axis.title()} evidence", ""]
        items = result.get("top_metrics", {}).get(axis, [])
        if not items:
            lines += ["_(no supported metric evidence)_", ""]
            continue
        lines += ["| Metric | Value | Unit | Samples |", "| --- | ---: | --- | ---: |"]
        for item in items:
            lines.append(
                f"| `{item.get('name')}` | {item.get('value')} | "
                f"{item.get('unit') or ''} | {item.get('samples') or 0} |"
            )
        lines.append("")
    lines += [
        "## Interpretation boundary",
        "",
        "This classification is heuristic and limited to metrics present in the report. "
        "Use the detailed report, source, workload FLOPs/bytes, and repeated timing before "
        "making a Roofline or end-to-end performance claim.",
        "",
    ]
    return "\n".join(lines)


def analyze(report: str, out_dir: str, ncu_bin: str, ncu_num: int, source: str = "") -> dict:
    report = os.path.abspath(report)
    if not os.path.isfile(report):
        raise FileNotFoundError(report)
    resolved = shutil.which(ncu_bin) if not os.path.isfile(ncu_bin) else os.path.abspath(ncu_bin)
    if not resolved:
        raise FileNotFoundError(f"ncu executable not found: {ncu_bin}")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary_rc, summary, summary_err = _run_import(
        resolved, report, ["--print-summary", "per-kernel"]
    )
    details_rc, details, details_err = _run_import(resolved, report, ["--page", "details"])
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    (out / "summary.stderr.txt").write_text(summary_err, encoding="utf-8")
    (out / "details.txt").write_text(details, encoding="utf-8")
    (out / "details.stderr.txt").write_text(details_err, encoding="utf-8")

    raw_rc, csv_text, raw_err = _import_metrics_csv(resolved, report)
    (out / "raw.csv").write_text(csv_text, encoding="utf-8")
    (out / "raw.stderr.txt").write_text(raw_err, encoding="utf-8")

    backend, freshness, hints = _source_identity(source, report)
    rows = _parse_ncu_csv(csv_text, kernel_name_hints=hints) if raw_rc == 0 else []
    aggregate = _aggregate_across_kernels(rows)
    top = _rank_by_axis(aggregate, ncu_num)
    scores = {axis: _axis_score(items) for axis, items in top.items()}
    evidenced = {axis: score for axis, score in scores.items() if score is not None}
    primary = max(evidenced, key=evidenced.get) if evidenced else None
    kernel_names = sorted({
        kernel
        for info in aggregate.values()
        for kernel in (info.get("kernels") or [])
    })

    result = {
        "status": "success" if raw_rc == 0 and aggregate else "partial",
        "report": report,
        "source": os.path.abspath(source) if source else None,
        "backend": backend,
        "kernel_names": kernel_names,
        "freshness_assessment": freshness,
        "ncu_bin": resolved,
        "summary_returncode": summary_rc,
        "details_returncode": details_rc,
        "raw_returncode": raw_rc,
        "metric_count": len(aggregate),
        "analysis_quality": "heuristic",
        "axis_scores": scores,
        "primary_axis": primary,
        "top_metrics": top,
        "artifacts": {
            "summary": str(out / "summary.txt"),
            "details": str(out / "details.txt"),
            "raw_csv": str(out / "raw.csv"),
        },
    }
    (out / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "analysis.md").write_text(_render_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze an existing .ncu-rep report")
    p.add_argument("report")
    p.add_argument("--out-dir", default="")
    p.add_argument("--ncu-bin", default="ncu")
    p.add_argument("--ncu-num", type=int, default=5)
    p.add_argument("--source", default="", help="Optional current kernel source for report identity hints")
    args = p.parse_args()
    out_dir = args.out_dir or f"{os.path.splitext(os.path.abspath(args.report))[0]}_analysis"
    try:
        result = analyze(args.report, out_dir, args.ncu_bin, args.ncu_num, args.source)
    except (FileNotFoundError, OSError) as e:
        print(json.dumps({"status": "error", "error": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["status"] == "success" else 2)


if __name__ == "__main__":
    main()
