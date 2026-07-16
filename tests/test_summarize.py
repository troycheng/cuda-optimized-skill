from __future__ import annotations

import copy
import contextlib
import argparse
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZE_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "summarize.py"
)
STATE_PATH = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "state.py"


def _load_summarize():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_summarize_test", SUMMARIZE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_state():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_state_summarize_integration", STATE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _statistics(status: str = "confirmed_win", *, estimate: float = 6.0) -> dict:
    intervals = {
        "confirmed_win": (estimate, 4.0, 8.0),
        "confirmed_loss": (-6.0, -8.0, -4.0),
        "inconclusive": (0.1, -1.0, 1.0),
    }
    estimate_pct, ci_low, ci_high = intervals[status]
    return {
        "status": status,
        "statistic": "median_paired_improvement_pct",
        "direction": "lower",
        "min_effect_pct": 1.0,
        "confidence": 0.95,
        "estimate_pct": estimate_pct,
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "valid_pairs": 9,
        "invalid_pairs": 1,
        "improvements_pct": [estimate_pct] * 9,
    }


def _full_win_state() -> dict:
    kernel = _statistics(estimate=6.0)
    workload = _statistics(estimate=4.0)
    return {
        "schema_version": 2,
        "run_dir": "/tmp/cuda-run",
        "input_hash": "a" * 64,
        "mode": "full",
        "budget": {
            "name": "balanced",
            "max_seconds": 900,
            "max_rounds": 3,
            "branches": 4,
            "sanitizer_mode": "targeted",
        },
        "workload": {
            "kind": "command",
            "source_hash": "b" * 64,
            "objective": {
                "primary_metric": {"name": "latency_ms", "direction": "lower"},
                "min_effect_pct": 1.0,
                "constraints": [
                    {"name": "memory_mb", "max_regression_pct": 2.0}
                ],
            },
        },
        "baseline_file_original": "/src/baseline.py",
        "ref_file": "/src/ref.py",
        "backend": "triton",
        "dims": {"M": 128, "N": 256},
        "confidence": 0.95,
        "min_effect_pct": 1.0,
        "env": {
            "gpus": [
                {
                    "name": "NVIDIA GeForce RTX 5090",
                    "sm_arch": "sm_120",
                    "compute_capability": "12.0",
                }
            ],
            "nvcc": {"version": "13.0"},
            "ncu": {
                "version": "2026.1",
                "metrics_query_available": True,
                "can_read_counters": True,
            },
        },
        "best_file": "/tmp/cuda-run/iterv1/kernel.py",
        "candidates": {},
        "best_metric_ms": 0.42,
        "best_kernel_statistics": kernel,
        "best_workload_statistics": workload,
        "correctness": {"status": "passed"},
        "sass_verification": {"status": "passed"},
        "sanitizer_coverage": "available",
        "compiler_evidence": {
            "status": "available",
            "stages": ["source", "ttir", "ttgir", "llvm_ir", "ptx", "sass"],
        },
        "history": [
            {
                "iter": 1,
                "status": "end_to_end_win",
                "validation_passed": True,
                "statistics": kernel,
            }
        ],
        "frontier": [
            {
                "iter": 1,
                "status": "kernel_only_win",
                "candidate_file": "/tmp/cuda-run/iterv1/frontier.py",
            }
        ],
        "rejected_candidates": [
            {"id": "b2", "status": "rejected_correctness"}
        ],
        "raw_artifacts": {
            "kernel_paired_samples": "/tmp/cuda-run/iterv1/paired_samples.jsonl",
            "workload_paired_samples": (
                "/tmp/cuda-run/iterv1/workload/paired_samples.jsonl"
            ),
        },
        "resume": {
            "status": "complete",
            "checkpoint": "/tmp/cuda-run/checkpoint.json",
        },
        "terminal_decision": {
            "iteration": 1,
            "input_hash": "a" * 64,
            "status": "end_to_end_win",
            "mode": "full",
            "candidate_file": "/tmp/cuda-run/iterv1/kernel.py",
            "candidate_sha256": "c" * 64,
            "candidate_id": "b1",
            "decision_sha256": "f" * 64,
            "statistics": copy.deepcopy(kernel),
            "workload_statistics": copy.deepcopy(workload),
            "constraints": [
                {
                    "name": "memory_mb",
                    "status": "passed",
                    "estimate_pct": 0.5,
                    "ci_low_pct": 0.1,
                    "ci_high_pct": 0.9,
                    "max_regression_pct": 2.0,
                    "cap_pct": 2.0,
                    "values_pct": [0.5],
                }
            ],
            "correctness": {"status": "passed"},
            "sass": {"status": "passed"},
            "sanitizer": {"status": "passed", "coverage": "complete"},
            "compiler_evidence": copy.deepcopy({
                "status": "available",
                "stages": ["source", "ttir", "ttgir", "llvm_ir", "ptx", "sass"],
            }),
            "resume": copy.deepcopy({
                "status": "complete",
                "checkpoint": "/tmp/cuda-run/checkpoint.json",
            }),
            "kernel_paired_samples": {
                "schema_version": 2,
                "kind": "kernel",
                "path": "/tmp/cuda-run/iterv1/paired_samples.jsonl",
                "sha256": "d" * 64,
                "pairs": 10,
                "input_hash": "a" * 64,
                "iteration": 1,
                "candidate_id": "b1",
                "candidate_file": "/tmp/cuda-run/iterv1/kernel.py",
                "candidate_sha256": "c" * 64,
            },
            "workload_paired_samples": {
                "schema_version": 2,
                "kind": "workload",
                "path": "/tmp/cuda-run/iterv1/workload/paired_samples.jsonl",
                "sha256": "e" * 64,
                "pairs": 10,
                "input_hash": "a" * 64,
                "iteration": 1,
                "candidate_id": "b1",
                "candidate_file": "/tmp/cuda-run/iterv1/kernel.py",
                "candidate_sha256": "c" * 64,
            },
        },
    }


def _minimal_state(run_dir: Path) -> dict:
    return {
        "schema_version": 2,
        "run_dir": str(run_dir),
        "input_hash": "a" * 64,
        "budget": {},
        "candidates": {},
        "mode": "kernel-only",
        "workload": None,
        "history": [],
    }


class SummarizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summarize = _load_summarize()
        self.full_win_state = _full_win_state()

    def test_full_win_has_separate_kernel_and_workload_sections(self) -> None:
        text = self.summarize.render_text(self.full_win_state)

        self.assertIn("# Result: end_to_end_win", text)
        self.assertIn("## Kernel evidence", text)
        self.assertIn("estimate: 6.000%", text)
        self.assertIn("CI: [4.000%, 8.000%]", text)
        self.assertIn("pairs: 9 valid / 1 invalid", text)
        self.assertIn("correctness: passed", text)
        self.assertIn("SASS: passed", text)
        self.assertIn("## Real workload evidence", text)
        self.assertIn("primary KPI: latency_ms (lower)", text)
        self.assertIn("constraint: memory_mb <= 2.000% regression", text)
        self.assertIn("paired_samples.jsonl", text)

    def test_real_state_update_renders_the_same_bound_terminal_evidence(self) -> None:
        state_module = _load_state()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_dir = root / "run"
            iter_dir = run_dir / "iterv1"
            iter_dir.mkdir(parents=True)
            baseline = run_dir / "baseline.py"
            baseline.write_text("# baseline\n", encoding="utf-8")
            candidate = iter_dir / "kernel.py"
            candidate.write_text("# candidate\n", encoding="utf-8")
            candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            kernel_statistics = _statistics(estimate=6.0)
            workload_statistics = _statistics(estimate=4.0)
            kernel_statistics.update(
                ci_low_pct=6.0,
                ci_high_pct=6.0,
                valid_pairs=9,
                invalid_pairs=1,
                improvements_pct=[6.0] * 9,
            )
            workload_statistics.update(
                ci_low_pct=4.0,
                ci_high_pct=4.0,
                valid_pairs=10,
                invalid_pairs=0,
                improvements_pct=[4.0] * 10,
            )
            workload = {
                "kind": "command",
                "source": ["/bin/echo"],
                "source_hash": "b" * 64,
                "cases": [],
                "objective": {
                    "primary_metric": {"name": "latency_ms", "direction": "lower"},
                    "min_effect_pct": 1.0,
                    "constraints": [
                        {"name": "memory_mb", "max_regression_pct": 2.0}
                    ],
                },
            }
            state_path = run_dir / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "run_dir": str(run_dir),
                        "input_hash": "a" * 64,
                        "budget": {
                            "name": "balanced",
                            "max_seconds": 900,
                            "max_rounds": 3,
                        },
                        "candidates": {},
                        "mode": "full",
                        "workload": workload,
                        "min_effect_pct": 1.0,
                        "best_file": str(baseline),
                        "best_metric_ms": 2.0,
                        "best_kernel_statistics": None,
                        "best_workload_statistics": None,
                        "selected_methods": [],
                        "effective_methods": [],
                        "ineffective_methods": [],
                        "implementation_failed_methods": [],
                        "history": [],
                        "roofline_history": [],
                        "frontier": [],
                    }
                ),
                encoding="utf-8",
            )

            def paired_metadata(kind: str) -> dict:
                path = iter_dir / kind / "paired_samples.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                classifier = (
                    {
                        "direction": "lower",
                        "min_effect_pct": 1.0,
                        "confidence": 0.95,
                        "bootstrap_samples": 20,
                        "seed": 0,
                    }
                    if kind == "kernel"
                    else {
                        "objective": workload["objective"],
                        "objective_sha256": hashlib.sha256(
                            json.dumps(
                                workload["objective"],
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                        "confidence": 0.95,
                        "bootstrap_samples": 20,
                        "seed": 0,
                    }
                )
                records = [
                    {
                        "schema_version": 2,
                        "kind": kind,
                        "input_hash": "a" * 64,
                        "iteration": 1,
                        "candidate_id": "b1",
                        "candidate_file": str(candidate),
                        "candidate_sha256": candidate_sha,
                        "classifier": classifier,
                        "pair_index": index,
                        "pair": (
                            {
                                "baseline": 100.0,
                                "candidate": 94.0,
                                "valid": index < 9,
                            }
                            if kind == "kernel"
                            else {
                                "block": index,
                                "order": "AB",
                                "case": None,
                                "baseline_metrics": {
                                    "latency_ms": 100.0,
                                    "memory_mb": 100.0,
                                },
                                "candidate_metrics": {
                                    "latency_ms": 96.0,
                                    "memory_mb": 100.5,
                                },
                                "valid": True,
                                "attempts": {"baseline": 1, "candidate": 1},
                                "attempt_records": {
                                    "baseline": [],
                                    "candidate": [],
                                },
                            }
                        ),
                    }
                    for index in range(10)
                ]
                path.write_text(
                    "".join(
                        json.dumps(record, separators=(",", ":")) + "\n"
                        for record in records
                    ),
                    encoding="utf-8",
                )
                return {
                    "schema_version": 2,
                    "kind": kind,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "pairs": 10,
                    "input_hash": "a" * 64,
                    "iteration": 1,
                    "candidate_id": "b1",
                    "candidate_file": str(candidate),
                    "candidate_sha256": candidate_sha,
                    "classifier": classifier,
                }

            constraint = {
                "name": "memory_mb",
                "status": "passed",
                "estimate_pct": 0.5,
                "ci_low_pct": 0.5,
                "ci_high_pct": 0.5,
                "max_regression_pct": 2.0,
                "cap_pct": 2.0,
                "values_pct": [0.5] * 10,
            }
            decision_path = iter_dir / "decision.json"
            decision_path.write_text(
                json.dumps(
                    {
                        "status": "end_to_end_win",
                        "candidate_id": "b1",
                        "candidate_file": str(candidate),
                        "candidate_sha256": candidate_sha,
                        "statistics": kernel_statistics,
                        "workload_statistics": workload_statistics,
                        "constraints": [constraint],
                        "kernel_paired_samples": paired_metadata("kernel"),
                        "workload_paired_samples": paired_metadata("workload"),
                    }
                ),
                encoding="utf-8",
            )
            bench_path = iter_dir / "bench.json"
            bench_path.write_text(
                json.dumps(
                    {
                        "correctness": {"passed": True},
                        "kernel": {"average_ms": 1.0},
                        "reference": {"average_ms": 2.0},
                        "compiler_evidence": {
                            "status": "available",
                            "stages": ["source", "ptx", "sass"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            methods_path = iter_dir / "methods.json"
            methods_path.write_text('{"methods":[]}', encoding="utf-8")
            sass_path = iter_dir / "sass_check.json"
            sass_path.write_text(
                '{"status":"passed","checks":[]}', encoding="utf-8"
            )
            checkpoint_path = run_dir / "checkpoint.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "input_hash": "a" * 64,
                        "iteration": 1,
                        "stage": "decision",
                        "status": "in_progress",
                        "budget": {"remaining_seconds": 400.0},
                        "stage_evidence": {
                            "candidate_sanitizer": {
                                "status": "passed",
                                "coverage": "complete",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                state=str(state_path),
                iter=1,
                kernel=str(candidate),
                bench=str(bench_path),
                methods_json=str(methods_path),
                attribution=None,
                sass_check=str(sass_path),
                retries=0,
                skip_validation=True,
                allow_ineffective=False,
                decision=str(decision_path),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                state_module.cmd_update(args)
            updated = json.loads(state_path.read_text("utf-8"))
            text = self.summarize.render_text(updated)
            summary_path = run_dir / "summary.md"
            with contextlib.redirect_stdout(io.StringIO()):
                self.summarize.render(str(state_path), str(summary_path))
            self.assertTrue(summary_path.is_file())
            decision_bytes = decision_path.read_bytes()
            decision_path.write_text("{}", encoding="utf-8")
            rejected_decision_summary = run_dir / "rejected-decision-summary.md"
            with self.assertRaisesRegex(ValueError, "decision.*sha256|sha256.*decision"):
                self.summarize.render(
                    str(state_path), str(rejected_decision_summary)
                )
            self.assertFalse(rejected_decision_summary.exists())
            decision_path.write_bytes(decision_bytes)
            kernel_raw = Path(
                updated["terminal_decision"]["kernel_paired_samples"]["path"]
            )
            kernel_raw.write_text(
                kernel_raw.read_text("utf-8") + "{}\n", encoding="utf-8"
            )
            rejected_summary = run_dir / "rejected-summary.md"
            with self.assertRaisesRegex(ValueError, "sha256|record|paired"):
                self.summarize.render(str(state_path), str(rejected_summary))
            self.assertFalse(rejected_summary.exists())

        self.assertIn("# Result: end_to_end_win", text)
        self.assertIn("budget preset: balanced", text)
        self.assertIn("constraint result: memory_mb: passed", text)
        self.assertIn("compiler coverage: available", text)
        self.assertIn("SASS: passed", text)
        self.assertIn("sanitizer coverage: complete", text)
        self.assertIn("kernel/paired_samples.jsonl", text)
        self.assertIn("workload/paired_samples.jsonl", text)
        self.assertEqual(updated["terminal_decision"]["candidate_id"], "b1")
        self.assertEqual(updated["terminal_decision"]["resume"]["stage"], "decision")

    def test_answer_first_sections_have_a_fixed_order(self) -> None:
        text = self.summarize.render_text(self.full_win_state)
        headings = (
            "# Result: end_to_end_win",
            "## Frozen inputs and environment",
            "## Kernel evidence",
            "## Real workload evidence",
            "## Evidence coverage",
            "## Candidate outcomes",
            "## Raw artifacts and resume",
        )

        offsets = [text.index(heading) for heading in headings]
        self.assertEqual(offsets, sorted(offsets))
        result_block = text[: offsets[1]]
        self.assertIn("mode: full", result_block)
        self.assertIn("budget preset: balanced", result_block)
        self.assertIn("budget limit: 900.000 seconds", result_block)

    def test_missing_workload_cannot_render_end_to_end_win(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["workload"] = None
        state["best_workload_statistics"] = None

        text = self.summarize.render_text(state)

        self.assertIn("# Result: kernel_only_win", text)
        self.assertIn("No user workload was supplied", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_kernel_only_mode_keeps_inner_and_end_to_end_results_distinct(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state.update(mode="kernel-only", workload=None)
        state["best_workload_statistics"] = None
        state["history"][-1]["status"] = "kernel_only_win"
        state["terminal_decision"].update(
            mode="kernel-only",
            status="kernel_only_win",
            workload_statistics=None,
            constraints=[],
        )

        text = self.summarize.render_text(state)

        self.assertIn("# Result: kernel_only_win", text)
        self.assertIn("mode: kernel-only", text)
        self.assertIn("No user workload was supplied", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_ncu_or_sanitizer_degradation_is_prominent(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["env"]["ncu"].update(
            can_read_counters=False,
            counter_access_error="ERR_NVGPUCTRPERM",
        )
        state["terminal_decision"]["sanitizer"] = {
            "status": "unavailable",
            "coverage": "unavailable",
        }

        text = self.summarize.render_text(state)

        self.assertIn("WARNING: profiler coverage degraded: ERR_NVGPUCTRPERM", text)
        self.assertIn("WARNING: sanitizer coverage: unavailable", text)

    def test_contradictory_win_is_fail_safe_inconclusive(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["terminal_decision"]["workload_statistics"] = _statistics(
            "inconclusive"
        )

        text = self.summarize.render_text(state)

        self.assertIn("# Result: inconclusive", text)
        self.assertIn("WARNING: contradictory terminal evidence", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_empty_workload_snapshot_cannot_support_end_to_end_win(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["workload"] = {}

        text = self.summarize.render_text(state)

        self.assertIn("# Result: inconclusive", text)
        self.assertIn("workload is malformed", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_status_only_statistics_cannot_support_a_win(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["terminal_decision"]["statistics"] = {"status": "confirmed_win"}

        text = self.summarize.render_text(state)

        self.assertIn("# Result: inconclusive", text)
        self.assertIn("lacks confirmed kernel and workload statistics", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_conflicting_terminal_sources_are_fail_safe(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["terminal_decision"]["status"] = "confirmed_loss"
        state["terminal_decision"]["statistics"] = _statistics("confirmed_loss")

        text = self.summarize.render_text(state)

        self.assertIn("# Result: confirmed_loss", text)
        self.assertIn("estimate: -6.000%", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_latest_terminal_decision_never_mixes_historical_best_evidence(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["best_kernel_statistics"] = _statistics("confirmed_win", estimate=9.0)
        state["best_workload_statistics"] = _statistics(
            "confirmed_win", estimate=8.0
        )
        state["terminal_decision"].update(
            iteration=2,
            status="confirmed_loss",
            statistics=_statistics("confirmed_loss"),
            workload_statistics=None,
            constraints=[],
        )

        text = self.summarize.render_text(state)

        headline_evidence = text[text.index("## Kernel evidence"):text.index("## Real workload evidence")]
        self.assertIn("estimate: -6.000%", headline_evidence)
        self.assertNotIn("estimate: 9.000%", headline_evidence)
        self.assertIn("## Historical best evidence", text)
        self.assertIn("estimate: 9.000%", text)

    def test_later_no_win_history_invalidates_an_older_terminal_win(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["history"].append(
            {
                "event": "decision_record",
                "iter": 2,
                "status": "no_confirmed_kernel_win",
                "decision_sha256": "1" * 64,
            }
        )

        text = self.summarize.render_text(state)

        self.assertIn("# Result: inconclusive", text)
        self.assertIn("newer decision", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_newer_checkpoint_candidate_state_invalidates_an_older_win(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["terminal_decision"]["resume"].update(
            iteration=2,
            candidate_status="inconclusive",
            candidate_id=None,
            input_hash=state["input_hash"],
        )

        text = self.summarize.render_text(state)

        self.assertIn("# Result: inconclusive", text)
        self.assertIn("newer checkpoint", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_missing_evidence_is_explicit_and_never_fabricated(self) -> None:
        state = {
            "schema_version": 2,
            "run_dir": "/tmp/run",
            "input_hash": "x",
            "mode": "full",
            "terminal_result": "inconclusive",
            "budget": {},
            "workload": {"kind": "command"},
            "env": {},
            "history": [],
            "frontier": [],
        }

        text = self.summarize.render_text(state)

        self.assertIn("kernel statistics: not recorded", text)
        self.assertIn("correctness: not recorded", text)
        self.assertIn("SASS: not recorded", text)
        self.assertIn("primary KPI: not recorded", text)
        self.assertIn("profiler coverage: not recorded", text)
        self.assertIn("sanitizer coverage: not recorded", text)
        self.assertIn("compiler coverage: not recorded", text)
        self.assertNotIn("correctness: passed", text)

    def test_malformed_types_and_hostile_text_are_escaped_not_executed(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["history"] = "not-a-list"
        state["best_kernel_statistics"] = "forged"
        state["terminal_decision"]["statistics"] = "forged"
        state["run_dir"] = "/tmp/<script>alert(1)</script>\n# forged"
        state["best_file"] = "x`\n# injected <img src=x>"
        state["dims"] = {"bad": "</code><script>x</script>"}

        text = self.summarize.render_text(state)

        self.assertIn("# Result: inconclusive", text)
        self.assertNotIn("<script>", text)
        self.assertNotIn("<img", text)
        self.assertNotIn("\n# injected", text)
        self.assertNotIn("\n# forged", text)
        self.assertIn("kernel statistics: not recorded", text)

    def test_untrusted_text_cannot_create_commonmark_links_or_images(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["baseline_file_original"] = "![track](https://evil.invalid/pixel)"
        state["best_file"] = "[click](https://evil.invalid/)"

        text = self.summarize.render_text(state)

        self.assertNotIn("![track](https://evil.invalid/pixel)", text)
        self.assertNotIn("[click](https://evil.invalid/)", text)
        self.assertIn(r"\!\[track\]\(https://evil.invalid/pixel\)", text)

    def test_raw_artifact_links_encode_markdown_delimiters(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["terminal_decision"]["kernel_paired_samples"]["path"] = (
            "/tmp/run/a) [forged](https://example.invalid)/paired_samples.jsonl"
        )

        text = self.summarize.render_text(state)

        self.assertIn("[kernel paired_samples.jsonl]", text)
        self.assertIn("a%29%20%5Bforged%5D%28https%3A", text)
        self.assertNotIn("[forged](https://example.invalid)", text)

    def test_missing_raw_sample_paths_are_never_guessed(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        del state["raw_artifacts"]
        state["terminal_decision"].pop("kernel_paired_samples")
        state["terminal_decision"].pop("workload_paired_samples")

        text = self.summarize.render_text(state)

        self.assertIn("kernel paired_samples.jsonl: not recorded", text)
        self.assertIn("workload paired_samples.jsonl: not recorded", text)
        self.assertNotIn("conventional location", text)

    def test_real_constraint_evidence_includes_status_and_ci(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["terminal_decision"]["constraints"] = [
            {
                "name": "memory_mb",
                "status": "passed",
                "estimate_pct": 0.5,
                "ci_low_pct": 0.1,
                "ci_high_pct": 0.9,
                "max_regression_pct": 2.0,
            }
        ]

        text = self.summarize.render_text(state)

        self.assertIn("constraint result: memory_mb: passed", text)
        self.assertIn("estimate 0.500%, CI [0.100%, 0.900%]", text)

    def test_persisted_candidate_map_feeds_rejected_and_inconclusive_counts(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state.pop("rejected_candidates")
        state["candidates"] = {
            "iter-1": {
                "candidate_file": "/tmp/cuda-run/iterv1/kernel.py",
                "status": "end_to_end_win",
            },
            "iter-2": {
                "candidate_file": "/tmp/cuda-run/iterv2/kernel.py",
                "status": "rejected_correctness",
            },
            "iter-3": {
                "candidate_file": "/tmp/cuda-run/iterv3/kernel.py",
                "status": "inconclusive",
            },
        }

        text = self.summarize.render_text(state)

        self.assertIn("- rejected: 1", text)
        self.assertIn("/tmp/cuda-run/iterv2/kernel.py: rejected_correctness", text)
        self.assertIn("- inconclusive: 1", text)
        self.assertIn("/tmp/cuda-run/iterv3/kernel.py: inconclusive", text)

    def test_render_text_is_pure_and_render_delegates_to_it(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        with mock.patch("builtins.open", side_effect=AssertionError("unexpected I/O")):
            text = self.summarize.render_text(state)
        self.assertIn("# Result: end_to_end_win", text)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            state_path = root / "state.json"
            output_path = root / "summary.md"
            io_state = _minimal_state(root)
            state_path.write_text(json.dumps(io_state), encoding="utf-8")
            with mock.patch.object(
                self.summarize, "render_text", return_value="delegated\n"
            ) as render_text:
                self.summarize.render(str(state_path), str(output_path))

            render_text.assert_called_once_with(io_state)
            self.assertEqual(output_path.read_text("utf-8"), "delegated\n")

    def test_render_rejects_win_when_raw_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            state_path = root / "state.json"
            output_path = root / "summary.md"
            state_path.write_text(
                json.dumps(self.full_win_state), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "decision_json|paired_samples|artifact|file"
            ):
                self.summarize.render(str(state_path), str(output_path))

            self.assertFalse(output_path.exists())

    def test_render_rejects_state_and_output_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            real_state = root / "real-state.json"
            real_state.write_text(json.dumps(_minimal_state(root)), encoding="utf-8")
            linked_state = root / "state.json"
            output = root / "summary.md"
            target = root / "target.md"
            target.write_text("keep me", encoding="utf-8")
            try:
                linked_state.symlink_to(real_state)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(ValueError, "state.*symlink"):
                self.summarize.render(str(linked_state), str(output))

            linked_state.unlink()
            linked_state.write_text(real_state.read_text("utf-8"), encoding="utf-8")
            output.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "output.*symlink"):
                self.summarize.render(str(linked_state), str(output))
            self.assertEqual(target.read_text("utf-8"), "keep me")

    def test_render_rejects_symlink_in_any_parent_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            real = root / "real"
            real.mkdir()
            state = real / "state.json"
            state.write_text(json.dumps(_minimal_state(real)), encoding="utf-8")
            linked = root / "linked"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(ValueError, "parent.*symlink"):
                self.summarize.render(
                    str(linked / "state.json"), str(root / "summary.md")
                )
            with self.assertRaisesRegex(ValueError, "parent.*symlink"):
                self.summarize.render(
                    str(state), str(linked / "summary.md")
                )

    def test_render_atomically_replaces_and_fsyncs_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            state_path = root / "state.json"
            output_path = root / "summary.md"
            state_path.write_text(
                json.dumps(_minimal_state(root)), encoding="utf-8"
            )
            real_replace = os.replace
            real_fsync = os.fsync
            with mock.patch.object(
                self.summarize.os, "replace", wraps=real_replace
            ) as replace, mock.patch.object(
                self.summarize.os, "fsync", wraps=real_fsync
            ) as fsync, contextlib.redirect_stdout(io.StringIO()):
                self.summarize.render(str(state_path), str(output_path))

            replace.assert_called_once()
            self.assertGreaterEqual(fsync.call_count, 2)
            self.assertIn("# Result: inconclusive", output_path.read_text("utf-8"))
            self.assertEqual(list(root.glob(".summary.md.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
