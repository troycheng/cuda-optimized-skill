from __future__ import annotations

import copy
import contextlib
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


def _load_summarize():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_summarize_test", SUMMARIZE_PATH
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
    workload["statistic"] = "median_paired_workload_improvement_pct"
    return {
        "schema_version": 2,
        "run_dir": "/tmp/cuda-run",
        "input_hash": "a" * 64,
        "mode": "full",
        "terminal_result": "end_to_end_win",
        "budget": {
            "preset": "balanced",
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
        state.update(mode="kernel-only", workload=None, terminal_result="kernel_only_win")
        state["best_workload_statistics"] = None
        state["history"][-1]["status"] = "kernel_only_win"

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
        state["sanitizer_coverage"] = "unavailable"

        text = self.summarize.render_text(state)

        self.assertIn("WARNING: profiler coverage degraded: ERR_NVGPUCTRPERM", text)
        self.assertIn("WARNING: sanitizer coverage: unavailable", text)

    def test_contradictory_win_is_fail_safe_inconclusive(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["best_workload_statistics"] = _statistics("inconclusive")

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
        state["best_kernel_statistics"] = {"status": "confirmed_win"}
        state["history"][-1]["statistics"] = {"status": "confirmed_win"}

        text = self.summarize.render_text(state)

        self.assertIn("# Result: inconclusive", text)
        self.assertIn("lacks confirmed kernel and workload statistics", text)
        self.assertNotIn("# Result: end_to_end_win", text)

    def test_conflicting_terminal_sources_are_fail_safe(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["history"][-1]["status"] = "confirmed_loss"

        text = self.summarize.render_text(state)

        self.assertIn("# Result: inconclusive", text)
        self.assertIn("terminal sources disagree", text)

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
        state["terminal_result"] = ["end_to_end_win"]
        state["history"] = "not-a-list"
        state["best_kernel_statistics"] = "forged"
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

    def test_raw_artifact_links_encode_markdown_delimiters(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["raw_artifacts"]["kernel_paired_samples"] = (
            "/tmp/run/a) [forged](https://example.invalid)/paired_samples.jsonl"
        )

        text = self.summarize.render_text(state)

        self.assertIn("[kernel paired_samples.jsonl]", text)
        self.assertIn("a%29%20%5Bforged%5D%28https%3A", text)
        self.assertNotIn("[forged](https://example.invalid)", text)

    def test_conventional_raw_sample_links_are_transparently_unverified(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        del state["raw_artifacts"]

        text = self.summarize.render_text(state)

        self.assertIn(
            "[kernel paired_samples.jsonl](</tmp/cuda-run/iterv1/paired_samples.jsonl>)",
            text,
        )
        self.assertIn("conventional location; existence not verified", text)

    def test_real_constraint_evidence_includes_status_and_ci(self) -> None:
        state = copy.deepcopy(self.full_win_state)
        state["terminal_decision"] = {
            "status": "end_to_end_win",
            "statistics": copy.deepcopy(state["best_kernel_statistics"]),
            "workload_statistics": copy.deepcopy(
                state["best_workload_statistics"]
            ),
            "constraints": [
                {
                    "name": "memory_mb",
                    "status": "passed",
                    "estimate_pct": 0.5,
                    "ci_low_pct": 0.1,
                    "ci_high_pct": 0.9,
                    "max_regression_pct": 2.0,
                }
            ],
        }

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
            root = Path(tmp)
            state_path = root / "state.json"
            output_path = root / "summary.md"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with mock.patch.object(
                self.summarize, "render_text", return_value="delegated\n"
            ) as render_text:
                self.summarize.render(str(state_path), str(output_path))

            render_text.assert_called_once_with(state)
            self.assertEqual(output_path.read_text("utf-8"), "delegated\n")

    def test_render_rejects_state_and_output_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_state = root / "real-state.json"
            real_state.write_text(json.dumps(self.full_win_state), encoding="utf-8")
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

    def test_render_atomically_replaces_and_fsyncs_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state.json"
            output_path = root / "summary.md"
            state_path.write_text(
                json.dumps(self.full_win_state), encoding="utf-8"
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
            self.assertIn("# Result: end_to_end_win", output_path.read_text("utf-8"))
            self.assertEqual(list(root.glob(".summary.md.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
