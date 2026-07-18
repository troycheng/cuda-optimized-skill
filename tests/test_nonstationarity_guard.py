from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nonstationarity_guard  # noqa: E402


SHA_A = "a" * 64


def digest(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def design() -> dict:
    return {
        "schema_version": 1,
        "metric": {"name": "latency", "unit": "ms", "direction": "lower"},
        "measurement": {
            "mode": "time_windows",
            "minimum_duration_ms": 900.0,
            "maximum_duration_ms": 1100.0,
            "burn_in_rows_per_segment": 1,
            "minimum_complete_blocks": 4,
        },
        "assignment_method": "site_randomized_balanced",
        "blocks": [
            {"block_id": "b1", "order": "AB"},
            {"block_id": "b2", "order": "BA"},
            {"block_id": "b3", "order": "BA"},
            {"block_id": "b4", "order": "AB"},
        ],
        "state_dimensions": [
            {
                "name": "queue_depth",
                "unit": "requests",
                "epsilon": 1.0,
                "pair_max_absolute": 2.0,
                "pair_max_percent": 20.0,
                "phase_max_absolute": 2.0,
                "phase_max_percent": 20.0,
            }
        ],
    }


def series(frozen: dict | None = None) -> dict:
    frozen = frozen or design()
    rows = []
    sequence = 0
    for block in frozen["blocks"]:
        roles = ("baseline", "candidate") if block["order"] == "AB" else ("candidate", "baseline")
        for segment_index, role in enumerate(roles):
            for _ in range(frozen["measurement"]["burn_in_rows_per_segment"]):
                rows.append({
                    "sequence_index": sequence,
                    "block_id": block["block_id"],
                    "segment_index": segment_index,
                    "role": role,
                    "phase": "burn_in",
                    "duration_ms": 1000.0,
                    "metric_value": None,
                    "usable": True,
                    "states": {"queue_depth": 10.0},
                })
                sequence += 1
            rows.append({
                "sequence_index": sequence,
                "block_id": block["block_id"],
                "segment_index": segment_index,
                "role": role,
                "phase": "timed",
                "duration_ms": 1000.0,
                "metric_value": 100.0 if role == "baseline" else 95.0,
                "usable": True,
                "states": {"queue_depth": 10.0},
            })
            sequence += 1
    return {
        "schema_version": 1,
        "design_sha256": digest(frozen),
        "source_artifact": {"path": "serving-series.jsonl", "sha256": SHA_A},
        "metric": copy.deepcopy(frozen["metric"]),
        "measurement_mode": frozen["measurement"]["mode"],
        "observations": rows,
    }


class NonstationarityModelTests(unittest.TestCase):
    def test_accepts_balanced_predeclared_blocks_with_comparable_state(self) -> None:
        frozen = design()
        verdict = nonstationarity_guard.evaluate(
            frozen, series(frozen), expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
        )
        self.assertEqual(verdict["status"], "comparable_paired_state")
        self.assertEqual(verdict["complete_blocks"], 4)
        self.assertEqual(verdict["reasons"], [])
        self.assertFalse(verdict["performance_gain_claimed"])

    def test_pair_state_mismatch_is_inconclusive_not_a_speedup(self) -> None:
        frozen = design()
        observed = series(frozen)
        candidate = next(
            row for row in observed["observations"]
            if row["block_id"] == "b1" and row["role"] == "candidate" and row["phase"] == "timed"
        )
        candidate["states"]["queue_depth"] = 20.0
        verdict = nonstationarity_guard.evaluate(
            frozen, observed, expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
        )
        self.assertEqual(verdict["status"], "inconclusive_nonstationary")
        self.assertIn("state_pair_mismatch", verdict["reasons"])
        self.assertFalse(verdict["performance_gain_claimed"])

    def test_burn_in_to_timed_step_is_inconclusive(self) -> None:
        frozen = design()
        observed = series(frozen)
        burn_in = next(
            row for row in observed["observations"]
            if row["block_id"] == "b1" and row["role"] == "candidate" and row["phase"] == "burn_in"
        )
        timed = next(
            row for row in observed["observations"]
            if row["block_id"] == "b1" and row["role"] == "candidate" and row["phase"] == "timed"
        )
        burn_in["states"]["queue_depth"] = 20.0
        timed["states"]["queue_depth"] = 10.0
        verdict = nonstationarity_guard.evaluate(
            frozen, observed, expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
        )
        self.assertEqual(verdict["status"], "inconclusive_nonstationary")
        self.assertIn("phase_shift", verdict["reasons"])

    def test_count_windows_and_duration_drift_are_inconclusive(self) -> None:
        frozen = design()
        observed = series(frozen)
        observed["observations"][0]["duration_ms"] = 1500.0
        verdict = nonstationarity_guard.evaluate(
            frozen, observed, expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
        )
        self.assertIn("duration_out_of_bounds", verdict["reasons"])
        count_design = design()
        count_design["measurement"]["mode"] = "count_windows"
        count_series = series(count_design)
        verdict = nonstationarity_guard.evaluate(
            count_design, count_series, expected_design_sha256=digest(count_design), anchor_sha256=SHA_A
        )
        self.assertEqual(verdict["status"], "inconclusive_nonstationary")
        self.assertIn("unsupported_measurement_mode", verdict["reasons"])

    def test_rejects_unbalanced_assignment_reordering_and_metric_drift(self) -> None:
        unbalanced = design()
        for block in unbalanced["blocks"]:
            block["order"] = "AB"
        with self.assertRaisesRegex(ValueError, "balanced"):
            nonstationarity_guard.evaluate(
                unbalanced, series(unbalanced), expected_design_sha256=digest(unbalanced), anchor_sha256=SHA_A
            )
        frozen = design()
        reordered = series(frozen)
        reordered["observations"][0], reordered["observations"][1] = (
            reordered["observations"][1], reordered["observations"][0]
        )
        with self.assertRaisesRegex(ValueError, "chronological plan"):
            nonstationarity_guard.evaluate(
                frozen, reordered, expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
            )
        drifted = series(frozen)
        drifted["metric"]["name"] = "qps"
        with self.assertRaisesRegex(ValueError, "metric drift"):
            nonstationarity_guard.evaluate(
                frozen, drifted, expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
            )

    def test_posthoc_row_deletion_and_undeclared_state_fail_input_validation(self) -> None:
        frozen = design()
        deleted = series(frozen)
        deleted["observations"].pop(3)
        with self.assertRaisesRegex(ValueError, "chronological plan"):
            nonstationarity_guard.evaluate(
                frozen, deleted, expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
            )
        extra_state = series(frozen)
        extra_state["observations"][0]["states"]["cache_hit_rate"] = 0.9
        with self.assertRaisesRegex(ValueError, "state dimensions"):
            nonstationarity_guard.evaluate(
                frozen, extra_state, expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
            )

    def test_unusable_rows_cannot_satisfy_the_minimum_complete_blocks(self) -> None:
        frozen = design()
        observed = series(frozen)
        observed["observations"][0]["usable"] = False
        verdict = nonstationarity_guard.evaluate(
            frozen, observed, expected_design_sha256=digest(frozen), anchor_sha256=SHA_A
        )
        self.assertEqual(verdict["complete_blocks"], 3)
        self.assertIn("unusable_observation", verdict["reasons"])
        self.assertIn("insufficient_complete_blocks", verdict["reasons"])


class NonstationarityCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.design_payload = design()
        raw = b"chronological serving source\n"
        (self.root / "serving-series.jsonl").write_bytes(raw)
        self.series_payload = series(self.design_payload)
        self.series_payload["source_artifact"]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.design_path = self.root / "design.json"
        self.series_path = self.root / "series.json"
        self.design_path.write_text(json.dumps(self.design_payload), encoding="utf-8")
        self.series_path.write_text(json.dumps(self.series_payload), encoding="utf-8")
        self.anchor_path = self.root / "nonstationarity-anchor.json"
        self.cli = SCRIPTS / "nonstationarity_guard.py"
        initialized = subprocess.run(
            [
                sys.executable,
                str(self.cli),
                "init",
                "--design",
                str(self.design_path),
                "--anchor",
                str(self.anchor_path),
            ],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(initialized.returncode, 0, initialized.stderr)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_cli(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.cli), "check", "--anchor", str(self.anchor_path), "--series", str(self.series_path)],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_rehashes_source_and_emits_read_only_verdict(self) -> None:
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "comparable_paired_state")
        self.assertEqual(payload["anchor_sha256"], digest(json.loads(self.anchor_path.read_text())))
        self.assertEqual(payload["source_sha256"], self.series_payload["source_artifact"]["sha256"])
        self.assertFalse(payload["performance_gain_claimed"])

    def test_cli_rejects_raw_source_drift_and_symlinks(self) -> None:
        (self.root / "serving-series.jsonl").write_bytes(b"changed\n")
        drifted = self.run_cli()
        self.assertEqual(drifted.returncode, 2)
        self.assertIn("source artifact digest", drifted.stderr)
        (self.root / "serving-series.jsonl").unlink()
        outside = self.root / "outside.jsonl"
        outside.write_bytes(b"chronological serving source\n")
        (self.root / "serving-series.jsonl").symlink_to(outside)
        linked = self.run_cli()
        self.assertEqual(linked.returncode, 2)
        self.assertIn("a symlink, or unsafe", linked.stderr)

    def test_cli_rejects_duplicate_json_keys_and_unsafe_source_paths(self) -> None:
        self.design_path.write_text(
            '{"schema_version":1,"schema_version":1}', encoding="utf-8"
        )
        duplicate = self.run_cli()
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("duplicate JSON key", duplicate.stderr)

        self.design_path.write_text(json.dumps(self.design_payload), encoding="utf-8")
        self.series_payload["source_artifact"]["path"] = "../serving-series.jsonl"
        self.series_path.write_text(json.dumps(self.series_payload), encoding="utf-8")
        unsafe = self.run_cli()
        self.assertEqual(unsafe.returncode, 2)
        self.assertIn("safe relative artifact path", unsafe.stderr)

    def test_create_once_anchor_blocks_posthoc_tolerance_relaxation(self) -> None:
        repeated = subprocess.run(
            [sys.executable, str(self.cli), "init", "--design", str(self.design_path), "--anchor", str(self.anchor_path)],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("already exists", repeated.stderr)

        relaxed = copy.deepcopy(self.design_payload)
        for dimension in relaxed["state_dimensions"]:
            for key in ("pair_max_absolute", "pair_max_percent", "phase_max_absolute", "phase_max_percent"):
                dimension[key] = 1000.0
        self.design_path.write_text(json.dumps(relaxed), encoding="utf-8")
        self.series_payload["design_sha256"] = digest(relaxed)
        self.series_path.write_text(json.dumps(self.series_payload), encoding="utf-8")
        posthoc = self.run_cli()
        self.assertEqual(posthoc.returncode, 2)
        self.assertIn("frozen design artifact digest", posthoc.stderr)


if __name__ == "__main__":
    unittest.main()
