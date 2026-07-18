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

import direction_guard  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def portfolio() -> dict:
    return {
        "schema_version": 1,
        "objective": {
            "claim_layer": "kernel",
            "metric_name": "latency",
            "metric_unit": "us",
            "metric_direction": "lower",
            "metric_kind": "additive_time",
            "minimum_effect_absolute": 2.0,
            "minimum_effect_percent": 1.0,
        },
        "environment_sha256": SHA_A,
        "measurement_window_sha256": SHA_B,
        "directions": [
            {
                "id": "selector",
                "claim_layer": "kernel",
                "bottleneck_class": "kernel",
                "target_identity_sha256": SHA_C,
                "component_id": "selector",
                "metric_name": "latency",
                "metric_unit": "us",
                "metric_direction": "lower",
                "metric_kind": "additive_time",
                "total_metric": 500.0,
                "component_metric": 27.5,
                "evidence_sha256": SHA_D,
            },
            {
                "id": "gather",
                "claim_layer": "kernel",
                "bottleneck_class": "framework",
                "target_identity_sha256": SHA_C,
                "component_id": "gather",
                "metric_name": "latency",
                "metric_unit": "us",
                "metric_direction": "lower",
                "metric_kind": "additive_time",
                "total_metric": 500.0,
                "component_metric": 90.0,
                "evidence_sha256": SHA_E,
            },
        ],
    }


class DirectionModelTests(unittest.TestCase):
    def test_freezes_closed_identity_without_mechanism_text(self) -> None:
        frozen = direction_guard.freeze_lineage(portfolio())
        self.assertEqual(frozen["schema_version"], 1)
        self.assertEqual(len(frozen["direction_families"]), 2)
        self.assertNotIn("mechanism", json.dumps(frozen))
        renamed = portfolio()
        renamed["directions"][0]["mechanism"] = "new eight-way launch"
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            direction_guard.freeze_lineage(renamed)

    def test_rejects_duplicate_keys_boolean_numbers_and_unknown_taxonomy(self) -> None:
        duplicate = b'{"schema_version":1,"schema_version":1}'
        with self.assertRaisesRegex(ValueError, "duplicate"):
            direction_guard.load_json_bytes(duplicate, "portfolio")
        bad = portfolio()
        bad["directions"][0]["component_metric"] = True
        with self.assertRaisesRegex(ValueError, "finite number"):
            direction_guard.freeze_lineage(bad)
        bad = portfolio()
        bad["directions"][0]["bottleneck_class"] = "magic"
        with self.assertRaisesRegex(ValueError, "bottleneck_class"):
            direction_guard.freeze_lineage(bad)

    def test_uses_full_elimination_upper_bound_and_switches_to_larger_direction(self) -> None:
        snapshot = portfolio()
        lineage = direction_guard.freeze_lineage(snapshot)
        decision = direction_guard.decide_direction(snapshot, lineage, "selector")
        self.assertEqual(decision["upper_bound_absolute"], 27.5)
        self.assertEqual(decision["upper_bound_percent"], 5.5)
        self.assertEqual(decision["action"], "switch_to_higher_impact")
        self.assertEqual(decision["recommended_direction_id"], "gather")
        self.assertFalse(decision["performance_gain_claimed"])

    def test_closes_direction_when_even_full_elimination_misses_frozen_floor(self) -> None:
        snapshot = portfolio()
        snapshot["objective"]["minimum_effect_absolute"] = 30.0
        snapshot["objective"]["minimum_effect_percent"] = 6.0
        lineage = direction_guard.freeze_lineage(snapshot)
        decision = direction_guard.decide_direction(snapshot, lineage, "selector")
        self.assertEqual(decision["action"], "close_direction")
        self.assertEqual(decision["state"], "closed")

    def test_cross_layer_throughput_and_composite_metrics_are_unrankable(self) -> None:
        for mutation in ("cross_layer", "throughput", "composite"):
            snapshot = portfolio()
            if mutation == "cross_layer":
                snapshot["directions"][0]["claim_layer"] = "serving"
            else:
                snapshot["objective"]["metric_kind"] = mutation
                snapshot["directions"][0]["metric_kind"] = mutation
                snapshot["directions"][1]["metric_kind"] = mutation
            lineage = direction_guard.freeze_lineage(snapshot)
            decision = direction_guard.decide_direction(snapshot, lineage, "selector")
            self.assertEqual(decision["action"], "unrankable", mutation)
            self.assertIsNone(decision["upper_bound_absolute"])
            self.assertIsNone(decision["upper_bound_percent"])

    def test_closed_family_requires_materially_new_reopen_evidence(self) -> None:
        snapshot = portfolio()
        lineage = direction_guard.freeze_lineage(snapshot)
        closed = direction_guard.decide_direction(
            snapshot, lineage, "selector", request="close"
        )
        blocked = direction_guard.decide_direction(
            snapshot, lineage, "selector", previous=closed
        )
        self.assertEqual(blocked["action"], "direction_closed")
        with self.assertRaisesRegex(ValueError, "new evidence"):
            direction_guard.decide_direction(
                snapshot, lineage, "selector", previous=closed, request="reopen"
            )
        changed = copy.deepcopy(snapshot)
        changed["measurement_window_sha256"] = SHA_E
        changed["directions"][0]["evidence_sha256"] = SHA_A
        reopened = direction_guard.decide_direction(
            changed, lineage, "selector", previous=closed, request="reopen"
        )
        self.assertEqual(reopened["transition"], "reopen")
        self.assertNotEqual(reopened["action"], "direction_closed")


class DirectionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.portfolio = self.root / "portfolio.json"
        self.portfolio.write_text(json.dumps(portfolio()), encoding="utf-8")
        self.run_dir = self.root / "run"
        self.cli = SCRIPTS / "direction_guard.py"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.cli), *args],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_creates_canonical_hash_chained_ledger_and_status(self) -> None:
        init = self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        )
        self.assertEqual(init.returncode, 0, init.stderr)
        first = self.run_cli(
            "check",
            "--portfolio",
            str(self.portfolio),
            "--run-dir",
            str(self.run_dir),
            "--direction-id",
            "selector",
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.run_cli(
            "check",
            "--portfolio",
            str(self.portfolio),
            "--run-dir",
            str(self.run_dir),
            "--direction-id",
            "gather",
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        decisions = sorted((self.run_dir / "direction-decisions").glob("*.json"))
        self.assertEqual([path.name for path in decisions], [
            "decision-0001.json", "decision-0002.json"
        ])
        first_payload = json.loads(decisions[0].read_text())
        second_payload = json.loads(decisions[1].read_text())
        expected = hashlib.sha256(decisions[0].read_bytes()).hexdigest()
        self.assertEqual(second_payload["previous_decision_sha256"], expected)
        self.assertEqual(first_payload["decision_index"], 1)
        status = self.run_cli("status", "--run-dir", str(self.run_dir))
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)["decision_count"], 2)

    def test_cli_rejects_second_init_chain_gaps_and_symlinked_ledger(self) -> None:
        self.assertEqual(self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        ).returncode, 0)
        self.assertEqual(self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        ).returncode, 2)
        decisions = self.run_dir / "direction-decisions"
        decisions.mkdir()
        (decisions / "decision-0002.json").write_text("{}", encoding="utf-8")
        failed = self.run_cli("status", "--run-dir", str(self.run_dir))
        self.assertEqual(failed.returncode, 2)
        (decisions / "decision-0002.json").unlink()
        target = self.root / "outside"
        target.mkdir()
        decisions.rmdir()
        decisions.symlink_to(target, target_is_directory=True)
        failed = self.run_cli(
            "check",
            "--portfolio",
            str(self.portfolio),
            "--run-dir",
            str(self.run_dir),
            "--direction-id",
            "selector",
        )
        self.assertEqual(failed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
