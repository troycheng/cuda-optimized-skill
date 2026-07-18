from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import direction_guard  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


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
        "environment_artifact": {"path": "environment.json", "sha256": SHA_A},
        "measurement_window_artifact": {"path": "window.json", "sha256": SHA_B},
        "directions": [
            {
                "id": "selector",
                "claim_layer": "kernel",
                "bottleneck_class": "kernel",
                "target_artifact": {"path": "target.json", "sha256": SHA_C},
                "component_artifact": {"path": "selector-component.json", "sha256": SHA_D},
                "source_artifact": {"path": "selector-profile.json", "sha256": SHA_A},
                "component_id": "selector",
                "metric_name": "latency",
                "metric_unit": "us",
                "metric_direction": "lower",
                "metric_kind": "additive_time",
                "total_metric": 500.0,
                "component_metric": 27.5,
                "evidence_artifact": {"path": "selector-evidence.json", "sha256": SHA_D},
            },
            {
                "id": "gather",
                "claim_layer": "kernel",
                "bottleneck_class": "framework",
                "target_artifact": {"path": "target.json", "sha256": SHA_C},
                "component_artifact": {"path": "gather-component.json", "sha256": SHA_E},
                "source_artifact": {"path": "gather-profile.json", "sha256": SHA_B},
                "component_id": "gather",
                "metric_name": "latency",
                "metric_unit": "us",
                "metric_direction": "lower",
                "metric_kind": "additive_time",
                "total_metric": 500.0,
                "component_metric": 90.0,
                "evidence_artifact": {"path": "gather-evidence.json", "sha256": SHA_E},
            },
        ],
    }


class DirectionModelTests(unittest.TestCase):
    def test_freezes_closed_identity_without_mechanism_text(self) -> None:
        frozen = direction_guard.freeze_lineage(portfolio())
        self.assertEqual(frozen["schema_version"], 1)
        self.assertEqual(len(frozen["direction_families"]), 2)
        self.assertNotIn("mechanism", json.dumps(frozen))
        renamed_id = portfolio()
        renamed_id["directions"][0]["component_id"] = "selector-renamed"
        self.assertEqual(
            frozen["direction_families"][0]["direction_family_key"],
            direction_guard.freeze_lineage(renamed_id)["direction_families"][0]["direction_family_key"],
        )
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
        bad = portfolio()
        bad["directions"][1]["total_metric"] = 501.0
        with self.assertRaisesRegex(ValueError, "same total_metric"):
            direction_guard.freeze_lineage(bad)

    def test_uses_full_elimination_upper_bound_and_switches_to_larger_direction(self) -> None:
        snapshot = portfolio()
        lineage = direction_guard.freeze_lineage(snapshot)
        decision = direction_guard.decide_direction(snapshot, lineage, "selector")
        self.assertEqual(decision["upper_bound_absolute"], 27.5)
        self.assertEqual(decision["upper_bound_percent"], 5.5)
        self.assertEqual(decision["action"], "switch_to_higher_impact")
        self.assertEqual(decision["recommended_direction_id"], "gather")
        self.assertFalse(decision["admitted"])
        self.assertFalse(decision["performance_gain_claimed"])

    def test_rejects_a_snapshot_that_removes_a_frozen_direction_family(self) -> None:
        snapshot = portfolio()
        lineage = direction_guard.freeze_lineage(snapshot)
        snapshot["directions"] = snapshot["directions"][:1]
        with self.assertRaisesRegex(ValueError, "preserve frozen direction family set"):
            direction_guard.decide_direction(snapshot, lineage, "selector")

    def test_closed_families_are_not_recommended_as_leaders(self) -> None:
        snapshot = portfolio()
        lineage = direction_guard.freeze_lineage(snapshot)
        gather = direction_guard._validate_portfolio(snapshot)["directions"][1]
        closed = direction_guard.decide_direction(
            snapshot, lineage, "gather", request="close"
        )
        decision = direction_guard.decide_direction(
            snapshot,
            lineage,
            "selector",
            latest_by_family={gather["direction_family_key"]: closed},
        )
        self.assertEqual(decision["action"], "admit_direction")
        self.assertEqual(decision["recommended_direction_id"], "selector")

    def test_equal_ceiling_does_not_switch_by_mutable_display_id(self) -> None:
        snapshot = portfolio()
        snapshot["directions"][0]["component_metric"] = 90.0
        lineage = direction_guard.freeze_lineage(snapshot)
        original = direction_guard.decide_direction(snapshot, lineage, "selector")
        renamed = copy.deepcopy(snapshot)
        renamed["directions"][0]["id"] = "aaaa"
        changed = direction_guard.decide_direction(renamed, lineage, "aaaa")
        self.assertEqual(original["action"], "admit_direction")
        self.assertEqual(changed["action"], "admit_direction")

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
            self.assertFalse(decision["admitted"])
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
        stale_source = copy.deepcopy(snapshot)
        stale_source["measurement_window_artifact"]["sha256"] = SHA_E
        stale_source["directions"][0]["evidence_artifact"]["sha256"] = SHA_A
        stale_source["directions"][0]["component_metric"] = 33.0
        with self.assertRaisesRegex(ValueError, "new raw source"):
            direction_guard.decide_direction(
                stale_source,
                lineage,
                "selector",
                previous=closed,
                family_history=[closed],
                closed_decision_sha256=SHA_C,
                request="reopen",
            )
        changed = copy.deepcopy(stale_source)
        changed["directions"][0]["source_artifact"]["sha256"] = SHA_F
        reopened = direction_guard.decide_direction(
            changed,
            lineage,
            "selector",
            previous=closed,
            family_history=[closed],
            closed_decision_sha256=SHA_C,
            request="reopen",
        )
        self.assertEqual(reopened["transition"], "reopen")
        self.assertEqual(reopened["reopen_reason"], "new_measurement_window")
        self.assertEqual(reopened["closed_decision_sha256"], SHA_C)
        self.assertEqual(reopened["source_sha256"], SHA_F)
        self.assertNotEqual(reopened["action"], "direction_closed")

    def test_reopen_cannot_reuse_any_evidence_from_the_family_history(self) -> None:
        snapshot = portfolio()
        lineage = direction_guard.freeze_lineage(snapshot)
        first_closed = direction_guard.decide_direction(
            snapshot, lineage, "selector", request="close"
        )
        changed = copy.deepcopy(snapshot)
        changed["measurement_window_artifact"]["sha256"] = SHA_E
        changed["directions"][0]["evidence_artifact"]["sha256"] = SHA_A
        changed["directions"][0]["source_artifact"]["sha256"] = SHA_F
        changed["directions"][0]["component_metric"] = 33.0
        reopened = direction_guard.decide_direction(
            changed,
            lineage,
            "selector",
            previous=first_closed,
            family_history=[first_closed],
            closed_decision_sha256=SHA_C,
            request="reopen",
        )
        second_closed = direction_guard.decide_direction(
            changed, lineage, "selector", previous=reopened, request="close"
        )
        replayed = copy.deepcopy(changed)
        replayed["measurement_window_artifact"]["sha256"] = SHA_C
        replayed["directions"][0]["evidence_artifact"]["sha256"] = SHA_D
        replayed["directions"][0]["source_artifact"]["sha256"] = SHA_C
        replayed["directions"][0]["component_metric"] = 40.0
        with self.assertRaisesRegex(ValueError, "used earlier"):
            direction_guard.decide_direction(
                replayed,
                lineage,
                "selector",
                previous=second_closed,
                family_history=[first_closed, reopened, second_closed],
                closed_decision_sha256=SHA_E,
                request="reopen",
            )

    def test_reopen_requires_material_upper_bound_increase_and_frozen_total(self) -> None:
        snapshot = portfolio()
        lineage = direction_guard.freeze_lineage(snapshot)
        closed = direction_guard.decide_direction(snapshot, lineage, "selector", request="close")
        changed = copy.deepcopy(snapshot)
        changed["measurement_window_artifact"]["sha256"] = SHA_E
        changed["directions"][0]["evidence_artifact"]["sha256"] = SHA_A
        changed["directions"][0]["source_artifact"]["sha256"] = SHA_F
        changed["directions"][0]["total_metric"] = 250.0
        changed["directions"][1]["total_metric"] = 250.0
        unchanged = direction_guard.decide_direction(changed, lineage, "selector")
        self.assertEqual(unchanged["upper_bound_percent"], 5.5)
        with self.assertRaisesRegex(ValueError, "material upper-bound increase"):
            direction_guard.decide_direction(
                changed,
                lineage,
                "selector",
                previous=closed,
                family_history=[closed],
                closed_decision_sha256=SHA_C,
                request="reopen",
            )
        changed["directions"][0]["component_metric"] = 33.0
        reopened = direction_guard.decide_direction(
            changed,
            lineage,
            "selector",
            previous=closed,
            family_history=[closed],
            closed_decision_sha256=SHA_C,
            request="reopen",
        )
        self.assertGreater(reopened["upper_bound_absolute"], closed["upper_bound_absolute"])

    def test_mixed_portfolio_ranks_only_the_comparable_subset(self) -> None:
        snapshot = portfolio()
        endpoint = copy.deepcopy(snapshot["directions"][0])
        endpoint.update({
            "id": "endpoint",
            "claim_layer": "serving",
            "component_id": "endpoint",
            "metric_name": "qps",
            "metric_unit": "requests_per_second",
            "metric_direction": "higher",
            "metric_kind": "throughput",
            "component_metric": 400.0,
            "evidence_artifact": {"path": "endpoint.json", "sha256": SHA_A},
        })
        snapshot["directions"].append(endpoint)
        lineage = direction_guard.freeze_lineage(snapshot)
        selector = direction_guard.decide_direction(snapshot, lineage, "selector")
        endpoint_result = direction_guard.decide_direction(snapshot, lineage, "endpoint")
        self.assertEqual(selector["action"], "switch_to_higher_impact")
        self.assertEqual(selector["recommended_direction_id"], "gather")
        self.assertEqual(endpoint_result["action"], "unrankable")
        self.assertFalse(endpoint_result["admitted"])


class DirectionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        payload = portfolio()
        artifacts = {
            "environment.json": b"environment\n",
            "window.json": b"window\n",
            "target.json": b"target\n",
            "selector-component.json": b"selector-component\n",
            "gather-component.json": b"gather-component\n",
            "selector-profile.json": b"selector-profile\n",
            "gather-profile.json": b"gather-profile\n",
        }
        payload["environment_artifact"]["sha256"] = hashlib.sha256(artifacts["environment.json"]).hexdigest()
        payload["measurement_window_artifact"]["sha256"] = hashlib.sha256(artifacts["window.json"]).hexdigest()
        for item in payload["directions"]:
            item["target_artifact"]["sha256"] = hashlib.sha256(artifacts["target.json"]).hexdigest()
            component_raw = artifacts[item["component_artifact"]["path"]]
            item["component_artifact"]["sha256"] = hashlib.sha256(component_raw).hexdigest()
            source_path = item["evidence_artifact"]["path"].replace("evidence", "profile")
            item["source_artifact"] = {
                "path": source_path,
                "sha256": hashlib.sha256(artifacts[source_path]).hexdigest(),
            }
            evidence = {
                "schema_version": 1,
                "source_artifact": item["source_artifact"],
                "component_artifact_sha256": item["component_artifact"]["sha256"],
                "target_artifact_sha256": item["target_artifact"]["sha256"],
                "measurement_window_sha256": payload["measurement_window_artifact"]["sha256"],
                "claim_layer": item["claim_layer"],
                "bottleneck_class": item["bottleneck_class"],
                "metric_name": item["metric_name"],
                "metric_unit": item["metric_unit"],
                "metric_direction": item["metric_direction"],
                "metric_kind": item["metric_kind"],
                "total_metric": item["total_metric"],
                "component_metric": item["component_metric"],
            }
            raw = (json.dumps(evidence, sort_keys=True) + "\n").encode()
            artifacts[item["evidence_artifact"]["path"]] = raw
            item["evidence_artifact"]["sha256"] = hashlib.sha256(raw).hexdigest()
        for name, raw in artifacts.items():
            (self.root / name).write_bytes(raw)
        self.portfolio = self.root / "portfolio.json"
        self.portfolio.write_text(json.dumps(payload), encoding="utf-8")
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
            "--expected-tail-sha256",
            hashlib.sha256((self.run_dir / "direction-decisions" / "decision-0001.json").read_bytes()).hexdigest(),
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
        self.assertFalse(first_payload["admitted"])
        status = self.run_cli("status", "--run-dir", str(self.run_dir))
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)["decision_count"], 2)

    def test_cli_requires_the_callers_last_seen_tail_before_append(self) -> None:
        self.assertEqual(self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        ).returncode, 0)
        first = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector"
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        missing = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "gather"
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("expected tail", missing.stderr)
        status = json.loads(self.run_cli("status", "--run-dir", str(self.run_dir)).stdout)
        self.assertRegex(status["ledger_tail_sha256"], r"^[a-f0-9]{64}$")
        stale = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "gather", "--expected-tail-sha256", SHA_A
        )
        self.assertEqual(stale.returncode, 2)

    def test_cli_reopen_records_reason_and_exact_closed_decision(self) -> None:
        self.assertEqual(self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        ).returncode, 0)
        closed = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector", "--request", "close",
        )
        self.assertEqual(closed.returncode, 0, closed.stderr)
        closed_path = self.run_dir / "direction-decisions" / "decision-0001.json"
        closed_sha = hashlib.sha256(closed_path.read_bytes()).hexdigest()
        (self.root / "window.json").write_bytes(b"new-window\n")
        payload = json.loads(self.portfolio.read_text())
        window_sha = hashlib.sha256((self.root / "window.json").read_bytes()).hexdigest()
        payload["measurement_window_artifact"]["sha256"] = window_sha
        payload["directions"][0]["component_metric"] = 33.0
        for item in payload["directions"]:
            source_path = self.root / item["source_artifact"]["path"]
            source_path.write_bytes(f"new-{item['id']}-profile\n".encode())
            item["source_artifact"]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            evidence_path = self.root / item["evidence_artifact"]["path"]
            evidence = json.loads(evidence_path.read_text())
            evidence["source_artifact"] = item["source_artifact"]
            evidence["measurement_window_sha256"] = window_sha
            evidence["component_metric"] = item["component_metric"]
            raw = (json.dumps(evidence, sort_keys=True) + "\n").encode()
            evidence_path.write_bytes(raw)
            item["evidence_artifact"]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.portfolio.write_text(json.dumps(payload), encoding="utf-8")
        reopened = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector", "--request", "reopen",
            "--expected-tail-sha256", closed_sha,
        )
        self.assertEqual(reopened.returncode, 0, reopened.stderr)
        decision = json.loads(reopened.stdout)
        self.assertEqual(decision["reopen_reason"], "new_measurement_window")
        self.assertEqual(decision["closed_decision_sha256"], closed_sha)

    def test_status_rejects_malformed_records_and_changed_external_tail(self) -> None:
        self.assertEqual(self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        ).returncode, 0)
        first = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector"
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        decision_path = self.run_dir / "direction-decisions" / "decision-0001.json"
        original_tail = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        payload = json.loads(decision_path.read_text())
        payload["unexpected"] = True
        decision_path.write_text(json.dumps(payload), encoding="utf-8")
        malformed = self.run_cli("status", "--run-dir", str(self.run_dir))
        self.assertEqual(malformed.returncode, 2)
        payload.pop("unexpected")
        decision_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
        changed = self.run_cli(
            "status", "--run-dir", str(self.run_dir),
            "--expected-tail-sha256", original_tail
        )
        self.assertEqual(changed.returncode, 2)
        self.assertIn("expected tail", changed.stderr)

    def test_status_rejects_forged_reopen_reference_and_reason(self) -> None:
        self.assertEqual(self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        ).returncode, 0)
        closed = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector", "--request", "close",
        )
        self.assertEqual(closed.returncode, 0, closed.stderr)
        closed_path = self.run_dir / "direction-decisions" / "decision-0001.json"
        closed_sha = hashlib.sha256(closed_path.read_bytes()).hexdigest()
        (self.root / "window.json").write_bytes(b"new-window\n")
        payload = json.loads(self.portfolio.read_text())
        window_sha = hashlib.sha256((self.root / "window.json").read_bytes()).hexdigest()
        payload["measurement_window_artifact"]["sha256"] = window_sha
        payload["directions"][0]["component_metric"] = 33.0
        for item in payload["directions"]:
            source_path = self.root / item["source_artifact"]["path"]
            source_path.write_bytes(f"new-{item['id']}-profile\n".encode())
            item["source_artifact"]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            evidence_path = self.root / item["evidence_artifact"]["path"]
            evidence = json.loads(evidence_path.read_text())
            evidence["source_artifact"] = item["source_artifact"]
            evidence["measurement_window_sha256"] = window_sha
            evidence["component_metric"] = item["component_metric"]
            raw = (json.dumps(evidence, sort_keys=True) + "\n").encode()
            evidence_path.write_bytes(raw)
            item["evidence_artifact"]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.portfolio.write_text(json.dumps(payload), encoding="utf-8")
        reopened = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector", "--request", "reopen",
            "--expected-tail-sha256", closed_sha,
        )
        self.assertEqual(reopened.returncode, 0, reopened.stderr)
        decision_path = self.run_dir / "direction-decisions" / "decision-0002.json"
        decision = json.loads(decision_path.read_text())
        decision["closed_decision_sha256"] = SHA_F
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        forged = self.run_cli("status", "--run-dir", str(self.run_dir))
        self.assertEqual(forged.returncode, 2)
        self.assertIn("latest closed family decision", forged.stderr)
        decision["closed_decision_sha256"] = closed_sha
        decision["reopen_reason"] = "new_target_identity"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        wrong_reason = self.run_cli("status", "--run-dir", str(self.run_dir))
        self.assertEqual(wrong_reason.returncode, 2)
        self.assertIn("reopen reason", wrong_reason.stderr)

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

    def test_cli_rehashes_bound_artifacts_and_rejects_drift(self) -> None:
        init = self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        )
        self.assertEqual(init.returncode, 0, init.stderr)
        (self.root / "selector-evidence.json").write_text("changed\n", encoding="utf-8")
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
        self.assertIn("artifact digest", failed.stderr)

    def test_cli_rejects_portfolio_metrics_not_bound_by_evidence(self) -> None:
        init = self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        )
        self.assertEqual(init.returncode, 0, init.stderr)
        payload = json.loads(self.portfolio.read_text())
        payload["directions"][0]["component_metric"] = 100.0
        self.portfolio.write_text(json.dumps(payload), encoding="utf-8")
        failed = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector",
        )
        self.assertEqual(failed.returncode, 2)
        self.assertIn("evidence field component_metric", failed.stderr)

    def test_cli_rehashes_the_raw_source_bound_by_normalized_evidence(self) -> None:
        init = self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        )
        self.assertEqual(init.returncode, 0, init.stderr)
        (self.root / "selector-profile.json").write_bytes(b"changed-profile\n")
        failed = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector",
        )
        self.assertEqual(failed.returncode, 2)
        self.assertIn("source artifact digest", failed.stderr)

    def test_ledger_hashes_the_same_bytes_it_parses_without_rescanning(self) -> None:
        self.assertEqual(self.run_cli(
            "init", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir)
        ).returncode, 0)
        first = self.run_cli(
            "check", "--portfolio", str(self.portfolio), "--run-dir", str(self.run_dir),
            "--direction-id", "selector",
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        lineage = direction_guard._validate_lineage(
            direction_guard.load_json_strict(self.run_dir / "direction-lineage.json")
        )
        with mock.patch.object(
            direction_guard.artifact_store,
            "sha256_file",
            side_effect=AssertionError("ledger files must not be rescanned"),
        ):
            records, hashes = direction_guard._load_ledger(self.run_dir, lineage)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(hashes), 1)


if __name__ == "__main__":
    unittest.main()
