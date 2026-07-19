from __future__ import annotations

import copy
import importlib.util
import inspect
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
MODULE_PATH = SCRIPTS / "stability_calibration.py"
CONTRACT_PATH = SCRIPTS / "workload_contract.py"
ENVIRONMENT_SHA = "b" * 64
SOURCE_SHA = "c" * 64
SEAL_KEY = b"stability-controller-secret" * 2


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _blocks(values):
    return [
        {
            "pair_id": f"pair-{index}",
            "first": first,
            "second": second,
            "valid": True,
        }
        for index, (first, second) in enumerate(values, 1)
    ]


class StabilityCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load(MODULE_PATH, "cuda_v3_stability")
        cls.contract = _load(CONTRACT_PATH, "cuda_v3_stability_contract")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.contract_path = self._freeze_contract("default")

    def tearDown(self):
        self.temp.cleanup()

    def _freeze_contract(
        self,
        name: str,
        *,
        minimum_practical_effect_pct: float = 1.0,
        stability: dict | None = None,
    ) -> Path:
        project = self.root / f"project-{name}"
        environment = self.root / f"environment-{name}"
        project.mkdir()
        environment.mkdir()
        (project / "kernels").mkdir()
        (project / "workload.json").write_text('{"name":"demo"}\n', "utf-8")
        (project / "reference.py").write_text("def reference(x): return x\n", "utf-8")
        policy = {
            "confidence": 0.95,
            "power": 0.8,
            "bootstrap_samples": 2000,
            "min_valid_pairs": 4,
            "seed": 17,
            "audit_every_candidates": 1,
        }
        if stability is not None:
            policy.update(stability)
        draft = {
            "schema_version": "cuda-optimizer/workload-contract-draft-v1",
            "run_id": f"stability-{name}",
            "parent_run": None,
            "requested_claim": "workload",
            "project_root": str(project),
            "artifacts": [
                {"role": "workload_manifest", "path": "workload.json"},
                {"role": "correctness_reference", "path": "reference.py"},
            ],
            "workload": {
                "argv": ["python3", "workload.py"],
                "input_distribution": "representative-test-snapshot",
                "representative_cases": ["prefill", "decode"],
            },
            "objective": {
                "metric": "request_latency",
                "unit": "ms",
                "direction": "lower",
                "aggregation": "median",
                "minimum_practical_effect_pct": minimum_practical_effect_pct,
                "constraints": ["correctness"],
            },
            "budget": {
                "preset": "balanced",
                "max_seconds": 10800,
                "max_candidates": 24,
            },
            "stability": policy,
            "mutation": {
                "project_paths": ["kernels"],
                "environment_root": str(environment),
                "host_policy": "recommend_only",
            },
            "evidence": {"max_age_seconds": 1800},
        }
        path = self.root / f"contract-{name}.json"
        self.contract.freeze_contract(draft, path)
        return path

    def _calibrate(self, values, *, contract_path: Path | None = None, **overrides):
        arguments = {
            "contract_path": contract_path or self.contract_path,
            "blocks": _blocks(values),
            "hard_guardrails_passed": True,
            "environment_sha256": ENVIRONMENT_SHA,
            "source_sha256": SOURCE_SHA,
            "recorded_at": 100.0,
            "controller_seal_key": SEAL_KEY,
        }
        arguments.update(overrides)
        return self.module.calibrate(**arguments)

    def test_policy_is_derived_only_from_verified_frozen_contract(self):
        parameters = inspect.signature(self.module.calibrate).parameters
        for forbidden in (
            "minimum_practical_effect_pct",
            "confidence",
            "power",
            "bootstrap_samples",
            "min_valid_pairs",
            "seed",
            "contract_sha256",
        ):
            self.assertNotIn(forbidden, parameters)

        contract = self._freeze_contract(
            "policy", minimum_practical_effect_pct=3.0,
            stability={"confidence": 0.9, "power": 0.85, "audit_every_candidates": 7},
        )
        result = self._calibrate([(100.0, 100.1)] * 4, contract_path=contract)
        self.assertEqual(result["minimum_practical_effect_pct"], 3.0)
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["power"], 0.85)
        self.assertEqual(result["audit_every_candidates"], 7)
        self.assertEqual(result["contract_sha256"], self.contract.verify_frozen_contract(contract)["contract_sha256"])

    def test_low_noise_baseline_is_green_and_reports_detectability(self):
        result = self._calibrate(
            [(100.0, 100.1), (100.0, 99.9), (100.0, 100.2), (100.0, 99.8)],
        )

        self.assertEqual(result["environment_state"], "green")
        self.assertTrue(result["measurable"])
        self.assertLessEqual(result["noise_ci_high_pct"], 1.0)
        self.assertLessEqual(result["minimum_detectable_effect_pct"], 1.0)
        self.assertEqual(result["decision_threshold_pct"], 1.0)
        self.assertEqual(result["mde_method"], "paired_log_ratio_normal_approximation")
        self.assertEqual(result["reasons"], [])

    def test_noise_or_mde_above_contract_mpe_is_yellow(self):
        values = [(100.0, 103.0), (100.0, 97.0), (100.0, 104.0), (100.0, 96.0)]
        yellow = self._calibrate(values)
        tolerant_contract = self._freeze_contract(
            "tolerant", minimum_practical_effect_pct=10.0
        )
        green = self._calibrate(values, contract_path=tolerant_contract)

        self.assertEqual(yellow["environment_state"], "yellow")
        self.assertFalse(yellow["measurable"])
        self.assertTrue(
            {"noise_exceeds_minimum_practical_effect", "mde_exceeds_minimum_practical_effect"}
            & set(yellow["reasons"])
        )
        self.assertEqual(green["environment_state"], "green")
        self.assertEqual(green["decision_threshold_pct"], 10.0)

    def test_noise_and_mde_are_order_invariant_and_tiny_bootstrap_is_rejected(self):
        contract = self._freeze_contract("order", minimum_practical_effect_pct=5.0)
        forward = self._calibrate([(100.0, 102.0)] * 4, contract_path=contract)
        reverse = self._calibrate([(102.0, 100.0)] * 4, contract_path=contract)
        self.assertEqual(forward["noise_ci_high_pct"], reverse["noise_ci_high_pct"])
        self.assertAlmostEqual(
            forward["minimum_detectable_effect_pct"],
            reverse["minimum_detectable_effect_pct"],
        )
        with self.assertRaisesRegex(ValueError, "bootstrap_samples"):
            self._freeze_contract("tiny", stability={"bootstrap_samples": 1})

    def test_insufficient_pairs_pause_and_hard_guardrail_failure_is_red(self):
        insufficient = self._calibrate([(100.0, 100.1), (100.0, 99.9)])
        red = self._calibrate(
            [(100.0, 100.1)] * 4,
            hard_guardrails_passed=False,
        )

        self.assertEqual(insufficient["environment_state"], "yellow")
        self.assertIn("insufficient_valid_pairs", insufficient["reasons"])
        self.assertEqual(red["environment_state"], "red")
        self.assertFalse(red["measurable"])
        self.assertEqual(red["reasons"], ["hard_guardrail_failed"])

    def test_zero_valid_pairs_are_yellow_or_red_without_invented_statistics(self):
        blocks = _blocks([(100.0, 100.1)] * 4)
        for block in blocks:
            block["valid"] = False
        yellow = self._calibrate([], blocks=blocks)
        red = self._calibrate(
            [], blocks=blocks, hard_guardrails_passed=False
        )

        self.assertEqual(yellow["environment_state"], "yellow")
        self.assertEqual(yellow["reasons"], ["insufficient_valid_pairs"])
        self.assertEqual(yellow["valid_pairs"], 0)
        self.assertIsNone(yellow["baseline_median"])
        self.assertIsNone(yellow["noise_median_pct"])
        self.assertIsNone(yellow["minimum_detectable_effect_pct"])
        self.assertEqual(red["environment_state"], "red")
        self.assertEqual(red["reasons"], ["hard_guardrail_failed"])
        with self.assertRaisesRegex(ValueError, "green|measurable"):
            self.module.audit(
                yellow,
                contract_path=self.contract_path,
                blocks=_blocks([(100.0, 100.0)] * 4),
                hard_guardrails_passed=True,
                environment_sha256=ENVIRONMENT_SHA,
                source_sha256=SOURCE_SHA,
                recorded_at=200.0,
                controller_seal_key=SEAL_KEY,
            )

        partial = _blocks([(100.0, 100.0)] * 4 + [(10000.0, 10000.0)])
        partial[-1]["valid"] = False
        clean = self._calibrate([], blocks=partial)
        self.assertEqual(clean["baseline_median"], 100.0)

    def test_periodic_replay_uses_same_contract_and_rejects_tampering(self):
        anchor = self._calibrate(
            [(100.0, 100.1), (100.0, 99.9), (100.0, 100.2), (100.0, 99.8)],
        )
        stable = self.module.audit(
            anchor,
            contract_path=self.contract_path,
            blocks=_blocks([(100.1, 100.0), (100.0, 100.2), (99.9, 100.0), (100.2, 100.1)]),
            hard_guardrails_passed=True,
            environment_sha256=ENVIRONMENT_SHA,
            source_sha256=SOURCE_SHA,
            recorded_at=200.0,
            controller_seal_key=SEAL_KEY,
        )
        shifted = self.module.audit(
            anchor,
            contract_path=self.contract_path,
            blocks=_blocks([(103.0, 103.1), (103.0, 102.9), (103.0, 103.2), (103.0, 102.8)]),
            hard_guardrails_passed=True,
            environment_sha256=ENVIRONMENT_SHA,
            source_sha256=SOURCE_SHA,
            recorded_at=200.0,
            controller_seal_key=SEAL_KEY,
        )

        self.assertEqual(stable["environment_state"], "green")
        self.assertEqual(shifted["environment_state"], "yellow")
        self.assertIn("baseline_shift_exceeds_calibrated_noise", shifted["reasons"])
        other_contract = self._freeze_contract("other", minimum_practical_effect_pct=2.0)
        with self.assertRaisesRegex(ValueError, "contract|policy"):
            self.module.audit(
                anchor,
                contract_path=other_contract,
                blocks=_blocks([(100.0, 100.0)] * 4),
                hard_guardrails_passed=True,
                environment_sha256=ENVIRONMENT_SHA,
                source_sha256="f" * 64,
                recorded_at=200.0,
                controller_seal_key=SEAL_KEY,
            )

        tampered = copy.deepcopy(anchor)
        tampered["minimum_practical_effect_pct"] = 100.0
        tampered["noise_ci_high_pct"] = 100.0
        unsigned = dict(tampered)
        unsigned.pop("calibration_sha256")
        unsigned.pop("controller_attestation")
        tampered["calibration_sha256"] = self.module._canonical_digest(unsigned)
        with self.assertRaisesRegex(ValueError, "attestation|controller"):
            self.module.audit(
                tampered,
                contract_path=self.contract_path,
                blocks=_blocks([(100.0, 100.0)] * 4),
                hard_guardrails_passed=True,
                environment_sha256=ENVIRONMENT_SHA,
                source_sha256="f" * 64,
                recorded_at=200.0,
                controller_seal_key=SEAL_KEY,
            )

        for field, value in (
            ("source_sha256", "d" * 64),
            ("environment_sha256", "e" * 64),
        ):
            arguments = {
                "source_sha256": SOURCE_SHA,
                "environment_sha256": ENVIRONMENT_SHA,
            }
            arguments[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "source|environment|identity"
            ):
                self.module.audit(
                    anchor,
                    contract_path=self.contract_path,
                    blocks=_blocks([(100.0, 100.0)] * 4),
                    hard_guardrails_passed=True,
                    recorded_at=200.0,
                    controller_seal_key=SEAL_KEY,
                    **arguments,
                )

    def test_contract_artifact_drift_duplicate_nonfinite_and_time_reversal_fail_closed(self):
        verified = self.contract.verify_frozen_contract(self.contract_path)
        (Path(verified["project_root"]) / "workload.json").write_text("changed\n", "utf-8")
        with self.assertRaisesRegex(ValueError, "identity|sha256|changed"):
            self._calibrate([(100.0, 100.0)] * 4)

        clean_contract = self._freeze_contract("clean")
        duplicate = _blocks([(100.0, 100.0), (100.0, 100.0)])
        duplicate[1]["pair_id"] = duplicate[0]["pair_id"]
        for blocks in (
            duplicate,
            _blocks([(100.0, math.nan)] * 4),
            _blocks([(100.0, 0.0)] * 4),
        ):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                self._calibrate([], contract_path=clean_contract, blocks=blocks)

        anchor = self._calibrate(
            [(100.0, 100.0)] * 4, contract_path=clean_contract, recorded_at=100.0
        )
        with self.assertRaisesRegex(ValueError, "time|recorded"):
            self.module.audit(
                anchor,
                contract_path=clean_contract,
                blocks=_blocks([(100.0, 100.0)] * 4),
                hard_guardrails_passed=True,
                environment_sha256=ENVIRONMENT_SHA,
                source_sha256=SOURCE_SHA,
                recorded_at=99.0,
                controller_seal_key=SEAL_KEY,
            )


if __name__ == "__main__":
    unittest.main()
