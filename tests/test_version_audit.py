import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "version_audit.py"


def valid_payload():
    frozen = {
        "source_sha256": "1" * 64,
        "onnx_sha256": "2" * 64,
        "build_recipe_sha256": "3" * 64,
        "request_corpus_sha256": "4" * 64,
        "correctness_contract_sha256": "5" * 64,
        "benchmark_design_sha256": "6" * 64,
        "model_config_sha256": "7" * 64,
        "custom_backend_sha256": "8" * 64,
        "gpu_uuid": "GPU-fixed",
        "driver_version": "fixed-driver",
        "clock_policy": "locked-3000",
    }
    return {
        "schema_version": "cuda-version-audit/v1",
        "baseline": {
            "frozen": frozen,
            "stack": {
                "image_digest": "sha256:baseline",
                "triton_version": "2.63",
                "tensorrt_version": "10.14",
                "cuda_version": "13.1",
            },
            "derived": {
                "plugin_sha256": "9" * 64,
                "engine_sha256": "a" * 64,
                "timing_cache_sha256": "b" * 64,
            },
        },
        "candidate": {
            "frozen": dict(frozen),
            "stack": {
                "image_digest": "sha256:candidate",
                "triton_version": "2.69",
                "tensorrt_version": "10.16",
                "cuda_version": "13.2",
            },
            "derived": {
                "plugin_sha256": "c" * 64,
                "engine_sha256": "d" * 64,
                "timing_cache_sha256": "e" * 64,
            },
        },
        "fresh_build_per_stack": True,
        "fresh_timing_cache_per_stack": True,
        "self_repeat_stable_per_stack": True,
        "reuse_engine_across_stacks": False,
        "correctness": {"passed": True, "evidence_id": "correctness-valid"},
        "timing_started": True,
        "measurement_evidence_ids": ["timing-valid"],
        "invalid_evidence_ids": ["timing-invalid"],
    }


class VersionAuditTests(unittest.TestCase):
    def run_audit(self, payload=None, *, raw=None, output_symlink=False):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = root / "input.json"
            output = root / "output.json"
            source.write_text(raw if raw is not None else json.dumps(payload))
            if output_symlink:
                target = root / "outside.json"
                target.write_text("unchanged")
                output.symlink_to(target)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(source), "--out", str(output)],
                text=True,
                capture_output=True,
            )
            report = None
            if output.exists() and not output.is_symlink():
                report = json.loads(output.read_text())
            outside = (root / "outside.json").read_text() if output_symlink else None
            return result, report, outside

    def test_accepts_pure_fresh_stack_upgrade(self):
        result, report, _ = self.run_audit(valid_payload())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["passed"])

    def test_rejects_confounded_custom_backend(self):
        payload = valid_payload()
        payload["candidate"]["frozen"]["custom_backend_sha256"] = "f" * 64
        result, report, _ = self.run_audit(payload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frozen_mismatch:custom_backend_sha256", report["reasons"])

    def test_rejects_timing_before_correctness_or_self_repeat_stability(self):
        for field in ("correctness", "self_repeat_stable_per_stack"):
            payload = valid_payload()
            if field == "correctness":
                payload[field]["passed"] = False
                reason = "timing_started_before_correctness_passed"
            else:
                payload[field] = False
                reason = "timing_started_before_self_repeat_stability"
            result, report, _ = self.run_audit(payload)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(reason, report["reasons"])

    def test_rejects_cross_stack_engine_reuse(self):
        payload = valid_payload()
        payload["reuse_engine_across_stacks"] = True
        result, report, _ = self.run_audit(payload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("engine_reused_across_stacks", report["reasons"])

    def test_invalid_evidence_is_quarantined_from_correctness_and_timing(self):
        for field, evidence_id in (
            ("measurement_evidence_ids", "timing-invalid"),
            ("correctness", "correctness-valid"),
        ):
            payload = valid_payload()
            if field == "correctness":
                payload["invalid_evidence_ids"].append(evidence_id)
            else:
                payload[field].append(evidence_id)
            result, report, _ = self.run_audit(payload)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid_evidence_referenced:%s" % evidence_id, report["reasons"])

    def test_rejects_unknown_duplicate_and_nonfinite_json(self):
        payload = valid_payload()
        payload["unknown"] = True
        result, report, _ = self.run_audit(payload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected_field:root.unknown", report["reasons"])

        raw = json.dumps(valid_payload())[:-1] + ',"schema_version":"duplicate"}'
        result, report, _ = self.run_audit(raw=raw)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate JSON key", result.stderr)
        self.assertIsNone(report)

        raw = json.dumps(valid_payload()).replace('"timing_started": true', '"timing_started": NaN')
        result, report, _ = self.run_audit(raw=raw)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JSON number must be finite", result.stderr)
        self.assertIsNone(report)

    def test_rejects_symlink_input_and_output_without_touching_target(self):
        result, report, outside = self.run_audit(valid_payload(), output_symlink=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(report)
        self.assertEqual(outside, "unchanged")

    def test_timing_evidence_requires_timing_started(self):
        payload = valid_payload()
        payload["timing_started"] = False
        result, report, _ = self.run_audit(payload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("measurement_evidence_without_timing", report["reasons"])


if __name__ == "__main__":
    unittest.main()
