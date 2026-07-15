from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "cuda-kernel-optimizer"
CATALOG = SKILL / "references" / "optimization_catalog.md"
REGISTRY = SKILL / "references" / "method_registry.json"
COMPATIBILITY = SKILL / "references" / "compatibility.md"
VALIDATOR = SKILL / "scripts" / "validate_methods.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_validate_methods", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompatibilityTests(unittest.TestCase):
    def test_deprecated_and_private_triton_claims_are_removed(self) -> None:
        catalog = CATALOG.read_text(encoding="utf-8")
        self.assertNotIn("allow_tf32=True", catalog)
        self.assertNotIn("num_consumer_groups", catalog)
        self.assertNotIn("num_buffers_warp_spec", catalog)
        self.assertNotIn("make_block_ptr` 触发 TMA", catalog)
        self.assertIn('input_precision="tf32"', catalog)
        self.assertIn("make_tensor_descriptor", catalog)
        self.assertIn("fork-specific", catalog)

    def test_blackwell_features_are_explicit_per_architecture(self) -> None:
        arch = json.loads(REGISTRY.read_text(encoding="utf-8"))["arch_feature_map"]
        for sm in ("sm_103", "sm_110", "sm_120", "sm_121"):
            self.assertIn(sm, arch)

        for sm in ("sm_103", "sm_110"):
            self.assertIn("tma", arch[sm])
            self.assertIn("tcgen05", arch[sm])
            self.assertIn("tmem", arch[sm])
            self.assertNotIn("wgmma", arch[sm])

        self.assertNotIn("wgmma", arch["sm_100"])

        for sm in ("sm_120", "sm_121"):
            self.assertIn("tma", arch[sm])
            self.assertIn("block_scaling", arch[sm])
            self.assertNotIn("tcgen05", arch[sm])
            self.assertNotIn("tmem", arch[sm])
            self.assertNotIn("wgmma", arch[sm])

        method = json.loads(REGISTRY.read_text(encoding="utf-8"))["methods"][
            "compute.block_scaled_precision"
        ]
        self.assertEqual(method["required_features"], ["block_scaling"])

    def test_validator_uses_exact_required_features(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        validator = _load_validator()
        missing = validator._missing_required_features(
            registry,
            "sm_120",
            registry["methods"]["compute.gemm_softmax_interleave"],
        )
        self.assertIn("wgmma", missing)

    def test_version_targets_and_capability_rules_are_documented(self) -> None:
        text = COMPATIBILITY.read_text(encoding="utf-8")
        for expected in (
            "2026-07-15",
            "CUDA Toolkit 13.3",
            "Nsight Compute 2026.2.1",
            "Triton 3.7.1",
            "CUTLASS 4.6.1",
            "CUTLASS_NVCC_ARCHS",
        ):
            self.assertIn(expected, text)
        self.assertIn("never infer", text.lower())


if __name__ == "__main__":
    unittest.main()
