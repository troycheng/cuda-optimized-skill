from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "ablate.py"


def _load_ablate():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_ablate_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AblateEvidenceTests(unittest.TestCase):
    def test_dotted_method_publishes_bound_rounded_evidence(self) -> None:
        module = _load_ablate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run = root / "run"
            iteration = run / "iterv1"
            ablation = iteration / "ablations" / "memory_coalesce"
            ablation.mkdir(parents=True)
            ref = root / "ref.py"
            ref.write_text("# ref\n", encoding="utf-8")
            champion = iteration / "bench.json"
            champion.write_text(json.dumps({
                "correctness": {"passed": True},
                "kernel": {"average_ms": 10.123456},
            }), encoding="utf-8")
            kernel = ablation / "kernel.py"
            kernel.write_text("# ablated\n", encoding="utf-8")
            (iteration / "methods.json").write_text(
                json.dumps({"methods": [{"id": "memory.coalesce"}]}), encoding="utf-8"
            )
            state = root / "state.json"
            state.write_text(json.dumps({
                "run_dir": str(run), "ref_file": str(ref), "dims": {},
                "ptr_size": 8, "noise_threshold_pct": 2.0,
            }), encoding="utf-8")
            result = {
                "correctness": {"passed": True},
                "kernel": {"average_ms": 12.654321},
            }

            def fake_bench(*_args):
                bench = ablation / "bench.json"
                bench.write_text(json.dumps(result), encoding="utf-8")
                return result

            with mock.patch.object(module, "_bench_kernel", side_effect=fake_bench):
                output = module.run(str(state), 1)
            item = output["attributions"][0]
            bench = ablation / "bench.json"
            self.assertEqual(item["method_id"], "memory.coalesce")
            self.assertEqual(item["ablated_kernel"], str(kernel.resolve()))
            self.assertEqual(item["ablated_kernel_sha256"], hashlib.sha256(kernel.read_bytes()).hexdigest())
            self.assertEqual(item["champion_bench"], str(champion.resolve()))
            self.assertEqual(item["champion_bench_sha256"], hashlib.sha256(champion.read_bytes()).hexdigest())
            self.assertEqual(item["ablated_bench"], str(bench.resolve()))
            self.assertEqual(item["ablated_bench_sha256"], hashlib.sha256(bench.read_bytes()).hexdigest())
            self.assertEqual(item["champion_ms"], round(10.123456, 4))
            self.assertEqual(item["ablated_ms"], round(12.654321, 4))
            self.assertEqual(item["attribution_ms"], round(12.654321 - 10.123456, 4))
            self.assertEqual(
                item["attribution_pct"],
                round((12.654321 - 10.123456) / 10.123456 * 100.0, 2),
            )


if __name__ == "__main__":
    unittest.main()
