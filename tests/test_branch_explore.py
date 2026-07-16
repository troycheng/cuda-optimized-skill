from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BRANCH_EXPLORE_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "branch_explore.py"
)


def _load_branch_explore():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_branch_explore", BRANCH_EXPLORE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bench(average: float, median: float, p95: float, cv_pct: float) -> dict:
    return {
        "correctness": {"passed": True},
        "kernel": {
            "average_ms": average,
            "median_ms": median,
            "p95_ms": p95,
            "cv_pct": cv_pct,
        },
    }


class BranchExploreTests(unittest.TestCase):
    def _state(self, root: Path, noise_threshold_pct: float = 2.0) -> Path:
        run_dir = root / "run"
        for branch in (1, 2):
            branch_dir = run_dir / "iterv1" / "branches" / f"b{branch}"
            branch_dir.mkdir(parents=True)
            (branch_dir / "kernel.py").write_text(f"# branch {branch}\n", encoding="utf-8")
        state_path = root / "state.json"
        state_path.write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "ref_file": str(root / "reference.py"),
                    "dims": {},
                    "ptr_size": 0,
                    "branches": 2,
                    "noise_threshold_pct": noise_threshold_pct,
                }
            ),
            encoding="utf-8",
        )
        return state_path

    def test_median_stable_branch_wins_over_better_average(self) -> None:
        branch_explore = _load_branch_explore()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            branch_explore,
            "_bench_kernel",
            side_effect=[
                _bench(average=0.90, median=1.05, p95=1.10, cv_pct=15.0),
                _bench(average=1.00, median=0.95, p95=1.01, cv_pct=1.0),
            ],
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                output = branch_explore.run(str(self._state(Path(tmp))), iteration=1)

        self.assertEqual(output["champion"]["branch_index"], 2)
        self.assertEqual(output["champion"]["ms"], 0.95)
        self.assertEqual(output["champion"]["average_ms"], 1.00)

    def test_candidates_are_annotated_against_noise_threshold(self) -> None:
        branch_explore = _load_branch_explore()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            branch_explore,
            "_bench_kernel",
            side_effect=[
                _bench(average=1.00, median=1.00, p95=1.02, cv_pct=1.0),
                _bench(average=1.02, median=1.015, p95=1.04, cv_pct=1.2),
            ],
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                output = branch_explore.run(
                    str(self._state(Path(tmp), noise_threshold_pct=2.0)), iteration=1
                )

        by_branch = {item["branch_index"]: item for item in output["branches"]}
        self.assertTrue(by_branch[1]["within_noise_of_champion"])
        self.assertTrue(by_branch[2]["within_noise_of_champion"])
        self.assertAlmostEqual(by_branch[2]["delta_pct_from_champion"], 1.5)


if __name__ == "__main__":
    unittest.main()
