from __future__ import annotations

import copy
import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRED_STATS_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "paired_stats.py"
)


def _load_paired_stats():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_paired_stats", PAIRED_STATS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ImprovementPctTests(unittest.TestCase):
    def test_lower_reports_two_percent_improvement(self) -> None:
        paired_stats = _load_paired_stats()
        self.assertAlmostEqual(
            paired_stats.improvement_pct(100, 98, "lower"),
            2.0,
        )

    def test_higher_reverses_the_same_observation(self) -> None:
        paired_stats = _load_paired_stats()
        self.assertAlmostEqual(
            paired_stats.improvement_pct(100, 98, "higher"),
            -2.0,
        )

    def test_negative_baseline_uses_its_absolute_magnitude(self) -> None:
        paired_stats = _load_paired_stats()
        self.assertAlmostEqual(
            paired_stats.improvement_pct(-100, -98, "higher"),
            2.0,
        )

    def test_invalid_baselines_are_rejected_with_parameter_name(self) -> None:
        paired_stats = _load_paired_stats()
        for value in (0, True, "100", math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "baseline"
            ):
                paired_stats.improvement_pct(value, 98, "lower")

    def test_invalid_candidates_are_rejected_with_parameter_name(self) -> None:
        paired_stats = _load_paired_stats()
        for value in (True, "98", math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "candidate"
            ):
                paired_stats.improvement_pct(100, value, "lower")

    def test_invalid_direction_is_rejected(self) -> None:
        paired_stats = _load_paired_stats()
        for direction in ("smaller", "LOWER", "", True):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                paired_stats.improvement_pct(100, 98, direction)


class BootstrapMedianCiTests(unittest.TestCase):
    def test_fixed_seed_is_reproducible_and_bounds_are_ordered(self) -> None:
        paired_stats = _load_paired_stats()
        first = paired_stats.bootstrap_median_ci(
            [1.0, 2.0, 3.0, 20.0], confidence=0.9, samples=200, seed=17
        )
        second = paired_stats.bootstrap_median_ci(
            [1.0, 2.0, 3.0, 20.0], confidence=0.9, samples=200, seed=17
        )

        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_percentiles_linearly_interpolate_between_bootstrap_statistics(
        self,
    ) -> None:
        paired_stats = _load_paired_stats()
        self.assertEqual(
            paired_stats.bootstrap_median_ci(
                [0.0, 10.0], confidence=0.5, samples=4, seed=0
            ),
            (3.75, 6.25),
        )

    def test_invalid_value_sequences_are_rejected(self) -> None:
        paired_stats = _load_paired_stats()
        for values in (
            [],
            [True],
            ["1"],
            [math.nan],
            [math.inf],
            [-math.inf],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                paired_stats.bootstrap_median_ci(values)

    def test_invalid_confidence_is_rejected(self) -> None:
        paired_stats = _load_paired_stats()
        for confidence in (
            0,
            1,
            -0.1,
            1.1,
            True,
            "0.95",
            math.nan,
            math.inf,
            -math.inf,
        ):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                paired_stats.bootstrap_median_ci([1.0], confidence=confidence)

    def test_invalid_sample_count_is_rejected(self) -> None:
        paired_stats = _load_paired_stats()
        for samples in (0, -1, True, 1.5, "10"):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                paired_stats.bootstrap_median_ci([1.0], samples=samples)


class ClassifyPairsTests(unittest.TestCase):
    def test_lower_two_percent_improvement_is_confirmed_win(self) -> None:
        paired_stats = _load_paired_stats()
        result = paired_stats.classify_pairs(
            [{"baseline": 100.0, "candidate": 98.0} for _ in range(6)],
            direction="lower",
            min_effect_pct=1.5,
            bootstrap_samples=200,
            seed=3,
        )

        self.assertEqual(result["status"], "confirmed_win")
        self.assertEqual(result["estimate_pct"], 2.0)
        self.assertEqual(result["ci_low_pct"], 2.0)
        self.assertEqual(result["ci_high_pct"], 2.0)
        self.assertEqual(result["improvements_pct"], [2.0] * 6)

    def test_higher_direction_on_same_data_is_confirmed_loss(self) -> None:
        paired_stats = _load_paired_stats()
        result = paired_stats.classify_pairs(
            [{"baseline": 100.0, "candidate": 98.0} for _ in range(6)],
            direction="higher",
            min_effect_pct=1.5,
            bootstrap_samples=200,
            seed=3,
        )

        self.assertEqual(result["status"], "confirmed_loss")
        self.assertEqual(result["estimate_pct"], -2.0)
        self.assertEqual(result["ci_low_pct"], -2.0)
        self.assertEqual(result["ci_high_pct"], -2.0)

    def test_mixed_99_101_noise_is_inconclusive(self) -> None:
        paired_stats = _load_paired_stats()
        pairs = [
            {"baseline": 100.0, "candidate": candidate}
            for candidate in ([99.0, 101.0] * 10)
        ]
        result = paired_stats.classify_pairs(
            pairs,
            direction="lower",
            min_effect_pct=0.0,
            bootstrap_samples=2000,
            seed=11,
        )

        self.assertEqual(result["status"], "inconclusive")
        self.assertLess(result["ci_low_pct"], 0.0)
        self.assertGreater(result["ci_high_pct"], 0.0)

    def test_invalid_blocks_are_excluded_and_counted(self) -> None:
        paired_stats = _load_paired_stats()
        pairs = [
            {"baseline": 100.0, "candidate": 98.0},
            {"valid": False},
            {"valid": False, "baseline": 0, "candidate": math.nan},
        ]
        result = paired_stats.classify_pairs(
            pairs,
            direction="lower",
            min_effect_pct=1.5,
            bootstrap_samples=100,
        )

        self.assertEqual(result["valid_pairs"], 1)
        self.assertEqual(result["invalid_pairs"], 2)
        self.assertEqual(result["improvements_pct"], [2.0])

    def test_no_valid_pairs_returns_empty_inconclusive_result(self) -> None:
        paired_stats = _load_paired_stats()
        result = paired_stats.classify_pairs(
            [{"valid": False}, {"valid": False}],
            direction="lower",
            min_effect_pct=1.0,
            bootstrap_samples=100,
        )

        self.assertEqual(
            result,
            {
                "status": "inconclusive",
                "statistic": "median_paired_improvement_pct",
                "direction": "lower",
                "min_effect_pct": 1.0,
                "confidence": 0.95,
                "estimate_pct": None,
                "ci_low_pct": None,
                "ci_high_pct": None,
                "valid_pairs": 0,
                "invalid_pairs": 2,
                "improvements_pct": [],
            },
        )

    def test_valid_pair_requires_baseline_and_candidate(self) -> None:
        paired_stats = _load_paired_stats()
        for pair, parameter in (({"candidate": 1.0}, "baseline"), ({"baseline": 1.0}, "candidate")):
            with self.subTest(pair=pair), self.assertRaisesRegex(
                ValueError, parameter
            ):
                paired_stats.classify_pairs(
                    [pair],
                    direction="lower",
                    min_effect_pct=0.0,
                    bootstrap_samples=10,
                )

    def test_invalid_minimum_effect_is_rejected(self) -> None:
        paired_stats = _load_paired_stats()
        for min_effect_pct in (
            -0.1,
            True,
            "1",
            math.nan,
            math.inf,
            -math.inf,
        ):
            with self.subTest(
                min_effect_pct=min_effect_pct
            ), self.assertRaises(ValueError):
                paired_stats.classify_pairs(
                    [{"baseline": 100.0, "candidate": 98.0}],
                    direction="lower",
                    min_effect_pct=min_effect_pct,
                    bootstrap_samples=10,
                )

    def test_classification_validates_bootstrap_parameters(self) -> None:
        paired_stats = _load_paired_stats()
        pairs = [{"baseline": 100.0, "candidate": 98.0}]
        for keyword, value in (
            ("confidence", 1.0),
            ("confidence", math.nan),
            ("bootstrap_samples", 0),
            ("bootstrap_samples", True),
        ):
            with self.subTest(keyword=keyword, value=value), self.assertRaises(
                ValueError
            ):
                paired_stats.classify_pairs(
                    pairs,
                    direction="lower",
                    min_effect_pct=0.0,
                    **{keyword: value},
                )

    def test_input_pairs_are_not_modified(self) -> None:
        paired_stats = _load_paired_stats()
        pairs = [
            {"baseline": 100.0, "candidate": 98.0, "metadata": {"block": 1}},
            {"valid": False, "metadata": {"reason": "warmup"}},
        ]
        original = copy.deepcopy(pairs)

        paired_stats.classify_pairs(
            pairs,
            direction="lower",
            min_effect_pct=1.0,
            bootstrap_samples=100,
            seed=9,
        )

        self.assertEqual(pairs, original)

    def test_result_contains_the_required_metadata(self) -> None:
        paired_stats = _load_paired_stats()
        result = paired_stats.classify_pairs(
            [{"baseline": 100.0, "candidate": 98.0}],
            direction="lower",
            min_effect_pct=1.0,
            confidence=0.9,
            bootstrap_samples=10,
        )

        self.assertEqual(
            set(result),
            {
                "status",
                "statistic",
                "direction",
                "min_effect_pct",
                "confidence",
                "estimate_pct",
                "ci_low_pct",
                "ci_high_pct",
                "valid_pairs",
                "invalid_pairs",
                "improvements_pct",
            },
        )
        self.assertEqual(result["statistic"], "median_paired_improvement_pct")
        self.assertEqual(result["direction"], "lower")
        self.assertEqual(result["min_effect_pct"], 1.0)
        self.assertEqual(result["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
