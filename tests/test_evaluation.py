"""Unit tests for src/c2dti/evaluation.py.

Each test uses simple hand-calculable inputs so we can verify correctness
without relying on an external statistics library.
"""

import math
import unittest

import numpy as np

from src.c2dti.evaluation import (
    compute_ci,
    compute_mse,
    compute_pearson,
    compute_rmse,
    compute_spearman,
    evaluate_predictions,
)


class TestComputeMSE(unittest.TestCase):
    def test_perfect_predictions_give_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(compute_mse(y, y), 0.0)

    def test_constant_offset(self):
        # MSE of (1,2,3) vs (2,3,4) = mean([1,1,1]) = 1.0
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        self.assertAlmostEqual(compute_mse(y_true, y_pred), 1.0)

    def test_known_value(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 0.0])
        # errors: -1, 1  → squared: 1, 1  → mean: 1.0
        self.assertAlmostEqual(compute_mse(y_true, y_pred), 1.0)

    def test_accepts_2d_input(self):
        # Function should flatten 2D arrays before computing
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(compute_mse(y_true, y_pred), 0.0)


class TestComputeRMSE(unittest.TestCase):
    def test_rmse_is_sqrt_of_mse(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 0.0])
        mse = compute_mse(y_true, y_pred)
        self.assertAlmostEqual(compute_rmse(y_true, y_pred), math.sqrt(mse))

    def test_perfect_predictions_give_zero(self):
        y = np.array([0.5, 0.7, 0.9])
        self.assertAlmostEqual(compute_rmse(y, y), 0.0)


class TestComputePearson(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(compute_pearson(y, y), 1.0, places=6)

    def test_perfect_negative_correlation(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        self.assertAlmostEqual(compute_pearson(y_true, y_pred), -1.0, places=6)

    def test_zero_correlation_constant_pred(self):
        # Constant predictions — correlation is undefined, should return 0
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        self.assertAlmostEqual(compute_pearson(y_true, y_pred), 0.0)

    def test_single_element_returns_zero(self):
        self.assertAlmostEqual(compute_pearson(np.array([1.0]), np.array([2.0])), 0.0)

    def test_result_clipped_to_minus_one_one(self):
        # Even with floating point noise the result should stay in [-1, 1]
        y_true = np.array([1.0, 2.0, 3.0])
        result = compute_pearson(y_true, y_true * 1e15)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)


class TestComputeSpearman(unittest.TestCase):
    def test_monotone_increasing(self):
        # Perfect rank agreement → spearman = 1.0
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(compute_spearman(y_true, y_pred), 1.0, places=6)

    def test_monotone_decreasing(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([30.0, 20.0, 10.0])
        self.assertAlmostEqual(compute_spearman(y_true, y_pred), -1.0, places=6)

    def test_with_ties(self):
        # Spearman should handle tied ranks without crashing
        y_true = np.array([1.0, 1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 1.0, 2.0, 3.0])
        result = compute_spearman(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_single_element_returns_zero(self):
        self.assertAlmostEqual(compute_spearman(np.array([1.0]), np.array([2.0])), 0.0)


class TestComputeCI(unittest.TestCase):
    def test_perfect_ranking(self):
        # When predicted order matches true order exactly, CI = 1.0
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(compute_ci(y_true, y_pred), 1.0, places=6)

    def test_reversed_ranking(self):
        # Perfect reversal → CI = 0.0
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(compute_ci(y_true, y_pred), 0.0, places=6)

    def test_all_true_values_equal_returns_half(self):
        # When all true values are identical there are no positive pairs → returns 0.5
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([0.1, 0.5, 0.9])
        self.assertAlmostEqual(compute_ci(y_true, y_pred), 0.5, places=6)

    def test_single_pair(self):
        # One positive pair, predicted correctly
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([0.3, 0.7])
        self.assertAlmostEqual(compute_ci(y_true, y_pred), 1.0, places=6)

    def test_tied_predictions_count_as_half_concordant(self):
        # y_true[0] > y_true[1] but y_pred[0] == y_pred[1] → 0.5 concordance
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([0.5, 0.5])
        self.assertAlmostEqual(compute_ci(y_true, y_pred), 0.5, places=6)

    def test_result_in_range(self):
        rng = np.random.RandomState(0)
        y_true = rng.rand(100)
        y_pred = rng.rand(100)
        result = compute_ci(y_true, y_pred)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_small_input_returns_zero(self):
        self.assertAlmostEqual(compute_ci(np.array([1.0]), np.array([1.0])), 0.0)


class TestEvaluatePredictions(unittest.TestCase):
    def test_returns_all_keys(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        result = evaluate_predictions(y_true, y_pred)
        for key in ("mse", "rmse", "pearson", "spearman", "ci", "n_evaluated"):
            self.assertIn(key, result)

    def test_n_evaluated_matches_input(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = evaluate_predictions(y_true, y_pred)
        self.assertEqual(result["n_evaluated"], 3)

    def test_nan_entries_are_excluded(self):
        # y_true[1] is NaN — that pair should be skipped
        y_true = np.array([1.0, float("nan"), 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = evaluate_predictions(y_true, y_pred)
        self.assertEqual(result["n_evaluated"], 2)
        # The two remaining pairs are perfect → mse = 0
        self.assertAlmostEqual(result["mse"], 0.0)

    def test_empty_after_nan_filter_returns_none_values(self):
        y_true = np.array([float("nan"), float("nan")])
        y_pred = np.array([1.0, 2.0])
        result = evaluate_predictions(y_true, y_pred)
        self.assertIsNone(result["mse"])
        self.assertEqual(result["n_evaluated"], 0)

    def test_2d_arrays_are_accepted(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = evaluate_predictions(y_true, y_pred)
        self.assertEqual(result["n_evaluated"], 4)
        self.assertAlmostEqual(result["mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
