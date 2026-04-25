"""Evaluation metrics for Drug-Target Interaction (DTI) predictions.

Standard regression metrics used across the DTI literature:
  - MSE / RMSE   : penalise large prediction errors
  - Pearson      : linear correlation between predicted and true affinities
  - Spearman     : rank correlation (order-preserving quality)
  - CI           : Concordance Index — probability that a randomly chosen pair
                   is ranked in the same order as the true affinities.
                   This is the primary metric used by MINDG, DeepDTA, etc.

Usage:
    from src.c2dti.evaluation import evaluate_predictions
    metrics = evaluate_predictions(y_true, y_pred)
    # metrics = {"mse": ..., "rmse": ..., "pearson": ..., "spearman": ..., "ci": ...}
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error between true and predicted values.

    Lower is better. Heavily penalises large individual errors.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — same unit as the target values."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient between true and predicted values.

    Ranges from -1 (perfect negative) to +1 (perfect positive).
    Measures linear agreement; independent of scale.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) < 2:
        return 0.0

    # Center each array around its mean
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()

    numerator = np.dot(yt, yp)
    denominator = np.sqrt(np.dot(yt, yt) * np.dot(yp, yp))

    if denominator < 1e-12:
        # One or both arrays are constant — correlation is undefined; return 0
        return 0.0

    return float(np.clip(numerator / denominator, -1.0, 1.0))


def _rank_with_ties(arr: np.ndarray) -> np.ndarray:
    """Convert a 1-D array to average ranks, handling ties correctly.

    Example: [3, 1, 1, 2] → [4.0, 1.5, 1.5, 3.0]
    """
    n = len(arr)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Replace tied ranks with their average
    sorted_arr = arr[order]
    i = 0
    while i < n:
        j = i
        # Find the run of identical values
        while j < n - 1 and sorted_arr[j] == sorted_arr[j + 1]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient.

    Converts both arrays to ranks then computes Pearson on those ranks.
    Captures monotonic (not just linear) relationships.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) < 2:
        return 0.0

    return compute_pearson(_rank_with_ties(y_true), _rank_with_ties(y_pred))


def compute_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_sample: int = 5000,
    seed: int = 0,
) -> float:
    """Concordance Index (CI) for DTI evaluation.

    CI = P(pred[i] > pred[j] | true[i] > true[j])

    That is: what fraction of all pairs ordered by the true affinity are
    also ordered correctly by the predicted affinity?

    A CI of 1.0 means perfect ranking; 0.5 means random ranking.

    For large inputs (n > max_sample), a random subsample is drawn to keep
    computation feasible (the metric is stochastic in that case).

    This follows the formulation from:
        Özturk et al. (2018), DeepDTA; and the original C-statistic literature.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    n = len(y_true)
    if n < 2:
        return 0.0

    # Subsample when the dataset is very large to avoid O(n²) memory
    if n > max_sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_sample, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        n = max_sample

    # Broadcasting: diff_true[i,j] = y_true[i] - y_true[j]
    diff_true = y_true[:, None] - y_true[None, :]  # (n, n)
    diff_pred = y_pred[:, None] - y_pred[None, :]  # (n, n)

    # All pairs where the true affinity of i is strictly greater than j
    positive_pairs = diff_true > 0
    total = int(positive_pairs.sum())

    if total == 0:
        # All true values are identical — CI is undefined; return 0.5 (random)
        return 0.5

    concordant = int(np.sum(positive_pairs & (diff_pred > 0)))
    tied_pred = int(np.sum(positive_pairs & (diff_pred == 0)))

    return float((concordant + 0.5 * tied_pred) / total)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compute all DTI evaluation metrics in a single call.

    Automatically filters out NaN / Inf entries before computing.

    Returns:
        dict with keys: mse, rmse, pearson, spearman, ci
        Values are None if there are no valid entries to evaluate.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # Only evaluate on entries that are finite in both arrays
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {
            "mse": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "ci": None,
            "n_evaluated": 0,
        }

    return {
        "mse": compute_mse(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "pearson": compute_pearson(y_true, y_pred),
        "spearman": compute_spearman(y_true, y_pred),
        "ci": compute_ci(y_true, y_pred),
        "n_evaluated": int(len(y_true)),
    }
