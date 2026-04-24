"""Binary classification metrics for DTI.

This file is independent from regression evaluation so existing CI/RMSE paths
remain untouched.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    # Prevent division-by-zero and keep metric APIs numerically stable.
    if abs(float(denominator)) < 1e-12:
        return float(default)
    return float(numerator) / float(denominator)


def _binary_confusion_counts(
    y_true: np.ndarray,
    y_pred_label: np.ndarray,
) -> Tuple[int, int, int, int]:
    # Compute TP/FP/TN/FN once so all confusion-derived metrics share counts.
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred_label = np.asarray(y_pred_label, dtype=np.int64).ravel()

    tp = int(np.sum((y_true == 1) & (y_pred_label == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_label == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_label == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred_label == 0)))
    return tp, fp, tn, fn


def compute_accuracy(y_true: np.ndarray, y_pred_label: np.ndarray) -> float:
    # Accuracy measures the fraction of total predictions that are correct.
    tp, fp, tn, fn = _binary_confusion_counts(y_true, y_pred_label)
    return _safe_divide(tp + tn, tp + fp + tn + fn)


def compute_precision(y_true: np.ndarray, y_pred_label: np.ndarray) -> float:
    # Precision answers: among predicted positives, how many are truly positive?
    tp, fp, _, _ = _binary_confusion_counts(y_true, y_pred_label)
    return _safe_divide(tp, tp + fp)


def compute_recall(y_true: np.ndarray, y_pred_label: np.ndarray) -> float:
    # Recall (sensitivity) answers: among true positives, how many are found?
    tp, _, _, fn = _binary_confusion_counts(y_true, y_pred_label)
    return _safe_divide(tp, tp + fn)


def compute_specificity(y_true: np.ndarray, y_pred_label: np.ndarray) -> float:
    # Specificity answers: among true negatives, how many are correctly rejected?
    _, fp, tn, _ = _binary_confusion_counts(y_true, y_pred_label)
    return _safe_divide(tn, tn + fp)


def compute_f1(y_true: np.ndarray, y_pred_label: np.ndarray) -> float:
    # F1 harmonically combines precision and recall into one balanced score.
    precision = compute_precision(y_true, y_pred_label)
    recall = compute_recall(y_true, y_pred_label)
    return _safe_divide(2.0 * precision * recall, precision + recall)


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Compute AUROC from rank statistics without third-party dependencies.
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    # Assign average ranks for tied scores to keep AUROC unbiased.
    i = 0
    rank_pos = 1.0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank_pos + (rank_pos + (j - i))) / 2.0
        ranks[i : j + 1] = avg_rank
        rank_pos += (j - i + 1)
        i = j + 1

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ranks_original = ranks[inv]

    sum_ranks_pos = float(np.sum(ranks_original[y_true == 1]))
    u_stat = sum_ranks_pos - (pos * (pos + 1) / 2.0)
    return _safe_divide(u_stat, pos * neg, default=0.5)


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Compute area under precision-recall curve using step-wise integration.
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / float(pos)

    # Add endpoints so integration starts at recall 0.
    precision = np.concatenate(([1.0], precision.astype(np.float64)))
    recall = np.concatenate(([0.0], recall.astype(np.float64)))

    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """Evaluate binary labels and scores with classification metrics.

    Args:
        y_true: Ground-truth binary labels in {0,1}.
        y_score: Model probabilities/scores in [0,1].
        threshold: Decision threshold to convert scores into class labels.
    """

    # Keep only finite pairs and coerce labels into strict binary space.
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(y_true) == 0:
        return {
            "auroc": None,
            "auprc": None,
            "f1": None,
            "accuracy": None,
            "sensitivity": None,
            "specificity": None,
            "precision": None,
            "threshold": float(threshold),
            "n_evaluated": 0,
            "n_positive": 0,
            "n_negative": 0,
        }

    y_true_bin = (y_true >= 0.5).astype(np.int64)
    y_pred_label = (y_score >= float(threshold)).astype(np.int64)

    n_pos = int(np.sum(y_true_bin == 1))
    n_neg = int(np.sum(y_true_bin == 0))

    return {
        "auroc": compute_auroc(y_true_bin, y_score),
        "auprc": compute_auprc(y_true_bin, y_score),
        "f1": compute_f1(y_true_bin, y_pred_label),
        "accuracy": compute_accuracy(y_true_bin, y_pred_label),
        "sensitivity": compute_recall(y_true_bin, y_pred_label),
        "specificity": compute_specificity(y_true_bin, y_pred_label),
        "precision": compute_precision(y_true_bin, y_pred_label),
        "threshold": float(threshold),
        "n_evaluated": int(len(y_true_bin)),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }
