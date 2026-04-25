"""Utility helpers for simple DTI feature and matrix summaries."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def build_string_feature_matrix(items: List[str], vector_size: int = 16) -> np.ndarray:
    """Convert strings into simple normalized hashed feature vectors.

    This keeps the first dataset-backed pipeline lightweight while still giving the
    model a deterministic numeric representation for drugs and targets.
    """
    if not items:
        return np.zeros((0, vector_size), dtype=np.float32)

    features = np.zeros((len(items), vector_size), dtype=np.float32)
    for row_index, item in enumerate(items):
        text = item or ""
        if not text:
            continue

        # Hash each character into a fixed-size vector so variable-length
        # identifiers or sequences become a stable numeric representation.
        for char in text:
            features[row_index, ord(char) % vector_size] += 1.0

        norm = np.linalg.norm(features[row_index])
        if norm > 0:
            features[row_index] /= norm

    return features


def summarize_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """Return compact summary statistics for a prediction or interaction matrix."""
    if matrix.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "mean": float(np.mean(matrix)),
        "std": float(np.std(matrix)),
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
    }
