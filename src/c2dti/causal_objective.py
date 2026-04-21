"""Causal objective helpers for the C2DTI pipeline."""

from typing import Dict, Any, Optional

import numpy as np


def validate_causal_config(causal_cfg: Optional[Dict[str, Any]]) -> list:
    """
    Validate causal configuration keys and values.
    
    Args:
        causal_cfg: Causal configuration dict or None.
    
    Returns:
        List of error strings. Empty list if valid.
    """
    errors = []
    
    if causal_cfg is None:
        return errors
    
    if not isinstance(causal_cfg, dict):
        errors.append("causal config must be a mapping or null")
        return errors
    
    enabled = causal_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append("causal.enabled must be a boolean")
    
    weight = causal_cfg.get("weight", 0.0)
    if not isinstance(weight, (int, float)) or weight < 0.0:
        errors.append("causal.weight must be a non-negative number")
    
    return errors


def compute_causal_score(
    enabled: bool = False,
    weight: float = 0.0
) -> Optional[float]:
    """
    Compute a placeholder causal consistency score.
    
    Purpose:
      This is a minimal placeholder that will be extended later with
      real causal computations (e.g., cross-view agreement, perturbation effects).
    
    Args:
        enabled: Whether causal objective is active.
        weight: Weight/importance of causal term (future use).
    
    Returns:
        A placeholder causal score if enabled, otherwise None.
    """
    if not enabled:
        return None
    
    # Placeholder: in real implementation, this will compute cross-view
    # causal agreement, perturbation robustness, or similar metrics.
    placeholder_score = 0.5
    
    return placeholder_score


def compute_causal_reliability_score(
    baseline_predictions: np.ndarray,
    perturbed_predictions: np.ndarray,
    weight: float = 1.0,
) -> float:
    """Measure how stable predictions remain after perturbation.

    A smaller change between baseline and perturbed predictions means the
    predictor is more reliable under intervention-style stress.
    """
    if baseline_predictions.shape != perturbed_predictions.shape:
        raise ValueError("baseline and perturbed predictions must have the same shape")

    if baseline_predictions.size == 0:
        return 0.0

    # Use mean absolute change as the instability term, then convert it to a
    # bounded reliability score in [0, 1].
    mean_abs_shift = float(np.mean(np.abs(baseline_predictions - perturbed_predictions)))
    scaled_shift = min(1.0, max(0.0, mean_abs_shift * max(weight, 0.0)))
    return float(1.0 - scaled_shift)
