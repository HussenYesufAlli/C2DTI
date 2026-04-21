"""
Causal objective module for C2DTI.

Purpose:
  Provides an optional causal consistency objective that can be enabled
  in configuration without breaking baseline behavior. When disabled,
  the baseline runs unchanged. When enabled, computes a placeholder
  causal consistency score and logs it.

Non-breaking design:
  - Baseline behavior is preserved when causal objective is disabled.
  - Can be extended later with real causal computations.
"""

from typing import Dict, Any, Optional


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
