"""Perturbation helpers for causal reliability scoring."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.c2dti.dataset_loader import DTIDataset


def perturb_dataset_interactions(
    dataset: DTIDataset,
    strength: float = 0.1,
    seed: int = 42,
) -> DTIDataset:
    """Create a perturbed dataset by masking a fraction of interactions.

    This is the first concrete causal stress test: if predictions stay stable
    after modest perturbation, the model receives a higher reliability score.
    """
    clipped_strength = float(min(1.0, max(0.0, strength)))
    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=dataset.interactions.shape) >= clipped_strength
    perturbed_interactions = (dataset.interactions * mask).astype(np.float32)

    return replace(
        dataset,
        interactions=perturbed_interactions,
        metadata={
            **dataset.metadata,
            "perturbation_strength": clipped_strength,
            "perturbation_seed": seed,
        },
    )