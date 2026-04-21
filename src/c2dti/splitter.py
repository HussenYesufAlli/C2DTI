"""Train/test split strategies for Drug-Target Interaction datasets.

Why splitting matters (beginner explanation):
---------------------------------------------
When we evaluate a model, we want to know how well it performs on pairs it
has NEVER seen during training.  If we evaluate on the same data we trained
on, the model can simply memorise the answers — which gives artificially
high metrics that won't hold in the real world.

A split divides all known drug-target pairs into two non-overlapping groups:
  - train set  : pairs the model is allowed to learn from
  - test set   : pairs the model is evaluated on (never seen during training)

Three split strategies are provided, matching those used in the DTI literature:

  random     : randomly assign a fraction of known pairs to the test set.
               Simple and fast; used when no drug/target structure is assumed.

  cold_drug  : test set contains drugs that never appear in training.
               Simulates the real-world scenario of predicting for a NEW drug.
               This is harder than random split (harder generalisation).

  cold_target: test set contains targets that never appear in training.
               Simulates predicting for a NEW protein/target.

MINDG, DeepDTA, and GraphDTA all report results under these three protocols,
so implementing them lets us produce directly comparable numbers.

Usage:
    from src.c2dti.splitter import split_dataset
    train_mask, test_mask = split_dataset(dataset, strategy="random", test_ratio=0.2, seed=42)
    # Both masks are boolean numpy arrays of shape (n_drugs, n_targets).
    # True  = this pair belongs to this split.
    # False = this pair does NOT belong to this split.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.c2dti.dataset_loader import DTIDataset


def _random_split(
    known_mask: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly assign known pairs to train and test sets.

    Steps:
      1. Find every position (i, j) where we have an observed affinity value.
      2. Shuffle those positions.
      3. The last `test_ratio` fraction goes to the test set; the rest to train.

    Returns:
        train_mask : bool array shape (n_drugs, n_targets)
        test_mask  : bool array shape (n_drugs, n_targets)
    """
    rng = np.random.RandomState(seed)
    known_indices = np.argwhere(known_mask)  # shape (n_known, 2)

    n_known = len(known_indices)
    n_test = max(1, int(round(n_known * test_ratio)))

    # Shuffle and split
    shuffled = rng.permutation(n_known)
    test_idx = known_indices[shuffled[-n_test:]]
    train_idx = known_indices[shuffled[:-n_test]]

    train_mask = np.zeros_like(known_mask, dtype=bool)
    test_mask = np.zeros_like(known_mask, dtype=bool)

    if len(train_idx) > 0:
        train_mask[train_idx[:, 0], train_idx[:, 1]] = True
    if len(test_idx) > 0:
        test_mask[test_idx[:, 0], test_idx[:, 1]] = True

    return train_mask, test_mask


def _cold_drug_split(
    known_mask: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by holding out entire drugs in the test set.

    Steps:
      1. Find all drugs that have at least one known interaction.
      2. Randomly select `test_ratio` of those drugs for the test set.
      3. All known pairs involving a test drug go to test; the rest go to train.
      4. Any drug that has NO known interactions is ignored (not in either split).

    This forces the model to generalise to drugs it has never seen — a much
    harder and more realistic evaluation scenario.

    Returns:
        train_mask, test_mask — same semantics as _random_split
    """
    rng = np.random.RandomState(seed)
    n_drugs = known_mask.shape[0]

    # Only consider drugs that appear in at least one known pair
    active_drugs = np.where(known_mask.any(axis=1))[0]
    n_test_drugs = max(1, int(round(len(active_drugs) * test_ratio)))

    shuffled = rng.permutation(len(active_drugs))
    test_drug_indices = active_drugs[shuffled[-n_test_drugs:]]
    train_drug_indices = active_drugs[shuffled[:-n_test_drugs]]

    test_drug_set = set(test_drug_indices.tolist())
    train_drug_set = set(train_drug_indices.tolist())

    train_mask = np.zeros_like(known_mask, dtype=bool)
    test_mask = np.zeros_like(known_mask, dtype=bool)

    for d in range(n_drugs):
        if d in test_drug_set:
            test_mask[d, :] = known_mask[d, :]
        elif d in train_drug_set:
            train_mask[d, :] = known_mask[d, :]

    return train_mask, test_mask


def _cold_target_split(
    known_mask: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by holding out entire targets in the test set.

    Mirror of _cold_drug_split but operates on the target (column) axis.
    Forces the model to generalise to proteins it has never seen.

    Returns:
        train_mask, test_mask — same semantics as _random_split
    """
    rng = np.random.RandomState(seed)
    n_targets = known_mask.shape[1]

    active_targets = np.where(known_mask.any(axis=0))[0]
    n_test_targets = max(1, int(round(len(active_targets) * test_ratio)))

    shuffled = rng.permutation(len(active_targets))
    test_target_indices = active_targets[shuffled[-n_test_targets:]]
    train_target_indices = active_targets[shuffled[:-n_test_targets]]

    test_target_set = set(test_target_indices.tolist())
    train_target_set = set(train_target_indices.tolist())

    train_mask = np.zeros_like(known_mask, dtype=bool)
    test_mask = np.zeros_like(known_mask, dtype=bool)

    for t in range(n_targets):
        if t in test_target_set:
            test_mask[:, t] = known_mask[:, t]
        elif t in train_target_set:
            train_mask[:, t] = known_mask[:, t]

    return train_mask, test_mask


def split_dataset(
    dataset: DTIDataset,
    strategy: str = "random",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split known drug-target pairs into train and test masks.

    Args:
        dataset    : DTIDataset to split; NaN entries in interactions are ignored.
        strategy   : one of "random", "cold_drug", "cold_target".
        test_ratio : fraction of known pairs (or drugs/targets) to hold out (0 < ratio < 1).
        seed       : random seed for reproducibility.

    Returns:
        (train_mask, test_mask) — each a boolean array of shape (n_drugs, n_targets).
        train_mask[i, j] = True  means pair (i, j) is in the training set.
        test_mask[i, j]  = True  means pair (i, j) is in the test set.
        These two masks are always disjoint (no pair is in both sets).

    Raises:
        ValueError: if strategy is not recognised or test_ratio is out of range.
    """
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be between 0 and 1 exclusive, got {test_ratio}")

    valid_strategies = {"random", "cold_drug", "cold_target"}
    normalized = strategy.strip().lower()
    if normalized not in valid_strategies:
        raise ValueError(
            f"Unknown split strategy: {strategy!r}. "
            f"Must be one of: {', '.join(sorted(valid_strategies))}"
        )

    # Build known-entry mask — only entries with finite values participate in the split
    known_mask = np.isfinite(dataset.interactions)

    if normalized == "random":
        return _random_split(known_mask, test_ratio, seed)
    elif normalized == "cold_drug":
        return _cold_drug_split(known_mask, test_ratio, seed)
    else:  # cold_target
        return _cold_target_split(known_mask, test_ratio, seed)
