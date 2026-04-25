"""IRM (Invariant Risk Minimization) and Counterfactual loss helpers — Pillar 4.

Beginner-friendly overview
---------------------------
Pillar 4 has two parts:

1. IRM — Invariant Risk Minimization
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Imagine we split all drugs into N groups (called "environments").
   In environment 0 we have drug group 0, in environment 1 drug group 1, etc.
   For each environment we compute the prediction error (MSE).
   IRM penalty = variance of those per-environment errors.

   Why does this matter?
     - If the model is truly learning causal drug-target relationships, it should
       perform equally well across all drug groups (low variance = invariant).
     - If one environment has much higher error, the model is exploiting some
       spurious correlation that only holds for certain drugs.

   Formula:
     L_IRM = Var_e [ MSE(predictions_e, labels_e) ]

2. Counterfactual Loss
   ~~~~~~~~~~~~~~~~~~~~~~
   For each POSITIVE pair (drug_i, target_j, y=1), we create a "counterfactual"
   pair by swapping target_j for a RANDOM target_j' where j' ≠ j.
   This counterfactual pair is labelled y=0 (hard negative).
   The model should score these pairs near 0.

   Formula:
     CF positions = indices in the prediction matrix for (drug_i, target_j') pairs
     L_CF = mean( predictions[ CF_positions ] )   ← we want this to be 0

   Why does this matter?
     It forces the model to NOT generalise from drug_i → any target: it should
     only score high when the specific drug-target pair is biologically relevant.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_irm_penalty(
    predictions: np.ndarray,
    labels: np.ndarray,
    drug_indices: np.ndarray,
    target_indices: np.ndarray,
    n_envs: int = 4,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute the numpy IRM penalty across drug-split environments.

    How it works:
      1. Sort unique drugs and split them into n_envs equally-sized groups.
      2. For each group (environment), collect the (drug_i, target_j) pairs
         that belong to that environment (based on drug_i membership).
      3. Compute MSE(predictions, labels) within that environment.
      4. Return the variance of the n_envs per-environment MSEs as the IRM penalty.

    A low variance means the model is equally accurate across all drug groups
    → the representations are invariant to which drug cluster we're in.

    Args:
        predictions:   1D array of predicted interaction scores.
        labels:        1D array of true interaction labels (same length).
        drug_indices:  1D int array — row index of each pair (which drug).
        target_indices: 1D int array — col index of each pair (which target).
        n_envs:        Number of drug-split environments (default 4).
        seed:          Random seed for reproducible drug–environment assignment.

    Returns:
        Dict with keys: env_mses (per-env MSE list), l_irm (variance), per_env_n (sample counts).
    """
    if len(predictions) == 0:
        return {"env_mses": [], "l_irm": 0.0, "per_env_n": []}

    # Assign each unique drug index to an environment deterministically.
    unique_drugs = np.unique(drug_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_drugs)  # shuffle so assignment is random but reproducible
    env_assignment = {int(d): int(i % n_envs) for i, d in enumerate(unique_drugs)}

    env_mses: List[float] = []
    per_env_n: List[int] = []

    for env_id in range(n_envs):
        # Select pairs whose drug belongs to this environment.
        env_mask = np.array([env_assignment[int(d)] == env_id for d in drug_indices])
        if env_mask.sum() == 0:
            # If an environment has no samples, skip it (can happen with small datasets).
            continue

        p_env = predictions[env_mask].astype(np.float64)
        y_env = labels[env_mask].astype(np.float64)
        mse_e = float(np.mean((p_env - y_env) ** 2))
        env_mses.append(mse_e)
        per_env_n.append(int(env_mask.sum()))

    if len(env_mses) < 2:
        # Need at least 2 environments to compute variance.
        irm_penalty = 0.0
    else:
        irm_penalty = float(np.var(env_mses))

    return {
        "env_mses": env_mses,
        "l_irm": irm_penalty,
        "per_env_n": per_env_n,
    }


def compute_counterfactual_loss(
    predictions: np.ndarray,
    labels: np.ndarray,
    drug_indices: np.ndarray,
    target_indices: np.ndarray,
    n_drugs: int,
    n_targets: int,
    pos_threshold: float = 0.5,
    n_cf_pairs: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute the counterfactual loss for Pillar 4.

    For each positive pair (drug_i, target_j, y=1):
      - Pick a random different target j' where (drug_i, target_j') is not positive.
      - Record the model's prediction for (drug_i, target_j').
      - The model should score this near 0 — we want to discourage it from
        saying 'drug_i binds anything', which would be a spurious shortcut.

    L_CF = mean( predictions at CF positions )

    Lower L_CF = model is correctly distinguishing real positives from
    counterfactual hard negatives.

    Args:
        predictions:    1D array of predicted interaction scores (all known pairs).
        labels:         1D array of true interaction labels.
        drug_indices:   1D int array — row index of each pair.
        target_indices: 1D int array — col index of each pair.
        n_drugs:        Total number of unique drugs in the dataset.
        n_targets:      Total number of unique targets in the dataset.
        pos_threshold:  Label value above which a pair is considered positive.
        n_cf_pairs:     Maximum number of CF pairs to sample (for speed).
        seed:           Random seed.

    Returns:
        Dict with keys: l_cf (mean prediction at CF positions), n_cf_sampled.
    """
    if len(predictions) == 0:
        return {"l_cf": 0.0, "n_cf_sampled": 0}

    rng = np.random.default_rng(seed)

    # Build a lookup: (drug_i, target_j) → prediction score.
    pair_to_pred = {
        (int(d), int(t)): float(p)
        for d, t, p in zip(drug_indices, target_indices, predictions)
    }

    # Build the set of known positive pairs.
    positive_pairs: List[Tuple[int, int]] = [
        (int(d), int(t))
        for d, t, y in zip(drug_indices, target_indices, labels)
        if float(y) > pos_threshold
    ]

    if len(positive_pairs) == 0:
        return {"l_cf": 0.0, "n_cf_sampled": 0}

    # Sample up to n_cf_pairs positive pairs to create CF negatives from.
    max_sample = min(len(positive_pairs), n_cf_pairs)
    sampled_indices = rng.choice(len(positive_pairs), max_sample, replace=False)
    sampled_positives = [positive_pairs[i] for i in sampled_indices]

    cf_preds: List[float] = []
    all_targets = list(range(n_targets))

    for drug_i, target_j in sampled_positives:
        # Pick a random target_j' different from target_j.
        other_targets = [t for t in all_targets if t != target_j]
        if not other_targets:
            continue
        target_j_prime = int(rng.choice(other_targets))
        cf_key = (drug_i, target_j_prime)

        # Only proceed if the CF pair exists in our known pairs.
        # (The interaction matrix may not be fully observed.)
        if cf_key in pair_to_pred:
            cf_preds.append(pair_to_pred[cf_key])

    if len(cf_preds) == 0:
        return {"l_cf": 0.0, "n_cf_sampled": 0}

    l_cf = float(np.mean(cf_preds))
    return {"l_cf": l_cf, "n_cf_sampled": len(cf_preds)}
