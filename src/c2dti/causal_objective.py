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

    mode = causal_cfg.get("mode", "reliability")
    if not isinstance(mode, str) or mode.strip().lower() not in {"reliability", "cross_view", "mas", "irm_cf", "unified"}:
        errors.append("causal.mode must be one of: reliability, cross_view, mas, irm_cf, unified")

    # Optional nested model configs used only in cross_view mode.
    for key in ("sequence_model", "graph_model"):
        if key in causal_cfg and not isinstance(causal_cfg.get(key), dict):
            errors.append(f"causal.{key} must be a mapping when provided")
    
    return errors


def compute_causal_score(
    enabled: bool = False,
    weight: float = 0.0
) -> Optional[float]:
    """
    Compute a baseline causal consistency score.
    
    Purpose:
            This is a minimal baseline that will be extended later with
            full causal computations (e.g., cross-view agreement, perturbation effects).
    
    Args:
        enabled: Whether causal objective is active.
        weight: Weight/importance of causal term (future use).
    
    Returns:
        A baseline causal score if enabled, otherwise None.
    """
    if not enabled:
        return None
    
    # Baseline stub: full implementation will compute cross-view
    # causal agreement, perturbation robustness, or similar metrics.
    baseline_score = 0.5
    
    return baseline_score


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


def compute_cross_view_causal_metrics(
    p_seq: np.ndarray,
    p_graph: np.ndarray,
    p_seq_pert: np.ndarray,
    p_graph_pert: np.ndarray,
    weight: float = 1.0,
) -> Dict[str, float]:
    """Compute cross-view causal agreement terms from the 4-pillar objective.

    Implements the three consistency terms used in Pillar 2:
      L_xview = MSE(p_seq, p_graph)
             + MSE(p_seq_pert, p_graph)
             + MSE(p_seq, p_graph_pert)

    Returns each component and a bounded causal score in [0, 1].
    """
    shapes = {p_seq.shape, p_graph.shape, p_seq_pert.shape, p_graph_pert.shape}
    if len(shapes) != 1:
        raise ValueError("All cross-view prediction matrices must have identical shape")

    if p_seq.size == 0:
        return {
            "mse_seq_graph": 0.0,
            "mse_seqpert_graph": 0.0,
            "mse_seq_graphpert": 0.0,
            "l_xview": 0.0,
            "l_xview_weighted": 0.0,
            "causal_score": 0.0,
        }

    mse_seq_graph = float(np.mean((p_seq - p_graph) ** 2))
    mse_seqpert_graph = float(np.mean((p_seq_pert - p_graph) ** 2))
    mse_seq_graphpert = float(np.mean((p_seq - p_graph_pert) ** 2))

    l_xview = mse_seq_graph + mse_seqpert_graph + mse_seq_graphpert
    weighted = float(max(weight, 0.0)) * l_xview

    # Convert disagreement loss into bounded agreement score.
    # Higher agreement (lower loss) -> score closer to 1.
    causal_score = float(1.0 / (1.0 + weighted))

    return {
        "mse_seq_graph": mse_seq_graph,
        "mse_seqpert_graph": mse_seqpert_graph,
        "mse_seq_graphpert": mse_seq_graphpert,
        "l_xview": float(l_xview),
        "l_xview_weighted": weighted,
        "causal_score": causal_score,
    }


def compute_mas_losses(
    drug_embeddings: np.ndarray,
    prot_embeddings: np.ndarray,
    mask_ratio: float = 0.15,
    seed: int = 42,
    weight: float = 1.0,
) -> Dict[str, float]:
    """Compute Masked AutoEncoder Self-supervision losses — Pillar 3 of C2DTI.

    Beginner-friendly explanation:
      We measure how much structural information the embeddings carry by asking:
      'If I hide 15% of the dimensions, can I reconstruct them from the rest?'
      Low reconstruction error = embeddings are rich and structured (good).
      High reconstruction error = embeddings are noisy/sparse (bad).

    This is computed separately for drug and protein embeddings, then combined.

    Formula (from the C2DTI 4-Pillar paper plan):
      L_MAS_drug = MSE( reconstruct(e_drug[M_drug]), e_drug[M_drug] )
      L_MAS_prot = MSE( reconstruct(e_prot[M_prot]), e_prot[M_prot] )
      L_MAS      = L_MAS_drug + L_MAS_prot
      mas_score  = 1 / (1 + weight * L_MAS)

    We use a numpy least-squares decoder: no GPU or PyTorch required.

    Args:
        drug_embeddings: (n_drugs, d) frozen drug embedding matrix.
        prot_embeddings: (n_prots, d) frozen protein embedding matrix.
        mask_ratio: Fraction of dims to mask per modality (default 0.15).
        seed: Random seed for reproducibility (drug uses seed, prot uses seed+1).
        weight: How much to scale the loss when computing mas_score.

    Returns:
        Dict with keys: mas_drug_loss, mas_prot_loss, l_mas, l_mas_weighted, mas_score.
    """
    # Import here to avoid circular dependency: causal_objective imports backbones.
    from src.c2dti.backbones import MASHead

    # --- Drug modality MAS ---
    # Seed is fixed so the mask is reproducible across runs.
    head_drug = MASHead(mask_ratio=mask_ratio, seed=seed)
    head_drug.fit(drug_embeddings)
    mas_drug = head_drug.reconstruct_loss(drug_embeddings)

    # --- Protein modality MAS ---
    # Use seed+1 to get a different mask from the drug head.
    head_prot = MASHead(mask_ratio=mask_ratio, seed=seed + 1)
    head_prot.fit(prot_embeddings)
    mas_prot = head_prot.reconstruct_loss(prot_embeddings)

    l_mas = mas_drug + mas_prot
    weighted = float(max(weight, 0.0)) * l_mas

    # Convert loss into a bounded score in [0, 1].
    # Lower MAS loss → score closer to 1 → better embedding quality.
    mas_score = float(1.0 / (1.0 + weighted))

    return {
        "mas_drug_loss": mas_drug,
        "mas_prot_loss": mas_prot,
        "l_mas": float(l_mas),
        "l_mas_weighted": weighted,
        "mas_score": mas_score,
    }


def compute_irm_cf_losses(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_drugs: int,
    n_targets: int,
    n_envs: int = 4,
    pos_threshold: float = 0.5,
    n_cf_pairs: int = 1000,
    irm_weight: float = 1.0,
    cf_weight: float = 1.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute Pillar 4: IRM penalty + Counterfactual loss.

    Beginner-friendly explanation:

      IRM part:
        We split drugs into `n_envs` groups. If the model truly learned causal
        relationships, its prediction error should be similar in all groups.
        Variance across group errors = IRM penalty. Lower = more invariant model.

      Counterfactual part:
        For each positive drug-target pair, we swap the target for a random one.
                The model should score this counterfactual pair low. If it scores it high, it's using
        a drug-level shortcut ('drug X always binds strongly') rather than learning
        the specific drug-target interaction.

      Combined score:
        irm_cf_score = 1 / (1 + irm_weight * L_IRM + cf_weight * L_CF)

    Args:
        predictions:  1D array of predicted scores (flattened over known pairs).
        labels:       1D array of true labels (same length as predictions).
        n_drugs:      Number of unique drugs in the dataset.
        n_targets:    Number of unique targets in the dataset.
        n_envs:       Number of drug-split IRM environments.
        pos_threshold: Label threshold for a pair to be considered positive.
        n_cf_pairs:   Max counterfactual pairs to sample.
        irm_weight:   Scaling weight for L_IRM in the combined score.
        cf_weight:    Scaling weight for L_CF in the combined score.
        seed:         Random seed.

    Returns:
        Dict with: l_irm, l_cf, env_mses, n_cf_sampled, irm_cf_score.
    """
    from src.c2dti.irm_loss import compute_irm_penalty, compute_counterfactual_loss

    # Build flat drug/target index arrays that correspond to the predictions vector.
    # Assumes predictions are ordered as: row 0 all targets, row 1 all targets, ...
    # i.e. the same order as np.ravel() on the interaction matrix.
    drug_indices  = np.repeat(np.arange(n_drugs),  n_targets).astype(np.int32)
    target_indices = np.tile(np.arange(n_targets), n_drugs).astype(np.int32)

    # Truncate index arrays to match the actual number of predictions
    # (in case only known/observed pairs were predicted).
    n_pairs = len(predictions)
    drug_indices  = drug_indices[:n_pairs]
    target_indices = target_indices[:n_pairs]

    irm_result = compute_irm_penalty(
        predictions=predictions,
        labels=labels,
        drug_indices=drug_indices,
        target_indices=target_indices,
        n_envs=n_envs,
        seed=seed,
    )

    cf_result = compute_counterfactual_loss(
        predictions=predictions,
        labels=labels,
        drug_indices=drug_indices,
        target_indices=target_indices,
        n_drugs=n_drugs,
        n_targets=n_targets,
        pos_threshold=pos_threshold,
        n_cf_pairs=n_cf_pairs,
        seed=seed,
    )

    l_irm = float(irm_result["l_irm"])
    l_cf  = float(cf_result["l_cf"])

    combined = float(max(irm_weight, 0.0)) * l_irm + float(max(cf_weight, 0.0)) * l_cf
    irm_cf_score = float(1.0 / (1.0 + combined))

    return {
        "l_irm": l_irm,
        "l_cf": l_cf,
        "env_mses": irm_result["env_mses"],
        "n_cf_sampled": cf_result["n_cf_sampled"],
        "irm_cf_score": irm_cf_score,
    }
