"""Shared causal runtime helpers for both regression and binary runners.

This module keeps the 4-pillar causal implementation in one place so
continuous and binary execution paths stay behaviorally aligned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.c2dti.backbones import load_frozen_entity_embeddings
from src.c2dti.causal_objective import (
    compute_causal_reliability_score,
    compute_cross_view_causal_metrics,
    compute_irm_cf_losses,
    compute_mas_losses,
)
from src.c2dti.dti_model import create_predictor
from src.c2dti.perturbation import perturb_dataset_interactions
from src.c2dti.unified_scorer import UnifiedC2DTIScorer


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_runtime_path(raw_path: str) -> Path:
    """Resolve relative config paths against the C2DTI repository root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def compute_causal_outputs(
    *,
    dataset: Any,
    predictions: np.ndarray,
    train_mask: Optional[np.ndarray],
    causal_cfg: Dict[str, Any],
    perturbation_cfg: Optional[Dict[str, Any]] = None,
    predictor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute causal payload for summary.json across all causal modes.

    Args:
        dataset: Loaded dataset dataclass (regression or binary) with fields:
            drugs, targets, interactions, metadata.
        predictions: Model prediction matrix for the same dataset.
        train_mask: Optional boolean train mask used by split logic.
        causal_cfg: Full causal config mapping from YAML.
        perturbation_cfg: Optional perturbation settings mapping.
        predictor: Optional baseline predictor instance from the main run path.

    Returns:
        Dict containing summary-ready keys. Always includes:
          - "causal": structured metrics payload
          - "causal_score": scalar score in [0, 1] (best effort)
    """
    perturbation_cfg = perturbation_cfg or {}
    causal_weight = float(causal_cfg.get("weight", 0.0))
    causal_mode = str(causal_cfg.get("mode", "reliability")).strip().lower()

    perturbed_dataset = perturb_dataset_interactions(
        dataset,
        strength=perturbation_cfg.get("strength", 0.1),
        seed=perturbation_cfg.get("seed", 42),
    )

    if causal_mode == "cross_view":
        # Pillar 2: compare sequence and graph predictions before/after perturbation.
        seq_cfg = causal_cfg.get("sequence_model", {"name": "dual_frozen_backbone"})
        graph_cfg = causal_cfg.get("graph_model", {"name": "mixhop_propagation"})

        seq_predictor = create_predictor(seq_cfg)
        graph_predictor = create_predictor(graph_cfg)

        p_seq = seq_predictor.fit_predict(dataset, train_mask=train_mask)
        p_graph = graph_predictor.fit_predict(dataset, train_mask=train_mask)
        p_seq_pert = seq_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)
        p_graph_pert = graph_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)

        cross_view = compute_cross_view_causal_metrics(
            p_seq=p_seq,
            p_graph=p_graph,
            p_seq_pert=p_seq_pert,
            p_graph_pert=p_graph_pert,
            weight=causal_weight if causal_weight > 0 else 1.0,
        )
        return {
            "causal": {
                "mode": "cross_view",
                "sequence_model": str(seq_cfg.get("name", "dual_frozen_backbone")),
                "graph_model": str(graph_cfg.get("name", "mixhop_propagation")),
                "metrics": cross_view,
            },
            "causal_score": cross_view["causal_score"],
        }

    if causal_mode == "mas":
        # Pillar 3: reconstruct masked dimensions in each modality.
        mas_cfg = causal_cfg.get("mas_config", {})
        drug_npz = mas_cfg.get("drug_npz_path")
        prot_npz = mas_cfg.get("prot_npz_path")
        emb_dim = int(mas_cfg.get("embedding_dim", 768))
        mask_ratio = float(mas_cfg.get("mask_ratio", 0.15))
        mas_seed = int(mas_cfg.get("seed", 42))

        drug_npz_resolved = str(resolve_runtime_path(drug_npz)) if drug_npz else None
        prot_npz_resolved = str(resolve_runtime_path(prot_npz)) if prot_npz else None

        drug_embeddings = load_frozen_entity_embeddings(
            entities=dataset.drugs,
            npz_path=drug_npz_resolved,
            default_dim=emb_dim,
        )
        prot_embeddings = load_frozen_entity_embeddings(
            entities=dataset.targets,
            npz_path=prot_npz_resolved,
            default_dim=emb_dim,
        )

        mas_metrics = compute_mas_losses(
            drug_embeddings=drug_embeddings,
            prot_embeddings=prot_embeddings,
            mask_ratio=mask_ratio,
            seed=mas_seed,
            weight=causal_weight if causal_weight > 0 else 1.0,
        )
        return {
            "causal": {
                "mode": "mas",
                "embedding_dim": emb_dim,
                "mask_ratio": mask_ratio,
                "n_drugs": len(dataset.drugs),
                "n_targets": len(dataset.targets),
                "metrics": mas_metrics,
            },
            "causal_score": mas_metrics["mas_score"],
        }

    if causal_mode == "irm_cf":
        # Pillar 4: invariant risk + counterfactual rejection.
        irm_cf_cfg = causal_cfg.get("irm_cf_config", {})
        n_envs = int(irm_cf_cfg.get("n_envs", 4))
        pos_threshold = float(irm_cf_cfg.get("pos_threshold", 0.5))
        n_cf_pairs = int(irm_cf_cfg.get("n_cf_pairs", 1000))
        irm_weight = float(irm_cf_cfg.get("irm_weight", 1.0))
        cf_weight = float(irm_cf_cfg.get("cf_weight", 1.0))
        irm_cf_seed = int(irm_cf_cfg.get("seed", 42))

        flat_preds = predictions.ravel()
        flat_labels = dataset.interactions.ravel()

        irm_cf_metrics = compute_irm_cf_losses(
            predictions=flat_preds,
            labels=flat_labels,
            n_drugs=len(dataset.drugs),
            n_targets=len(dataset.targets),
            n_envs=n_envs,
            pos_threshold=pos_threshold,
            n_cf_pairs=n_cf_pairs,
            irm_weight=irm_weight,
            cf_weight=cf_weight,
            seed=irm_cf_seed,
        )
        return {
            "causal": {
                "mode": "irm_cf",
                "n_envs": n_envs,
                "pos_threshold": pos_threshold,
                "n_cf_pairs_requested": n_cf_pairs,
                "metrics": irm_cf_metrics,
            },
            "causal_score": irm_cf_metrics["irm_cf_score"],
        }

    if causal_mode == "unified":
        # Unified mode executes active Pillar 2/3/4 terms in one objective.
        scorer = UnifiedC2DTIScorer(causal_cfg)

        seq_cfg = causal_cfg.get("sequence_model", {"name": "dual_frozen_backbone"})
        graph_cfg = causal_cfg.get("graph_model", {"name": "mixhop_propagation"})
        seq_predictor = create_predictor(seq_cfg)
        graph_predictor = create_predictor(graph_cfg)
        p_seq = seq_predictor.fit_predict(dataset, train_mask=train_mask)
        p_graph = graph_predictor.fit_predict(dataset, train_mask=train_mask)
        p_seq_pert = seq_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)
        p_graph_pert = graph_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)

        mas_cfg = causal_cfg.get("mas_config", {})
        emb_dim = int(mas_cfg.get("embedding_dim", 768))
        drug_npz = mas_cfg.get("drug_npz_path")
        prot_npz = mas_cfg.get("prot_npz_path")
        drug_npz_resolved = str(resolve_runtime_path(drug_npz)) if drug_npz else None
        prot_npz_resolved = str(resolve_runtime_path(prot_npz)) if prot_npz else None
        drug_embeddings = load_frozen_entity_embeddings(
            dataset.drugs,
            npz_path=drug_npz_resolved,
            default_dim=emb_dim,
        )
        prot_embeddings = load_frozen_entity_embeddings(
            dataset.targets,
            npz_path=prot_npz_resolved,
            default_dim=emb_dim,
        )

        unified = scorer.score(
            predictions=predictions,
            labels=dataset.interactions,
            n_drugs=len(dataset.drugs),
            n_targets=len(dataset.targets),
            seq_predictions=p_seq,
            graph_predictions=p_graph,
            seq_predictions_pert=p_seq_pert,
            graph_predictions_pert=p_graph_pert,
            drug_embeddings=drug_embeddings,
            prot_embeddings=prot_embeddings,
        )
        return {
            "causal": {
                "mode": "unified",
                "lambdas": {
                    "xview": scorer.lambda_xview,
                    "mas": scorer.lambda_mas,
                    "irm": scorer.lambda_irm,
                    "cf": scorer.lambda_cf,
                },
                "metrics": unified,
            },
            "causal_score": unified["unified_causal_score"],
        }

    # Reliability mode is the default safety fallback for unknown/legacy modes.
    # Use the same predictor class as the main path when provided to avoid
    # accidental behavior drift between primary prediction and reliability mode.
    reliability_predictor = predictor if predictor is not None else create_predictor({"name": "simple_baseline"})
    perturbed_predictions = reliability_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)
    reliability = compute_causal_reliability_score(
        baseline_predictions=predictions,
        perturbed_predictions=perturbed_predictions,
        weight=causal_weight if causal_weight > 0 else 1.0,
    )
    return {
        "causal": {
            "mode": "reliability",
            "metric": "prediction_stability",
        },
        "causal_score": reliability,
    }
