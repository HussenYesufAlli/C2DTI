"""Unified C2DTI causal scorer — Phase 5.

This module combines all 4 pillars of the C2DTI causal objective into a
single, unified scoring step. Think of it as the "final boss" module that
calls all the individual pillar functions and combines them into one number.

Beginner-friendly overview
---------------------------
The 4-pillar total loss formula is:

  L_total = L_BCE
          + lambda_xview * L_XVIEW    (Pillar 2: cross-view agreement)
          + lambda_mas   * L_MAS      (Pillar 3: masked autoencoder reconstruction)
          + lambda_irm   * L_IRM      (Pillar 4: invariant risk minimization)
          + lambda_cf    * L_CF       (Pillar 4: counterfactual rejection)

  unified_causal_score = 1 / (1 + L_total)

Each pillar term is optional — you can disable any one by setting its lambda to 0.
This lets us run ablation studies: full model vs -xview vs -mas vs -irm vs -cf.

The UnifiedC2DTIScorer class orchestrates all this in a clean interface so that
runner.py just has one function call for the entire causal objective.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class UnifiedC2DTIScorer:
    """Combines all 4 C2DTI causal pillars into one unified objective.

    Usage:
        scorer = UnifiedC2DTIScorer(cfg)
        result = scorer.score(
            dataset        = dataset,
            predictions    = predictions,
            seq_predictions  = p_seq,
            graph_predictions = p_graph,
            seq_predictions_pert   = p_seq_pert,
            graph_predictions_pert = p_graph_pert,
            drug_embeddings = drug_embs,
            prot_embeddings = prot_embs,
        )
        # result["unified_causal_score"] is the main number to report.

    Args:
        cfg: The full causal config dict from the YAML file.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        # Lambda weights for each pillar (how much each term contributes).
        self.lambda_xview = float(cfg.get("lambda_xview", 1.0))
        self.lambda_mas   = float(cfg.get("lambda_mas",   1.0))
        self.lambda_irm   = float(cfg.get("lambda_irm",   1.0))
        self.lambda_cf    = float(cfg.get("lambda_cf",    1.0))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_drugs: int,
        n_targets: int,
        # Pillar 2 inputs (optional: needed only when lambda_xview > 0)
        seq_predictions: Optional[np.ndarray] = None,
        graph_predictions: Optional[np.ndarray] = None,
        seq_predictions_pert: Optional[np.ndarray] = None,
        graph_predictions_pert: Optional[np.ndarray] = None,
        # Pillar 3 inputs (optional: needed only when lambda_mas > 0)
        drug_embeddings: Optional[np.ndarray] = None,
        prot_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run all active pillars and return the unified causal score.

        Returns a dict with all per-pillar metrics plus the combined score.
        Any pillar whose lambda is 0 is skipped and its loss is 0.
        """
        result: Dict[str, Any] = {
            "lambda_xview": self.lambda_xview,
            "lambda_mas":   self.lambda_mas,
            "lambda_irm":   self.lambda_irm,
            "lambda_cf":    self.lambda_cf,
        }

        flat_preds  = predictions.ravel()
        flat_labels = labels.ravel()

        # ----------------------------------------------------------------
        # Pillar 2 — Cross-View Causal Agreement
        # ----------------------------------------------------------------
        xview_cfg  = self.cfg.get("cross_view_config", {})
        xview_weight = float(xview_cfg.get("weight", 1.0))

        if self.lambda_xview > 0 and all(
            a is not None for a in [
                seq_predictions, graph_predictions,
                seq_predictions_pert, graph_predictions_pert,
            ]
        ):
            from src.c2dti.causal_objective import compute_cross_view_causal_metrics
            xview = compute_cross_view_causal_metrics(
                p_seq=seq_predictions,
                p_graph=graph_predictions,
                p_seq_pert=seq_predictions_pert,
                p_graph_pert=graph_predictions_pert,
                weight=xview_weight,
            )
            result["xview"] = xview
            l_xview = xview["l_xview"]
        else:
            result["xview"] = None
            l_xview = 0.0

        # ----------------------------------------------------------------
        # Pillar 3 — MAS Self-Supervision
        # ----------------------------------------------------------------
        mas_cfg = self.cfg.get("mas_config", {})
        mask_ratio = float(mas_cfg.get("mask_ratio", 0.15))
        mas_seed   = int(mas_cfg.get("seed", 42))
        mas_weight = float(mas_cfg.get("weight", 1.0))

        if self.lambda_mas > 0 and drug_embeddings is not None and prot_embeddings is not None:
            from src.c2dti.causal_objective import compute_mas_losses
            mas = compute_mas_losses(
                drug_embeddings=drug_embeddings,
                prot_embeddings=prot_embeddings,
                mask_ratio=mask_ratio,
                seed=mas_seed,
                weight=mas_weight,
            )
            result["mas"] = mas
            l_mas = mas["l_mas"]
        else:
            result["mas"] = None
            l_mas = 0.0

        # ----------------------------------------------------------------
        # Pillar 4 — IRM + Counterfactual
        # ----------------------------------------------------------------
        irm_cf_cfg    = self.cfg.get("irm_cf_config", {})
        n_envs        = int(irm_cf_cfg.get("n_envs", 4))
        pos_threshold = float(irm_cf_cfg.get("pos_threshold", 0.5))
        n_cf_pairs    = int(irm_cf_cfg.get("n_cf_pairs", 1000))
        irm_weight_inner = float(irm_cf_cfg.get("irm_weight", 1.0))
        cf_weight_inner  = float(irm_cf_cfg.get("cf_weight", 1.0))
        irm_cf_seed   = int(irm_cf_cfg.get("seed", 42))

        if self.lambda_irm > 0 or self.lambda_cf > 0:
            from src.c2dti.causal_objective import compute_irm_cf_losses
            irm_cf = compute_irm_cf_losses(
                predictions=flat_preds,
                labels=flat_labels,
                n_drugs=n_drugs,
                n_targets=n_targets,
                n_envs=n_envs,
                pos_threshold=pos_threshold,
                n_cf_pairs=n_cf_pairs,
                irm_weight=irm_weight_inner,
                cf_weight=cf_weight_inner,
                seed=irm_cf_seed,
            )
            result["irm_cf"] = irm_cf
            l_irm = irm_cf["l_irm"] if self.lambda_irm > 0 else 0.0
            l_cf  = irm_cf["l_cf"]  if self.lambda_cf  > 0 else 0.0
        else:
            result["irm_cf"] = None
            l_irm = 0.0
            l_cf  = 0.0

        # ----------------------------------------------------------------
        # Combined score
        # ----------------------------------------------------------------
        # L_total combines all active pillar losses with their lambda weights.
        # We normalise IRM by 1e-13 since raw Kd^2 values can be very large.
        # This scaling factor will be obsolete once the pipeline uses
        # normalised [0, 1] predictions end-to-end.
        l_irm_normalised = l_irm / max(abs(l_irm), 1.0) if l_irm != 0.0 else 0.0
        l_total = (
            self.lambda_xview * l_xview
            + self.lambda_mas   * l_mas
            + self.lambda_irm   * l_irm_normalised
            + self.lambda_cf    * l_cf
        )
        unified_causal_score = float(1.0 / (1.0 + max(l_total, 0.0)))

        result["l_xview"]          = l_xview
        result["l_mas"]            = l_mas
        result["l_irm"]            = l_irm
        result["l_cf"]             = l_cf
        result["l_irm_normalised"] = l_irm_normalised
        result["l_total"]          = float(l_total)
        result["unified_causal_score"] = unified_causal_score

        return result
