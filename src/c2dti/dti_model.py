"""DTI predictor interfaces and implementations.

Available predictors:
    - SimpleMatrixDTIPredictor: fast non-trainable baseline using row/col means.
    - MatrixFactorizationDTIPredictor: trainable bilinear model.
    - MixHopPropagationDTIPredictor: MixHop-style multi-hop propagation baseline.
    - InteractionCrossAttentionDTIPredictor: interaction-aware cross-attention multimodal model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from src.c2dti.backbones import load_frozen_entity_embeddings
from src.c2dti.data_utils import build_string_feature_matrix
from src.c2dti.dataset_loader import DTIDataset


def _row_softmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(np.clip(shifted, -50.0, 50.0))
    denom = np.sum(ex, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return ex / denom


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid used by binary end-to-end predictors."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert matrix rows to probability mass, preserving zeros."""
    denom = np.sum(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _cosine_affinity(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Build cosine similarity matrix from feature matrix."""
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, eps)
    sim = x @ x.T
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def _topk_adjacency(sim: np.ndarray, top_k: int) -> np.ndarray:
    """Keep top-k neighbors per row and return row-normalized adjacency."""
    n = sim.shape[0]
    k = int(max(1, min(top_k, max(1, n - 1))))
    adj = np.zeros_like(sim, dtype=np.float64)
    for i in range(n):
        row = sim[i].copy()
        row[i] = 0.0
        if k < n:
            idx = np.argpartition(row, -k)[-k:]
        else:
            idx = np.arange(n)
        adj[i, idx] = row[idx]
    return _row_normalize(adj)


def _prepare_training_view(y: np.ndarray, train_mask: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Create training matrix and known-mask view used by multiple predictors."""
    y = np.asarray(y, dtype=np.float64)
    if train_mask is not None:
        known_mask = train_mask & ~np.isnan(y)
    else:
        known_mask = ~np.isnan(y)

    if int(known_mask.sum()) == 0:
        return np.zeros_like(y, dtype=np.float64), known_mask

    global_mean = float(np.nanmean(y[known_mask]))
    y_train = np.where(known_mask, y, global_mean)
    return y_train, known_mask


def _normalize_objective_name(objective: Optional[str]) -> str:
    """Normalize objective aliases into canonical names.

    Canonical names used in this module:
      - "auto"
      - "binary_classification"
      - "regression"
    """
    if objective is None:
        return "auto"

    name = str(objective).strip().lower()
    aliases = {
        "auto": "auto",
        "binary": "binary_classification",
        "binary_classification": "binary_classification",
        "classification": "binary_classification",
        "regression": "regression",
        "continuous": "regression",
    }
    if name not in aliases:
        raise ValueError(
            f"Unsupported objective: {objective!r}. "
            "Must be one of: auto, binary_classification, regression"
        )
    return aliases[name]


def _infer_objective_from_labels(y: np.ndarray, known_mask: np.ndarray) -> str:
    """Infer objective from observed labels.

    If all observed labels are in {0, 1}, treat as binary classification,
    otherwise treat as regression.
    """
    observed = np.asarray(y, dtype=np.float64)[known_mask]
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return "regression"

    is_binary = np.all(np.isclose(observed, 0.0) | np.isclose(observed, 1.0))
    return "binary_classification" if bool(is_binary) else "regression"


def _resolve_objective(configured_objective: str, y: np.ndarray, known_mask: np.ndarray) -> str:
    """Resolve objective from config and data labels."""
    if configured_objective == "auto":
        return _infer_objective_from_labels(y, known_mask)
    return configured_objective


class DTIPredictor(ABC):
    """Abstract interface for DTI predictors used by the runner."""

    @abstractmethod
    def fit_predict(
        self,
        dataset: DTIDataset,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit the predictor on a dataset and return a prediction matrix.

        Args:
            dataset    : the full DTI dataset (drugs, targets, interactions).
            train_mask : optional boolean array of shape (n_drugs, n_targets).
                         When provided, the model may only observe entries where
                         train_mask[i, j] is True during training.
                         If None, all non-NaN entries are used (backward-compatible).
        """


class SimpleMatrixDTIPredictor(DTIPredictor):
    """Small baseline predictor using row, column, and feature priors.

    This gives the project a concrete end-to-end prediction stage without
    committing to a heavy learning architecture yet.
    """

    def fit_predict(
        self,
        dataset: DTIDataset,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a dense prediction matrix from the observed interactions.

        train_mask restricts which entries are used to compute row/col/global means.
        If None, all non-NaN entries are used.
        """
        interactions = dataset.interactions.astype(np.float32)
        if interactions.size == 0:
            return interactions.copy()

        # Apply train_mask: hide test entries so the baseline only uses training pairs
        if train_mask is not None:
            interactions = np.where(train_mask, interactions, np.nan)

        # Robust mean computation for cold splits where entire rows/cols may be masked.
        observed = ~np.isnan(interactions)
        observed_count = int(np.sum(observed))
        global_mean = float(np.nanmean(interactions)) if observed_count > 0 else 0.5

        row_sum = np.nansum(interactions, axis=1, keepdims=True)
        row_count = np.sum(observed, axis=1, keepdims=True)
        row_means = np.divide(
            row_sum,
            row_count,
            out=np.full_like(row_sum, np.nan),
            where=row_count > 0,
        )

        col_sum = np.nansum(interactions, axis=0, keepdims=True)
        col_count = np.sum(observed, axis=0, keepdims=True)
        col_means = np.divide(
            col_sum,
            col_count,
            out=np.full_like(col_sum, np.nan),
            where=col_count > 0,
        )

        row_means = np.where(np.isnan(row_means), global_mean, row_means)
        col_means = np.where(np.isnan(col_means), global_mean, col_means)

        drug_features = build_string_feature_matrix(dataset.drugs)
        target_features = build_string_feature_matrix(dataset.targets)

        # Feature similarity acts as a lightweight prior that nudges the model
        # beyond pure averaging when identifiers/sequences share structure.
        feature_prior = np.matmul(drug_features, target_features.T)
        if feature_prior.size > 0 and float(feature_prior.max()) > 0.0:
            feature_prior = feature_prior / float(feature_prior.max())

        predictions = (0.4 * row_means) + (0.4 * col_means) + (0.1 * global_mean) + (0.1 * feature_prior)
        return np.clip(predictions.astype(np.float32), 0.0, 1.0)


class DualFrozenBackbonePredictor(DTIPredictor):
    """Phase-1 sequence-view predictor with frozen drug/protein embeddings.

    This implementation is intentionally lightweight and non-breaking:
    - Loads frozen embeddings from NPZ files when available.
    - Falls back to deterministic hash features when NPZ files are missing.
    - Uses train-mask-only calibration so evaluation remains leakage-safe.
    """

    def __init__(
        self,
        chemberta_npz_path: Optional[str] = None,
        ankh_npz_path: Optional[str] = None,
        fusion_alpha: float = 0.7,
        max_calibration_samples: int = 200000,
        seed: int = 42,
    ) -> None:
        self.chemberta_npz_path = chemberta_npz_path
        self.ankh_npz_path = ankh_npz_path
        self.fusion_alpha = float(np.clip(fusion_alpha, 0.0, 1.0))
        self.max_calibration_samples = int(max(1000, max_calibration_samples))
        self.seed = int(seed)

    def _align_embedding_dims(
        self,
        drug_emb: np.ndarray,
        target_emb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align drug/target embedding dimensions to a shared latent space.

        Why this is needed:
                - Pretrained tables can have different widths (for example,
          ChemBERTa=384 and ANKH=768).
        - The sequence-view prior uses a cosine-like dot product, which requires
          the same feature dimension on both sides.

        Strategy:
        - If widths already match, return unchanged.
        - Otherwise, apply deterministic random projection to both modalities into
          a shared dimension (the smaller original width). This keeps behavior
          stable and avoids hard failures while preserving as much information as
          possible in a non-breaking way.
        """
        d_drug = int(drug_emb.shape[1])
        d_target = int(target_emb.shape[1])
        if d_drug == d_target:
            return drug_emb, target_emb

        shared_dim = min(d_drug, d_target)
        rng = np.random.RandomState(self.seed)

        # Scale by sqrt(input_dim) to keep projected magnitudes numerically stable.
        proj_drug = rng.normal(0.0, 1.0, size=(d_drug, shared_dim)).astype(np.float64) / np.sqrt(max(d_drug, 1))
        proj_target = rng.normal(0.0, 1.0, size=(d_target, shared_dim)).astype(np.float64) / np.sqrt(max(d_target, 1))

        print(
            "[DualFrozenBackbone] Embedding dimension mismatch detected "
            f"(drug={d_drug}, target={d_target}). Projecting both to shared_dim={shared_dim}."
        )

        drug_aligned = drug_emb @ proj_drug
        target_aligned = target_emb @ proj_target
        return drug_aligned, target_aligned

    def fit_predict(
        self,
        dataset: DTIDataset,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        y = dataset.interactions.astype(np.float64)
        if y.size == 0:
            return y.astype(np.float32)

        if train_mask is not None:
            known_mask = train_mask & ~np.isnan(y)
        else:
            known_mask = ~np.isnan(y)

        # Frozen embeddings (or deterministic fallback) for both modalities.
        drug_emb = load_frozen_entity_embeddings(
            entities=dataset.drugs,
            npz_path=self.chemberta_npz_path,
            default_dim=768,
        ).astype(np.float64)
        target_emb = load_frozen_entity_embeddings(
            entities=dataset.targets,
            npz_path=self.ankh_npz_path,
            default_dim=768,
        ).astype(np.float64)

        drug_emb, target_emb = self._align_embedding_dims(drug_emb, target_emb)

        # Cosine-like interaction prior from normalized frozen embeddings.
        dn = drug_emb / np.maximum(np.linalg.norm(drug_emb, axis=1, keepdims=True), 1e-12)
        tn = target_emb / np.maximum(np.linalg.norm(target_emb, axis=1, keepdims=True), 1e-12)
        sim = dn @ tn.T
        sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)

        # Interaction statistics from training view only.
        y_masked = np.where(known_mask, y, np.nan)
        global_mean = float(np.nanmean(y_masked)) if int(known_mask.sum()) > 0 else 0.5
        # Compute row/column means with explicit counts to avoid RuntimeWarning
        # when a cold-split row/column has no observed training labels.
        row_sum = np.nansum(y_masked, axis=1, keepdims=True)
        row_count = np.sum(~np.isnan(y_masked), axis=1, keepdims=True)
        row_means = np.divide(
            row_sum,
            row_count,
            out=np.full_like(row_sum, np.nan),
            where=row_count > 0,
        )

        col_sum = np.nansum(y_masked, axis=0, keepdims=True)
        col_count = np.sum(~np.isnan(y_masked), axis=0, keepdims=True)
        col_means = np.divide(
            col_sum,
            col_count,
            out=np.full_like(col_sum, np.nan),
            where=col_count > 0,
        )
        row_means = np.where(np.isnan(row_means), global_mean, row_means)
        col_means = np.where(np.isnan(col_means), global_mean, col_means)
        stats_prior = 0.5 * row_means + 0.5 * col_means

        # Sequence-view fusion: frozen prior + statistical prior.
        alpha = self.fusion_alpha
        base_pred = (alpha * sim) + ((1.0 - alpha) * stats_prior)
        base_pred = np.clip(base_pred, 0.0, 1.0)

        # Calibrate on train entries only (linear head equivalent).
        if int(known_mask.sum()) == 0:
            return base_pred.astype(np.float32)

        idx = np.flatnonzero(known_mask)
        rng = np.random.RandomState(self.seed)
        if idx.size > self.max_calibration_samples:
            idx = rng.choice(idx, size=self.max_calibration_samples, replace=False)

        y_flat = y.ravel()
        base_flat = base_pred.ravel()
        sim_flat = sim.ravel()
        row_flat = np.repeat(row_means[:, 0], y.shape[1])
        col_flat = np.tile(col_means[0, :], y.shape[0])

        x = np.stack(
            [
                np.ones_like(base_flat[idx]),
                base_flat[idx],
                sim_flat[idx],
                row_flat[idx],
                col_flat[idx],
            ],
            axis=1,
        )
        beta, *_ = np.linalg.lstsq(x, y_flat[idx], rcond=None)

        pred = (
            beta[0]
            + beta[1] * base_pred
            + beta[2] * sim
            + beta[3] * row_means
            + beta[4] * col_means
        )
        return np.clip(pred, 0.0, 1.0).astype(np.float32)


class MatrixFactorizationDTIPredictor(DTIPredictor):
    """Trainable matrix factorisation predictor for DTI affinity prediction.

    How it works (beginner explanation):
    -------------------------------------
    Think of drugs as rows and targets as columns in a grid of affinities.
    Instead of using the raw values directly, we learn a *compressed
    representation* (called an embedding) for each drug and each target.

    A drug embedding is a small vector of numbers — say 32 numbers — that
    captures what makes a drug unique.  Same for a target.
    The predicted affinity between drug d and target t is just the dot
    product (inner product) of their two embedding vectors.

    We start with random embeddings and iteratively adjust them so that
    the dot products match the observed affinity values.  This adjustment
    is gradient descent:
      1. Compute predictions = P @ Q.T      (P = drug embeddings, Q = target embeddings)
      2. Compute error       = predictions - true_affinities  (only on known entries)
      3. Compute gradient    = how much each embedding should change to reduce error
      4. Update embeddings   = embeddings - lr * gradient
    Repeat for `epochs` iterations.

    After training, the embeddings can reconstruct the full affinity matrix
    (including unseen drug-target pairs), which is the *generalisation* we want.

    Hyperparameters (set via model config in YAML):
      latent_dim : size of each embedding vector (default 32)
      epochs     : number of gradient descent steps (default 100)
      lr         : learning rate, step size for each update (default 0.01)
      seed       : random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        epochs: int = 100,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

        # These are populated during fit_predict and exposed for the runner
        self._drug_embeddings: Optional[np.ndarray] = None    # (n_drugs,  latent_dim)
        self._target_embeddings: Optional[np.ndarray] = None  # (n_targets, latent_dim)
        self.train_loss_history: List[float] = []             # MSE per epoch

    def fit_predict(
        self,
        dataset: DTIDataset,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Train the factorisation model and return the predicted affinity matrix.

        Steps:
          1. Extract the observed affinity matrix Y from the dataset.
          2. Build a training mask: either the caller-supplied train_mask, or
             all non-NaN entries (backward-compatible when no split is used).
          3. Randomly initialise drug embeddings P and target embeddings Q.
          4. Run gradient descent on TRAIN entries only for `epochs` steps.
          5. Apply sigmoid to map raw dot-product scores to [0, 1].
          6. Return the full predicted matrix (all drug-target pairs).
        """
        rng = np.random.RandomState(self.seed)
        n_drugs = len(dataset.drugs)
        n_targets = len(dataset.targets)

        # Y is the ground-truth affinity matrix, float64 for numerical stability
        Y = dataset.interactions.astype(np.float64)

        # Decide which entries the model is allowed to see during training:
        #   - If train_mask is provided (split was performed), train only on those entries.
        #   - Otherwise, train on all non-NaN entries (old behaviour, fully backward-compatible).
        if train_mask is not None:
            # Combine: entry must be in train_mask AND have an observed value
            known_mask = train_mask & ~np.isnan(Y)
        else:
            known_mask = ~np.isnan(Y)

        # Initialise small random embeddings to break symmetry
        scale = 0.1
        P = rng.randn(n_drugs, self.latent_dim).astype(np.float64) * scale
        Q = rng.randn(n_targets, self.latent_dim).astype(np.float64) * scale

        self.train_loss_history = []

        for _epoch in range(self.epochs):
            # Forward pass: predicted affinity = P @ Q.T
            Y_hat = P @ Q.T  # shape (n_drugs, n_targets)

            # Error only on observed (known) entries; zero elsewhere
            error = np.where(known_mask, Y_hat - Y, 0.0)

            # MSE on known entries
            n_known = int(known_mask.sum())
            loss = float(np.sum(error ** 2) / max(n_known, 1))
            self.train_loss_history.append(loss)

            # Gradients of MSE w.r.t. P and Q:
            #   dL/dP[i,:] = (2/n) * sum_j error[i,j] * Q[j,:]
            #              = (2/n) * error[i,:] @ Q
            #   dL/dQ[j,:] = (2/n) * sum_i error[i,j] * P[i,:]
            #              = (2/n) * error[:,j] @ P = (2/n) * error.T @ P
            factor = 2.0 / max(n_known, 1)
            grad_P = factor * (error @ Q)
            grad_Q = factor * (error.T @ P)

            P -= self.lr * grad_P
            Q -= self.lr * grad_Q

        # Store trained embeddings so the runner can checkpoint them
        self._drug_embeddings = P
        self._target_embeddings = Q

        # Final raw predictions
        raw = P @ Q.T  # (n_drugs, n_targets)

        # Sigmoid maps any scalar score to (0, 1), matching affinity/probability scale
        predictions = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
        return predictions.astype(np.float32)

    def save_checkpoint(self, run_dir: Path) -> Path:
        """Save the trained drug and target embeddings to disk.

        Embeddings are saved as a NumPy .npz archive inside run_dir/checkpoints/.
        They can be reloaded with: np.load(checkpoint_path)
        """
        if self._drug_embeddings is None or self._target_embeddings is None:
            raise RuntimeError("Cannot save checkpoint before fit_predict is called.")

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "embeddings.npz"

        np.savez(
            str(checkpoint_path),
            drug_embeddings=self._drug_embeddings,
            target_embeddings=self._target_embeddings,
        )
        return checkpoint_path


class MixHopPropagationDTIPredictor(DTIPredictor):
    """MixHop-style propagation baseline for drug-target matrices.

    Beginner view:
    - Build a drug similarity graph and a target similarity graph.
    - Propagate known interactions through multiple hops (0, 1, 2, ...).
    - Blend hop outputs with learned-style fixed weights.
    """

    def __init__(
        self,
        top_k: int = 8,
        hop_weights: Optional[List[float]] = None,
        objective: str = "auto",
    ) -> None:
        self.top_k = int(max(1, top_k))
        self.hop_weights = hop_weights if hop_weights is not None else [0.6, 0.3, 0.1]
        self.objective = _normalize_objective_name(objective)

    def fit_predict(
        self,
        dataset: DTIDataset,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        y = dataset.interactions.astype(np.float64)
        if y.size == 0:
            return y.astype(np.float32)

        y_train, known_mask = _prepare_training_view(y, train_mask)

        drug_features = build_string_feature_matrix(dataset.drugs)
        target_features = build_string_feature_matrix(dataset.targets)
        drug_adj = _topk_adjacency(_cosine_affinity(drug_features), top_k=self.top_k)
        target_adj = _topk_adjacency(_cosine_affinity(target_features), top_k=self.top_k)

        n_drugs = drug_adj.shape[0]
        n_targets = target_adj.shape[0]
        drug_power = np.eye(n_drugs, dtype=np.float64)
        target_power = np.eye(n_targets, dtype=np.float64)

        weights = np.asarray(self.hop_weights, dtype=np.float64)
        if weights.size == 0:
            weights = np.asarray([1.0], dtype=np.float64)
        weights = weights / np.maximum(weights.sum(), 1e-12)

        pred = np.zeros_like(y_train, dtype=np.float64)
        for w in weights:
            pred += float(w) * (drug_power @ y_train @ target_power.T)
            drug_power = drug_power @ drug_adj
            target_power = target_power @ target_adj

        objective = _resolve_objective(self.objective, y, known_mask)
        if objective == "binary_classification":
            pred = np.clip(pred, 0.0, 1.0)
        return pred.astype(np.float32)


class InteractionCrossAttentionDTIPredictor(DTIPredictor):
    """Interaction-aware cross-attention multimodal predictor.

    This model fuses two signals on the same train split:
    1) a matrix-factorization interaction score,
    2) a cross-attention interaction prior computed from drug/target features.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        epochs: int = 100,
        lr: float = 0.01,
        seed: int = 42,
        attention_temperature: float = 1.0,
        top_k: int = 8,
        objective: str = "auto",
    ) -> None:
        self.latent_dim = int(max(1, latent_dim))
        self.epochs = int(max(1, epochs))
        self.lr = float(lr)
        self.seed = int(seed)
        self.attention_temperature = float(max(1e-6, attention_temperature))
        self.top_k = int(max(1, top_k))
        self.objective = _normalize_objective_name(objective)

    def fit_predict(
        self,
        dataset: DTIDataset,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        y = dataset.interactions.astype(np.float64)
        if y.size == 0:
            return y.astype(np.float32)

        y_train, known_mask = _prepare_training_view(y, train_mask)

        # Branch A: trainable matrix-factorization interaction score.
        mf = MatrixFactorizationDTIPredictor(
            latent_dim=self.latent_dim,
            epochs=self.epochs,
            lr=self.lr,
            seed=self.seed,
        )
        mf_score = mf.fit_predict(dataset, train_mask=train_mask).astype(np.float64)

        # Branch B: interaction-aware cross-attention over modality features.
        drug_features = build_string_feature_matrix(dataset.drugs).astype(np.float64)
        target_features = build_string_feature_matrix(dataset.targets).astype(np.float64)
        rng = np.random.RandomState(self.seed)

        w_q = rng.randn(drug_features.shape[1], self.latent_dim) * 0.1
        w_k = rng.randn(target_features.shape[1], self.latent_dim) * 0.1

        q = np.tanh(drug_features @ w_q)
        k = np.tanh(target_features @ w_k)

        scale = np.sqrt(float(self.latent_dim)) * self.attention_temperature
        logits = (q @ k.T) / max(scale, 1e-8)

        attn_drug_to_target = _row_softmax(logits)
        attn_target_to_drug = _row_softmax(logits.T).T
        attn = 0.5 * (attn_drug_to_target + attn_target_to_drug)

        # Top-k sparsification keeps the interaction prior focused.
        if self.top_k > 0:
            n_targets = attn.shape[1]
            k_keep = int(max(1, min(self.top_k, n_targets)))
            sparse_attn = np.zeros_like(attn)
            for i in range(attn.shape[0]):
                idx = np.argpartition(attn[i], -k_keep)[-k_keep:]
                sparse_attn[i, idx] = attn[i, idx]
            attn = _row_normalize(sparse_attn)

        # Calibrate branch fusion on training entries only.
        objective = _resolve_objective(self.objective, y, known_mask)
        if int(known_mask.sum()) == 0:
            if objective == "binary_classification":
                return np.clip(mf_score, 0.0, 1.0).astype(np.float32)
            return mf_score.astype(np.float32)

        x1 = mf_score[known_mask]
        x2 = attn[known_mask]
        y_obs = y_train[known_mask]

        x = np.stack([np.ones_like(x1), x1, x2], axis=1)
        beta, *_ = np.linalg.lstsq(x, y_obs, rcond=None)

        pred = beta[0] + beta[1] * mf_score + beta[2] * attn
        if objective == "binary_classification":
            pred = np.clip(pred, 0.0, 1.0)
        return pred.astype(np.float32)


class EndToEndCharEncoderPredictor(DTIPredictor):
    """True end-to-end predictor trained directly from raw strings.

    Beginner-friendly summary:
    - Input is raw text for each entity: drug SMILES and protein sequence.
    - We learn trainable character embeddings for drugs and targets.
    - Each entity vector is the mean of its character embeddings.
    - Interaction score is a learned compatibility between the two vectors.
    - All parameters are optimized jointly from the DTI objective.

    This is intentionally independent from frozen NPZ embeddings so users can
    run a full end-to-end training path without breaking existing predictors.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        epochs: int = 30,
        lr: float = 0.05,
        max_drug_len: int = 160,
        max_target_len: int = 2048,
        l2: float = 1e-5,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = int(max(8, embedding_dim))
        self.epochs = int(max(1, epochs))
        self.lr = float(max(1e-6, lr))
        self.max_drug_len = int(max(8, max_drug_len))
        self.max_target_len = int(max(8, max_target_len))
        self.l2 = float(max(0.0, l2))
        self.seed = int(seed)
        self.train_loss_history: list[float] = []

    def _build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build char vocabulary with 0 as PAD token id."""
        chars = sorted({ch for text in texts for ch in str(text)})
        vocab = {ch: i + 1 for i, ch in enumerate(chars)}
        return vocab

    def _encode_texts(self, texts: List[str], vocab: Dict[str, int], max_len: int) -> np.ndarray:
        """Encode strings into padded char-id arrays of shape (n_entities, max_len)."""
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = [vocab.get(ch, 0) for ch in str(text)[:max_len]]
            if ids:
                out[i, : len(ids)] = ids
        return out

    def _mean_embed(self, token_ids: np.ndarray, embedding_table: np.ndarray) -> np.ndarray:
        """Pool char embeddings by mean over non-pad characters."""
        emb = embedding_table[token_ids]  # (n, L, d)
        mask = (token_ids != 0).astype(np.float64)[..., None]
        counts = np.maximum(mask.sum(axis=1), 1.0)
        pooled = (emb * mask).sum(axis=1) / counts
        return pooled

    def _accumulate_char_grads(
        self,
        token_ids: np.ndarray,
        d_entity: np.ndarray,
        vocab_size: int,
    ) -> np.ndarray:
        """Backprop mean-pooling gradients into per-character embedding table."""
        grad_table = np.zeros((vocab_size, d_entity.shape[1]), dtype=np.float64)
        for i in range(token_ids.shape[0]):
            ids = token_ids[i]
            nz = ids[ids != 0]
            if nz.size == 0:
                continue
            g = d_entity[i] / float(nz.size)
            for idx in nz:
                grad_table[idx] += g
        return grad_table

    def fit_predict(
        self,
        dataset: DTIDataset,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit end-to-end parameters and return dense prediction matrix."""
        y = dataset.interactions.astype(np.float64)
        if y.size == 0:
            return y.astype(np.float32)

        if train_mask is not None:
            known_mask = train_mask & ~np.isnan(y)
        else:
            known_mask = ~np.isnan(y)

        if int(known_mask.sum()) == 0:
            return np.zeros_like(y, dtype=np.float32)

        # Detect binary vs regression target space from training labels.
        y_obs = y[known_mask]
        unique_vals = np.unique(y_obs)
        is_binary = set(unique_vals.tolist()).issubset({0.0, 1.0})

        # For regression, optimize in standardized space then invert at the end.
        if is_binary:
            y_train = y.copy()
            y_scale = 1.0
            y_shift = 0.0
        else:
            y_shift = float(np.mean(y_obs))
            y_scale = float(np.std(y_obs) + 1e-8)
            y_train = (y - y_shift) / y_scale

        rng = np.random.RandomState(self.seed)

        drug_vocab = self._build_vocab(dataset.drugs)
        target_vocab = self._build_vocab(dataset.targets)

        drug_tokens = self._encode_texts(dataset.drugs, drug_vocab, self.max_drug_len)
        target_tokens = self._encode_texts(dataset.targets, target_vocab, self.max_target_len)

        d_vocab_size = len(drug_vocab) + 1
        t_vocab_size = len(target_vocab) + 1
        d = self.embedding_dim

        # Trainable parameters.
        drug_char_emb = rng.normal(0.0, 0.05, size=(d_vocab_size, d))
        target_char_emb = rng.normal(0.0, 0.05, size=(t_vocab_size, d))
        drug_bias = np.zeros((len(dataset.drugs),), dtype=np.float64)
        target_bias = np.zeros((len(dataset.targets),), dtype=np.float64)

        self.train_loss_history = []

        for _ in range(self.epochs):
            # Forward: raw strings -> trainable pooled vectors -> pair logits.
            drug_vec = self._mean_embed(drug_tokens, drug_char_emb)
            target_vec = self._mean_embed(target_tokens, target_char_emb)
            logits = (drug_vec @ target_vec.T) + drug_bias[:, None] + target_bias[None, :]

            if is_binary:
                pred = _sigmoid(logits)
                # BCE derivative wrt logits: pred - y
                grad_logits = np.zeros_like(logits)
                grad_logits[known_mask] = pred[known_mask] - y_train[known_mask]

                p = np.clip(pred[known_mask], 1e-8, 1.0 - 1e-8)
                t = y_train[known_mask]
                data_loss = float(-np.mean(t * np.log(p) + (1.0 - t) * np.log(1.0 - p)))
            else:
                pred = logits
                grad_logits = np.zeros_like(logits)
                grad_logits[known_mask] = 2.0 * (pred[known_mask] - y_train[known_mask])
                data_loss = float(np.mean((pred[known_mask] - y_train[known_mask]) ** 2))

            # Normalize gradient by number of observed entries.
            n_obs = float(max(1, int(known_mask.sum())))
            grad_logits /= n_obs

            d_drug_vec = grad_logits @ target_vec
            d_target_vec = grad_logits.T @ drug_vec
            d_drug_bias = grad_logits.sum(axis=1)
            d_target_bias = grad_logits.sum(axis=0)

            d_drug_char_emb = self._accumulate_char_grads(drug_tokens, d_drug_vec, d_vocab_size)
            d_target_char_emb = self._accumulate_char_grads(target_tokens, d_target_vec, t_vocab_size)

            # L2 regularization for stable optimization.
            if self.l2 > 0.0:
                d_drug_char_emb += self.l2 * drug_char_emb
                d_target_char_emb += self.l2 * target_char_emb
                d_drug_bias += self.l2 * drug_bias
                d_target_bias += self.l2 * target_bias

            drug_char_emb -= self.lr * d_drug_char_emb
            target_char_emb -= self.lr * d_target_char_emb
            drug_bias -= self.lr * d_drug_bias
            target_bias -= self.lr * d_target_bias

            self.train_loss_history.append(data_loss)

        # Final inference with trained parameters.
        drug_vec = self._mean_embed(drug_tokens, drug_char_emb)
        target_vec = self._mean_embed(target_tokens, target_char_emb)
        logits = (drug_vec @ target_vec.T) + drug_bias[:, None] + target_bias[None, :]

        if is_binary:
            return _sigmoid(logits).astype(np.float32)

        return (logits * y_scale + y_shift).astype(np.float32)


def create_predictor(model_name_or_cfg: Union[str, Dict]) -> DTIPredictor:
    """Create the configured predictor implementation.

    Accepts either:
      - a plain model name string (legacy / simple usage), or
            - a model config dict as read from YAML (preferred for dataset-backed runs).

    Supported model names:
      "simple_baseline"        -> SimpleMatrixDTIPredictor (fast, non-trainable)
            "dual_frozen_backbone"   -> DualFrozenBackbonePredictor (Phase-1 frozen embedding model)
        "end_to_end_char_encoder" -> EndToEndCharEncoderPredictor (true trainable raw-string pipeline)
      "matrix_factorization"   -> MatrixFactorizationDTIPredictor (trainable)
            "mixhop_propagation"     -> MixHopPropagationDTIPredictor
            "interaction_cross_attention" -> InteractionCrossAttentionDTIPredictor
    """
    # Normalise: accept both string and dict
    if isinstance(model_name_or_cfg, dict):
        model_cfg = model_name_or_cfg
        model_name = model_cfg.get("name", "simple_baseline")
    else:
        model_cfg = {}
        model_name = model_name_or_cfg or "simple_baseline"

    normalized_name = str(model_name).strip().lower()

    if normalized_name == "simple_baseline":
        return SimpleMatrixDTIPredictor()

    if normalized_name == "dual_frozen_backbone":
        return DualFrozenBackbonePredictor(
            chemberta_npz_path=model_cfg.get("chemberta_npz_path"),
            ankh_npz_path=model_cfg.get("ankh_npz_path"),
            fusion_alpha=float(model_cfg.get("fusion_alpha", 0.7)),
            max_calibration_samples=int(model_cfg.get("max_calibration_samples", 200000)),
            seed=int(model_cfg.get("seed", 42)),
        )

    if normalized_name == "end_to_end_char_encoder":
        return EndToEndCharEncoderPredictor(
            embedding_dim=int(model_cfg.get("embedding_dim", 64)),
            epochs=int(model_cfg.get("epochs", 30)),
            lr=float(model_cfg.get("lr", 0.05)),
            max_drug_len=int(model_cfg.get("max_drug_len", 160)),
            max_target_len=int(model_cfg.get("max_target_len", 2048)),
            l2=float(model_cfg.get("l2", 1e-5)),
            seed=int(model_cfg.get("seed", 42)),
        )

    if normalized_name == "matrix_factorization":
        return MatrixFactorizationDTIPredictor(
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            epochs=int(model_cfg.get("epochs", 100)),
            lr=float(model_cfg.get("lr", 0.01)),
            seed=int(model_cfg.get("seed", 42)),
        )

    if normalized_name == "mixhop_propagation":
        return MixHopPropagationDTIPredictor(
            top_k=int(model_cfg.get("top_k", 8)),
            hop_weights=list(model_cfg.get("hop_weights", [0.6, 0.3, 0.1])),
            objective=str(model_cfg.get("objective", "auto")),
        )

    if normalized_name == "interaction_cross_attention":
        return InteractionCrossAttentionDTIPredictor(
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            epochs=int(model_cfg.get("epochs", 100)),
            lr=float(model_cfg.get("lr", 0.01)),
            seed=int(model_cfg.get("seed", 42)),
            attention_temperature=float(model_cfg.get("attention_temperature", 1.0)),
            top_k=int(model_cfg.get("top_k", 8)),
            objective=str(model_cfg.get("objective", "auto")),
        )

    raise ValueError(
        f"Unsupported model.name: {model_name!r}. "
        "Must be one of: simple_baseline, dual_frozen_backbone, end_to_end_char_encoder, matrix_factorization, mixhop_propagation, interaction_cross_attention"
    )
