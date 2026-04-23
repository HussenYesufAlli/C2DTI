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

        # nanmean ignores masked-out (NaN) entries when computing averages
        row_means = np.nanmean(interactions, axis=1, keepdims=True)
        col_means = np.nanmean(interactions, axis=0, keepdims=True)
        global_mean = float(np.nanmean(interactions))

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

        # Cosine-like interaction prior from normalized frozen embeddings.
        dn = drug_emb / np.maximum(np.linalg.norm(drug_emb, axis=1, keepdims=True), 1e-12)
        tn = target_emb / np.maximum(np.linalg.norm(target_emb, axis=1, keepdims=True), 1e-12)
        sim = dn @ tn.T
        sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)

        # Interaction statistics from training view only.
        y_masked = np.where(known_mask, y, np.nan)
        global_mean = float(np.nanmean(y_masked)) if int(known_mask.sum()) > 0 else 0.5
        row_means = np.nanmean(y_masked, axis=1, keepdims=True)
        col_means = np.nanmean(y_masked, axis=0, keepdims=True)
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
            # Combine: entry must be in train_mask AND have a real value
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

        # Sigmoid maps any real number to (0, 1), matching affinity/probability scale
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
    ) -> None:
        self.top_k = int(max(1, top_k))
        self.hop_weights = hop_weights if hop_weights is not None else [0.6, 0.3, 0.1]

    def fit_predict(
        self,
        dataset: DTIDataset,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        y = dataset.interactions.astype(np.float64)
        if y.size == 0:
            return y.astype(np.float32)

        y_train, _ = _prepare_training_view(y, train_mask)

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

        return np.clip(pred, 0.0, 1.0).astype(np.float32)


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
    ) -> None:
        self.latent_dim = int(max(1, latent_dim))
        self.epochs = int(max(1, epochs))
        self.lr = float(lr)
        self.seed = int(seed)
        self.attention_temperature = float(max(1e-6, attention_temperature))
        self.top_k = int(max(1, top_k))

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
        if int(known_mask.sum()) == 0:
            return np.clip(mf_score, 0.0, 1.0).astype(np.float32)

        x1 = mf_score[known_mask]
        x2 = attn[known_mask]
        y_obs = y_train[known_mask]

        x = np.stack([np.ones_like(x1), x1, x2], axis=1)
        beta, *_ = np.linalg.lstsq(x, y_obs, rcond=None)

        pred = beta[0] + beta[1] * mf_score + beta[2] * attn
        return np.clip(pred, 0.0, 1.0).astype(np.float32)


def create_predictor(model_name_or_cfg: Union[str, Dict]) -> DTIPredictor:
    """Create the configured predictor implementation.

    Accepts either:
      - a plain model name string (legacy / simple usage), or
      - a model config dict as read from YAML (preferred for real runs).

    Supported model names:
      "simple_baseline"        -> SimpleMatrixDTIPredictor (fast, non-trainable)
            "dual_frozen_backbone"   -> DualFrozenBackbonePredictor (Phase-1 frozen embedding model)
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
        )

    if normalized_name == "interaction_cross_attention":
        return InteractionCrossAttentionDTIPredictor(
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            epochs=int(model_cfg.get("epochs", 100)),
            lr=float(model_cfg.get("lr", 0.01)),
            seed=int(model_cfg.get("seed", 42)),
            attention_temperature=float(model_cfg.get("attention_temperature", 1.0)),
            top_k=int(model_cfg.get("top_k", 8)),
        )

    raise ValueError(
        f"Unsupported model.name: {model_name!r}. "
        "Must be one of: simple_baseline, dual_frozen_backbone, matrix_factorization, mixhop_propagation, interaction_cross_attention"
    )
