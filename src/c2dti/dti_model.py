"""DTI predictor interfaces and implementations.

Two predictors are provided:
  - SimpleMatrixDTIPredictor  : fast non-trainable baseline using row/col means.
  - MatrixFactorizationDTIPredictor : trainable bilinear model that learns
    low-dimensional drug and target embeddings via gradient descent on MSE loss.
    This is the first real learnable component in the C2DTI pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from src.c2dti.data_utils import build_string_feature_matrix
from src.c2dti.dataset_loader import DTIDataset


class DTIPredictor(ABC):
    """Abstract interface for DTI predictors used by the runner."""

    @abstractmethod
    def fit_predict(self, dataset: DTIDataset) -> np.ndarray:
        """Fit the predictor on a dataset and return a prediction matrix."""


class SimpleMatrixDTIPredictor(DTIPredictor):
    """Small baseline predictor using row, column, and feature priors.

    This gives the project a concrete end-to-end prediction stage without
    committing to a heavy learning architecture yet.
    """

    def fit_predict(self, dataset: DTIDataset) -> np.ndarray:
        """Generate a dense prediction matrix from the observed interactions."""
        interactions = dataset.interactions.astype(np.float32)
        if interactions.size == 0:
            return interactions.copy()

        row_means = interactions.mean(axis=1, keepdims=True)
        col_means = interactions.mean(axis=0, keepdims=True)
        global_mean = float(interactions.mean())

        drug_features = build_string_feature_matrix(dataset.drugs)
        target_features = build_string_feature_matrix(dataset.targets)

        # Feature similarity acts as a lightweight prior that nudges the model
        # beyond pure averaging when identifiers/sequences share structure.
        feature_prior = np.matmul(drug_features, target_features.T)
        if feature_prior.size > 0 and float(feature_prior.max()) > 0.0:
            feature_prior = feature_prior / float(feature_prior.max())

        predictions = (0.4 * row_means) + (0.4 * col_means) + (0.1 * global_mean) + (0.1 * feature_prior)
        return np.clip(predictions.astype(np.float32), 0.0, 1.0)


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

    def fit_predict(self, dataset: DTIDataset) -> np.ndarray:
        """Train the factorisation model and return the predicted affinity matrix.

        Steps:
          1. Extract the observed affinity matrix Y from the dataset.
          2. Build a mask of known (non-NaN) entries to train only on real data.
          3. Randomly initialise drug embeddings P and target embeddings Q.
          4. Run gradient descent for `epochs` steps, updating P and Q.
          5. Apply sigmoid to map raw dot-product scores to [0, 1].
          6. Return the full predicted matrix (all drug-target pairs).
        """
        rng = np.random.RandomState(self.seed)
        n_drugs = len(dataset.drugs)
        n_targets = len(dataset.targets)

        # Y is the ground-truth affinity matrix, float64 for numerical stability
        Y = dataset.interactions.astype(np.float64)

        # known_mask marks every entry that has a real (non-NaN) value
        # We only compute loss on these entries; unknown entries are ignored
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


def create_predictor(model_name_or_cfg: Union[str, Dict]) -> DTIPredictor:
    """Create the configured predictor implementation.

    Accepts either:
      - a plain model name string (legacy / simple usage), or
      - a model config dict as read from YAML (preferred for real runs).

    Supported model names:
      "simple_baseline"        → SimpleMatrixDTIPredictor (fast, non-trainable)
      "matrix_factorization"   → MatrixFactorizationDTIPredictor (trainable)
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

    if normalized_name == "matrix_factorization":
        return MatrixFactorizationDTIPredictor(
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            epochs=int(model_cfg.get("epochs", 100)),
            lr=float(model_cfg.get("lr", 0.01)),
            seed=int(model_cfg.get("seed", 42)),
        )

    raise ValueError(
        f"Unsupported model.name: {model_name!r}. "
        "Must be one of: simple_baseline, matrix_factorization"
    )