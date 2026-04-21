"""Simple DTI predictor interfaces and baseline implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

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


def create_predictor(model_name: str) -> DTIPredictor:
    """Create the configured predictor implementation."""
    normalized_name = (model_name or "simple_baseline").strip().lower()
    if normalized_name == "simple_baseline":
        return SimpleMatrixDTIPredictor()

    raise ValueError(f"Unsupported model.name: {model_name}")