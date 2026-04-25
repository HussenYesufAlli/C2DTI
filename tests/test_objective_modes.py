"""Tests for objective-mode behavior across multimodal predictors.

Covers the two requested settings:
1) objective modes: binary classification vs regression
2) model families: mixhop_propagation and interaction_cross_attention
"""

import unittest

import numpy as np

from src.c2dti.dataset_loader import DTIDataset
from src.c2dti.dti_model import (
    InteractionCrossAttentionDTIPredictor,
    MixHopPropagationDTIPredictor,
    create_predictor,
)


def _binary_dataset() -> DTIDataset:
    interactions = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    drugs = [f"drug_{i}" for i in range(interactions.shape[0])]
    targets = [f"target_{j}" for j in range(interactions.shape[1])]
    return DTIDataset(drugs=drugs, targets=targets, interactions=interactions, metadata={"source": "synthetic"})


def _regression_dataset() -> DTIDataset:
    # Values intentionally > 1 so clipping bugs are easy to detect.
    interactions = np.array(
        [
            [2.3, 3.1, 4.2],
            [4.8, 5.7, 6.9],
            [7.4, 8.2, 9.1],
            [3.6, 4.4, 7.8],
        ],
        dtype=np.float32,
    )
    drugs = [f"drug_{i}" for i in range(interactions.shape[0])]
    targets = [f"target_{j}" for j in range(interactions.shape[1])]
    return DTIDataset(drugs=drugs, targets=targets, interactions=interactions, metadata={"source": "synthetic"})


class TestMixHopObjectiveModes(unittest.TestCase):
    def test_mixhop_binary_objective_outputs_in_unit_interval(self):
        dataset = _binary_dataset()
        predictor = MixHopPropagationDTIPredictor(objective="binary_classification", top_k=2)
        pred = predictor.fit_predict(dataset)

        self.assertEqual(pred.shape, dataset.interactions.shape)
        self.assertEqual(pred.dtype, np.float32)
        self.assertTrue(np.all(pred >= 0.0))
        self.assertTrue(np.all(pred <= 1.0))

    def test_mixhop_regression_objective_not_forced_to_unit_interval(self):
        dataset = _regression_dataset()
        predictor = MixHopPropagationDTIPredictor(objective="regression", top_k=2)
        pred = predictor.fit_predict(dataset)

        self.assertEqual(pred.shape, dataset.interactions.shape)
        self.assertEqual(pred.dtype, np.float32)
        self.assertGreater(float(pred.max()), 1.0)


class TestInteractionCrossAttentionObjectiveModes(unittest.TestCase):
    def test_interaction_cross_attention_binary_objective_outputs_in_unit_interval(self):
        dataset = _binary_dataset()
        predictor = InteractionCrossAttentionDTIPredictor(
            objective="binary_classification",
            latent_dim=8,
            epochs=20,
            lr=0.05,
            seed=0,
            top_k=2,
        )
        pred = predictor.fit_predict(dataset)

        self.assertEqual(pred.shape, dataset.interactions.shape)
        self.assertEqual(pred.dtype, np.float32)
        self.assertTrue(np.all(pred >= 0.0))
        self.assertTrue(np.all(pred <= 1.0))

    def test_interaction_cross_attention_regression_objective_not_forced_to_unit_interval(self):
        dataset = _regression_dataset()
        predictor = InteractionCrossAttentionDTIPredictor(
            objective="regression",
            latent_dim=8,
            epochs=20,
            lr=0.05,
            seed=0,
            top_k=2,
        )
        pred = predictor.fit_predict(dataset)

        self.assertEqual(pred.shape, dataset.interactions.shape)
        self.assertEqual(pred.dtype, np.float32)
        self.assertGreater(float(pred.max()), 1.0)


class TestFactoryObjectiveForwarding(unittest.TestCase):
    def test_factory_forwards_objective_for_mixhop(self):
        predictor = create_predictor({"name": "mixhop_propagation", "objective": "regression"})
        self.assertIsInstance(predictor, MixHopPropagationDTIPredictor)
        self.assertEqual(predictor.objective, "regression")

    def test_factory_forwards_objective_for_interaction_cross_attention(self):
        predictor = create_predictor({"name": "interaction_cross_attention", "objective": "binary"})
        self.assertIsInstance(predictor, InteractionCrossAttentionDTIPredictor)
        self.assertEqual(predictor.objective, "binary_classification")


if __name__ == "__main__":
    unittest.main()
