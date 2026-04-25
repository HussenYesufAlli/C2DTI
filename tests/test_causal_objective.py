import sys
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2dti.causal_objective import (
    validate_causal_config,
    compute_causal_score,
    compute_causal_reliability_score,
)


class TestCausalConfigValidation(unittest.TestCase):
    def test_validate_causal_config_none_is_valid(self) -> None:
        errors = validate_causal_config(None)
        self.assertEqual(errors, [])

    def test_validate_causal_config_disabled_is_valid(self) -> None:
        cfg = {"enabled": False}
        errors = validate_causal_config(cfg)
        self.assertEqual(errors, [])

    def test_validate_causal_config_enabled_with_weight(self) -> None:
        cfg = {"enabled": True, "weight": 0.5}
        errors = validate_causal_config(cfg)
        self.assertEqual(errors, [])

    def test_validate_causal_config_negative_weight_is_invalid(self) -> None:
        cfg = {"enabled": True, "weight": -0.1}
        errors = validate_causal_config(cfg)
        self.assertIn("causal.weight must be a non-negative number", errors)

    def test_validate_causal_config_non_bool_enabled_is_invalid(self) -> None:
        cfg = {"enabled": "yes"}
        errors = validate_causal_config(cfg)
        self.assertIn("causal.enabled must be a boolean", errors)


class TestCausalScoreComputation(unittest.TestCase):
    def test_compute_causal_score_disabled_returns_none(self) -> None:
        score = compute_causal_score(enabled=False, weight=0.0)
        self.assertIsNone(score)

    def test_compute_causal_score_enabled_returns_value(self) -> None:
        score = compute_causal_score(enabled=True, weight=0.5)
        self.assertIsNotNone(score)
        self.assertIsInstance(score, float)

    def test_compute_causal_score_enabled_placeholder_value(self) -> None:
        score = compute_causal_score(enabled=True, weight=1.0)
        self.assertEqual(score, 0.5)

    def test_compute_causal_reliability_score_returns_bounded_value(self) -> None:
        baseline = np.array([[0.9, 0.2], [0.4, 0.8]], dtype=np.float32)
        perturbed = np.array([[0.8, 0.3], [0.5, 0.7]], dtype=np.float32)
        score = compute_causal_reliability_score(baseline, perturbed, weight=1.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_compute_causal_reliability_score_rejects_shape_mismatch(self) -> None:
        baseline = np.array([[0.9, 0.2]], dtype=np.float32)
        perturbed = np.array([[0.8], [0.3]], dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_causal_reliability_score(baseline, perturbed, weight=1.0)


if __name__ == "__main__":
    unittest.main()
