import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2dti.config_validation import validate_config


class TestConfigValidation(unittest.TestCase):
    def test_valid_minimal_config(self) -> None:
        cfg = {
            "name": "C2DTI_MINIMAL",
            "protocol": "P0",
            "output": {"base_dir": "outputs"},
        }
        self.assertEqual(validate_config(cfg), [])

    def test_missing_required_keys(self) -> None:
        cfg = {}
        errors = validate_config(cfg)
        self.assertIn("Missing required key: name", errors)
        self.assertIn("Missing required key: protocol", errors)
        self.assertIn("Missing required key: output.base_dir", errors)

    def test_valid_real_pipeline_config(self) -> None:
        cfg = {
            "name": "C2DTI_DAVIS_REAL",
            "protocol": "P1",
            "output": {"base_dir": "outputs"},
            "dataset": {"name": "DAVIS", "path": "data/davis"},
            "model": {"name": "simple_baseline"},
            "perturbation": {"strength": 0.2, "seed": 7},
            "causal": {"enabled": True, "weight": 1.0},
        }
        self.assertEqual(validate_config(cfg), [])

    def test_invalid_dataset_config_is_rejected(self) -> None:
        cfg = {
            "name": "C2DTI_INVALID_DATASET",
            "protocol": "P1",
            "output": {"base_dir": "outputs"},
            "dataset": {"name": "UNKNOWN"},
        }
        errors = validate_config(cfg)
        self.assertIn("dataset.name must be one of: BindingDB, BindingDB_Kd, DAVIS, KIBA", errors)
        self.assertIn("dataset.path is required when dataset config is provided", errors)

    def test_invalid_allow_placeholder_type_is_rejected(self) -> None:
        cfg = {
            "name": "C2DTI_INVALID_ALLOW_PLACEHOLDER",
            "protocol": "P1",
            "output": {"base_dir": "outputs"},
            "dataset": {
                "name": "DAVIS",
                "path": "data/davis",
                "allow_placeholder": "no",
            },
        }
        errors = validate_config(cfg)
        self.assertIn("dataset.allow_placeholder must be a boolean", errors)

    def test_model_objective_accepts_regression(self) -> None:
        cfg = {
            "name": "C2DTI_OBJECTIVE_OK",
            "protocol": "P1",
            "output": {"base_dir": "outputs"},
            "model": {"name": "mixhop_propagation", "objective": "regression"},
        }
        self.assertEqual(validate_config(cfg), [])

    def test_model_objective_invalid_value_is_rejected(self) -> None:
        cfg = {
            "name": "C2DTI_OBJECTIVE_BAD",
            "protocol": "P1",
            "output": {"base_dir": "outputs"},
            "model": {"name": "interaction_cross_attention", "objective": "multiclass"},
        }
        errors = validate_config(cfg)
        self.assertIn(
            "model.objective must be one of: auto, binary_classification, regression",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
