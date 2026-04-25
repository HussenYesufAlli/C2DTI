"""Unit tests for MatrixFactorizationDTIPredictor training contract.

These tests verify:
  - The predictor trains and produces predictions with the correct shape.
  - Training loss decreases over epochs (the model actually learns).
  - train_loss_history is recorded per epoch.
  - save_checkpoint writes a readable .npz file with both embedding arrays.
  - create_predictor correctly instantiates the model from a config dict.
  - Integration: runner.py stores training stats and checkpoint_path in summary.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.c2dti.dataset_loader import DTIDataset
from src.c2dti.dti_model import MatrixFactorizationDTIPredictor, create_predictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_dataset(n_drugs: int = 5, n_targets: int = 4, seed: int = 0) -> DTIDataset:
    """Create a small synthetic DTI dataset for fast unit tests."""
    rng = np.random.RandomState(seed)
    interactions = rng.rand(n_drugs, n_targets).astype(np.float32)
    drugs = [f"drug_{i}" for i in range(n_drugs)]
    targets = [f"target_{j}" for j in range(n_targets)]
    return DTIDataset(
        drugs=drugs,
        targets=targets,
        interactions=interactions,
        metadata={"source": "synthetic"},
    )


def _dataset_with_nans(n_drugs: int = 4, n_targets: int = 4, seed: int = 1) -> DTIDataset:
    """Dataset where some entries are NaN (unknown affinity)."""
    rng = np.random.RandomState(seed)
    interactions = rng.rand(n_drugs, n_targets).astype(np.float32)
    # Mark 25 % of entries as unknown
    mask = rng.rand(n_drugs, n_targets) < 0.25
    interactions[mask] = float("nan")
    drugs = [f"drug_{i}" for i in range(n_drugs)]
    targets = [f"target_{j}" for j in range(n_targets)]
    return DTIDataset(
        drugs=drugs,
        targets=targets,
        interactions=interactions,
        metadata={"source": "synthetic_with_nan"},
    )


# ---------------------------------------------------------------------------
# Prediction shape and value range
# ---------------------------------------------------------------------------

class TestMatrixFactorizationPredictions(unittest.TestCase):
    def setUp(self):
        self.dataset = _small_dataset()
        self.predictor = MatrixFactorizationDTIPredictor(
            latent_dim=8, epochs=10, lr=0.05, seed=0
        )

    def test_output_shape_matches_input(self):
        predictions = self.predictor.fit_predict(self.dataset)
        n_drugs = len(self.dataset.drugs)
        n_targets = len(self.dataset.targets)
        self.assertEqual(predictions.shape, (n_drugs, n_targets))

    def test_output_dtype_is_float32(self):
        predictions = self.predictor.fit_predict(self.dataset)
        self.assertEqual(predictions.dtype, np.float32)

    def test_predictions_in_zero_one_range(self):
        # Sigmoid output should always be strictly within (0, 1)
        predictions = self.predictor.fit_predict(self.dataset)
        self.assertTrue(np.all(predictions > 0.0))
        self.assertTrue(np.all(predictions < 1.0))

    def test_predictions_are_finite(self):
        predictions = self.predictor.fit_predict(self.dataset)
        self.assertTrue(np.all(np.isfinite(predictions)))


# ---------------------------------------------------------------------------
# Training loss behaviour
# ---------------------------------------------------------------------------

class TestMatrixFactorizationTrainingLoss(unittest.TestCase):
    def test_loss_history_has_correct_length(self):
        dataset = _small_dataset()
        epochs = 20
        predictor = MatrixFactorizationDTIPredictor(latent_dim=4, epochs=epochs, lr=0.05, seed=7)
        predictor.fit_predict(dataset)
        self.assertEqual(len(predictor.train_loss_history), epochs)

    def test_loss_decreases_over_training(self):
        """The model should reduce MSE over 200 epochs on a simple dataset."""
        dataset = _small_dataset(n_drugs=8, n_targets=6, seed=42)
        predictor = MatrixFactorizationDTIPredictor(latent_dim=16, epochs=200, lr=0.05, seed=0)
        predictor.fit_predict(dataset)
        history = predictor.train_loss_history
        # Loss at the end should be lower than at the start
        self.assertLess(history[-1], history[0])

    def test_loss_history_empty_before_fit(self):
        predictor = MatrixFactorizationDTIPredictor()
        self.assertEqual(predictor.train_loss_history, [])

    def test_loss_history_is_positive(self):
        dataset = _small_dataset()
        predictor = MatrixFactorizationDTIPredictor(latent_dim=4, epochs=10, lr=0.01, seed=0)
        predictor.fit_predict(dataset)
        for loss in predictor.train_loss_history:
            self.assertGreaterEqual(loss, 0.0)

    def test_loss_with_nan_entries_does_not_explode(self):
        """NaN affinities should be masked; loss should remain finite."""
        dataset = _dataset_with_nans()
        predictor = MatrixFactorizationDTIPredictor(latent_dim=4, epochs=10, lr=0.01, seed=3)
        predictor.fit_predict(dataset)
        for loss in predictor.train_loss_history:
            self.assertTrue(math.isfinite(loss), f"Non-finite loss: {loss}")


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

class TestMatrixFactorizationCheckpoint(unittest.TestCase):
    def test_save_checkpoint_creates_npz_file(self):
        dataset = _small_dataset()
        predictor = MatrixFactorizationDTIPredictor(latent_dim=4, epochs=5, lr=0.01, seed=0)
        predictor.fit_predict(dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint_path = predictor.save_checkpoint(run_dir)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(checkpoint_path.suffix == ".npz")

    def test_checkpoint_contains_both_embeddings(self):
        dataset = _small_dataset()
        predictor = MatrixFactorizationDTIPredictor(latent_dim=4, epochs=5, lr=0.01, seed=0)
        predictor.fit_predict(dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = predictor.save_checkpoint(Path(tmpdir))
            data = np.load(str(checkpoint_path))
            self.assertIn("drug_embeddings", data)
            self.assertIn("target_embeddings", data)

    def test_checkpoint_embedding_shapes(self):
        n_drugs, n_targets, latent_dim = 5, 4, 8
        dataset = _small_dataset(n_drugs=n_drugs, n_targets=n_targets)
        predictor = MatrixFactorizationDTIPredictor(latent_dim=latent_dim, epochs=5, lr=0.01, seed=0)
        predictor.fit_predict(dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = predictor.save_checkpoint(Path(tmpdir))
            data = np.load(str(checkpoint_path))
            self.assertEqual(data["drug_embeddings"].shape, (n_drugs, latent_dim))
            self.assertEqual(data["target_embeddings"].shape, (n_targets, latent_dim))

    def test_save_checkpoint_before_fit_raises(self):
        predictor = MatrixFactorizationDTIPredictor()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                predictor.save_checkpoint(Path(tmpdir))


# ---------------------------------------------------------------------------
# create_predictor factory
# ---------------------------------------------------------------------------

class TestCreatePredictor(unittest.TestCase):
    def test_string_simple_baseline_returns_simple_predictor(self):
        from src.c2dti.dti_model import SimpleMatrixDTIPredictor
        p = create_predictor("simple_baseline")
        self.assertIsInstance(p, SimpleMatrixDTIPredictor)

    def test_dict_simple_baseline(self):
        from src.c2dti.dti_model import SimpleMatrixDTIPredictor
        p = create_predictor({"name": "simple_baseline"})
        self.assertIsInstance(p, SimpleMatrixDTIPredictor)

    def test_dict_matrix_factorization_with_defaults(self):
        p = create_predictor({"name": "matrix_factorization"})
        self.assertIsInstance(p, MatrixFactorizationDTIPredictor)
        self.assertEqual(p.latent_dim, 32)
        self.assertEqual(p.epochs, 100)
        self.assertAlmostEqual(p.lr, 0.01)
        self.assertEqual(p.seed, 42)

    def test_dict_matrix_factorization_with_custom_hyperparams(self):
        p = create_predictor({
            "name": "matrix_factorization",
            "latent_dim": 16,
            "epochs": 50,
            "lr": 0.001,
            "seed": 99,
        })
        self.assertEqual(p.latent_dim, 16)
        self.assertEqual(p.epochs, 50)
        self.assertAlmostEqual(p.lr, 0.001)
        self.assertEqual(p.seed, 99)

    def test_unknown_model_name_raises(self):
        with self.assertRaises(ValueError):
            create_predictor("does_not_exist")


# ---------------------------------------------------------------------------
# Runner integration: summary contains training + evaluation + checkpoint
# ---------------------------------------------------------------------------

class TestRunnerTrainingIntegration(unittest.TestCase):
    """Verify that run_once writes training stats and checkpoint into summary.json."""

    def _build_config(self, tmpdir: str, model_name: str, latent_dim: int, epochs: int) -> str:
        """Write a temporary YAML config for one run and return its path."""
        import yaml
        model_section: dict = {"name": model_name}
        # Only include matrix factorization hyperparams for the relevant model
        if model_name == "matrix_factorization":
            model_section.update({"latent_dim": latent_dim, "epochs": epochs, "lr": 0.05, "seed": 0})
        cfg = {
            "name": f"test_training_{model_name}",
            "protocol": "P0",
            "output": {"base_dir": tmpdir},
            "model": model_section,
        }
        config_path = Path(tmpdir) / "test_config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        return str(config_path)

    def test_run_once_with_matrix_factorization_succeeds_without_dataset(self):
        """run_once with matrix_factorization and no dataset should return 0."""
        from src.c2dti.runner import run_once
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._build_config(tmpdir, "matrix_factorization", latent_dim=8, epochs=5)
            result = run_once(cfg_path)
            self.assertEqual(result, 0)

    def test_run_once_with_simple_baseline_succeeds(self):
        from src.c2dti.runner import run_once
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._build_config(tmpdir, "simple_baseline", latent_dim=0, epochs=0)
            result = run_once(cfg_path)
            self.assertEqual(result, 0)

    def test_matrix_factorization_run_produces_checkpoint_on_real_dataset(self):
        """With a real (synthetic) dataset, run_once should write checkpoint_path to summary."""
        import yaml
        from src.c2dti.runner import run_once
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a tiny synthetic DAVIS-style dataset on disk
            data_dir = Path(tmpdir) / "data" / "davis"
            data_dir.mkdir(parents=True)
            (data_dir / "drug_smiles.txt").write_text("drugA\ndrugB\ndrugC\n")
            (data_dir / "target_sequences.txt").write_text("targetX\ntargetY\n")
            (data_dir / "Y.txt").write_text("0.9 0.2\n0.3 0.8\n0.5 0.6\n")

            cfg = {
                "name": "test_mf_with_dataset",
                "protocol": "P1",
                "output": {"base_dir": tmpdir},
                "dataset": {
                    "name": "DAVIS",
                    "path": str(data_dir),
                    "allow_placeholder": False,
                },
                "model": {
                    "name": "matrix_factorization",
                    "latent_dim": 4,
                    "epochs": 10,
                    "lr": 0.05,
                    "seed": 0,
                },
            }
            cfg_path = Path(tmpdir) / "test_mf_cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))

            result = run_once(str(cfg_path))
            self.assertEqual(result, 0)

            # Find the run directory and read the summary
            runs_dir = Path(tmpdir) / "runs"
            run_dirs = list(runs_dir.iterdir())
            self.assertEqual(len(run_dirs), 1)
            summary_path = run_dirs[0] / "summary.json"
            summary = json.loads(summary_path.read_text())

            # Training block should be present with the expected keys
            self.assertIn("training", summary)
            training = summary["training"]
            self.assertIn("epochs_completed", training)
            self.assertIn("loss_start", training)
            self.assertIn("loss_final", training)
            self.assertIn("loss_min", training)
            self.assertEqual(training["epochs_completed"], 10)

            # Evaluation metrics should be present
            self.assertIn("evaluation_metrics", summary)
            metrics = summary["evaluation_metrics"]
            for key in ("mse", "rmse", "pearson", "spearman", "ci"):
                self.assertIn(key, metrics)
                self.assertIsNotNone(metrics[key])

            # Checkpoint should have been saved
            self.assertIn("checkpoint_path", summary)
            self.assertTrue(Path(summary["checkpoint_path"]).exists())


# Need math.isfinite in test class above
import math  # noqa: E402 (imported here to avoid shadowing top-level imports)

if __name__ == "__main__":
    unittest.main()
