"""Unit tests for src/c2dti/splitter.py.

Tests cover:
  - random split: correct counts, disjoint masks, reproducibility, NaN handling
  - cold_drug split: test drugs never appear in train
  - cold_target split: test targets never appear in train
  - split_dataset factory: routing, invalid inputs
  - runner integration: summary.json contains split block and test-only metrics
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.c2dti.dataset_loader import DTIDataset
from src.c2dti.splitter import split_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(
    n_drugs: int = 10,
    n_targets: int = 8,
    seed: int = 0,
    nan_fraction: float = 0.0,
) -> DTIDataset:
    """Synthetic DTIDataset for testing."""
    rng = np.random.RandomState(seed)
    interactions = rng.rand(n_drugs, n_targets).astype(np.float32)
    if nan_fraction > 0:
        mask = rng.rand(n_drugs, n_targets) < nan_fraction
        interactions[mask] = float("nan")
    drugs = [f"drug_{i}" for i in range(n_drugs)]
    targets = [f"target_{j}" for j in range(n_targets)]
    return DTIDataset(
        drugs=drugs,
        targets=targets,
        interactions=interactions,
        metadata={"source": "synthetic"},
    )


# ---------------------------------------------------------------------------
# Random split
# ---------------------------------------------------------------------------

class TestRandomSplit(unittest.TestCase):
    def setUp(self):
        self.ds = _make_dataset(n_drugs=10, n_targets=8)
        self.n_known = 80  # no NaNs → all 80 entries are known

    def test_masks_are_disjoint(self):
        """No pair should appear in both train and test."""
        train, test = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=0)
        overlap = train & test
        self.assertEqual(int(overlap.sum()), 0)

    def test_masks_cover_all_known_entries(self):
        """Every known entry should be in train or test."""
        train, test = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=0)
        union = train | test
        self.assertEqual(int(union.sum()), self.n_known)

    def test_test_size_is_approximately_correct(self):
        """Test set should be roughly test_ratio * n_known entries."""
        _, test = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=0)
        n_test = int(test.sum())
        expected = int(round(self.n_known * 0.2))
        self.assertEqual(n_test, expected)

    def test_reproducibility_with_same_seed(self):
        train1, test1 = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=7)
        train2, test2 = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=7)
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)

    def test_different_seeds_give_different_splits(self):
        _, test1 = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=1)
        _, test2 = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=2)
        self.assertFalse(np.array_equal(test1, test2))

    def test_nan_entries_not_in_either_mask(self):
        """NaN entries (unknown affinities) must not appear in train or test."""
        ds = _make_dataset(nan_fraction=0.2, seed=5)
        train, test = split_dataset(ds, strategy="random", test_ratio=0.2, seed=0)
        known = np.isfinite(ds.interactions)
        # All entries in train/test must be known
        self.assertTrue(np.all(known | ~train))
        self.assertTrue(np.all(known | ~test))

    def test_output_dtype_is_bool(self):
        train, test = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=0)
        self.assertEqual(train.dtype, bool)
        self.assertEqual(test.dtype, bool)

    def test_output_shape_matches_dataset(self):
        train, test = split_dataset(self.ds, strategy="random", test_ratio=0.2, seed=0)
        self.assertEqual(train.shape, (10, 8))
        self.assertEqual(test.shape, (10, 8))


# ---------------------------------------------------------------------------
# Cold drug split
# ---------------------------------------------------------------------------

class TestColdDrugSplit(unittest.TestCase):
    def setUp(self):
        self.ds = _make_dataset(n_drugs=10, n_targets=8)

    def test_masks_are_disjoint(self):
        train, test = split_dataset(self.ds, strategy="cold_drug", test_ratio=0.2, seed=0)
        self.assertEqual(int((train & test).sum()), 0)

    def test_test_drugs_absent_from_train(self):
        """No drug row that appears in test should also appear in train."""
        train, test = split_dataset(self.ds, strategy="cold_drug", test_ratio=0.2, seed=0)
        # For each drug row that has any test entry, that row must be all-False in train
        for drug_idx in range(self.ds.interactions.shape[0]):
            if test[drug_idx, :].any():
                self.assertFalse(train[drug_idx, :].any(),
                    f"drug_{drug_idx} appears in both train and test")

    def test_train_drugs_absent_from_test(self):
        train, test = split_dataset(self.ds, strategy="cold_drug", test_ratio=0.2, seed=0)
        for drug_idx in range(self.ds.interactions.shape[0]):
            if train[drug_idx, :].any():
                self.assertFalse(test[drug_idx, :].any(),
                    f"drug_{drug_idx} appears in both train and test")

    def test_approximately_correct_test_size(self):
        """About test_ratio of drugs should be held out."""
        train, test = split_dataset(self.ds, strategy="cold_drug", test_ratio=0.2, seed=0)
        n_test_drugs = int(test.any(axis=1).sum())
        expected = max(1, int(round(10 * 0.2)))
        self.assertEqual(n_test_drugs, expected)


# ---------------------------------------------------------------------------
# Cold target split
# ---------------------------------------------------------------------------

class TestColdTargetSplit(unittest.TestCase):
    def setUp(self):
        self.ds = _make_dataset(n_drugs=8, n_targets=10)

    def test_masks_are_disjoint(self):
        train, test = split_dataset(self.ds, strategy="cold_target", test_ratio=0.2, seed=0)
        self.assertEqual(int((train & test).sum()), 0)

    def test_test_targets_absent_from_train(self):
        train, test = split_dataset(self.ds, strategy="cold_target", test_ratio=0.2, seed=0)
        for target_idx in range(self.ds.interactions.shape[1]):
            if test[:, target_idx].any():
                self.assertFalse(train[:, target_idx].any(),
                    f"target_{target_idx} appears in both train and test")

    def test_train_targets_absent_from_test(self):
        train, test = split_dataset(self.ds, strategy="cold_target", test_ratio=0.2, seed=0)
        for target_idx in range(self.ds.interactions.shape[1]):
            if train[:, target_idx].any():
                self.assertFalse(test[:, target_idx].any(),
                    f"target_{target_idx} appears in both train and test")

    def test_approximately_correct_test_size(self):
        train, test = split_dataset(self.ds, strategy="cold_target", test_ratio=0.2, seed=0)
        n_test_targets = int(test.any(axis=0).sum())
        expected = max(1, int(round(10 * 0.2)))
        self.assertEqual(n_test_targets, expected)


# ---------------------------------------------------------------------------
# split_dataset factory / validation
# ---------------------------------------------------------------------------

class TestSplitDatasetFactory(unittest.TestCase):
    def setUp(self):
        self.ds = _make_dataset()

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            split_dataset(self.ds, strategy="unknown_strategy")

    def test_test_ratio_zero_raises(self):
        with self.assertRaises(ValueError):
            split_dataset(self.ds, test_ratio=0.0)

    def test_test_ratio_one_raises(self):
        with self.assertRaises(ValueError):
            split_dataset(self.ds, test_ratio=1.0)

    def test_test_ratio_negative_raises(self):
        with self.assertRaises(ValueError):
            split_dataset(self.ds, test_ratio=-0.1)

    def test_case_insensitive_strategy(self):
        # "RANDOM" should work the same as "random"
        train, test = split_dataset(self.ds, strategy="RANDOM", test_ratio=0.2, seed=0)
        self.assertEqual(int((train & test).sum()), 0)


# ---------------------------------------------------------------------------
# Config validation for split section
# ---------------------------------------------------------------------------

class TestSplitConfigValidation(unittest.TestCase):
    def _validate(self, split_cfg):
        from src.c2dti.config_validation import _validate_split_config
        return _validate_split_config(split_cfg)

    def test_none_returns_no_errors(self):
        self.assertEqual(self._validate(None), [])

    def test_valid_random_config(self):
        self.assertEqual(self._validate({"strategy": "random", "test_ratio": 0.2, "seed": 42}), [])

    def test_valid_cold_drug_config(self):
        self.assertEqual(self._validate({"strategy": "cold_drug"}), [])

    def test_valid_cold_target_config(self):
        self.assertEqual(self._validate({"strategy": "cold_target", "test_ratio": 0.3}), [])

    def test_invalid_strategy(self):
        errors = self._validate({"strategy": "warm_drug"})
        self.assertTrue(any("strategy" in e for e in errors))

    def test_test_ratio_out_of_range(self):
        errors = self._validate({"test_ratio": 1.5})
        self.assertTrue(any("test_ratio" in e for e in errors))

    def test_test_ratio_zero(self):
        errors = self._validate({"test_ratio": 0.0})
        self.assertTrue(any("test_ratio" in e for e in errors))

    def test_seed_must_be_int(self):
        errors = self._validate({"seed": "not_an_int"})
        self.assertTrue(any("seed" in e for e in errors))

    def test_not_a_dict_returns_error(self):
        errors = self._validate("random")
        self.assertTrue(len(errors) > 0)


# ---------------------------------------------------------------------------
# Runner integration: split produces test-only metrics in summary.json
# ---------------------------------------------------------------------------

class TestRunnerSplitIntegration(unittest.TestCase):
    def _run_with_split(self, strategy: str) -> dict:
        """Run run_once with a synthetic DAVIS dataset and a split config."""
        import yaml
        from src.c2dti.runner import run_once
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data" / "davis"
            data_dir.mkdir(parents=True)
            # 6 drugs × 5 targets = 30 known pairs (enough for a meaningful split)
            (data_dir / "drug_smiles.txt").write_text("\n".join(f"drug_{i}" for i in range(6)) + "\n")
            (data_dir / "target_sequences.txt").write_text("\n".join(f"tgt_{j}" for j in range(5)) + "\n")
            rows = ["0.9 0.2 0.5 0.1 0.8",
                    "0.3 0.7 0.4 0.6 0.2",
                    "0.5 0.6 0.9 0.3 0.1",
                    "0.1 0.4 0.7 0.8 0.5",
                    "0.8 0.3 0.2 0.9 0.6",
                    "0.4 0.5 0.6 0.7 0.3"]
            (data_dir / "Y.txt").write_text("\n".join(rows) + "\n")

            cfg = {
                "name": f"test_split_{strategy}",
                "protocol": "P1",
                "output": {"base_dir": tmpdir},
                "dataset": {"name": "DAVIS", "path": str(data_dir), "allow_placeholder": False},
                "model": {"name": "matrix_factorization", "latent_dim": 4, "epochs": 10, "lr": 0.05, "seed": 0},
                "split": {"strategy": strategy, "test_ratio": 0.2, "seed": 42},
            }
            cfg_path = Path(tmpdir) / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))
            rc = run_once(str(cfg_path))
            self.assertEqual(rc, 0)

            runs_dir = Path(tmpdir) / "runs"
            run_dirs = list(runs_dir.iterdir())
            self.assertEqual(len(run_dirs), 1)
            return json.loads((run_dirs[0] / "summary.json").read_text())

    def test_random_split_summary_has_split_block(self):
        summary = self._run_with_split("random")
        self.assertIn("split", summary)
        split = summary["split"]
        self.assertEqual(split["strategy"], "random")
        self.assertIn("n_train", split)
        self.assertIn("n_test", split)
        self.assertGreater(split["n_train"], 0)
        self.assertGreater(split["n_test"], 0)
        # Train and test must not overlap
        self.assertEqual(split["n_train"] + split["n_test"], 30)  # 6×5 all known

    def test_random_split_has_test_and_train_metrics(self):
        summary = self._run_with_split("random")
        self.assertIn("evaluation_metrics", summary)   # test-set metrics
        self.assertIn("train_metrics", summary)        # train-set fit quality
        for key in ("mse", "rmse", "pearson", "spearman", "ci"):
            self.assertIn(key, summary["evaluation_metrics"])
            self.assertIn(key, summary["train_metrics"])

    def test_cold_drug_split_summary_has_split_block(self):
        summary = self._run_with_split("cold_drug")
        self.assertIn("split", summary)
        self.assertEqual(summary["split"]["strategy"], "cold_drug")
        self.assertIn("evaluation_metrics", summary)

    def test_cold_target_split_summary_has_split_block(self):
        summary = self._run_with_split("cold_target")
        self.assertIn("split", summary)
        self.assertEqual(summary["split"]["strategy"], "cold_target")
        self.assertIn("evaluation_metrics", summary)

    def test_no_split_config_still_produces_evaluation_metrics(self):
        """Backward compatibility: no split → evaluation on all known pairs."""
        import yaml
        from src.c2dti.runner import run_once
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data" / "davis"
            data_dir.mkdir(parents=True)
            (data_dir / "drug_smiles.txt").write_text("drugA\ndrugB\ndrugC\n")
            (data_dir / "target_sequences.txt").write_text("tgtX\ntgtY\n")
            (data_dir / "Y.txt").write_text("0.9 0.2\n0.3 0.8\n0.5 0.6\n")
            cfg = {
                "name": "test_no_split",
                "protocol": "P1",
                "output": {"base_dir": tmpdir},
                "dataset": {"name": "DAVIS", "path": str(data_dir), "allow_placeholder": False},
                "model": {"name": "matrix_factorization", "latent_dim": 4, "epochs": 5, "lr": 0.05, "seed": 0},
                # NO split section
            }
            cfg_path = Path(tmpdir) / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))
            rc = run_once(str(cfg_path))
            self.assertEqual(rc, 0)

            runs_dir = Path(tmpdir) / "runs"
            summary = json.loads(list(runs_dir.iterdir())[0].joinpath("summary.json").read_text())
            self.assertIn("evaluation_metrics", summary)
            self.assertNotIn("split", summary)
            self.assertNotIn("train_metrics", summary)


if __name__ == "__main__":
    unittest.main()
