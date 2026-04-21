"""
Unit tests for dataset_loader module.

Tests cover:
  - BindingDB loading from CSV
  - DAVIS loading from text files
  - KIBA loading from text files
  - Placeholder datasets (when files missing)
  - Factory function
  - Data validation
"""

import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

from src.c2dti.dataset_loader import (
    BindingDBLoader,
    DAVISLoader,
    DTIDataset,
    KIBALoader,
    create_dataset_loader,
    load_dti_dataset,
)


class TestBindingDBLoader(TestCase):
    """Test BindingDB dataset loading."""

    def test_load_valid_bindingdb_csv(self):
        """Load valid BindingDB CSV and validate output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bindingdb.csv"
            df = pd.DataFrame(
                {
                    "Drug_ID": ["D1", "D1", "D2"],
                    "Target_ID": ["T1", "T2", "T1"],
                    "Y": [1e-6, 1e-8, 1e-7],  # nM values
                }
            )
            df.to_csv(csv_path, index=False)

            loader = BindingDBLoader(csv_path)
            dataset = loader.load()

            # Validate structure
            assert isinstance(dataset, DTIDataset)
            assert len(dataset.drugs) == 2  # D1, D2
            assert len(dataset.targets) == 2  # T1, T2
            assert dataset.interactions.shape == (2, 2)
            assert dataset.metadata["source"] == "BindingDB"

    def test_bindingdb_placeholder_on_missing_file(self):
        """Return placeholder when CSV file not found."""
        loader = BindingDBLoader(Path("/nonexistent/file.csv"))
        dataset = loader.load()

        assert dataset.metadata.get("is_placeholder") is True
        assert len(dataset.drugs) > 0
        assert len(dataset.targets) > 0

    def test_bindingdb_placeholder_on_invalid_csv(self):
        """Return placeholder when CSV has invalid columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "invalid.csv"
            df = pd.DataFrame({"BadCol1": [1, 2], "BadCol2": [3, 4]})
            df.to_csv(csv_path, index=False)

            loader = BindingDBLoader(csv_path)
            dataset = loader.load()

            assert dataset.metadata.get("is_placeholder") is True

    def test_bindingdb_binarization(self):
        """Verify pKd binarization with threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bindingdb.csv"
            # Follow the existing MINDG BindingDB rule:
            # label = 1 when pKd <= threshold, else 0.
            # The pipeline expects Y in nM before converting to pKd.
            # 10 nM -> pKd ~= 7.96 -> 0, 1000 nM -> pKd ~= 6.00 -> 1.
            df = pd.DataFrame(
                {
                    "Drug_ID": ["D1", "D2"],
                    "Target_ID": ["T1", "T1"],
                    "Y": [10.0, 1000.0],
                }
            )
            df.to_csv(csv_path, index=False)

            loader = BindingDBLoader(csv_path, threshold_pkd=7.6)
            dataset = loader.load()

            assert dataset.interactions[0, 0] == 0  # D1-T1
            assert dataset.interactions[1, 0] == 1  # D2-T1


class TestDAVISLoader(TestCase):
    """Test DAVIS dataset loading."""

    def test_load_valid_davis_files(self):
        """Load valid DAVIS text files and validate output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._create_davis_files(data_dir)

            loader = DAVISLoader(data_dir)
            dataset = loader.load()

            assert len(dataset.drugs) == 2
            assert len(dataset.targets) == 3
            assert dataset.interactions.shape == (2, 3)
            assert dataset.metadata["source"] == "DAVIS"

    def test_davis_placeholder_on_missing_files(self):
        """Return placeholder when any required file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            # Only create one file
            (data_dir / "drug_smiles.txt").write_text("C\nCC\n")

            loader = DAVISLoader(data_dir)
            dataset = loader.load()

            assert dataset.metadata.get("is_placeholder") is True

    def test_davis_placeholder_on_shape_mismatch(self):
        """Return placeholder when Y matrix dimensions don't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "drug_smiles.txt").write_text("C\nCC\n")  # 2 drugs
            (data_dir / "target_sequences.txt").write_text("ACGT\nACGT\n")  # 2 targets
            # Y matrix has wrong shape (3x3)
            np.savetxt(
                data_dir / "Y.txt", np.random.rand(3, 3), fmt="%.4f"
            )

            loader = DAVISLoader(data_dir)
            dataset = loader.load()

            assert dataset.metadata.get("is_placeholder") is True

    @staticmethod
    def _create_davis_files(data_dir: Path):
        """Helper to create valid DAVIS test files."""
        (data_dir / "drug_smiles.txt").write_text("C\nCC\n")  # 2 SMILES
        (data_dir / "target_sequences.txt").write_text("ACGT\nACGT\nACGT\n")  # 3 sequences
        np.savetxt(data_dir / "Y.txt", np.random.rand(2, 3), fmt="%.4f")


class TestKIBALoader(TestCase):
    """Test KIBA dataset loading."""

    def test_load_valid_kiba_files(self):
        """Load valid KIBA text files and validate output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._create_kiba_files(data_dir)

            loader = KIBALoader(data_dir)
            dataset = loader.load()

            assert len(dataset.drugs) == 2
            assert len(dataset.targets) == 3
            assert dataset.interactions.shape == (2, 3)
            assert dataset.metadata["source"] == "KIBA"

    def test_kiba_placeholder_on_missing_files(self):
        """Return placeholder when any required file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            loader = KIBALoader(data_dir)
            dataset = loader.load()

            assert dataset.metadata.get("is_placeholder") is True

    @staticmethod
    def _create_kiba_files(data_dir: Path):
        """Helper to create valid KIBA test files."""
        (data_dir / "drug_smiles.txt").write_text("C\nCC\n")
        (data_dir / "target_sequences.txt").write_text("ACGT\nACGT\nACGT\n")
        np.savetxt(data_dir / "Y.txt", np.random.rand(2, 3), fmt="%.4f")


class TestFactoryFunction(TestCase):
    """Test dataset loader factory and convenience functions."""

    def test_create_bindingdb_loader(self):
        """Factory creates BindingDB loader correctly."""
        loader = create_dataset_loader("BindingDB", Path("/tmp/dummy.csv"))
        assert isinstance(loader, BindingDBLoader)

    def test_create_davis_loader(self):
        """Factory creates DAVIS loader correctly."""
        loader = create_dataset_loader("DAVIS", Path("/tmp/dummy"))
        assert isinstance(loader, DAVISLoader)

    def test_create_kiba_loader(self):
        """Factory creates KIBA loader correctly."""
        loader = create_dataset_loader("KIBA", Path("/tmp/dummy"))
        assert isinstance(loader, KIBALoader)

    def test_factory_case_insensitive(self):
        """Factory works with any case."""
        loader1 = create_dataset_loader("bindingdb", Path("/tmp/dummy.csv"))
        loader2 = create_dataset_loader("BindingDB", Path("/tmp/dummy.csv"))
        assert isinstance(loader1, BindingDBLoader)
        assert isinstance(loader2, BindingDBLoader)

    def test_factory_invalid_dataset(self):
        """Factory raises error for unknown dataset."""
        with self.assertRaises(ValueError):
            create_dataset_loader("UNKNOWN", Path("/tmp/dummy"))

    def test_convenience_load_function(self):
        """load_dti_dataset convenience function works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "drug_smiles.txt").write_text("C\nCC\n")
            (data_dir / "target_sequences.txt").write_text("ACGT\nACGT\n")
            np.savetxt(data_dir / "Y.txt", np.random.rand(2, 2), fmt="%.4f")

            dataset = load_dti_dataset("DAVIS", data_dir)
            assert isinstance(dataset, DTIDataset)
            assert len(dataset.drugs) == 2
            assert len(dataset.targets) == 2


class TestDTIDatasetStructure(TestCase):
    """Test DTIDataset dataclass and expected structure."""

    def test_dataset_structure_complete(self):
        """DTIDataset has all expected fields."""
        dataset = DTIDataset(
            drugs=["D1", "D2"],
            targets=["T1", "T2"],
            interactions=np.array([[1, 0], [0, 1]]),
            metadata={"source": "test"},
        )

        assert hasattr(dataset, "drugs")
        assert hasattr(dataset, "targets")
        assert hasattr(dataset, "interactions")
        assert hasattr(dataset, "metadata")

    def test_interactions_dtype_float32(self):
        """Interactions matrix uses float32 dtype."""
        dataset = DTIDataset(
            drugs=["D1"],
            targets=["T1"],
            interactions=np.array([[0.5]], dtype=np.float32),
            metadata={},
        )

        assert dataset.interactions.dtype == np.float32
