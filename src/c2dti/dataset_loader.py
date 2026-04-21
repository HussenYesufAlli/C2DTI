"""
Dataset loader for Drug-Target Interaction (DTI) benchmarks.

Supports BindingDB, DAVIS, and KIBA datasets with a unified interface.
Each dataset loader validates data format and returns standardized format:
  {
    'drugs': List[str],           # SMILES or identifiers
    'targets': List[str],         # Sequences or identifiers
    'interactions': np.ndarray,   # Shape (n_drugs, n_targets), binary or continuous
    'split_info': Dict[str, List[int]] (optional)
  }

Non-breaking design: Placeholder data returned if dataset files not found (enables testing
without actual data files). Real causal implementation can use this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DTIDataset:
    """Standard DTI dataset container."""
    drugs: List[str]  # Drug SMILES, IDs, or sequences
    targets: List[str]  # Target protein sequences or IDs
    interactions: np.ndarray  # Shape (n_drugs, n_targets); values 0/1 or continuous affinity
    metadata: Dict[str, Any]  # Dataset-specific metadata (threshold, source, etc.)


class DTIDatasetLoader(ABC):
    """Abstract base class for DTI dataset loaders."""

    @abstractmethod
    def load(self) -> DTIDataset:
        """Load and validate dataset. Return DTIDataset or raise error."""
        pass

    @staticmethod
    def _validate_file_exists(path: Path, dataset_name: str) -> bool:
        """Check if file exists; log warning if missing."""
        if not path.exists():
            print(f"[Warning] {dataset_name} file not found: {path}")
            return False
        return True


class BindingDBLoader(DTIDatasetLoader):
    """
    Load BindingDB dataset from CSV format.

    CSV must contain columns: Drug_ID (or Drug), Target_ID (or Target), Y (affinity).
    If file not found, returns minimal placeholder dataset for contract testing.
    """

    def __init__(self, csv_path: Path, threshold_pkd: float = 7.6):
        """
        Initialize BindingDB loader.

        Args:
            csv_path: Path to CSV file (columns: Drug_ID, Target_ID, Y)
            threshold_pkd: Threshold for binarization (default 7.6 from MINDG_CLASSA)
        """
        self.csv_path = Path(csv_path)
        self.threshold_pkd = threshold_pkd

    def load(self) -> DTIDataset:
        """Load BindingDB CSV and convert to standardized format."""
        if not self._validate_file_exists(self.csv_path, "BindingDB"):
            return self._placeholder_dataset("BindingDB")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"[Error] Failed to read BindingDB CSV: {e}")
            return self._placeholder_dataset("BindingDB")

        # Validate columns
        required = {"Drug_ID", "Target_ID", "Y"}
        cols = set(df.columns)
        if "Drug_ID" not in cols:
            if "Drug" in cols:
                df = df.rename(columns={"Drug": "Drug_ID"})
        if "Target_ID" not in cols:
            if "Target" in cols:
                df = df.rename(columns={"Target": "Target_ID"})

        missing = sorted(required.difference(set(df.columns)))
        if missing:
            print(f"[Error] BindingDB CSV missing columns: {missing}")
            return self._placeholder_dataset("BindingDB")

        # Extract unique drugs and targets
        drugs = sorted(df["Drug_ID"].unique().tolist())
        targets = sorted(df["Target_ID"].unique().tolist())

        # Create affinity matrix
        interactions = self._build_interaction_matrix(df, drugs, targets)

        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={
                "source": "BindingDB",
                "threshold_pkd": self.threshold_pkd,
                "n_samples": len(df),
            },
        )

    def _build_interaction_matrix(
        self, df: pd.DataFrame, drugs: List[str], targets: List[str]
    ) -> np.ndarray:
        """Build drug-target interaction matrix from DataFrame."""
        n_drugs, n_targets = len(drugs), len(targets)
        matrix = np.zeros((n_drugs, n_targets), dtype=np.float32)

        drug_idx_map = {drug: i for i, drug in enumerate(drugs)}
        target_idx_map = {target: j for j, target in enumerate(targets)}

        for _, row in df.iterrows():
            try:
                i = drug_idx_map[row["Drug_ID"]]
                j = target_idx_map[row["Target_ID"]]
                # Convert Y to pKd and binarize
                pkd = -np.log10(float(row["Y"]) * 1e-9 + 1e-10)
                label = 1 if pkd <= self.threshold_pkd else 0
                matrix[i, j] = label
            except (KeyError, ValueError):
                continue

        return matrix

    @staticmethod
    def _placeholder_dataset(source: str) -> DTIDataset:
        """Return minimal placeholder dataset for contract testing."""
        drugs = ["drug_0", "drug_1"]
        targets = ["target_0", "target_1"]
        interactions = np.array([[1, 0], [0, 1]], dtype=np.float32)
        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={"source": source, "is_placeholder": True},
        )


class DAVISLoader(DTIDatasetLoader):
    """
    Load DAVIS dataset from text files.

    Expected files:
      - drug_smiles.txt (one SMILES per line)
      - target_sequences.txt (one sequence per line)
      - Y.txt (interaction matrix, space or tab separated)

    If files not found, returns minimal placeholder dataset.
    """

    def __init__(self, data_dir: Path):
        """Initialize DAVIS loader."""
        self.data_dir = Path(data_dir)
        self.drug_file = self.data_dir / "drug_smiles.txt"
        self.target_file = self.data_dir / "target_sequences.txt"
        self.y_file = self.data_dir / "Y.txt"

    def load(self) -> DTIDataset:
        """Load DAVIS dataset from text files."""
        files_exist = all(
            self._validate_file_exists(f, f"DAVIS/{f.name}")
            for f in [self.drug_file, self.target_file, self.y_file]
        )
        if not files_exist:
            return self._placeholder_dataset("DAVIS")

        try:
            drugs = self._load_list(self.drug_file)
            targets = self._load_list(self.target_file)
            interactions = self._load_matrix(self.y_file)
        except Exception as e:
            print(f"[Error] Failed to load DAVIS dataset: {e}")
            return self._placeholder_dataset("DAVIS")

        if interactions.shape != (len(drugs), len(targets)):
            print(
                f"[Error] DAVIS Y matrix shape {interactions.shape} "
                f"does not match drugs ({len(drugs)}) x targets ({len(targets)})"
            )
            return self._placeholder_dataset("DAVIS")

        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={"source": "DAVIS", "n_drugs": len(drugs), "n_targets": len(targets)},
        )

    @staticmethod
    def _load_list(path: Path) -> List[str]:
        """Load list of items from file (one per line)."""
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _load_matrix(path: Path) -> np.ndarray:
        """Load interaction matrix from text file (space or tab separated)."""
        return np.loadtxt(path, dtype=np.float32)

    @staticmethod
    def _placeholder_dataset(source: str) -> DTIDataset:
        """Return minimal placeholder dataset."""
        drugs = ["drug_0", "drug_1"]
        targets = ["target_0", "target_1"]
        interactions = np.array([[0.8, 0.2], [0.3, 0.9]], dtype=np.float32)
        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={"source": source, "is_placeholder": True},
        )


class KIBALoader(DTIDatasetLoader):
    """
    Load KIBA dataset from text files.

    Same format as DAVIS:
      - drug_smiles.txt (one SMILES per line)
      - target_sequences.txt (one sequence per line)
      - Y.txt (interaction matrix, space or tab separated)

    If files not found, returns minimal placeholder dataset.
    """

    def __init__(self, data_dir: Path):
        """Initialize KIBA loader."""
        self.data_dir = Path(data_dir)
        self.drug_file = self.data_dir / "drug_smiles.txt"
        self.target_file = self.data_dir / "target_sequences.txt"
        self.y_file = self.data_dir / "Y.txt"

    def load(self) -> DTIDataset:
        """Load KIBA dataset from text files."""
        files_exist = all(
            self._validate_file_exists(f, f"KIBA/{f.name}")
            for f in [self.drug_file, self.target_file, self.y_file]
        )
        if not files_exist:
            return self._placeholder_dataset("KIBA")

        try:
            drugs = self._load_list(self.drug_file)
            targets = self._load_list(self.target_file)
            interactions = self._load_matrix(self.y_file)
        except Exception as e:
            print(f"[Error] Failed to load KIBA dataset: {e}")
            return self._placeholder_dataset("KIBA")

        if interactions.shape != (len(drugs), len(targets)):
            print(
                f"[Error] KIBA Y matrix shape {interactions.shape} "
                f"does not match drugs ({len(drugs)}) x targets ({len(targets)})"
            )
            return self._placeholder_dataset("KIBA")

        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={"source": "KIBA", "n_drugs": len(drugs), "n_targets": len(targets)},
        )

    @staticmethod
    def _load_list(path: Path) -> List[str]:
        """Load list of items from file (one per line)."""
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _load_matrix(path: Path) -> np.ndarray:
        """Load interaction matrix from text file."""
        return np.loadtxt(path, dtype=np.float32)

    @staticmethod
    def _placeholder_dataset(source: str) -> DTIDataset:
        """Return minimal placeholder dataset."""
        drugs = ["drug_0", "drug_1", "drug_2"]
        targets = ["target_0", "target_1", "target_2"]
        interactions = np.array(
            [[0.7, 0.2, 0.1], [0.3, 0.8, 0.4], [0.5, 0.6, 0.9]], dtype=np.float32
        )
        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={"source": source, "is_placeholder": True},
        )


def create_dataset_loader(dataset_name: str, data_path: Path) -> DTIDatasetLoader:
    """
    Factory function to create appropriate dataset loader.

    Args:
        dataset_name: One of "BindingDB", "DAVIS", "KIBA"
        data_path: Path to CSV file (BindingDB) or directory (DAVIS/KIBA)

    Returns:
        Appropriate DTIDatasetLoader instance

    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_name = dataset_name.upper()
    if dataset_name == "BINDINGDB":
        return BindingDBLoader(data_path)
    elif dataset_name == "DAVIS":
        return DAVISLoader(data_path)
    elif dataset_name == "KIBA":
        return KIBALoader(data_path)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of: BindingDB, DAVIS, KIBA"
        )


def load_dti_dataset(dataset_name: str, data_path: Path) -> DTIDataset:
    """
    Convenience function to load DTI dataset in one call.

    Args:
        dataset_name: One of "BindingDB", "DAVIS", "KIBA"
        data_path: Path to dataset files

    Returns:
        DTIDataset with drugs, targets, interactions, and metadata
    """
    loader = create_dataset_loader(dataset_name, data_path)
    return loader.load()
