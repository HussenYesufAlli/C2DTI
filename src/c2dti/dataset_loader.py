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

        # Normalise column names.
        # TDC returns "Drug (SMILES)" / "Target (sequence)"; the workspace CSV
        # uses "Drug" / "Target"; both are treated as the molecular content columns.
        col_map = {}
        if "Drug (SMILES)" in df.columns:
            col_map["Drug (SMILES)"] = "Drug"
        if "Target (sequence)" in df.columns:
            col_map["Target (sequence)"] = "Target"
        # Workspace CSV uses "Label" instead of "Y"
        if "Label" in df.columns and "Y" not in df.columns:
            col_map["Label"] = "Y"
        if col_map:
            df = df.rename(columns=col_map)

        # Require at minimum Drug_ID, Target_ID and an affinity column (Y).
        # Drug (SMILES) and Target (sequence) columns are used when present so
        # that build_string_feature_matrix sees real chemical content, not IDs.
        for required_col in ("Drug_ID", "Target_ID", "Y"):
            if required_col not in df.columns:
                print(f"[Error] BindingDB CSV missing required column: {required_col}")
                return self._placeholder_dataset("BindingDB")

        # Normalize potentially mixed-type IDs (numeric or string) into stable strings.
        # This avoids downstream feature builders receiving float IDs.
        dropna_cols = ["Drug_ID", "Target_ID", "Y"]
        df = df.dropna(subset=dropna_cols).copy()
        df["Drug_ID"] = df["Drug_ID"].astype(str)
        df["Target_ID"] = df["Target_ID"].astype(str)
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df = df.dropna(subset=["Y"])

        # Use SMILES / sequence as the drug/target representation when available.
        # This is critical: feature encoders (build_string_feature_matrix) hash
        # the string content, so hashing a SMILES gives chemical signal while
        # hashing a bare CID number gives none.
        has_smiles = "Drug" in df.columns
        has_seq = "Target" in df.columns
        if has_smiles:
            df["Drug"] = df["Drug"].fillna("").astype(str)
        if has_seq:
            df["Target"] = df["Target"].fillna("").astype(str)

        # Build sorted unique lists.
        # When SMILES / sequences are present, deduplicate on content (not on ID)
        # so the feature matrix rows correspond to distinct molecules.
        if has_smiles:
            # Map each unique SMILES to its Drug_ID for interaction matrix lookup
            smiles_series = df["Drug"]
            drugs = sorted(smiles_series.unique().tolist())
            drug_col = "Drug"
        else:
            drugs = sorted(df["Drug_ID"].unique().tolist())
            drug_col = "Drug_ID"

        if has_seq:
            seq_series = df["Target"]
            targets = sorted(seq_series.unique().tolist())
            target_col = "Target"
        else:
            targets = sorted(df["Target_ID"].unique().tolist())
            target_col = "Target_ID"

        # Create affinity matrix
        interactions = self._build_interaction_matrix(df, drugs, targets, drug_col, target_col)

        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={
                "source": "BindingDB",
                "threshold_pkd": self.threshold_pkd,
                "n_samples": len(df),
                "has_smiles": has_smiles,
                "has_sequences": has_seq,
            },
        )

    def _build_interaction_matrix(
        self,
        df: pd.DataFrame,
        drugs: List[str],
        targets: List[str],
        drug_col: str = "Drug_ID",
        target_col: str = "Target_ID",
    ) -> np.ndarray:
        """Build drug-target interaction matrix from DataFrame.

        drug_col / target_col specify which DataFrame column to use as the
        index key — either the content column (SMILES / sequence) or the ID.
        """
        n_drugs, n_targets = len(drugs), len(targets)
        matrix = np.zeros((n_drugs, n_targets), dtype=np.float32)

        drug_idx_map = {drug: i for i, drug in enumerate(drugs)}
        target_idx_map = {target: j for j, target in enumerate(targets)}

        for _, row in df.iterrows():
            try:
                i = drug_idx_map[row[drug_col]]
                j = target_idx_map[row[target_col]]
                # Y column: if it came from "Label" it is already binary (0/1).
                # If it is a raw affinity in nM, convert to pKd and binarize.
                y_val = float(row["Y"])
                if y_val <= 1.0:
                    # Looks like a pre-binarized label (0 or 1) — use directly
                    label = int(y_val)
                else:
                    # Raw affinity in nM — convert and threshold
                    pkd = -np.log10(y_val * 1e-9 + 1e-10)
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


class _MatrixCSVLoader(DTIDatasetLoader):
    """
    Shared loader for datasets stored as a flat CSV with columns:
      Drug_ID, Drug, Target_ID, Target, Y

    Drug_ID and Target_ID are integer positional indices.
    Drug is the SMILES string; Target is the amino acid sequence.
    Y is the raw affinity value.

    This is the canonical storage format for DAVIS and KIBA.
    """

    def __init__(self, data_dir: Path, source_name: str):
        """
        Args:
            data_dir: Directory containing <source_name.lower()>.csv
            source_name: Dataset name used in metadata and error messages.
        """
        self.data_dir = Path(data_dir)
        self.source_name = source_name
        self.csv_path = self.data_dir / f"{source_name.lower()}.csv"

    def load(self) -> DTIDataset:
        """Load dataset from flat CSV (Drug_ID, Drug, Target_ID, Target, Y)."""
        if not self._validate_file_exists(self.csv_path, self.source_name):
            return self._placeholder_dataset(self.source_name)

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"[Error] Failed to read {self.source_name} CSV: {e}")
            return self._placeholder_dataset(self.source_name)

        # Validate required columns are present
        required = {"Drug_ID", "Drug", "Target_ID", "Target", "Y"}
        missing = sorted(required.difference(set(df.columns)))
        if missing:
            print(f"[Error] {self.source_name} CSV missing columns: {missing}")
            return self._placeholder_dataset(self.source_name)

        # Ensure types are clean — Drug_ID / Target_ID are integers (positional),
        # Drug and Target are strings (SMILES / sequences).
        df = df.dropna(subset=["Drug_ID", "Drug", "Target_ID", "Target", "Y"]).copy()
        df["Drug_ID"]   = pd.to_numeric(df["Drug_ID"],   errors="coerce").astype("Int64")
        df["Target_ID"] = pd.to_numeric(df["Target_ID"], errors="coerce").astype("Int64")
        df["Drug"]      = df["Drug"].astype(str)
        df["Target"]    = df["Target"].astype(str)
        df["Y"]         = pd.to_numeric(df["Y"], errors="coerce")
        df = df.dropna(subset=["Drug_ID", "Target_ID", "Y"])

        # Reconstruct ordered lists: SMILES ordered by Drug_ID, sequences by Target_ID.
        # This preserves the original index ordering from the matrix.
        drug_map   = df.drop_duplicates("Drug_ID").sort_values("Drug_ID").set_index("Drug_ID")["Drug"]
        target_map = df.drop_duplicates("Target_ID").sort_values("Target_ID").set_index("Target_ID")["Target"]
        drugs   = drug_map.tolist()    # list of SMILES strings, index = Drug_ID
        targets = target_map.tolist()  # list of sequences,     index = Target_ID

        # Build affinity matrix (Drug_ID × Target_ID) from the flat rows
        n_drugs, n_targets = len(drugs), len(targets)
        interactions = np.zeros((n_drugs, n_targets), dtype=np.float32)
        for row in df.itertuples(index=False):
            i = int(row.Drug_ID)
            j = int(row.Target_ID)
            if 0 <= i < n_drugs and 0 <= j < n_targets:
                interactions[i, j] = float(row.Y)

        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={
                "source": self.source_name,
                "n_drugs": n_drugs,
                "n_targets": n_targets,
                "n_samples": len(df),
            },
        )

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


class DAVISLoader(_MatrixCSVLoader):
    """
    Load DAVIS dataset from flat CSV (data/davis/davis.csv).

    CSV columns: Drug_ID (int), Drug (SMILES), Target_ID (int),
                 Target (sequence), Y (affinity).
    """

    def __init__(self, data_dir: Path):
        super().__init__(data_dir, "DAVIS")


class KIBALoader(_MatrixCSVLoader):
    """
    Load KIBA dataset from flat CSV (data/kiba/kiba.csv).

    CSV columns: Drug_ID (int), Drug (SMILES), Target_ID (int),
                 Target (sequence), Y (affinity).
    """

    def __init__(self, data_dir: Path):
        super().__init__(data_dir, "KIBA")


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
