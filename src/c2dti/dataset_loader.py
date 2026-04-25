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

Non-breaking design: Fallback scaffold data is returned if dataset files are not found
(enables contract testing without source datasets). Full causal implementation can use this interface.
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
    If file not found, returns a minimal fallback dataset for contract testing.
    """

    def __init__(self, csv_path: Path, threshold_pkd: float = 7.6):
        """
        Initialize BindingDB loader.

        Args:
            csv_path: Path to CSV file (columns: Drug_ID, Target_ID, Y)
            threshold_pkd: Threshold for binarization (default 7.6 )
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
        """Return a minimal fallback dataset for contract testing."""
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
    Shared loader for DAVIS/KIBA datasets.

    Supported input layouts (non-breaking):
      1) Flat CSV with columns: Drug_ID, Drug, Target_ID, Target, Y
      2) Legacy directory format:
         - drug_smiles.txt
         - target_sequences.txt
         - Y.txt

    The loader auto-detects the layout so existing workflows keep working.
    """

    def __init__(self, data_dir: Path, source_name: str):
        """
        Args:
            data_dir: Directory containing a dataset CSV or a direct CSV path
            source_name: Dataset name used in metadata and error messages.
        """
        self.data_dir = Path(data_dir)
        self.source_name = source_name

        # Flat CSV path candidates (either direct file input or common directory naming).
        if self.data_dir.is_file() and self.data_dir.suffix.lower() == ".csv":
            self.csv_path = self.data_dir
        else:
            csv_candidates = [
                self.data_dir / f"{source_name.lower()}.csv",
                self.data_dir / f"{source_name.upper()}.csv",
            ]
            self.csv_path = next((p for p in csv_candidates if p.exists()), csv_candidates[0])

        # Legacy text-file layout paths.
        self.drug_txt_path = self.data_dir / "drug_smiles.txt"
        self.target_txt_path = self.data_dir / "target_sequences.txt"
        self.y_txt_path = self.data_dir / "Y.txt"

    def load(self) -> DTIDataset:
        """Load dataset from flat CSV or legacy text-file layout."""
        if self.csv_path.exists():
            return self._load_from_csv()

        # Fallback for historical DAVIS/KIBA contracts used in tests and
        # strict checks that still produce drug_smiles.txt/target_sequences.txt/Y.txt.
        if self.drug_txt_path.exists() and self.target_txt_path.exists() and self.y_txt_path.exists():
            return self._load_from_legacy_text()

        # Preserve prior warning behavior for missing primary path.
        self._validate_file_exists(self.csv_path, self.source_name)
        return self._placeholder_dataset(self.source_name)

    def _load_from_csv(self) -> DTIDataset:
        """Load dataset from flat CSV (Drug_ID, Drug, Target_ID, Target, Y)."""
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"[Error] Failed to read {self.source_name} CSV: {e}")
            return self._placeholder_dataset(self.source_name)

        required = {"Drug_ID", "Drug", "Target_ID", "Target", "Y"}
        missing = sorted(required.difference(set(df.columns)))
        if missing:
            print(f"[Error] {self.source_name} CSV missing columns: {missing}")
            return self._placeholder_dataset(self.source_name)

        df = df.dropna(subset=["Drug_ID", "Drug", "Target_ID", "Target", "Y"]).copy()
        df["Drug_ID"] = df["Drug_ID"].astype(str)
        df["Target_ID"] = df["Target_ID"].astype(str)
        df["Drug"] = df["Drug"].astype(str)
        df["Target"] = df["Target"].astype(str)
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df = df.dropna(subset=["Drug_ID", "Target_ID", "Y"])

        drug_map = df.drop_duplicates("Drug_ID").sort_values("Drug_ID").set_index("Drug_ID")["Drug"]
        target_map = df.drop_duplicates("Target_ID").sort_values("Target_ID").set_index("Target_ID")["Target"]
        drug_ids = drug_map.index.tolist()
        target_ids = target_map.index.tolist()
        drugs = drug_map.tolist()
        targets = target_map.tolist()
        drug_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
        target_idx = {target_id: idx for idx, target_id in enumerate(target_ids)}

        n_drugs, n_targets = len(drugs), len(targets)
        interactions = np.zeros((n_drugs, n_targets), dtype=np.float32)
        for row in df.itertuples(index=False):
            i = drug_idx.get(str(row.Drug_ID))
            j = target_idx.get(str(row.Target_ID))
            if i is not None and j is not None:
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
                "layout": "csv",
            },
        )

    def _load_from_legacy_text(self) -> DTIDataset:
        """Load dataset from legacy text files (drug_smiles/target_sequences/Y)."""
        try:
            drugs = [line.strip() for line in self.drug_txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            targets = [line.strip() for line in self.target_txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            y_raw = np.loadtxt(self.y_txt_path, dtype=np.float32)
        except Exception as e:
            print(f"[Error] Failed to read {self.source_name} legacy text files: {e}")
            return self._placeholder_dataset(self.source_name)

        if not drugs or not targets:
            print(f"[Error] {self.source_name} legacy text files contain no entities")
            return self._placeholder_dataset(self.source_name)

        if y_raw.ndim == 0:
            y = np.array([[float(y_raw)]], dtype=np.float32)
        elif y_raw.ndim == 1:
            if len(drugs) == 1:
                y = y_raw.reshape(1, -1).astype(np.float32)
            elif len(targets) == 1:
                y = y_raw.reshape(-1, 1).astype(np.float32)
            else:
                y = y_raw.reshape(1, -1).astype(np.float32)
        else:
            y = y_raw.astype(np.float32)

        expected_shape = (len(drugs), len(targets))
        if y.shape != expected_shape:
            print(
                f"[Error] {self.source_name} Y.txt shape mismatch: got {tuple(y.shape)}, "
                f"expected {expected_shape} from drug/target files"
            )
            return self._placeholder_dataset(self.source_name)

        return DTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=y,
            metadata={
                "source": self.source_name,
                "n_drugs": len(drugs),
                "n_targets": len(targets),
                "n_samples": int(y.size),
                "layout": "legacy_text",
            },
        )

    @staticmethod
    def _placeholder_dataset(source: str) -> DTIDataset:
        """Return a minimal fallback dataset."""
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
    Load DAVIS dataset from flat CSV (for example datasets/DAVIS.csv).

    CSV columns: Drug_ID (int), Drug (SMILES), Target_ID (int),
                 Target (sequence), Y (affinity).
    """

    def __init__(self, data_dir: Path):
        super().__init__(data_dir, "DAVIS")


class KIBALoader(_MatrixCSVLoader):
    """
    Load KIBA dataset from flat CSV (for example datasets/KIBA.csv).

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
    if dataset_name in {"BINDINGDB", "BINDINGDB_KD"}:
        return BindingDBLoader(data_path)
    elif dataset_name == "DAVIS":
        return DAVISLoader(data_path)
    elif dataset_name == "KIBA":
        return KIBALoader(data_path)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of: BindingDB, BindingDB_Kd, DAVIS, KIBA"
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
