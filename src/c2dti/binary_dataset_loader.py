"""Binary dataset loaders for C2DTI.

This module is intentionally separate from the regression loaders so the
continuous-affinity path stays unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class BinaryDTIDataset:
    """Container for binary DTI data used by the binary runner."""

    drugs: List[str]
    targets: List[str]
    interactions: np.ndarray
    metadata: Dict[str, Any]


class BinaryDTIDatasetLoader(ABC):
    """Base interface for binary dataset loaders."""

    @abstractmethod
    def load(self) -> BinaryDTIDataset:
        """Load one dataset and return a BinaryDTIDataset object."""

    @staticmethod
    def _validate_file_exists(path: Path, dataset_name: str) -> bool:
        # Keep file existence checks centralized for consistent error messages.
        if not path.exists():
            print(f"[Warning] {dataset_name} binary file not found: {path}")
            return False
        return True


class _BinaryMatrixCSVLoader(BinaryDTIDatasetLoader):
    """Shared loader for flat binary CSV files.

    Expected CSV columns:
      Drug_ID, Drug, Target_ID, Target, Label
    """

    def __init__(self, csv_path: Path, source_name: str):
        # Store the on-disk file path and source name for metadata.
        self.csv_path = Path(csv_path)
        self.source_name = source_name

    def load(self) -> BinaryDTIDataset:
        # Load CSV first, then validate schema before matrix construction.
        if not self._validate_file_exists(self.csv_path, self.source_name):
            return self._placeholder_dataset(self.source_name)

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as exc:
            print(f"[Error] Failed to read {self.source_name} binary CSV: {exc}")
            return self._placeholder_dataset(self.source_name)

        required = {"Drug_ID", "Drug", "Target_ID", "Target", "Label"}
        missing = sorted(required.difference(set(df.columns)))
        if missing:
            print(f"[Error] {self.source_name} binary CSV missing columns: {missing}")
            return self._placeholder_dataset(self.source_name)

        # Clean and normalize dtypes to stable string IDs and numeric labels.
        df = df.dropna(subset=["Drug_ID", "Drug", "Target_ID", "Target", "Label"]).copy()
        df["Drug_ID"] = df["Drug_ID"].astype(str)
        df["Target_ID"] = df["Target_ID"].astype(str)
        df["Drug"] = df["Drug"].astype(str)
        df["Target"] = df["Target"].astype(str)
        df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
        df = df.dropna(subset=["Label"])

        # Enforce strict binary labels so downstream metrics are correct.
        df = df[df["Label"].isin([0, 1, 0.0, 1.0])].copy()
        df["Label"] = df["Label"].astype(np.float32)

        # Build deterministic index orders from IDs to keep run reproducibility.
        drug_map = df.drop_duplicates("Drug_ID").sort_values("Drug_ID").set_index("Drug_ID")["Drug"]
        target_map = df.drop_duplicates("Target_ID").sort_values("Target_ID").set_index("Target_ID")["Target"]
        drug_ids = drug_map.index.tolist()
        target_ids = target_map.index.tolist()
        drugs = drug_map.tolist()
        targets = target_map.tolist()

        drug_idx = {d: i for i, d in enumerate(drug_ids)}
        target_idx = {t: j for j, t in enumerate(target_ids)}

        # Use NaN for unknown pairs so split/eval only uses observed rows.
        interactions = np.full((len(drugs), len(targets)), np.nan, dtype=np.float32)

        for row in df.itertuples(index=False):
            i = drug_idx.get(str(row.Drug_ID))
            j = target_idx.get(str(row.Target_ID))
            if i is None or j is None:
                continue
            interactions[i, j] = float(row.Label)

        return BinaryDTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={
                "source": self.source_name,
                "task_type": "binary_classification",
                "csv_path": str(self.csv_path),
                "n_rows": int(len(df)),
                "n_drugs": int(len(drugs)),
                "n_targets": int(len(targets)),
            },
        )

    @staticmethod
    def _placeholder_dataset(source: str) -> BinaryDTIDataset:
        # Return a tiny contract-testing dataset when files are missing.
        drugs = ["drug_0", "drug_1"]
        targets = ["target_0", "target_1"]
        interactions = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        return BinaryDTIDataset(
            drugs=drugs,
            targets=targets,
            interactions=interactions,
            metadata={
                "source": source,
                "task_type": "binary_classification",
                "is_placeholder": True,
            },
        )


class BinaryDAVISLoader(_BinaryMatrixCSVLoader):
    """Binary DAVIS loader."""

    def __init__(self, csv_path: Path):
        # Hard-code source name so metadata stays consistent across runs.
        super().__init__(csv_path=csv_path, source_name="DAVIS")


class BinaryKIBALoader(_BinaryMatrixCSVLoader):
    """Binary KIBA loader."""

    def __init__(self, csv_path: Path):
        # Hard-code source name so metadata stays consistent across runs.
        super().__init__(csv_path=csv_path, source_name="KIBA")


class BinaryBindingDBLoader(_BinaryMatrixCSVLoader):
    """Binary BindingDB loader."""

    def __init__(self, csv_path: Path):
        # Hard-code source name so metadata stays consistent across runs.
        super().__init__(csv_path=csv_path, source_name="BindingDB")


def create_binary_dataset_loader(dataset_name: str, csv_path: Path) -> BinaryDTIDatasetLoader:
    """Factory for binary loaders.

    Args:
        dataset_name: BindingDB, DAVIS, or KIBA.
        csv_path: Path to the binarized CSV for that dataset.
    """

    # Normalize dataset name so callers can use any capitalization.
    normalized = str(dataset_name).strip().upper()
    if normalized == "DAVIS":
        return BinaryDAVISLoader(csv_path)
    if normalized == "KIBA":
        return BinaryKIBALoader(csv_path)
    if normalized in {"BINDINGDB", "BINDINGDB_KD"}:
        return BinaryBindingDBLoader(csv_path)

    raise ValueError(
        f"Unknown binary dataset: {dataset_name}. Must be one of: BindingDB, BindingDB_Kd, DAVIS, KIBA"
    )


def load_binary_dti_dataset(dataset_name: str, csv_path: Path) -> BinaryDTIDataset:
    """Convenience API for binary DTI loading in one call."""

    # Keep external call sites small by wrapping factory + load.
    loader = create_binary_dataset_loader(dataset_name=dataset_name, csv_path=csv_path)
    return loader.load()
