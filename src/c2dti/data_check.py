"""Dataset precheck helpers for real C2DTI runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from src.c2dti.config_validation import validate_config
from src.c2dti.dataset_loader import DTIDataset, load_dti_dataset


def _required_dataset_files(dataset_name: str, dataset_path: Path) -> List[Path]:
    """Return the file paths a dataset must provide for a real run.

    This keeps the expected on-disk contract explicit so the precheck output
    can tell the user exactly which files must exist.
    """
    normalized_name = dataset_name.strip().upper()
    if normalized_name == "BINDINGDB":
        return [dataset_path]
    if normalized_name in {"DAVIS", "KIBA"}:
        return [
            dataset_path / "drug_smiles.txt",
            dataset_path / "target_sequences.txt",
            dataset_path / "Y.txt",
        ]
    return []


def summarize_dataset(dataset: DTIDataset) -> Dict[str, object]:
    """Build a small summary payload for validated dataset contents."""
    return {
        "dataset_name": dataset.metadata.get("source", "unknown"),
        "num_drugs": len(dataset.drugs),
        "num_targets": len(dataset.targets),
        "matrix_shape": list(dataset.interactions.shape),
        "is_placeholder": bool(dataset.metadata.get("is_placeholder", False)),
    }


def check_data(config_path: str) -> int:
    """Validate dataset availability and shape before a real run starts.

    Exit codes:
    - 0: dataset files are valid and loadable
    - 1: config path is missing
    - 2: config is invalid or does not define a dataset section
    - 3: dataset files are missing or invalid for a strict real run
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 1

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    errors = validate_config(cfg)
    if errors:
        print("[ERROR] Config validation failed:")
        for error in errors:
            print(f"- {error}")
        return 2

    dataset_cfg = cfg.get("dataset")
    if not dataset_cfg:
        print("[ERROR] dataset config is required for --check-data")
        return 2

    dataset_name = dataset_cfg["name"]
    dataset_path = Path(dataset_cfg["path"])
    required_files = _required_dataset_files(dataset_name, dataset_path)
    missing_files = [path for path in required_files if not path.exists()]

    print(f"[INFO] Checking dataset: {dataset_name}")
    print(f"[INFO] dataset.path={dataset_path}")
    for required_file in required_files:
        status = "OK" if required_file.exists() else "MISSING"
        print(f"[INFO] required_file[{status}]={required_file}")

    if missing_files:
        print("[ERROR] Dataset files are missing for a real run:")
        for missing_file in missing_files:
            print(f"- {missing_file}")
        return 3

    dataset = load_dti_dataset(dataset_name, dataset_path)
    dataset_summary = summarize_dataset(dataset)

    if bool(dataset_summary["is_placeholder"]):
        print("[ERROR] Dataset loader fell back to placeholder data")
        print("[ERROR] Real dataset files may exist but are invalid in format or shape")
        return 3

    print("[OK] Dataset check passed")
    print(f"dataset_name={dataset_summary['dataset_name']}")
    print(f"num_drugs={dataset_summary['num_drugs']}")
    print(f"num_targets={dataset_summary['num_targets']}")
    print(f"matrix_shape={dataset_summary['matrix_shape']}")
    return 0