"""Dataset precheck helpers for real C2DTI runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _dataset_schema_details(dataset_name: str, dataset_path: Path) -> Dict[str, Any]:
    """Describe the expected dataset structure for the JSON report."""
    normalized_name = dataset_name.strip().upper()
    if normalized_name == "BINDINGDB":
        return {
            "dataset_type": "csv",
            "path_kind": "file",
            "expected_path": str(dataset_path),
            "required_columns": ["Drug_ID", "Target_ID", "Y"],
            "optional_alias_columns": {
                "Drug_ID": ["Drug"],
                "Target_ID": ["Target"],
            },
            "notes": [
                "BindingDB path should point to a CSV file.",
                "The loader converts Y from nM-like affinity values into pKd-based binary labels.",
            ],
        }

    if normalized_name in {"DAVIS", "KIBA"}:
        return {
            "dataset_type": "directory",
            "path_kind": "directory",
            "expected_path": str(dataset_path),
            "required_files": [
                "drug_smiles.txt",
                "target_sequences.txt",
                "Y.txt",
            ],
            "file_descriptions": {
                "drug_smiles.txt": "One drug SMILES string per line.",
                "target_sequences.txt": "One target protein sequence per line.",
                "Y.txt": "Whitespace-delimited numeric interaction matrix with shape (num_drugs, num_targets).",
            },
            "notes": [
                f"{dataset_name} path should point to a directory.",
                "The number of rows in Y.txt must match the number of drug lines.",
                "The number of columns in Y.txt must match the number of target lines.",
            ],
        }

    return {
        "dataset_type": "unknown",
        "path_kind": "unknown",
        "expected_path": str(dataset_path),
        "notes": ["Unknown dataset; no schema details available."],
    }


def summarize_dataset(dataset: DTIDataset) -> Dict[str, object]:
    """Build a small summary payload for validated dataset contents."""
    return {
        "dataset_name": dataset.metadata.get("source", "unknown"),
        "num_drugs": len(dataset.drugs),
        "num_targets": len(dataset.targets),
        "matrix_shape": list(dataset.interactions.shape),
        "is_placeholder": bool(dataset.metadata.get("is_placeholder", False)),
    }


def _resolve_report_path(cfg: Dict[str, Any], config_path: Path) -> Path:
    """Build a stable JSON report path under output/checks.

    We keep the filename deterministic so each config has one reusable report
    that gets refreshed every time the user runs the precheck.
    """
    base_dir = Path(cfg.get("output", {}).get("base_dir", "outputs"))
    report_dir = base_dir / "checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{config_path.stem}_data_check.json"


def _write_report(report_path: Path, payload: Dict[str, Any]) -> None:
    """Persist the dataset precheck report as pretty JSON."""
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_required_file_report(required_files: List[Path]) -> List[Dict[str, str]]:
    """Describe each required dataset file and whether it exists."""
    report_rows: List[Dict[str, str]] = []
    for required_file in required_files:
        report_rows.append(
            {
                "path": str(required_file),
                "status": "ok" if required_file.exists() else "missing",
            }
        )
    return report_rows


def _emit_report(report_path: Optional[Path], payload: Dict[str, Any]) -> None:
    """Write the report when a valid config gives us a destination path."""
    if report_path is None:
        return

    _write_report(report_path, payload)
    print(f"report={report_path}")


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

    report_path = _resolve_report_path(cfg, cfg_path)

    dataset_cfg = cfg.get("dataset")
    if not dataset_cfg:
        print("[ERROR] dataset config is required for --check-data")
        _emit_report(
            report_path,
            {
                "status": "error",
                "exit_code": 2,
                "config": str(cfg_path),
                "reason": "dataset config is required for --check-data",
            },
        )
        return 2

    dataset_name = dataset_cfg["name"]
    dataset_path = Path(dataset_cfg["path"])
    required_files = _required_dataset_files(dataset_name, dataset_path)
    missing_files = [path for path in required_files if not path.exists()]
    file_report = _build_required_file_report(required_files)
    schema_details = _dataset_schema_details(dataset_name, dataset_path)

    print(f"[INFO] Checking dataset: {dataset_name}")
    print(f"[INFO] dataset.path={dataset_path}")
    for required_file in required_files:
        status = "OK" if required_file.exists() else "MISSING"
        print(f"[INFO] required_file[{status}]={required_file}")

    if missing_files:
        print("[ERROR] Dataset files are missing for a real run:")
        for missing_file in missing_files:
            print(f"- {missing_file}")
        _emit_report(
            report_path,
            {
                "status": "error",
                "exit_code": 3,
                "config": str(cfg_path),
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "dataset_schema": schema_details,
                "required_files": file_report,
                "missing_files": [str(path) for path in missing_files],
                "reason": "Dataset files are missing for a real run",
            },
        )
        return 3

    dataset = load_dti_dataset(dataset_name, dataset_path)
    dataset_summary = summarize_dataset(dataset)

    if bool(dataset_summary["is_placeholder"]):
        print("[ERROR] Dataset loader fell back to placeholder data")
        print("[ERROR] Real dataset files may exist but are invalid in format or shape")
        _emit_report(
            report_path,
            {
                "status": "error",
                "exit_code": 3,
                "config": str(cfg_path),
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "dataset_schema": schema_details,
                "required_files": file_report,
                "dataset_summary": dataset_summary,
                "reason": "Dataset loader fell back to placeholder data",
            },
        )
        return 3

    print("[OK] Dataset check passed")
    print(f"dataset_name={dataset_summary['dataset_name']}")
    print(f"num_drugs={dataset_summary['num_drugs']}")
    print(f"num_targets={dataset_summary['num_targets']}")
    print(f"matrix_shape={dataset_summary['matrix_shape']}")
    _emit_report(
        report_path,
        {
            "status": "ok",
            "exit_code": 0,
            "config": str(cfg_path),
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "dataset_schema": schema_details,
            "required_files": file_report,
            "dataset_summary": dataset_summary,
        },
    )
    return 0