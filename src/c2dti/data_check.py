"""Dataset precheck helpers for real C2DTI runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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


def _validate_bindingdb_content(dataset_path: Path) -> Dict[str, Any]:
    """Validate the BindingDB CSV header before the loader runs.

    This gives the user a precise report about column readiness instead of a
    generic placeholder fallback when the file exists but is malformed.
    """
    validation: Dict[str, Any] = {
        "status": "skipped",
        "available_columns": [],
        "num_data_rows": 0,
        "required_columns": ["Drug_ID", "Target_ID", "Y"],
        "optional_alias_columns": {
            "Drug_ID": ["Drug"],
            "Target_ID": ["Target"],
        },
        "resolved_columns": {},
        "missing_columns": [],
        "reason": "BindingDB CSV validation not run",
    }

    if not dataset_path.exists():
        validation["reason"] = "BindingDB CSV file does not exist"
        return validation

    try:
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header_row = next(reader, None)
            data_rows = sum(1 for row in reader if row and any(cell.strip() for cell in row))
    except Exception as exc:
        validation["status"] = "error"
        validation["reason"] = f"Failed to read BindingDB CSV header: {exc}"
        return validation

    if not header_row:
        validation["status"] = "error"
        validation["reason"] = "BindingDB CSV is empty"
        return validation

    available_columns = [column.strip() for column in header_row]
    validation["available_columns"] = available_columns
    validation["num_data_rows"] = data_rows

    resolved_columns: Dict[str, str] = {}
    missing_columns: List[str] = []
    for required_column in validation["required_columns"]:
        if required_column in available_columns:
            resolved_columns[required_column] = required_column
            continue

        aliases = validation["optional_alias_columns"].get(required_column, [])
        alias_match = next((alias for alias in aliases if alias in available_columns), None)
        if alias_match is not None:
            resolved_columns[required_column] = alias_match
        else:
            missing_columns.append(required_column)

    validation["resolved_columns"] = resolved_columns
    validation["missing_columns"] = missing_columns

    if missing_columns:
        validation["status"] = "error"
        validation["reason"] = "BindingDB CSV is missing required columns"
        return validation

    if data_rows == 0:
        validation["status"] = "error"
        validation["reason"] = "BindingDB CSV has no data rows"
        return validation

    validation["status"] = "ok"
    validation["reason"] = "BindingDB CSV contains the required columns"
    return validation


def _validate_dataset_content(dataset_name: str, dataset_path: Path) -> Dict[str, Any]:
    """Run dataset-specific content checks after file existence passes."""
    normalized_name = dataset_name.strip().upper()
    if normalized_name == "BINDINGDB":
        return _validate_bindingdb_content(dataset_path)

    if normalized_name in {"DAVIS", "KIBA"}:
        return _validate_sequence_matrix_content(normalized_name, dataset_path)

    return {
        "status": "skipped",
        "reason": f"No additional content validation implemented for {dataset_name}",
    }


def _count_non_empty_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _matrix_shape(raw_matrix: np.ndarray, n_drugs: int, n_targets: int) -> List[int]:
    """Infer matrix shape robustly for loadtxt outputs, including 1D edge cases."""
    if raw_matrix.ndim == 0:
        return [1, 1]

    if raw_matrix.ndim == 1:
        size = int(raw_matrix.shape[0])
        if size == 0:
            return [0, 0]
        if n_drugs == 1:
            return [1, size]
        if n_targets == 1:
            return [size, 1]
        return [1, size]

    return [int(raw_matrix.shape[0]), int(raw_matrix.shape[1])]


def _validate_sequence_matrix_content(dataset_name: str, dataset_path: Path) -> Dict[str, Any]:
    """Validate DAVIS/KIBA content: line counts and Y matrix shape consistency."""
    drug_file = dataset_path / "drug_smiles.txt"
    target_file = dataset_path / "target_sequences.txt"
    y_file = dataset_path / "Y.txt"

    validation: Dict[str, Any] = {
        "status": "error",
        "dataset": dataset_name,
        "num_drugs_from_file": 0,
        "num_targets_from_file": 0,
        "y_matrix_shape": [0, 0],
        "reason": "Sequence-matrix validation not completed",
    }

    try:
        n_drugs = _count_non_empty_lines(drug_file)
        n_targets = _count_non_empty_lines(target_file)
        raw_matrix = np.loadtxt(y_file, dtype=np.float32)
    except Exception as exc:
        validation["reason"] = f"Failed to parse sequence/matrix files: {exc}"
        return validation

    shape = _matrix_shape(raw_matrix, n_drugs=n_drugs, n_targets=n_targets)
    validation["num_drugs_from_file"] = n_drugs
    validation["num_targets_from_file"] = n_targets
    validation["y_matrix_shape"] = shape

    if shape != [n_drugs, n_targets]:
        validation["reason"] = (
            f"Y.txt shape {shape} does not match expected [{n_drugs}, {n_targets}] "
            f"from drug_smiles.txt and target_sequences.txt"
        )
        return validation

    if n_drugs == 0 or n_targets == 0:
        validation["reason"] = "drug_smiles.txt or target_sequences.txt has no non-empty rows"
        return validation

    validation["status"] = "ok"
    validation["reason"] = "Sequence counts and Y.txt shape are consistent"
    return validation


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

    content_validation = _validate_dataset_content(dataset_name, dataset_path)
    if content_validation.get("status") == "error":
        print("[ERROR] Dataset content validation failed")
        print(f"[ERROR] {content_validation.get('reason')}")
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
                "content_validation": content_validation,
                "reason": "Dataset content validation failed",
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
                "content_validation": content_validation,
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
            "content_validation": content_validation,
            "dataset_summary": dataset_summary,
        },
    )
    return 0