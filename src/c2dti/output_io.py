from pathlib import Path
from datetime import datetime
import csv
import json
from typing import List
import yaml
import numpy as np

def make_run_dir(base_dir: str, run_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / "runs" / f"{run_name}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def write_summary(run_dir: Path, payload: dict) -> Path:
    out = run_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def write_config_snapshot(run_dir: Path, cfg: dict) -> Path:
    out = run_dir / "config_snapshot.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def write_prediction_matrix(run_dir: Path, drugs: List[str], targets: List[str], matrix: np.ndarray) -> Path:
    """Persist the prediction matrix as a CSV artifact for inspection."""
    out = run_dir / "predictions.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_id", *targets])
        for drug_name, row in zip(drugs, matrix):
            writer.writerow([drug_name, *[f"{float(value):.6f}" for value in row]])
    return out

def append_registry(base_dir: str, row: dict) -> Path:
    reg = Path(base_dir) / "results_registry.csv"
    reg.parent.mkdir(parents=True, exist_ok=True)
    write_header = not reg.exists()
    fields = ["run_name", "protocol", "status", "summary_path", "config_snapshot_path", "created_at"]
    with reg.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return reg
