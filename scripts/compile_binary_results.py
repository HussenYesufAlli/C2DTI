#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "outputs_binary" / "runs"
OUT_DIR = ROOT / "outputs_binary" / "reports"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for binary result compilation."""
    parser = argparse.ArgumentParser(
        description="Compile binary C2DTI summaries into detailed and aggregate CSV tables."
    )
    parser.add_argument(
        "--prefix",
        default="C2DTI_BINARY_EVAL_",
        help="Run-name prefix to include (default: C2DTI_BINARY_EVAL_).",
    )
    return parser.parse_args()


def parse_identity(run_name: str) -> Dict[str, str]:
    """Parse run identity from C2DTI_BINARY_EVAL_<DATASET>_<SPLIT>_S<SEED>."""
    parts = run_name.split("_")
    if len(parts) < 6:
        return {
            "dataset": "UNKNOWN",
            "split": "UNKNOWN",
            "seed": "UNKNOWN",
        }

    # Supports both:
    #   C2DTI_BINARY_EVAL_DAVIS_RANDOM_S10
    #   C2DTI_BINARY_EVAL_DAVIS_COLD_DRUG_S10
    dataset = parts[3]
    split = "_".join(parts[4:-1])
    seed = parts[-1].lstrip("S")

    return {
        "dataset": dataset,
        "split": split,
        "seed": seed,
    }


def flatten_row(summary: Dict, summary_path: Path) -> Dict[str, object]:
    """Flatten one summary payload into a report row."""
    run_name = str(summary.get("run_name", ""))
    ident = parse_identity(run_name)
    metrics = summary.get("evaluation_metrics", {}) or {}

    return {
        "run_name": run_name,
        "dataset": ident["dataset"],
        "split": ident["split"],
        "seed": ident["seed"],
        "status": summary.get("status", ""),
        "threshold": metrics.get("threshold", summary.get("binary_threshold")),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "f1": metrics.get("f1"),
        "accuracy": metrics.get("accuracy"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "n_positive": metrics.get("n_positive"),
        "n_negative": metrics.get("n_negative"),
        "summary_path": str(summary_path),
    }


def safe_float(value: object) -> float:
    """Convert report values to float while preserving NaN semantics."""
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def mean_or_none(values: List[float]):
    """Return arithmetic mean for finite values, else None."""
    valid = [x for x in values if x == x]
    return (sum(valid) / len(valid)) if valid else None


def dedupe_latest_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Keep only the latest row for each logical run_name."""
    latest_by_name: Dict[str, Dict[str, object]] = {}
    for row in rows:
        run_name = str(row.get("run_name", ""))
        summary_path = str(row.get("summary_path", ""))
        prev = latest_by_name.get(run_name)
        if prev is None or summary_path > str(prev.get("summary_path", "")):
            latest_by_name[run_name] = row
    return list(latest_by_name.values())


def main() -> int:
    """Compile matching binary summaries into detailed and aggregate CSV files."""
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for run_dir in sorted(RUNS_DIR.glob("*")):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_name = str(payload.get("run_name", ""))
        if not run_name.startswith(args.prefix):
            continue

        rows.append(flatten_row(payload, summary_path))

    rows = dedupe_latest_rows(rows)

    if not rows:
        print(f"[INFO] No runs found with prefix={args.prefix}")
        return 0

    detailed_path = OUT_DIR / "binary_eval_matrix_detailed.csv"
    detailed_columns = [
        "run_name",
        "dataset",
        "split",
        "seed",
        "status",
        "threshold",
        "auroc",
        "auprc",
        "f1",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "n_positive",
        "n_negative",
        "summary_path",
    ]
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detailed_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        key = f"{row['dataset']}|{row['split']}"
        grouped.setdefault(key, []).append(row)

    aggregate_rows: List[Dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        dataset, split = key.split("|")

        aggregate_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "n_runs": len(group_rows),
                "auroc_mean": mean_or_none([safe_float(r["auroc"]) for r in group_rows]),
                "auprc_mean": mean_or_none([safe_float(r["auprc"]) for r in group_rows]),
                "f1_mean": mean_or_none([safe_float(r["f1"]) for r in group_rows]),
                "accuracy_mean": mean_or_none([safe_float(r["accuracy"]) for r in group_rows]),
                "sensitivity_mean": mean_or_none([safe_float(r["sensitivity"]) for r in group_rows]),
                "specificity_mean": mean_or_none([safe_float(r["specificity"]) for r in group_rows]),
                "precision_mean": mean_or_none([safe_float(r["precision"]) for r in group_rows]),
            }
        )

    aggregate_path = OUT_DIR / "binary_eval_matrix_aggregate.csv"
    aggregate_columns = [
        "dataset",
        "split",
        "n_runs",
        "auroc_mean",
        "auprc_mean",
        "f1_mean",
        "accuracy_mean",
        "sensitivity_mean",
        "specificity_mean",
        "precision_mean",
    ]
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)

    print(f"[OK] Wrote detailed table : {detailed_path}")
    print(f"[OK] Wrote aggregate table: {aggregate_path}")
    print(f"[INFO] Included runs      : {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
