#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "outputs" / "runs"
OUT_DIR = ROOT / "outputs" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Phase-6 C2DTI evaluation summaries into CSV tables."
    )
    parser.add_argument(
        "--prefix",
        default="C2DTI_EVAL_",
        help="Run-name prefix to include (default: C2DTI_EVAL_).",
    )
    return parser.parse_args()


def parse_identity(run_name: str) -> Dict[str, str]:
    """Parse run identity from names like:
    C2DTI_EVAL_DAVIS_RANDOM_FULL_S10
    """
    parts = run_name.split("_")
    # Expected format: C2DTI_EVAL_<DATASET>_<SPLIT>_<ABLATION>_S<SEED>
    # Example: C2DTI_EVAL_DAVIS_RANDOM_FULL_S10
    if len(parts) < 6:
        return {
            "dataset": "UNKNOWN",
            "split": "UNKNOWN",
            "ablation": "UNKNOWN",
            "seed": "UNKNOWN",
        }
    return {
        "dataset": parts[2],
        "split": parts[3],
        "ablation": parts[4],
        "seed": parts[5].lstrip("S"),
    }


def flatten_row(summary: Dict, summary_path: Path) -> Dict[str, object]:
    run_name = str(summary.get("run_name", ""))
    ident = parse_identity(run_name)
    eval_metrics = summary.get("evaluation_metrics", {}) or {}
    causal = summary.get("causal", {}) or {}
    causal_metrics = causal.get("metrics", {}) if isinstance(causal, dict) else {}

    return {
        "run_name": run_name,
        "dataset": ident["dataset"],
        "split": ident["split"],
        "ablation": ident["ablation"],
        "seed": ident["seed"],
        "status": summary.get("status", ""),
        "ci": eval_metrics.get("ci"),
        "rmse": eval_metrics.get("rmse"),
        "pearson": eval_metrics.get("pearson"),
        "spearman": eval_metrics.get("spearman"),
        "causal_score": summary.get("causal_score"),
        "l_total": causal_metrics.get("l_total"),
        "summary_path": str(summary_path),
    }


def safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def main() -> int:
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

    if not rows:
        print(f"[INFO] No runs found with prefix={args.prefix}")
        return 0

    # Detailed per-run table
    detailed_path = OUT_DIR / "eval_matrix_detailed.csv"
    detailed_columns = [
        "run_name",
        "dataset",
        "split",
        "ablation",
        "seed",
        "status",
        "ci",
        "rmse",
        "pearson",
        "spearman",
        "causal_score",
        "l_total",
        "summary_path",
    ]
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detailed_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Aggregate table: mean metrics across seeds per dataset/split/ablation
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        key = f"{row['dataset']}|{row['split']}|{row['ablation']}"
        grouped.setdefault(key, []).append(row)

    aggregate_rows: List[Dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        dataset, split, ablation = key.split("|")

        ci_vals = [safe_float(r["ci"]) for r in group_rows]
        rmse_vals = [safe_float(r["rmse"]) for r in group_rows]
        pearson_vals = [safe_float(r["pearson"]) for r in group_rows]
        spearman_vals = [safe_float(r["spearman"]) for r in group_rows]

        # Drop NaNs manually using x==x trick (NaN != NaN).
        ci_vals = [x for x in ci_vals if x == x]
        rmse_vals = [x for x in rmse_vals if x == x]
        pearson_vals = [x for x in pearson_vals if x == x]
        spearman_vals = [x for x in spearman_vals if x == x]

        def mean_or_none(vals: List[float]):
            return (sum(vals) / len(vals)) if vals else None

        aggregate_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "ablation": ablation,
                "n_runs": len(group_rows),
                "ci_mean": mean_or_none(ci_vals),
                "rmse_mean": mean_or_none(rmse_vals),
                "pearson_mean": mean_or_none(pearson_vals),
                "spearman_mean": mean_or_none(spearman_vals),
            }
        )

    aggregate_path = OUT_DIR / "eval_matrix_aggregate.csv"
    aggregate_columns = [
        "dataset",
        "split",
        "ablation",
        "n_runs",
        "ci_mean",
        "rmse_mean",
        "pearson_mean",
        "spearman_mean",
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
