"""
C2DTI Benchmark Dataset Importer
================================
Acquires DAVIS, KIBA, and BindingDB_Kd datasets and writes them into the
C2DTI data directory in the format expected by dataset_loader.py:

  DAVIS / KIBA  ->  data/<dataset>/drug_smiles.txt
                    data/<dataset>/target_sequences.txt
                    data/<dataset>/Y.txt  (dense float32 matrix)

  BindingDB_Kd  ->  data/bindingdb/bindingdb.csv  (Drug_ID, Target_ID, Y)

Acquisition order (for each dataset):
  1. TDC live download via tdc.multi_pred.DTI  [source-first preference]
  2. Workspace CSV fallback from MINDG_CLASSA/doc/dataset/

Run this script from the C2DTI project root or pass --root explicitly.
Example:
    conda run -n mindg-org python scripts/import_real_data.py --root /home/hussen/MINDG/C2DTI
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

# Workspace fallback paths (relative to this file's repo root)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_FALLBACK = {
    "DAVIS":       _WORKSPACE_ROOT / "MINDG_CLASSA" / "doc" / "dataset" / "DAVIS.csv",
    "KIBA":        _WORKSPACE_ROOT / "MINDG_CLASSA" / "doc" / "dataset" / "KIBA.csv",
    "BindingDB_Kd": _WORKSPACE_ROOT / "MINDG_CLASSA" / "doc" / "dataset" / "BindingDB_Kd.csv",
}


# ────────────────────────────────────────────────────────────────────────────
# Acquisition helpers
# ────────────────────────────────────────────────────────────────────────────

def _try_tdc_download(name: str) -> "pd.DataFrame | None":
    """
    Attempt a live download using PyTDC.
    Returns a DataFrame with columns [Drug_ID, Drug, Target_ID, Target, Y]
    or None if TDC is unavailable / download fails.
    """
    try:
        from tdc.multi_pred import DTI  # noqa: PLC0415
    except ImportError:
        print(f"  [TDC] Not installed in current env – skipping live download for {name}.")
        return None

    try:
        print(f"  [TDC] Downloading {name} from TDC …")
        data = DTI(name=name)
        df = data.get_data()
        print(f"  [TDC] Downloaded {name}: {len(df)} rows, columns={list(df.columns)}")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"  [TDC] Download failed for {name}: {exc}")
        return None


def _load_fallback(name: str) -> "pd.DataFrame | None":
    """
    Load from the pre-existing workspace CSV (originally sourced from TDC).
    Returns a DataFrame or None if the file does not exist.
    """
    path = _FALLBACK.get(name)
    if path is None or not path.exists():
        print(f"  [Fallback] CSV not found for {name}: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  [Fallback] Loaded {name} from workspace: {len(df)} rows, columns={list(df.columns)}")
    return df


def _acquire(name: str) -> "pd.DataFrame":
    """
    Acquire dataset with source-first, workspace-fallback strategy.
    Raises RuntimeError if both paths fail.
    """
    df = _try_tdc_download(name)
    if df is not None:
        return df
    df = _load_fallback(name)
    if df is not None:
        return df
    raise RuntimeError(
        f"Could not acquire dataset '{name}'. "
        "Neither TDC download nor workspace fallback succeeded."
    )


# ────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ────────────────────────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has standard column names:
      Drug_ID, Drug (SMILES), Target_ID, Target (sequence), Y.
    TDC uses: Drug_ID, Drug, Target_ID, Target, Y  (already standard).
    """
    col_map = {}
    if "Drug" in df.columns and "Drug_ID" not in df.columns:
        col_map["Drug"] = "Drug_ID"
    if "Target" in df.columns and "Target_ID" not in df.columns:
        col_map["Target"] = "Target_ID"
    if col_map:
        df = df.rename(columns=col_map)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Writers
# ────────────────────────────────────────────────────────────────────────────

def _write_davis_kiba(df: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    """
    Write DAVIS/KIBA dataset to C2DTI text-file format.

    For these datasets the full D×T matrix is dense (TDC provides all
    drug-target pairs). We:
      - Sort drugs by Drug_ID, targets by Target_ID (reproducible order).
      - Extract SMILES from the 'Drug' column, sequences from 'Target'.
      - Fill any missing pairs with 0.0 (unobserved == no interaction).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort unique drugs and targets for reproducibility
    drugs_sorted = sorted(df["Drug_ID"].unique().tolist())
    targets_sorted = sorted(df["Target_ID"].unique().tolist())

    # Build lookup: Drug_ID -> SMILES, Target_ID -> sequence
    # Use the first occurrence for each ID (should be unique in TDC data)
    drug_smiles = (
        df.drop_duplicates("Drug_ID")
          .set_index("Drug_ID")["Drug"]
          .to_dict()
    )
    target_seq = (
        df.drop_duplicates("Target_ID")
          .set_index("Target_ID")["Target"]
          .to_dict()
    )

    # Build dense Y matrix (n_drugs x n_targets), default 0.0
    n_drugs, n_targets = len(drugs_sorted), len(targets_sorted)
    drug_idx = {d: i for i, d in enumerate(drugs_sorted)}
    tgt_idx  = {t: j for j, t in enumerate(targets_sorted)}

    Y = np.zeros((n_drugs, n_targets), dtype=np.float32)
    for _, row in df.iterrows():
        i = drug_idx.get(row["Drug_ID"])
        j = tgt_idx.get(row["Target_ID"])
        if i is not None and j is not None:
            Y[i, j] = float(row["Y"])

    # Write files
    (out_dir / "drug_smiles.txt").write_text(
        "\n".join(drug_smiles[d] for d in drugs_sorted) + "\n", encoding="utf-8"
    )
    (out_dir / "target_sequences.txt").write_text(
        "\n".join(target_seq[t] for t in targets_sorted) + "\n", encoding="utf-8"
    )

    # np.savetxt for space-separated matrix
    np.savetxt(out_dir / "Y.txt", Y, fmt="%.6f")

    print(f"  [Write] {dataset_name} -> {out_dir}")
    print(f"          drugs={n_drugs}  targets={n_targets}  matrix={Y.shape}")


def _write_bindingdb(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Write BindingDB dataset to C2DTI CSV format (Drug_ID, Target_ID, Y).
    The BindingDB loader in C2DTI reads a CSV directly; we keep the raw Y
    (affinity in nM) – binarisation is handled inside the loader.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bindingdb.csv"

    # Normalise TDC column names to the format expected by BindingDBLoader.
    # TDC: "Drug (SMILES)" / "Target (sequence)"; workspace CSV: "Drug" / "Target".
    col_map = {}
    if "Drug (SMILES)" in df.columns:
        col_map["Drug (SMILES)"] = "Drug"
    if "Target (sequence)" in df.columns:
        col_map["Target (sequence)"] = "Target"
    if col_map:
        df = df.rename(columns=col_map)

    # Keep ID columns, molecular content columns (SMILES / sequence), and label.
    keep_cols = ["Drug_ID", "Target_ID", "Y"]
    for optional in ("Drug", "Target"):
        if optional in df.columns:
            keep_cols.append(optional)

    missing_required = [c for c in ["Drug_ID", "Target_ID", "Y"] if c not in df.columns]
    if missing_required:
        raise ValueError(f"BindingDB DataFrame missing columns: {missing_required}")

    clean_df = df[keep_cols].copy()
    # Drop rows with missing IDs/labels and force stable string IDs.
    clean_df = clean_df.dropna(subset=["Drug_ID", "Target_ID", "Y"])
    clean_df["Drug_ID"] = clean_df["Drug_ID"].astype(str)
    clean_df["Target_ID"] = clean_df["Target_ID"].astype(str)
    clean_df["Y"] = pd.to_numeric(clean_df["Y"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Y"])

    clean_df.to_csv(out_path, index=False)
    print(f"  [Write] BindingDB_Kd -> {out_path}")
    print(f"          rows={len(clean_df)}  unique_drugs={clean_df['Drug_ID'].nunique()}  "
          f"unique_targets={clean_df['Target_ID'].nunique()}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def run_import(c2dti_root: Path, force: bool = False) -> None:
    """
    Main entry-point: acquire all three datasets and write them to C2DTI data/.
    Already-populated directories are skipped unless --force is set.
    """
    data_root = c2dti_root / "data"

    for name, out_subdir, writer_key in [
        ("DAVIS",        data_root / "davis",      "matrix"),
        ("KIBA",         data_root / "kiba",       "matrix"),
        ("BindingDB_Kd", data_root / "bindingdb",  "csv"),
    ]:
        print(f"\n[Dataset] {name}")

        # Skip if already populated (unless --force)
        sentinel_file = (
            out_subdir / "Y.txt"
            if writer_key == "matrix"
            else out_subdir / "bindingdb.csv"
        )
        if sentinel_file.exists() and not force:
            existing_lines = sum(1 for _ in sentinel_file.open())
            if existing_lines > 5:
                    print(f"  [Skip] Source data already exists at {sentinel_file} "
                      f"({existing_lines} lines). Use --force to overwrite.")
                continue

        # Acquire
        df = _acquire(name)
        df = _normalise_columns(df)

        # Write
        if writer_key == "matrix":
            _write_davis_kiba(df, out_subdir, name)
        else:
            _write_bindingdb(df, out_subdir)

    print("\n[Done] Dataset import complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import benchmark datasets into C2DTI data/ directory."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="C2DTI project root (default: current directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing non-demo data files.",
    )
    args = parser.parse_args()

    c2dti_root = Path(args.root).resolve()
    if not (c2dti_root / "data").exists():
          print(f"[Error] data/ directory not found under {c2dti_root}. "
              "Create the expected dataset folders and files first.", file=sys.stderr)
        sys.exit(1)

    run_import(c2dti_root=c2dti_root, force=args.force)


if __name__ == "__main__":
    main()
