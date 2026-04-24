#!/usr/bin/env python3
"""
Extract ChemBERTa (drug) and ANKH (protein) embeddings for C2DTI datasets.

Key difference from MINDG_CLASSA scripts:
  - C2DTI uses SMILES strings as drug IDs and full AA sequences as target IDs.
  - So we save ids = SMILES / ids = sequences directly, matching C2DTI's entity lists.

Usage examples:
  # Drug embeddings for DAVIS
  python scripts/extract_embeddings.py --mode drug \
      --input-csv data/davis/davis.csv \
      --output-npz data/embeddings/DAVIS_chemberta_drug.npz

  # Protein embeddings for DAVIS
  python scripts/extract_embeddings.py --mode protein \
      --input-csv data/davis/davis.csv \
      --output-npz data/embeddings/DAVIS_ankh_target.npz
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Mean-pool token embeddings weighted by attention mask, return numpy float32."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
    return pooled.detach().cpu().numpy().astype(np.float32)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device string to torch.device. Raises if cuda requested but unavailable."""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    # auto: prefer cuda
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Drug (ChemBERTa) extraction
# ---------------------------------------------------------------------------

def extract_drug_embeddings(
    smiles_list: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Encode a list of SMILES strings with ChemBERTa and return (N, D) float32 array."""
    from transformers import AutoModel, AutoTokenizer

    print(f"[drug] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.eval()

    vectors = []
    current_bs = batch_size
    i = 0
    with torch.no_grad():
        pbar = tqdm(total=len(smiles_list), desc="ChemBERTa batches")
        while i < len(smiles_list):
            batch = smiles_list[i: i + current_bs]
            try:
                toks = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                hidden = model(**toks).last_hidden_state
                vectors.append(mean_pool(hidden, toks["attention_mask"]))
                i += len(batch)
                pbar.update(len(batch))
                del hidden, toks
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"  OOM — reducing batch size to {current_bs}")
        pbar.close()

    return np.concatenate(vectors, axis=0)


# ---------------------------------------------------------------------------
# Protein (ANKH) extraction
# ---------------------------------------------------------------------------

def normalize_seq(seq: str) -> str:
    """Normalize amino acid sequence: uppercase, replace ambiguous residues with X."""
    return re.sub(r"[UZOB]", "X", str(seq).strip().upper())


def extract_protein_embeddings(
    seqs: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Encode protein sequences with ANKH (T5 encoder) and return (N, D) float32 array."""
    from transformers import AutoTokenizer, T5EncoderModel

    print(f"[protein] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use T5EncoderModel to avoid decoder errors with ANKH checkpoints
    model = T5EncoderModel.from_pretrained(model_name).to(device=device, dtype=torch.float32)
    model.eval()

    vectors = []
    current_bs = batch_size
    i = 0
    normalized = [normalize_seq(s) for s in seqs]
    with torch.no_grad():
        pbar = tqdm(total=len(normalized), desc="ANKH batches")
        while i < len(normalized):
            batch = normalized[i: i + current_bs]
            try:
                toks = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                hidden = model(**toks).last_hidden_state
                vectors.append(mean_pool(hidden, toks["attention_mask"]))
                i += len(batch)
                pbar.update(len(batch))
                del hidden, toks
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"  OOM — reducing batch size to {current_bs}")
        pbar.close()

    return np.concatenate(vectors, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="C2DTI embedding extractor (SMILES/sequence as IDs)")
    p.add_argument("--mode", required=True, choices=["drug", "protein"],
                   help="'drug' = ChemBERTa on SMILES; 'protein' = ANKH on sequences")
    p.add_argument("--input-csv", required=True, help="Path to dataset CSV (e.g. data/davis/davis.csv)")
    p.add_argument("--output-npz", required=True, help="Where to save the NPZ file")
    p.add_argument("--drug-col", default="Drug", help="Column holding SMILES strings")
    p.add_argument("--target-col", default="Target", help="Column holding protein sequences")
    p.add_argument("--drug-model", default="DeepChem/ChemBERTa-77M-MTR")
    p.add_argument("--protein-model", default="ElnaggarLab/ankh-base")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=32,
                   help="Initial batch size (auto-halved on OOM)")
    p.add_argument("--max-length", type=int, default=0,
                   help="Max token length. 0 = use defaults (256 for drug, 1024 for protein)")
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    df = pd.read_csv(args.input_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)

    if args.mode == "drug":
        # Deduplicate by SMILES — SMILES string IS the ID in C2DTI
        unique_smiles = df[args.drug_col].dropna().drop_duplicates().astype(str).tolist()
        print(f"Unique drugs: {len(unique_smiles)}")
        max_len = args.max_length if args.max_length > 0 else 256
        emb = extract_drug_embeddings(
            smiles_list=unique_smiles,
            model_name=args.drug_model,
            device=device,
            batch_size=args.batch_size,
            max_length=max_len,
        )
        np.savez_compressed(
            args.output_npz,
            ids=np.array(unique_smiles, dtype=object),   # SMILES strings as IDs
            embeddings=emb,
            model_name=np.array([args.drug_model], dtype=object),
        )
        print(f"Saved: {args.output_npz}")
        print(f"Shape: {emb.shape}  (drugs × dim)")

    else:  # protein
        # Deduplicate by sequence — sequence IS the ID in C2DTI
        unique_seqs = df[args.target_col].dropna().drop_duplicates().astype(str).tolist()
        print(f"Unique proteins: {len(unique_seqs)}")
        max_len = args.max_length if args.max_length > 0 else 1024
        emb = extract_protein_embeddings(
            seqs=unique_seqs,
            model_name=args.protein_model,
            device=device,
            batch_size=args.batch_size,
            max_length=max_len,
        )
        np.savez_compressed(
            args.output_npz,
            ids=np.array(unique_seqs, dtype=object),     # sequences as IDs
            embeddings=emb,
            model_name=np.array([args.protein_model], dtype=object),
        )
        print(f"Saved: {args.output_npz}")
        print(f"Shape: {emb.shape}  (proteins × dim)")


if __name__ == "__main__":
    main()
