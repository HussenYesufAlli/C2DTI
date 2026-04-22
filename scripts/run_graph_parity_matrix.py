#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run.py"
TEMP_ROOT = ROOT / "configs" / "generated_graph_parity"

BASE_CONFIGS: List[Tuple[str, str, Path]] = [
    ("DAVIS", "mixhop_baseline", ROOT / "configs" / "davis_mixhop_propagation_gate.yaml"),
    ("DAVIS", "interaction_cross_attn", ROOT / "configs" / "davis_interaction_cross_attention_gate.yaml"),
    ("KIBA", "mixhop_baseline", ROOT / "configs" / "kiba_gate.yaml"),
    ("KIBA", "interaction_cross_attn", ROOT / "configs" / "kiba_gate.yaml"),
    ("BindingDB", "mixhop_baseline", ROOT / "configs" / "bindingdb_gate.yaml"),
    ("BindingDB", "interaction_cross_attn", ROOT / "configs" / "bindingdb_gate.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run the C2DTI graph parity matrix (2 branches x 3 datasets x 3 seeds)."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[10, 34, 42],
        help="Seeds for model and split. Default: 10 34 42",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "run-once"],
        default="dry-run",
        help="Runner mode to execute via scripts/run.py",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute commands. If omitted, print command sheet only.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def apply_branch_settings(cfg: Dict, dataset: str, branch: str, seed: int) -> Dict:
    out = copy.deepcopy(cfg)

    out.setdefault("dataset", {})
    out["dataset"]["name"] = dataset

    out.setdefault("split", {})
    out["split"]["seed"] = int(seed)

    out.setdefault("model", {})
    out["model"]["seed"] = int(seed)

    if branch == "mixhop_baseline":
        out["model"]["name"] = "mixhop_propagation"
        out["model"]["top_k"] = int(out["model"].get("top_k", 8))
        out["model"]["hop_weights"] = out["model"].get("hop_weights", [0.6, 0.3, 0.1])
        out["name"] = f"C2DTI_{dataset.upper()}_MIXHOP_PARITY_S{seed}"
    elif branch == "interaction_cross_attn":
        out["model"]["name"] = "interaction_cross_attention"
        out["model"]["latent_dim"] = int(out["model"].get("latent_dim", 16))
        out["model"]["epochs"] = int(out["model"].get("epochs", 5))
        out["model"]["lr"] = float(out["model"].get("lr", 0.01))
        out["model"]["attention_temperature"] = float(out["model"].get("attention_temperature", 1.0))
        out["model"]["top_k"] = int(out["model"].get("top_k", 8))
        out["name"] = f"C2DTI_{dataset.upper()}_INTERACTION_CROSS_ATTN_PARITY_S{seed}"
    else:
        raise ValueError(f"Unknown branch: {branch}")

    return out


def main() -> int:
    args = parse_args()
    mode_flag = "--dry-run" if args.mode == "dry-run" else "--run-once"

    commands: List[str] = []

    for dataset, branch, base_cfg_path in BASE_CONFIGS:
        if not base_cfg_path.exists():
            print(f"[ERROR] Missing base config: {base_cfg_path}")
            return 1

        base_cfg = load_yaml(base_cfg_path)
        for seed in args.seeds:
            cfg = apply_branch_settings(base_cfg, dataset, branch, seed)
            out_path = TEMP_ROOT / f"{dataset.lower()}_{branch}_seed{seed}.yaml"
            dump_yaml(out_path, cfg)

            cmd = f"cd {ROOT} && python {RUNNER} --config {out_path} {mode_flag}"
            commands.append(cmd)

    print("# C2DTI Graph Parity Matrix Commands")
    print(f"# Mode: {args.mode}")
    print(f"# Total runs: {len(commands)}")
    for idx, cmd in enumerate(commands, start=1):
        print(f"{idx:02d}. {cmd}")

    if not args.execute:
        print("\n[INFO] Dry mode only. Re-run with --execute to launch runs.")
        return 0

    for idx, cmd in enumerate(commands, start=1):
        print(f"\n[RUN {idx:02d}/{len(commands)}] {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"[ERROR] Run failed at index {idx} with exit code {result.returncode}")
            return result.returncode

    print("\n[OK] Completed all C2DTI parity matrix runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
