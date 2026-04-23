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
BASE_CFG = ROOT / "configs" / "davis_unified_causal_gate.yaml"
GEN_DIR = ROOT / "configs" / "generated_eval_matrix"

DATASETS: List[Tuple[str, str]] = [
    ("DAVIS", "data/davis"),
    ("KIBA", "data/kiba"),
    ("BindingDB", "data/bindingdb/bindingdb.csv"),
]
SPLITS = ["random", "cold_drug", "cold_target"]
ABLATIONS = ["full", "no_causal", "no_irm", "no_cf", "no_mas"]
SEEDS = [10, 34, 42]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally execute the C2DTI Phase-6 evaluation matrix "
            "(3 datasets x 3 splits x 5 ablations x 3 seeds = 135 runs)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "run-once"],
        default="dry-run",
        help="Runner mode passed to scripts/run.py",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute generated commands. If omitted, only print command sheet.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on number of runs to execute (0 means no cap).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def apply_ablation(causal_cfg: Dict, ablation: str) -> None:
    """Apply one ablation setting by adjusting causal lambdas.

    Non-breaking rule: we keep the same unified mode API and only zero out
    lambda terms. That way, all other behavior remains stable.
    """
    causal_cfg["enabled"] = True
    causal_cfg["mode"] = "unified"
    causal_cfg["lambda_xview"] = 1.0
    causal_cfg["lambda_mas"] = 1.0
    causal_cfg["lambda_irm"] = 1.0
    causal_cfg["lambda_cf"] = 1.0

    if ablation == "full":
        return
    if ablation == "no_causal":
        causal_cfg["lambda_xview"] = 0.0
        return
    if ablation == "no_irm":
        causal_cfg["lambda_irm"] = 0.0
        return
    if ablation == "no_cf":
        causal_cfg["lambda_cf"] = 0.0
        return
    if ablation == "no_mas":
        causal_cfg["lambda_mas"] = 0.0
        return
    raise ValueError(f"Unknown ablation: {ablation}")


def make_config(base_cfg: Dict, dataset: str, dataset_path: str, split: str, ablation: str, seed: int) -> Dict:
    cfg = copy.deepcopy(base_cfg)

    # Name encodes the full matrix identity for easy filtering later.
    cfg["name"] = f"C2DTI_EVAL_{dataset.upper()}_{split.upper()}_{ablation.upper()}_S{seed}"
    cfg["protocol"] = "P1"

    cfg.setdefault("dataset", {})
    cfg["dataset"]["name"] = dataset
    cfg["dataset"]["path"] = dataset_path
    cfg["dataset"]["allow_placeholder"] = True

    cfg.setdefault("split", {})
    cfg["split"]["strategy"] = split
    cfg["split"]["seed"] = int(seed)
    cfg["split"]["test_ratio"] = float(cfg["split"].get("test_ratio", 0.2))

    cfg.setdefault("model", {})
    cfg["model"]["seed"] = int(seed)

    cfg.setdefault("perturbation", {})
    cfg["perturbation"]["seed"] = int(seed)

    cfg.setdefault("causal", {})
    apply_ablation(cfg["causal"], ablation)

    # Keep sub-config seeds aligned for reproducibility.
    if isinstance(cfg["causal"].get("sequence_model"), dict):
        cfg["causal"]["sequence_model"]["seed"] = int(seed)
    if isinstance(cfg["causal"].get("mas_config"), dict):
        cfg["causal"]["mas_config"]["seed"] = int(seed)
    if isinstance(cfg["causal"].get("irm_cf_config"), dict):
        cfg["causal"]["irm_cf_config"]["seed"] = int(seed)

    return cfg


def build_commands(mode: str) -> List[Tuple[Path, str]]:
    if not BASE_CFG.exists():
        raise FileNotFoundError(f"Missing base config: {BASE_CFG}")

    mode_flag = "--dry-run" if mode == "dry-run" else "--run-once"
    base_cfg = load_yaml(BASE_CFG)
    commands: List[Tuple[Path, str]] = []

    for dataset, dataset_path in DATASETS:
        for split in SPLITS:
            for ablation in ABLATIONS:
                for seed in SEEDS:
                    cfg = make_config(base_cfg, dataset, dataset_path, split, ablation, seed)
                    cfg_name = f"{dataset.lower()}_{split}_{ablation}_seed{seed}.yaml"
                    cfg_path = GEN_DIR / cfg_name
                    save_yaml(cfg_path, cfg)
                    cmd = f"cd {ROOT} && python {RUNNER} --config {cfg_path} {mode_flag}"
                    commands.append((cfg_path, cmd))

    return commands


def main() -> int:
    args = parse_args()

    commands = build_commands(args.mode)
    total = len(commands)

    print("# C2DTI Phase-6 Evaluation Matrix")
    print(f"# Mode        : {args.mode}")
    print(f"# Config dir  : {GEN_DIR}")
    print(f"# Total runs  : {total}")

    for idx, (_, cmd) in enumerate(commands, start=1):
        print(f"{idx:03d}. {cmd}")

    if not args.execute:
        print("\n[INFO] Command sheet generated only. Use --execute to run.")
        return 0

    run_limit = args.max_runs if args.max_runs > 0 else total
    to_execute = commands[:run_limit]

    for idx, (cfg_path, cmd) in enumerate(to_execute, start=1):
        print(f"\n[RUN {idx:03d}/{len(to_execute)}] {cfg_path.name}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"[ERROR] Failed at {cfg_path.name} with code={result.returncode}")
            return result.returncode

    print("\n[OK] Evaluation matrix execution finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
