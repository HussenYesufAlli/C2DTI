#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_binary.py"
BASE_CFG = ROOT / "configs" / "davis_binary_baseline.yaml"
GEN_DIR = ROOT / "configs" / "generated_binary_eval_matrix"

DATASETS: List[Tuple[str, str]] = [
    ("DAVIS", "datasets/DAVIS_binary.csv"),
    ("KIBA", "datasets/KIBA_binary.csv"),
    ("BindingDB", "datasets/BindingDB_Kd_binary.csv"),
]
SPLITS = ["random", "cold_drug", "cold_target"]
SEEDS = [10, 34, 42]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for binary matrix generation and execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally execute the binary C2DTI evaluation matrix "
            "(3 datasets x 3 splits x 3 seeds = 27 runs)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "run-once"],
        default="dry-run",
        help="Runner mode passed to scripts/run_binary.py",
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
    """Load one YAML file and return mapping payload."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path, cfg: Dict) -> None:
    """Write one YAML config to disk, creating parent folder when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def make_config(base_cfg: Dict, dataset: str, dataset_path: str, split: str, seed: int) -> Dict:
    """Build one binary run config while keeping baseline behavior unchanged."""
    cfg = copy.deepcopy(base_cfg)

    # Encode identity into run_name so compile scripts can parse it deterministically.
    cfg["name"] = f"C2DTI_BINARY_EVAL_{dataset.upper()}_{split.upper()}_S{seed}"
    cfg["protocol"] = "P_binary"

    cfg.setdefault("dataset", {})
    cfg["dataset"]["name"] = dataset
    cfg["dataset"]["path"] = dataset_path
    cfg["dataset"]["allow_placeholder"] = False

    cfg.setdefault("split", {})
    cfg["split"]["strategy"] = split
    cfg["split"]["seed"] = int(seed)
    cfg["split"]["test_ratio"] = float(cfg["split"].get("test_ratio", 0.2))

    cfg.setdefault("model", {})
    cfg["model"]["seed"] = int(seed)

    cfg.setdefault("binary", {})
    cfg["binary"]["threshold"] = float(cfg["binary"].get("threshold", 0.5))

    cfg.setdefault("output", {})
    cfg["output"]["base_dir"] = str(cfg["output"].get("base_dir", "outputs_binary"))

    return cfg


def build_commands(mode: str) -> List[Tuple[Path, str]]:
    """Generate all matrix configs and shell commands for execution."""
    if not BASE_CFG.exists():
        raise FileNotFoundError(f"Missing base config: {BASE_CFG}")

    mode_flag = "--dry-run" if mode == "dry-run" else "--run-once"
    base_cfg = load_yaml(BASE_CFG)
    commands: List[Tuple[Path, str]] = []

    for dataset, dataset_path in DATASETS:
        for split in SPLITS:
            for seed in SEEDS:
                cfg = make_config(base_cfg, dataset, dataset_path, split, seed)
                cfg_name = f"{dataset.lower()}_{split}_seed{seed}.yaml"
                cfg_path = GEN_DIR / cfg_name
                save_yaml(cfg_path, cfg)
                cmd = f"cd {ROOT} && python {RUNNER} --config {cfg_path} {mode_flag}"
                commands.append((cfg_path, cmd))

    return commands


def main() -> int:
    """Entry point: print command sheet, and optionally execute the matrix."""
    args = parse_args()

    commands = build_commands(args.mode)
    total = len(commands)

    print("# C2DTI Binary Evaluation Matrix")
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

    print("\n[OK] Binary evaluation matrix execution finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
