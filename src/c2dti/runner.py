from pathlib import Path

import yaml


def dry_run(config_path: str) -> int:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 1

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    name = cfg.get("name", "unnamed")
    protocol = cfg.get("protocol", "P0")

    print("[OK] Dry-run passed")
    print(f"name={name}")
    print(f"protocol={protocol}")
    print(f"config={cfg_path}")
    return 0
