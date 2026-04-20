from pathlib import Path
import yaml
from src.c2dti.config_validation import validate_config

def dry_run(config_path: str) -> int:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 1

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    errors = validate_config(cfg)
    if errors:
        print("[ERROR] Config validation failed:")
        for e in errors:
            print(f"- {e}")
        return 2

    print("[OK] Dry-run passed")
    print(f"name={cfg.get('name')}")
    print(f"protocol={cfg.get('protocol')}")
    print(f"output.base_dir={cfg.get('output', {}).get('base_dir')}")
    print(f"config={cfg_path}")
    return 0
