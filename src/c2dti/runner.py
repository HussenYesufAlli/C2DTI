from pathlib import Path
from datetime import datetime
import yaml

from src.c2dti.config_validation import validate_config
from src.c2dti.output_io import make_run_dir, write_summary, write_config_snapshot, append_registry

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

def run_once(config_path: str) -> int:
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

    name = cfg.get("name", "unnamed")
    protocol = cfg.get("protocol", "P0")
    base_dir = cfg.get("output", {}).get("base_dir", "outputs")

    run_dir = make_run_dir(base_dir, name)
    config_snapshot = write_config_snapshot(run_dir, cfg)

    summary_payload = {
        "run_name": name,
        "protocol": protocol,
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Minimal run contract smoke step (no model training yet)."
    }
    summary_path = write_summary(run_dir, summary_payload)

    append_registry(
        base_dir=base_dir,
        row={
            "run_name": name,
            "protocol": protocol,
            "status": "completed",
            "summary_path": str(summary_path),
            "config_snapshot_path": str(config_snapshot),
            "created_at": summary_payload["created_at"],
        },
    )

    print("[OK] Run contract completed")
    print(f"run_dir={run_dir}")
    print(f"summary={summary_path}")
    print(f"config_snapshot={config_snapshot}")
    print(f"registry={Path(base_dir) / 'results_registry.csv'}")
    return 0
