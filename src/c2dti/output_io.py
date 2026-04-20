from pathlib import Path
from datetime import datetime
import csv
import json

def make_run_dir(base_dir: str, run_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / "runs" / f"{run_name}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def write_summary(run_dir: Path, payload: dict) -> Path:
    out = run_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out

def append_registry(base_dir: str, row: dict) -> Path:
    reg = Path(base_dir) / "results_registry.csv"
    reg.parent.mkdir(parents=True, exist_ok=True)
    write_header = not reg.exists()
    fields = ["run_name", "protocol", "status", "summary_path", "created_at"]
    with reg.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return reg
