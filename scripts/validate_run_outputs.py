import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_CONFIGS = [
    "configs/davis_real_pipeline_strict.yaml",
    "configs/bindingdb_real_pipeline_strict.yaml",
    "configs/kiba_real_pipeline_strict.yaml",
]


def _now_utc_iso() -> str:
    """Return a timezone-aware UTC timestamp with trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_report_path() -> Path:
    """Build a timestamped default report path under outputs/gates."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "gates" / f"validate_outputs_{stamp}.json"


def _resolve_configs(cli_configs: List[str]) -> List[str]:
    """Resolve config list from CLI; use defaults if none provided."""
    if cli_configs:
        return cli_configs
    return DEFAULT_CONFIGS


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config into a dictionary; return empty dict on empty files."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _latest_run_dir(base_dir: Path, run_name: str) -> Optional[Path]:
    """Find the latest run directory for the given run name."""
    runs_root = base_dir / "runs"
    if not runs_root.exists():
        return None
    matches = sorted(runs_root.glob(f"{run_name}-*"))
    if not matches:
        return None
    return matches[-1]


def _read_registry_rows(registry_path: Path) -> List[Dict[str, str]]:
    """Read registry CSV rows if file exists and is readable."""
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _validate_one_config(config_path: str) -> Dict[str, Any]:
    """Validate latest output contract for one config and return detailed result."""
    result: Dict[str, Any] = {
        "config": config_path,
        "status": "PASS",
        "errors": [],
    }

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"Config not found: {cfg_path}")
        return result

    cfg = _load_yaml(cfg_path)
    run_name = str(cfg.get("name", ""))
    base_dir = Path(cfg.get("output", {}).get("base_dir", "outputs"))
    has_dataset = bool(cfg.get("dataset"))

    if not run_name:
        result["status"] = "FAIL"
        result["errors"].append("Missing config name")
        return result

    run_dir = _latest_run_dir(base_dir, run_name)
    if run_dir is None:
        result["status"] = "FAIL"
        result["errors"].append(f"No run directory found for run_name={run_name}")
        return result

    summary_path = run_dir / "summary.json"
    snapshot_path = run_dir / "config_snapshot.yaml"
    predictions_path = run_dir / "predictions.csv"
    registry_path = base_dir / "results_registry.csv"

    result["run_name"] = run_name
    result["base_dir"] = str(base_dir)
    result["run_dir"] = str(run_dir)

    if not summary_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"Missing summary file: {summary_path}")
    if not snapshot_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"Missing config snapshot: {snapshot_path}")
    if has_dataset and not predictions_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"Missing predictions file: {predictions_path}")

    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_payload.get("status") != "completed":
                result["status"] = "FAIL"
                result["errors"].append(f"summary.status is not completed: {summary_payload.get('status')}")
        except Exception as exc:
            result["status"] = "FAIL"
            result["errors"].append(f"Invalid summary JSON: {exc}")

    rows = _read_registry_rows(registry_path)
    if not rows:
        result["status"] = "FAIL"
        result["errors"].append(f"Missing or empty registry: {registry_path}")
    else:
        matching = [row for row in rows if row.get("run_name") == run_name]
        if not matching:
            result["status"] = "FAIL"
            result["errors"].append(f"No registry row found for run_name={run_name}")

    return result


def _write_report(report_path: Path, payload: Dict[str, Any]) -> None:
    """Persist validator report for auditability."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional config list; defaults to strict DAVIS/BindingDB/KIBA configs.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for validation report JSON.",
    )
    args = parser.parse_args()

    configs = _resolve_configs(args.configs or [])
    report_path = Path(args.report_path) if args.report_path else _default_report_path()

    print("[INFO] Validating latest run output artifacts...")
    results: List[Dict[str, Any]] = []
    for config in configs:
        print(f"\n[INFO] --- validate: {config} ---")
        one = _validate_one_config(config)
        results.append(one)
        print(f"{one['status']} config={config}")
        for err in one["errors"]:
            print(f"- {err}")

    failed = [item for item in results if item["status"] == "FAIL"]
    overall_status = "PASS" if not failed else "FAIL"

    payload: Dict[str, Any] = {
        "started_at_utc": _now_utc_iso(),
        "finished_at_utc": _now_utc_iso(),
        "overall_status": overall_status,
        "failed_count": len(failed),
        "results": results,
    }
    _write_report(report_path, payload)

    print("\n[INFO] ===== output validation summary =====")
    for item in results:
        print(f"{item['status']} config={item['config']}")
    print(f"report={report_path}")

    if not failed:
        print("[OK] Run output validation passed")
        raise SystemExit(0)

    print(f"[ERROR] {len(failed)}/{len(results)} configs failed output validation")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
