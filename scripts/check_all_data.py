import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


DEFAULT_CONFIGS = [
    "configs/davis_real_pipeline_strict.yaml",
    "configs/bindingdb_real_pipeline_strict.yaml",
    "configs/kiba_real_pipeline_strict.yaml",
]


def _resolve_configs(cli_configs: List[str]) -> List[str]:
    """Resolve config list from CLI; use defaults if none provided."""
    if cli_configs:
        return cli_configs
    return DEFAULT_CONFIGS


def _run_check(config_path: str) -> int:
    """Run `--check-data` for one config and stream output."""
    cmd = [sys.executable, "scripts/run.py", "--config", config_path, "--check-data"]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _report_path_for_config(config_path: str) -> Optional[Path]:
    """Resolve expected JSON report path for one config file."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return None

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    base_dir = Path(cfg.get("output", {}).get("base_dir", "outputs"))
    return base_dir / "checks" / f"{cfg_path.stem}_data_check.json"


def _load_report(report_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load one JSON report if available."""
    if report_path is None or not report_path.exists():
        return None

    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _print_next_actions(results: List[Tuple[str, int]]) -> None:
    """Print a compact actionable checklist from generated reports."""
    failed_results = [(config, code) for config, code in results if code != 0]
    if not failed_results:
        print("[INFO] No follow-up actions required")
        return

    print("\n[INFO] ===== next actions checklist =====")
    for config, _ in failed_results:
        report_path = _report_path_for_config(config)
        report = _load_report(report_path)

        print(f"[INFO] config={config}")
        if report is None:
            print("- Re-run check and inspect terminal output (report not found or unreadable)")
            if report_path is not None:
                print(f"- Expected report path: {report_path}")
            continue

        reason = report.get("reason")
        if reason:
            print(f"- Reason: {reason}")

        missing_files = report.get("missing_files", [])
        if missing_files:
            for missing_file in missing_files:
                print(f"- Create file: {missing_file}")
            continue

        content_validation = report.get("content_validation", {})
        if content_validation.get("status") == "error":
            content_reason = content_validation.get("reason")
            if content_reason:
                print(f"- Fix data content: {content_reason}")

            dataset_name = str(report.get("dataset_name", "")).upper()
            dataset_path = str(report.get("dataset_path", ""))
            if "no data rows" in str(content_reason).lower() and dataset_name == "BINDINGDB":
                print(f"- Add at least one data row to: {dataset_path}")
                print("- Required columns: Drug_ID, Target_ID, Y")

            if "no non-empty rows" in str(content_reason).lower() and dataset_name in {"DAVIS", "KIBA"}:
                print(f"- Add at least one non-empty line in: {dataset_path}/drug_smiles.txt")
                print(f"- Add at least one non-empty line in: {dataset_path}/target_sequences.txt")
                print(f"- Fill {dataset_path}/Y.txt with a numeric matrix of shape [num_drugs, num_targets]")

            if "shape" in str(content_reason).lower() and dataset_name in {"DAVIS", "KIBA"}:
                expected_drugs = content_validation.get("num_drugs_from_file")
                expected_targets = content_validation.get("num_targets_from_file")
                if expected_drugs is not None and expected_targets is not None:
                    print(
                        f"- Update {dataset_path}/Y.txt to shape [{expected_drugs}, {expected_targets}] "
                        "to match sequence files"
                    )

            missing_columns = content_validation.get("missing_columns", [])
            for column in missing_columns:
                print(f"- Add missing column: {column}")
            continue

        print("- Inspect report details for remediation")
        if report_path is not None:
            print(f"- Report path: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional config list; defaults to strict DAVIS/BindingDB/KIBA configs.",
    )
    args = parser.parse_args()

    configs = _resolve_configs(args.configs or [])
    results: List[Tuple[str, int]] = []

    print("[INFO] Running strict dataset prechecks...")
    for config in configs:
        print(f"\n[INFO] --- checking: {config} ---")
        code = _run_check(config)
        results.append((config, code))

    print("\n[INFO] ===== strict precheck summary =====")
    failed = 0
    for config, code in results:
        status = "PASS" if code == 0 else "FAIL"
        if code != 0:
            failed += 1
        print(f"{status} code={code} config={config}")

    if failed == 0:
        print("[OK] All strict dataset prechecks passed")
        _print_next_actions(results)
        raise SystemExit(0)

    print(f"[ERROR] {failed}/{len(results)} strict prechecks failed")
    _print_next_actions(results)
    raise SystemExit(1)


if __name__ == "__main__":
    main()