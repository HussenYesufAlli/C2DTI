import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    "configs/davis_real_pipeline_strict.yaml",
    "configs/bindingdb_real_pipeline_strict.yaml",
    "configs/kiba_real_pipeline_strict.yaml",
]


def _resolve_configs(cli_configs: list[str]) -> list[str]:
    """Resolve config list from CLI; use defaults if none provided."""
    if cli_configs:
        return cli_configs
    return DEFAULT_CONFIGS


def _run_check(config_path: str) -> int:
    """Run `--check-data` for one config and stream output."""
    cmd = [sys.executable, "scripts/run.py", "--config", config_path, "--check-data"]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


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
    results: list[tuple[str, int]] = []

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
        raise SystemExit(0)

    print(f"[ERROR] {failed}/{len(results)} strict prechecks failed")
    raise SystemExit(1)


if __name__ == "__main__":
    main()