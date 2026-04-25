import argparse
import subprocess
import sys
from typing import List, Tuple


DEFAULT_CONFIGS = [
    "configs/davis_gate.yaml",
    "configs/bindingdb_gate.yaml",
    "configs/kiba_gate.yaml",
]


def _resolve_configs(cli_configs: List[str]) -> List[str]:
    """Resolve config list from CLI; use defaults if none provided."""
    if cli_configs:
        return cli_configs
    return DEFAULT_CONFIGS


def _run_once(config_path: str) -> int:
    """Run one config through the run-once path."""
    cmd = [sys.executable, "scripts/run.py", "--config", config_path, "--run-once"]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional config list; defaults to DAVIS/BindingDB/KIBA gate configs.",
    )
    args = parser.parse_args()

    configs = _resolve_configs(args.configs or [])
    results: List[Tuple[str, int]] = []

    print("[INFO] Running all configs with --run-once...")
    for config in configs:
        print(f"\n[INFO] --- run-once: {config} ---")
        code = _run_once(config)
        results.append((config, code))

    print("\n[INFO] ===== run-once summary =====")
    failed = 0
    for config, code in results:
        status = "PASS" if code == 0 else "FAIL"
        if code != 0:
            failed += 1
        print(f"{status} code={code} config={config}")

    if failed == 0:
        print("[OK] All run-once executions passed")
        raise SystemExit(0)

    print(f"[ERROR] {failed}/{len(results)} run-once executions failed")
    raise SystemExit(1)


if __name__ == "__main__":
    main()