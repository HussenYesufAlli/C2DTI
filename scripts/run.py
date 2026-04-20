import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2dti.runner import dry_run, run_once

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        raise SystemExit(dry_run(args.config))
    if args.run_once:
        raise SystemExit(run_once(args.config))

    print("[INFO] Choose --dry-run or --run-once")
    raise SystemExit(0)

if __name__ == "__main__":
    main()
