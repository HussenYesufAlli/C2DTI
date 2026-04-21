import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _default_report_path() -> Path:
    """Build a timestamped default report path under outputs/gates."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "gates" / f"gate_all_{stamp}.json"


def _run_step(name: str, command: Sequence[str]) -> Dict[str, Any]:
    """Run one gate step, stream output, and capture execution metadata."""
    print(f"[INFO] --- step: {name} ---")
    print(f"[INFO] command={' '.join(command)}")

    started = time.time()
    completed = subprocess.run(list(command), check=False)
    duration_sec = round(time.time() - started, 3)

    code = int(completed.returncode)
    status = "PASS" if code == 0 else "FAIL"
    print(f"[INFO] result={status} code={code} duration_sec={duration_sec}")

    return {
        "name": name,
        "command": list(command),
        "return_code": code,
        "status": status,
        "duration_sec": duration_sec,
    }


def _write_report(report_path: Path, payload: Dict[str, Any]) -> None:
    """Persist gate execution report so runs are easy to audit later."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _now_utc_iso() -> str:
    """Return a timezone-aware UTC timestamp with trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-cmd",
        nargs="+",
        default=["make", "verify"],
        help="Command tokens for the verify step. Default: make verify",
    )
    parser.add_argument(
        "--real-cmd",
        nargs="+",
        default=["make", "real-all"],
        help="Command tokens for strict real pipeline step. Default: make real-all",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for gate report JSON.",
    )
    args = parser.parse_args()

    report_path = Path(args.report_path) if args.report_path else _default_report_path()
    started_at = _now_utc_iso()

    print("[INFO] Running unified quality gate...")
    step_results: List[Dict[str, Any]] = []

    # Step order intentionally enforces fast feedback before real pipeline checks.
    step_results.append(_run_step("verify", args.verify_cmd))
    if step_results[-1]["return_code"] == 0:
        step_results.append(_run_step("real-all", args.real_cmd))
    else:
        step_results.append(
            {
                "name": "real-all",
                "command": list(args.real_cmd),
                "return_code": None,
                "status": "SKIPPED",
                "duration_sec": 0.0,
            }
        )
        print("[INFO] real-all skipped because verify failed")

    failed_count = sum(1 for step in step_results if step["status"] == "FAIL")
    overall_status = "PASS" if failed_count == 0 else "FAIL"

    report_payload: Dict[str, Any] = {
        "started_at_utc": started_at,
        "finished_at_utc": _now_utc_iso(),
        "workspace": str(Path.cwd()),
        "overall_status": overall_status,
        "failed_step_count": failed_count,
        "steps": step_results,
    }
    _write_report(report_path, report_payload)

    print("\n[INFO] ===== gate summary =====")
    for step in step_results:
        print(f"{step['status']} step={step['name']} code={step['return_code']}")
    print(f"report={report_path}")

    if overall_status == "PASS":
        print("[OK] Unified quality gate passed")
        raise SystemExit(0)

    print("[ERROR] Unified quality gate failed")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
