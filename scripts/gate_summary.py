import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return the newest file matching a pattern or None when absent."""
    files = list(directory.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file content into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def _status_line(label: str, status: str) -> str:
    """Build a compact status line used by the summary output."""
    return f"- {label}: {status}"


def _build_markdown(
    gate_report_path: Path,
    gate_report: Dict[str, Any],
    validate_report_path: Optional[Path],
    validate_report: Optional[Dict[str, Any]],
) -> str:
    """Generate a readable markdown summary from gate evidence reports."""
    lines: List[str] = []
    lines.append("# Gate Summary")
    lines.append("")
    lines.append(f"- gate_report: {gate_report_path}")
    lines.append(f"- gate_overall_status: {gate_report.get('overall_status', 'UNKNOWN')}")
    lines.append("")
    lines.append("## Gate Steps")

    for step in gate_report.get("steps", []):
        lines.append(
            _status_line(
                f"{step.get('name', 'unknown')} (code={step.get('return_code')})",
                str(step.get("status", "UNKNOWN")),
            )
        )

    if validate_report_path is not None and validate_report is not None:
        lines.append("")
        lines.append("## Output Validation")
        lines.append(f"- validate_report: {validate_report_path}")
        lines.append(
            f"- validate_overall_status: {validate_report.get('overall_status', 'UNKNOWN')}"
        )
        for item in validate_report.get("results", []):
            lines.append(
                _status_line(
                    str(item.get("config", "unknown-config")),
                    str(item.get("status", "UNKNOWN")),
                )
            )
    else:
        lines.append("")
        lines.append("## Output Validation")
        lines.append("- validate_report: not found")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gates-dir",
        default="outputs/gates",
        help="Directory containing gate report files.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional explicit output path for markdown summary.",
    )
    parser.add_argument(
        "--fail-on-nonpass",
        action="store_true",
        help="Return non-zero when gate or validation report status is not PASS.",
    )
    args = parser.parse_args()

    gates_dir = Path(args.gates_dir)
    summary_path = Path(args.summary_path) if args.summary_path else gates_dir / "latest_gate_summary.md"

    gate_report_path = _latest_file(gates_dir, "gate_all_*.json")
    if gate_report_path is None:
        print(f"[ERROR] No gate report found in: {gates_dir}")
        raise SystemExit(1)

    gate_report = _load_json(gate_report_path)

    validate_report_path = _latest_file(gates_dir, "validate_outputs_*.json")
    validate_report: Optional[Dict[str, Any]] = None
    if validate_report_path is not None:
        validate_report = _load_json(validate_report_path)

    markdown = _build_markdown(
        gate_report_path=gate_report_path,
        gate_report=gate_report,
        validate_report_path=validate_report_path,
        validate_report=validate_report,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(markdown, encoding="utf-8")

    print(f"[OK] Gate summary written: {summary_path}")

    if args.fail_on_nonpass:
        gate_ok = gate_report.get("overall_status") == "PASS"
        validate_ok = True if validate_report is None else validate_report.get("overall_status") == "PASS"
        if not (gate_ok and validate_ok):
            print("[ERROR] Non-pass status detected in latest reports")
            raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
