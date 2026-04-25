import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_utc_stamp() -> str:
    """Return compact UTC timestamp used in bundle file names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return newest matching file from a directory or None."""
    files = list(directory.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def _build_manifest(files: List[Path]) -> Dict[str, Any]:
    """Build a small manifest with relative paths and file sizes."""
    entries: List[Dict[str, Any]] = []
    for path in files:
        entries.append(
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "file_count": len(entries),
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gates-dir",
        default="outputs/gates",
        help="Directory containing gate evidence files.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="outputs/gates/bundles",
        help="Directory where bundles are written.",
    )
    parser.add_argument(
        "--bundle-path",
        default=None,
        help="Optional explicit tar.gz output path.",
    )
    args = parser.parse_args()

    gates_dir = Path(args.gates_dir)
    bundle_dir = Path(args.bundle_dir)

    if not gates_dir.exists():
        print(f"[ERROR] gates directory not found: {gates_dir}")
        raise SystemExit(1)

    latest_gate = _latest_file(gates_dir, "gate_all_*.json")
    if latest_gate is None:
        print(f"[ERROR] no gate report found in {gates_dir}")
        raise SystemExit(1)

    latest_validate = _latest_file(gates_dir, "validate_outputs_*.json")
    latest_summary = gates_dir / "latest_gate_summary.md"

    files_to_pack: List[Path] = [latest_gate]
    if latest_validate is not None:
        files_to_pack.append(latest_validate)
    if latest_summary.exists():
        files_to_pack.append(latest_summary)

    manifest = _build_manifest(files_to_pack)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(args.bundle_path) if args.bundle_path else bundle_dir / f"gate_bundle_{_now_utc_stamp()}.tar.gz"
    manifest_path = bundle_dir / f"{bundle_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with tarfile.open(bundle_path, "w:gz") as tf:
        for src in files_to_pack:
            tf.add(src, arcname=src.name)
        tf.add(manifest_path, arcname=manifest_path.name)

    print(f"[OK] Gate bundle written: {bundle_path}")
    print(f"manifest={manifest_path}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
