import argparse
from pathlib import Path
from typing import Dict, List


def _write_if_empty(path: Path, content: str, force: bool) -> bool:
    """Write content if file is empty/missing, or always when force is true."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text(content, encoding="utf-8")
        return True

    existing = path.read_text(encoding="utf-8")
    if force or not existing.strip():
        path.write_text(content, encoding="utf-8")
        return True

    return False


def fill_demo_data(root_dir: Path, force: bool = False) -> Dict[str, List[str]]:
    """Fill scaffolded dataset files with minimal valid demo content."""
    changed: List[str] = []
    skipped: List[str] = []

    file_payloads = {
        root_dir / "data" / "bindingdb" / "bindingdb.csv": (
            "Drug_ID,Target_ID,Y\n"
            "D1,T1,10.0\n"
            "D2,T2,1000.0\n"
        ),
        root_dir / "data" / "davis" / "drug_smiles.txt": "C\nCC\n",
        root_dir / "data" / "davis" / "target_sequences.txt": "AAAA\nBBBB\n",
        root_dir / "data" / "davis" / "Y.txt": "0.10 0.20\n0.30 0.40\n",
        root_dir / "data" / "kiba" / "drug_smiles.txt": "CCC\nCCCC\n",
        root_dir / "data" / "kiba" / "target_sequences.txt": "MMMM\nNNNN\n",
        root_dir / "data" / "kiba" / "Y.txt": "0.55 0.25\n0.15 0.85\n",
    }

    for file_path, payload in file_payloads.items():
        if _write_if_empty(file_path, payload, force=force):
            changed.append(str(file_path))
        else:
            skipped.append(str(file_path))

    return {
        "changed": changed,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root (default: current directory).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even non-empty files with demo content.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    result = fill_demo_data(root_dir=root_dir, force=args.force)

    print("[INFO] Demo dataset fill complete")
    print(f"root={root_dir}")
    print(f"changed_count={len(result['changed'])}")
    for path in result["changed"]:
        print(f"changed={path}")

    print(f"skipped_count={len(result['skipped'])}")
    for path in result["skipped"]:
        print(f"skipped={path}")

    print("[INFO] Next step: run `make check-data-all`.")


if __name__ == "__main__":
    main()