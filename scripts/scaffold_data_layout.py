import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def _scaffold_spec() -> List[Tuple[str, str]]:
    """Return required data files and starter content templates."""
    return [
        (
            "data/bindingdb/bindingdb.csv",
            "Drug_ID,Target_ID,Y\n",
        ),
        (
            "data/davis/drug_smiles.txt",
            "",
        ),
        (
            "data/davis/target_sequences.txt",
            "",
        ),
        (
            "data/davis/Y.txt",
            "",
        ),
        (
            "data/kiba/drug_smiles.txt",
            "",
        ),
        (
            "data/kiba/target_sequences.txt",
            "",
        ),
        (
            "data/kiba/Y.txt",
            "",
        ),
    ]


def scaffold_layout(root_dir: Path, overwrite: bool = False) -> Dict[str, List[str]]:
    """Create expected dataset paths without overwriting existing files by default."""
    created: List[str] = []
    skipped: List[str] = []

    for relative_path, starter_content in _scaffold_spec():
        file_path = root_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and not overwrite:
            skipped.append(str(file_path))
            continue

        file_path.write_text(starter_content, encoding="utf-8")
        created.append(str(file_path))

    return {
        "created": created,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=".",
        help="Project root where data/ should be scaffolded (default: current directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files with starter templates.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    result = scaffold_layout(root_dir=root_dir, overwrite=args.overwrite)

    print("[INFO] Dataset layout scaffold complete")
    print(f"root={root_dir}")
    print(f"created_count={len(result['created'])}")
    for path in result["created"]:
        print(f"created={path}")

    print(f"skipped_count={len(result['skipped'])}")
    for path in result["skipped"]:
        print(f"skipped={path}")

    print("[INFO] Next step: fill these files with real data, then run `make check-data-all`.")


if __name__ == "__main__":
    main()