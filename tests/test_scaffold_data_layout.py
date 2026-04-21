import subprocess
import sys
from pathlib import Path
import tempfile
import unittest


class TestScaffoldDataLayoutScript(unittest.TestCase):
    def test_scaffold_creates_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cmd = [
                sys.executable,
                "scripts/scaffold_data_layout.py",
                "--root",
                str(tmp_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0)

            expected_files = [
                tmp_path / "data" / "bindingdb" / "bindingdb.csv",
                tmp_path / "data" / "davis" / "drug_smiles.txt",
                tmp_path / "data" / "davis" / "target_sequences.txt",
                tmp_path / "data" / "davis" / "Y.txt",
                tmp_path / "data" / "kiba" / "drug_smiles.txt",
                tmp_path / "data" / "kiba" / "target_sequences.txt",
                tmp_path / "data" / "kiba" / "Y.txt",
            ]

            for file_path in expected_files:
                self.assertTrue(file_path.exists(), msg=f"missing scaffold file: {file_path}")

    def test_scaffold_does_not_overwrite_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bindingdb_csv = tmp_path / "data" / "bindingdb" / "bindingdb.csv"
            bindingdb_csv.parent.mkdir(parents=True, exist_ok=True)
            bindingdb_csv.write_text("custom\n", encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/scaffold_data_layout.py",
                "--root",
                str(tmp_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertEqual(bindingdb_csv.read_text(encoding="utf-8"), "custom\n")
            self.assertIn("skipped=", completed.stdout)