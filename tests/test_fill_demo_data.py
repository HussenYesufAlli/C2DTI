import subprocess
import sys
from pathlib import Path
import tempfile
import unittest


class TestFillDemoDataScript(unittest.TestCase):
    def test_fill_demo_data_populates_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = [
                tmp_path / "data" / "bindingdb" / "bindingdb.csv",
                tmp_path / "data" / "davis" / "drug_smiles.txt",
                tmp_path / "data" / "davis" / "target_sequences.txt",
                tmp_path / "data" / "davis" / "Y.txt",
                tmp_path / "data" / "kiba" / "drug_smiles.txt",
                tmp_path / "data" / "kiba" / "target_sequences.txt",
                tmp_path / "data" / "kiba" / "Y.txt",
            ]
            for file_path in paths:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("", encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/fill_demo_data.py",
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
            self.assertIn("changed_count=7", completed.stdout)
            self.assertIn("Drug_ID,Target_ID,Y", (tmp_path / "data" / "bindingdb" / "bindingdb.csv").read_text(encoding="utf-8"))

    def test_fill_demo_data_skips_non_empty_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bindingdb_csv = tmp_path / "data" / "bindingdb" / "bindingdb.csv"
            bindingdb_csv.parent.mkdir(parents=True, exist_ok=True)
            bindingdb_csv.write_text("custom\n", encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/fill_demo_data.py",
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
            self.assertIn("skipped_count=1", completed.stdout)