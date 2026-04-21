import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml


class TestCheckAllDataScript(unittest.TestCase):
    def test_check_all_data_passes_for_valid_davis_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "davis"
            data_dir.mkdir(parents=True, exist_ok=True)

            (data_dir / "drug_smiles.txt").write_text("C\nCC\n", encoding="utf-8")
            (data_dir / "target_sequences.txt").write_text("AAAA\nBBBB\n", encoding="utf-8")
            np.savetxt(data_dir / "Y.txt", np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), fmt="%.4f")

            cfg_path = tmp_path / "davis_valid.yaml"
            cfg = {
                "name": "C2DTI_DAVIS_ALL_CHECK",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "DAVIS",
                    "path": str(data_dir),
                    "allow_placeholder": False,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/check_all_data.py",
                "--configs",
                str(cfg_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertIn("PASS code=0", completed.stdout)
            self.assertIn("All strict dataset prechecks passed", completed.stdout)
            self.assertIn("No follow-up actions required", completed.stdout)

    def test_check_all_data_returns_1_when_any_config_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "davis_missing.yaml"
            cfg = {
                "name": "C2DTI_DAVIS_ALL_CHECK_FAIL",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "DAVIS",
                    "path": str(tmp_path / "missing_davis"),
                    "allow_placeholder": False,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/check_all_data.py",
                "--configs",
                str(cfg_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("FAIL code=3", completed.stdout)
            self.assertIn("strict prechecks failed", completed.stdout)
            self.assertIn("next actions checklist", completed.stdout)
            self.assertIn("Create file:", completed.stdout)