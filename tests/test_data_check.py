import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2dti.data_check import check_data


class TestDataCheck(unittest.TestCase):
    def test_check_data_passes_for_valid_davis_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "davis"
            data_dir.mkdir(parents=True, exist_ok=True)

            (data_dir / "drug_smiles.txt").write_text("C\nCC\n", encoding="utf-8")
            (data_dir / "target_sequences.txt").write_text("AAAA\nBBBB\nCCCC\n", encoding="utf-8")
            np.savetxt(data_dir / "Y.txt", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32), fmt="%.4f")

            cfg_path = tmp_path / "valid_davis.yaml"
            cfg = {
                "name": "C2DTI_DAVIS_CHECK",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
                "dataset": {"name": "DAVIS", "path": str(data_dir), "allow_placeholder": False},
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = check_data(str(cfg_path))
            self.assertEqual(code, 0)

            report_path = tmp_path / "outputs" / "checks" / "valid_davis_data_check.json"
            self.assertTrue(report_path.exists())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["exit_code"], 0)
            self.assertEqual(report["dataset_summary"]["matrix_shape"], [2, 3])

    def test_check_data_fails_when_required_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "missing_davis.yaml"
            cfg = {
                "name": "C2DTI_DAVIS_CHECK_FAIL",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
                "dataset": {"name": "DAVIS", "path": str(tmp_path / "missing_davis"), "allow_placeholder": False},
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = check_data(str(cfg_path))
            self.assertEqual(code, 3)

            report_path = tmp_path / "outputs" / "checks" / "missing_davis_data_check.json"
            self.assertTrue(report_path.exists())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["exit_code"], 3)
            self.assertEqual(len(report["missing_files"]), 3)

    def test_check_data_requires_dataset_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "no_dataset.yaml"
            cfg = {
                "name": "C2DTI_NO_DATASET",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = check_data(str(cfg_path))
            self.assertEqual(code, 2)

            report_path = tmp_path / "outputs" / "checks" / "no_dataset_data_check.json"
            self.assertTrue(report_path.exists())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["exit_code"], 2)