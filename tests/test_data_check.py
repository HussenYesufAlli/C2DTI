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
            self.assertEqual(report["dataset_schema"]["dataset_type"], "directory")
            self.assertIn("drug_smiles.txt", report["dataset_schema"]["required_files"])
            self.assertEqual(report["content_validation"]["status"], "ok")
            self.assertEqual(report["content_validation"]["y_matrix_shape"], [2, 3])

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
            self.assertEqual(report["dataset_schema"]["path_kind"], "directory")
            self.assertIn("Y.txt", report["dataset_schema"]["required_files"])

    def test_check_data_davis_invalid_matrix_shape_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "davis"
            data_dir.mkdir(parents=True, exist_ok=True)

            (data_dir / "drug_smiles.txt").write_text("C\nCC\n", encoding="utf-8")
            (data_dir / "target_sequences.txt").write_text("AAAA\nBBBB\n", encoding="utf-8")
            np.savetxt(data_dir / "Y.txt", np.array([[0.1, 0.2, 0.3]], dtype=np.float32), fmt="%.4f")

            cfg_path = tmp_path / "davis_bad_shape.yaml"
            cfg = {
                "name": "C2DTI_DAVIS_BAD_SHAPE",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
                "dataset": {"name": "DAVIS", "path": str(data_dir), "allow_placeholder": False},
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = check_data(str(cfg_path))
            self.assertEqual(code, 3)

            report_path = tmp_path / "outputs" / "checks" / "davis_bad_shape_data_check.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["content_validation"]["status"], "error")
            self.assertEqual(report["content_validation"]["num_drugs_from_file"], 2)
            self.assertEqual(report["content_validation"]["num_targets_from_file"], 2)
            self.assertEqual(report["content_validation"]["y_matrix_shape"], [1, 3])
            self.assertIn("does not match expected", report["content_validation"]["reason"])

    def test_check_data_bindingdb_report_includes_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "bindingdb_missing.yaml"
            cfg = {
                "name": "C2DTI_BINDINGDB_CHECK_FAIL",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "BindingDB",
                    "path": str(tmp_path / "bindingdb.csv"),
                    "allow_placeholder": False,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = check_data(str(cfg_path))
            self.assertEqual(code, 3)

            report_path = tmp_path / "outputs" / "checks" / "bindingdb_missing_data_check.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["dataset_schema"]["dataset_type"], "csv")
            self.assertEqual(report["dataset_schema"]["path_kind"], "file")
            self.assertEqual(report["dataset_schema"]["required_columns"], ["Drug_ID", "Target_ID", "Y"])

    def test_check_data_bindingdb_invalid_columns_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "bindingdb.csv"
            csv_path.write_text("BadCol1,BadCol2\n1,2\n", encoding="utf-8")

            cfg_path = tmp_path / "bindingdb_invalid_columns.yaml"
            cfg = {
                "name": "C2DTI_BINDINGDB_INVALID_COLUMNS",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "BindingDB",
                    "path": str(csv_path),
                    "allow_placeholder": False,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = check_data(str(cfg_path))
            self.assertEqual(code, 3)

            report_path = tmp_path / "outputs" / "checks" / "bindingdb_invalid_columns_data_check.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["content_validation"]["status"], "error")
            self.assertEqual(report["content_validation"]["available_columns"], ["BadCol1", "BadCol2"])
            self.assertEqual(report["content_validation"]["missing_columns"], ["Drug_ID", "Target_ID", "Y"])

    def test_check_data_bindingdb_valid_columns_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "bindingdb.csv"
            csv_path.write_text("Drug_ID,Target_ID,Y\nD1,T1,10.0\n", encoding="utf-8")

            cfg_path = tmp_path / "bindingdb_valid_columns.yaml"
            cfg = {
                "name": "C2DTI_BINDINGDB_VALID_COLUMNS",
                "protocol": "P1",
                "output": {"base_dir": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "BindingDB",
                    "path": str(csv_path),
                    "allow_placeholder": False,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = check_data(str(cfg_path))
            self.assertEqual(code, 0)

            report_path = tmp_path / "outputs" / "checks" / "bindingdb_valid_columns_data_check.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["content_validation"]["status"], "ok")
            self.assertEqual(
                report["content_validation"]["resolved_columns"],
                {"Drug_ID": "Drug_ID", "Target_ID": "Target_ID", "Y": "Y"},
            )

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