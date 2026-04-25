import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class TestValidateRunOutputsScript(unittest.TestCase):
    def test_validate_outputs_passes_after_run_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_base = tmp_path / "outputs"
            cfg_path = tmp_path / "ok.yaml"
            cfg = {
                "name": "C2DTI_VALIDATE_OK",
                "protocol": "P0",
                "output": {"base_dir": str(output_base)},
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            run_cmd = [
                sys.executable,
                "scripts/run.py",
                "--config",
                str(cfg_path),
                "--run-once",
            ]
            run_completed = subprocess.run(
                run_cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run_completed.returncode, 0)

            report_path = tmp_path / "validate_report.json"
            validate_cmd = [
                sys.executable,
                "scripts/validate_run_outputs.py",
                "--configs",
                str(cfg_path),
                "--report-path",
                str(report_path),
            ]
            completed = subprocess.run(
                validate_cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall_status"], "PASS")

    def test_validate_outputs_fails_when_no_run_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_base = tmp_path / "outputs"
            cfg_path = tmp_path / "missing_run.yaml"
            cfg = {
                "name": "C2DTI_VALIDATE_MISSING",
                "protocol": "P0",
                "output": {"base_dir": str(output_base)},
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            report_path = tmp_path / "validate_report_fail.json"
            cmd = [
                sys.executable,
                "scripts/validate_run_outputs.py",
                "--configs",
                str(cfg_path),
                "--report-path",
                str(report_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
