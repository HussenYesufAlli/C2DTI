import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import yaml


class TestRunAllOnceScript(unittest.TestCase):
    def test_run_all_once_passes_for_valid_minimal_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "ok.yaml"
            cfg = {
                "name": "C2DTI_RUN_ALL_ONCE_OK",
                "protocol": "P0",
                "output": {"base_dir": str(tmp_path / "outputs")},
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/run_all_once.py",
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
            self.assertIn("All run-once executions passed", completed.stdout)

    def test_run_all_once_returns_1_when_config_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_cfg = tmp_path / "missing.yaml"

            cmd = [
                sys.executable,
                "scripts/run_all_once.py",
                "--configs",
                str(missing_cfg),
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("FAIL code=1", completed.stdout)
            self.assertIn("run-once executions failed", completed.stdout)