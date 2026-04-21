import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGateAllScript(unittest.TestCase):
    def test_gate_all_passes_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "gate_report.json"

            cmd = [
                sys.executable,
                "scripts/gate_all.py",
                "--verify-cmd",
                "true",
                "--real-cmd",
                "true",
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

            self.assertEqual(completed.returncode, 0)
            self.assertTrue(report_path.exists())

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall_status"], "PASS")
            self.assertEqual(payload["failed_step_count"], 0)
            self.assertEqual(payload["steps"][0]["status"], "PASS")
            self.assertEqual(payload["steps"][1]["status"], "PASS")

    def test_gate_all_skips_real_when_verify_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "gate_report_fail.json"

            cmd = [
                sys.executable,
                "scripts/gate_all.py",
                "--verify-cmd",
                "false",
                "--real-cmd",
                "true",
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
            self.assertEqual(payload["failed_step_count"], 1)
            self.assertEqual(payload["steps"][0]["status"], "FAIL")
            self.assertEqual(payload["steps"][0]["return_code"], 1)
            self.assertEqual(payload["steps"][1]["status"], "SKIPPED")
            self.assertIsNone(payload["steps"][1]["return_code"])


if __name__ == "__main__":
    unittest.main()
