import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGateSummaryScript(unittest.TestCase):
    def test_gate_summary_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gates_dir = tmp_path / "gates"
            gates_dir.mkdir(parents=True, exist_ok=True)

            gate_payload = {
                "overall_status": "PASS",
                "steps": [
                    {"name": "verify", "status": "PASS", "return_code": 0},
                    {"name": "real-all", "status": "PASS", "return_code": 0},
                    {"name": "validate-outputs", "status": "PASS", "return_code": 0},
                ],
            }
            validate_payload = {
                "overall_status": "PASS",
                "results": [
                    {"config": "configs/davis_real_pipeline_strict.yaml", "status": "PASS"},
                ],
            }

            (gates_dir / "gate_all_20260421-120000.json").write_text(
                json.dumps(gate_payload), encoding="utf-8"
            )
            (gates_dir / "validate_outputs_20260421-120000.json").write_text(
                json.dumps(validate_payload), encoding="utf-8"
            )

            summary_path = gates_dir / "latest_gate_summary.md"
            cmd = [
                sys.executable,
                "scripts/gate_summary.py",
                "--gates-dir",
                str(gates_dir),
                "--summary-path",
                str(summary_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertTrue(summary_path.exists())
            text = summary_path.read_text(encoding="utf-8")
            self.assertIn("# Gate Summary", text)
            self.assertIn("gate_overall_status: PASS", text)
            self.assertIn("validate_overall_status: PASS", text)

    def test_gate_summary_fail_on_nonpass_returns_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gates_dir = tmp_path / "gates"
            gates_dir.mkdir(parents=True, exist_ok=True)

            gate_payload = {
                "overall_status": "FAIL",
                "steps": [
                    {"name": "verify", "status": "FAIL", "return_code": 2},
                ],
            }
            (gates_dir / "gate_all_20260421-120000.json").write_text(
                json.dumps(gate_payload), encoding="utf-8"
            )

            cmd = [
                sys.executable,
                "scripts/gate_summary.py",
                "--gates-dir",
                str(gates_dir),
                "--fail-on-nonpass",
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1)


if __name__ == "__main__":
    unittest.main()
