import csv
import sys
from pathlib import Path
import tempfile
import unittest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2dti.runner import run_once


class TestRunContract(unittest.TestCase):
    def test_run_once_writes_summary_snapshot_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "minimal.yaml"
            output_base = tmp_path / "outputs"

            cfg = {
                "name": "C2DTI_TEST",
                "protocol": "P0",
                "output": {"base_dir": str(output_base)},
            }
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = run_once(str(config_path))
            self.assertEqual(code, 0)

            runs_root = output_base / "runs"
            run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
            self.assertEqual(len(run_dirs), 1)

            run_dir = run_dirs[0]
            summary_path = run_dir / "summary.json"
            snapshot_path = run_dir / "config_snapshot.yaml"
            registry_path = output_base / "results_registry.csv"

            self.assertTrue(summary_path.exists())
            self.assertTrue(snapshot_path.exists())
            self.assertTrue(registry_path.exists())

            with registry_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_name"], "C2DTI_TEST")
            self.assertEqual(rows[0]["protocol"], "P0")
            self.assertEqual(rows[0]["status"], "completed")
            self.assertTrue(rows[0]["summary_path"].endswith("summary.json"))
            self.assertTrue(rows[0]["config_snapshot_path"].endswith("config_snapshot.yaml"))


if __name__ == "__main__":
    unittest.main()
