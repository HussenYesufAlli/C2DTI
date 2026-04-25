import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2dti.binary_runner import run_once_binary


class TestBinaryRunner(unittest.TestCase):
    def test_run_once_binary_writes_causal_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "davis_binary.csv"

            # Minimal binary matrix with overlapping drugs/targets.
            df = pd.DataFrame(
                {
                    "Drug_ID": ["D1", "D1", "D2", "D2"],
                    "Drug": ["CCO", "CCO", "CCN", "CCN"],
                    "Target_ID": ["T1", "T2", "T1", "T2"],
                    "Target": ["AAAA", "BBBB", "AAAA", "BBBB"],
                    "Label": [1, 0, 0, 1],
                }
            )
            df.to_csv(csv_path, index=False)

            cfg = {
                "name": "binary_causal_smoke",
                "protocol": "P_binary",
                "dataset": {
                    "name": "DAVIS",
                    "path": str(csv_path),
                    "allow_placeholder": False,
                },
                "model": {"name": "simple_baseline"},
                "split": {"strategy": "random", "test_ratio": 0.5, "seed": 42},
                "causal": {
                    "enabled": True,
                    "weight": 1.0,
                    "mode": "reliability",
                },
                "perturbation": {"strength": 0.1, "seed": 42},
                "binary": {"threshold": 0.5},
                "output": {"base_dir": str(tmp_path / "outputs_binary")},
            }
            cfg_path = tmp_path / "binary.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = run_once_binary(str(cfg_path))
            self.assertEqual(code, 0)

            runs_dir = tmp_path / "outputs_binary" / "runs"
            run_dir = next(runs_dir.iterdir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["task_type"], "binary_classification")
            self.assertIn("causal", summary)
            self.assertIn("causal_score", summary)
            self.assertGreaterEqual(float(summary["causal_score"]), 0.0)
            self.assertLessEqual(float(summary["causal_score"]), 1.0)


if __name__ == "__main__":
    unittest.main()
