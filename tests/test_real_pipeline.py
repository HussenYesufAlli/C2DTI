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

from src.c2dti.runner import run_once


class TestRealPipeline(unittest.TestCase):
    def test_run_once_real_pipeline_writes_predictions_and_causal_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "davis"
            data_dir.mkdir(parents=True, exist_ok=True)

            (data_dir / "drug_smiles.txt").write_text("C\nCC\n", encoding="utf-8")
            (data_dir / "target_sequences.txt").write_text("AAAA\nBBBB\nCCCC\n", encoding="utf-8")
            np.savetxt(data_dir / "Y.txt", np.array([[0.9, 0.2, 0.5], [0.4, 0.8, 0.1]], dtype=np.float32), fmt="%.4f")
            # Keep an explicit CSV fixture because DAVISLoader reads davis.csv in this pipeline.
            (data_dir / "davis.csv").write_text(
                "Drug_ID,Drug,Target_ID,Target,Y\n"
                "0,C,0,AAAA,0.9\n"
                "0,C,1,BBBB,0.2\n"
                "0,C,2,CCCC,0.5\n"
                "1,CC,0,AAAA,0.4\n"
                "1,CC,1,BBBB,0.8\n"
                "1,CC,2,CCCC,0.1\n",
                encoding="utf-8",
            )

            output_base = tmp_path / "outputs"
            config_path = tmp_path / "dataset.yaml"
            cfg = {
                "name": "C2DTI_DATASET_TEST",
                "protocol": "P1",
                "output": {"base_dir": str(output_base)},
                "dataset": {"name": "DAVIS", "path": str(data_dir)},
                "model": {"name": "simple_baseline"},
                "perturbation": {"strength": 0.3, "seed": 11},
                "causal": {"enabled": True, "weight": 1.0},
            }
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            code = run_once(str(config_path))
            self.assertEqual(code, 0)

            run_dir = next((output_base / "runs").iterdir())
            summary_path = run_dir / "summary.json"
            predictions_path = run_dir / "predictions.csv"

            self.assertTrue(summary_path.exists())
            self.assertTrue(predictions_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["dataset_name"], "DAVIS")
            self.assertEqual(summary["num_drugs"], 2)
            self.assertEqual(summary["num_targets"], 3)
            self.assertIn("prediction_stats", summary)
            self.assertIn("causal_score", summary)
            self.assertGreaterEqual(summary["causal_score"], 0.0)
            self.assertLessEqual(summary["causal_score"], 1.0)