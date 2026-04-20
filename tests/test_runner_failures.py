import sys
from pathlib import Path
import tempfile
import unittest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2dti.runner import dry_run, run_once


class TestRunnerFailurePaths(unittest.TestCase):
    def test_dry_run_missing_config_returns_1(self) -> None:
        code = dry_run("/tmp/does-not-exist-c2dti.yaml")
        self.assertEqual(code, 1)

    def test_run_once_missing_config_returns_1(self) -> None:
        code = run_once("/tmp/does-not-exist-c2dti.yaml")
        self.assertEqual(code, 1)

    def test_dry_run_invalid_config_returns_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "invalid.yaml"
            cfg_path.write_text(yaml.safe_dump({"name": "only_name"}), encoding="utf-8")
            code = dry_run(str(cfg_path))
            self.assertEqual(code, 2)

    def test_run_once_invalid_config_returns_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "invalid.yaml"
            cfg_path.write_text(yaml.safe_dump({"protocol": "P0"}), encoding="utf-8")
            code = run_once(str(cfg_path))
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
