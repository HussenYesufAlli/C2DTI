import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2dti.config_validation import validate_config


class TestConfigValidation(unittest.TestCase):
    def test_valid_minimal_config(self) -> None:
        cfg = {
            "name": "C2DTI_MINIMAL",
            "protocol": "P0",
            "output": {"base_dir": "outputs"},
        }
        self.assertEqual(validate_config(cfg), [])

    def test_missing_required_keys(self) -> None:
        cfg = {}
        errors = validate_config(cfg)
        self.assertIn("Missing required key: name", errors)
        self.assertIn("Missing required key: protocol", errors)
        self.assertIn("Missing required key: output.base_dir", errors)


if __name__ == "__main__":
    unittest.main()
