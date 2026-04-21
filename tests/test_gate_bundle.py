import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


class TestGateBundleScript(unittest.TestCase):
    def test_gate_bundle_writes_tar_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gates_dir = tmp_path / "gates"
            bundles_dir = tmp_path / "bundles"
            gates_dir.mkdir(parents=True, exist_ok=True)

            (gates_dir / "gate_all_20260421-120000.json").write_text("{}", encoding="utf-8")
            (gates_dir / "validate_outputs_20260421-120000.json").write_text("{}", encoding="utf-8")
            (gates_dir / "latest_gate_summary.md").write_text("# Gate Summary\n", encoding="utf-8")

            bundle_path = bundles_dir / "gate_bundle.tar.gz"
            cmd = [
                sys.executable,
                "scripts/gate_bundle.py",
                "--gates-dir",
                str(gates_dir),
                "--bundle-dir",
                str(bundles_dir),
                "--bundle-path",
                str(bundle_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertTrue(bundle_path.exists())

            members = []
            with tarfile.open(bundle_path, "r:gz") as tf:
                members = tf.getnames()

            self.assertIn("gate_all_20260421-120000.json", members)
            self.assertIn("validate_outputs_20260421-120000.json", members)
            self.assertIn("latest_gate_summary.md", members)

            manifest_files = list(bundles_dir.glob("*.manifest.json"))
            self.assertEqual(len(manifest_files), 1)
            manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["file_count"], 3)

    def test_gate_bundle_fails_without_gate_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gates_dir = tmp_path / "gates"
            gates_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "scripts/gate_bundle.py",
                "--gates-dir",
                str(gates_dir),
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
