import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_delivery import inspect_asset


class DeliveryAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_missing_asset_is_a_gap_and_not_a_claimed_hash_mismatch(self):
        result = inspect_asset(self.root, {"paths": ["missing"], "sha256": "0" * 64}, set())
        self.assertFalse(result["present"])
        self.assertIsNone(result["expected_sha256_match"])
        self.assertEqual(result["missing"], ["missing"])

    def test_reference_digest_detects_modified_bytes(self):
        (self.root / "model").write_bytes(b"original")
        (self.root / "manifest.json").write_text(json.dumps({"artifact": hashlib.sha256(b"original").hexdigest()}))
        spec = {"paths": ["model"], "expected": {"path": "manifest.json", "keys": ["artifact"]}}
        self.assertTrue(inspect_asset(self.root, spec, {"model"})["expected_sha256_match"])
        (self.root / "model").write_bytes(b"changed")
        self.assertFalse(inspect_asset(self.root, spec, {"model"})["expected_sha256_match"])

    def test_inventory_ignores_python_caches_and_tracks_every_real_file(self):
        (self.root / "source/__pycache__").mkdir(parents=True)
        (self.root / "source/a.py").write_bytes(b"pass\n")
        (self.root / "source/__pycache__/a.pyc").write_bytes(b"cache")
        result = inspect_asset(self.root, {"paths": ["source"]}, {"source/a.py"})
        self.assertEqual(result["file_count"], 1)
        self.assertEqual(result["bytes"], 5)
        self.assertTrue(result["files"][0]["in_git"])
        self.assertEqual(result, inspect_asset(self.root, {"paths": ["source"]}, {"source/a.py"}))
