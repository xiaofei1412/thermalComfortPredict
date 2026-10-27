import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

from scripts.install_demo import ROOT, install, sample
from scripts.package_delivery import package, public_paths
from scripts.verify_delivery import verify
from thermal_comfort.workbench import Workbench


class DeliveryReleaseTests(unittest.TestCase):
    def test_real_sample_verifies_without_engine_or_weights(self):
        result = verify()
        self.assertEqual(result['replayed_actions'], 5)
        self.assertTrue(result['c2_stop_decision_reproduced'])
        self.assertEqual(result['new_simulations'], 0)

    def test_install_is_idempotent_and_never_overwrites_different_history(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            path = install(store=store)
            before = path.read_bytes()
            self.assertEqual(install(store=store).read_bytes(), before)
            path.write_bytes(b'user history')
            with self.assertRaises(ValueError): install(store=store)
            self.assertEqual(path.read_bytes(), b'user history')

    def test_corrupt_archive_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root/'examples/workbench'; target.mkdir(parents=True)
            manifest = json.loads((ROOT/'examples/workbench/manifest.json').read_text())
            (target/'manifest.json').write_text(json.dumps(manifest))
            (target/'0a2e84c3cb66.json.gz').write_bytes(gzip.compress(b'changed'))
            with self.assertRaises(ValueError): sample(root)

    def test_archived_sample_load_does_not_call_simulator(self):
        class Offline:
            def call(self, *_):
                raise AssertionError('Offline replay must not contact BOPTEST')
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder); install(store=store)
            app = Workbench(store, client=Offline())
            self.assertEqual(app.sessions()[0]['steps'], 5)
            self.assertEqual(app.load('0a2e84c3cb66')['status'], 'completed')
            self.assertIsNone(app.state()['session'])

    def test_source_bundle_excludes_raw_data_outputs_and_cache(self):
        paths = {p.relative_to(ROOT).as_posix() for p in public_paths(ROOT)}
        self.assertIn('examples/workbench/0a2e84c3cb66.json.gz', paths)
        self.assertIn('requirements/workbench.lock.txt', paths)
        self.assertFalse(any(p.startswith(('outputs/', 'data/raw/', 'data/processed/')) or '__pycache__' in p for p in paths))

    @unittest.skipUnless(shutil.which('git'), 'Source packaging requires Git; runtime does not')
    def test_source_archive_uses_reviewed_index_not_dirty_working_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(['git','init','-q'],cwd=root,check=True,capture_output=True)
            (root/'README.md').write_bytes(b'reviewed\n')
            (root/'LICENSE').write_bytes(b'reviewed license\n')
            subprocess.run(['git','add','README.md','LICENSE'],cwd=root,check=True,capture_output=True)
            subprocess.run(['git','-c','user.name=Delivery Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],cwd=root,check=True,capture_output=True)
            (root/'README.md').write_bytes(b'unreviewed changes\n')
            (root/'LICENSE').write_bytes(b'unreviewed license\n')
            result = package(root,'source',root/'delivery.zip')
            with zipfile.ZipFile(root/'delivery.zip') as archive:
                self.assertEqual(archive.read('README.md'),b'reviewed\n')
                self.assertEqual(archive.read('LICENSE'),b'reviewed license\n')
            self.assertTrue(result['read_back_verified'])


if __name__ == '__main__':
    unittest.main()
