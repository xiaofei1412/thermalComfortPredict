import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_public_site import build
from scripts.verify_v2_delivery import ROOT


class PublicSiteTests(unittest.TestCase):
    def test_only_public_assets_are_published_with_a_relative_replay_route(self):
        frozen = (ROOT / 'workbench/v2/index.html').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'site'
            result = build(target)
            files = {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
            self.assertEqual(files, {'index.html', 'replay/index.html', 'replay/data.json', 'assets/demo.png', '.nojekyll'})
            self.assertEqual(len(json.loads((target / 'replay/data.json').read_bytes())['cases']), 45)
            self.assertIn("fetch('./data.json')", (target / 'replay/index.html').read_text(encoding='utf-8'))
            self.assertEqual(result['FMU_calls'], 0)
            with self.assertRaises(FileExistsError):
                build(target)
        self.assertEqual((ROOT / 'workbench/v2/index.html').read_bytes(), frozen)
