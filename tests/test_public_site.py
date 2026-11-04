import gzip
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
            self.assertEqual(files, {
                'index.html', 'zh-CN/index.html', 'assets/showcase.css', 'assets/demo.png',
                'replay/index.html', 'replay/data.json', '.nojekyll',
            })
            english = (target / 'index.html').read_text(encoding='utf-8')
            chinese = (target / 'zh-CN/index.html').read_text(encoding='utf-8')
            self.assertIn('<html lang="en">', english)
            self.assertIn('<html lang="zh-CN">', chinese)
            self.assertIn('href="zh-CN/" hreflang="zh-CN"', english)
            self.assertIn('href="../" hreflang="en"', chinese)
            self.assertIn('href="replay/"', english)
            self.assertIn('href="../replay/"', chinese)
            archive = (target / 'replay/data.json').read_bytes()
            self.assertEqual(archive, gzip.decompress((ROOT / 'examples/v2/replay.json.gz').read_bytes()))
            self.assertEqual(len(json.loads(archive)['cases']), 45)
            replay = frozen.decode('utf-8').replace("fetch('/data.json')", "fetch('./data.json')")
            self.assertEqual((target / 'replay/index.html').read_text(encoding='utf-8'), replay)
            self.assertEqual(result['published_files'], len(files))
            self.assertEqual(result['FMU_calls'], 0)
            with self.assertRaises(FileExistsError):
                build(target)
        self.assertEqual((ROOT / 'workbench/v2/index.html').read_bytes(), frozen)
