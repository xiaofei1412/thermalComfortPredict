import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_public_site import build
from scripts.verify_v2_delivery import ROOT


class PublicSiteTests(unittest.TestCase):
    def test_bilingual_site_links_to_one_public_replay_archive(self):
        frozen = (ROOT / 'workbench/v2/index.html').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'site'
            result = build(target)
            files = {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
            self.assertEqual(files, {
                'index.html', 'zh-CN/index.html', 'assets/showcase.css', 'assets/demo.png',
                'replay/index.html', 'replay/data.json', 'replay/copy.js',
                'replay/replay.js', 'replay/replay.css', '.nojekyll',
            })
            english = (target / 'index.html').read_text(encoding='utf-8')
            chinese = (target / 'zh-CN/index.html').read_text(encoding='utf-8')
            self.assertIn('<html lang="en">', english)
            self.assertIn('<html lang="zh-CN">', chinese)
            self.assertIn('href="zh-CN/" hreflang="zh-CN"', english)
            self.assertIn('href="../" hreflang="en"', chinese)
            self.assertIn('href="replay/"', english)
            self.assertNotIn('href="../replay/"', chinese)
            self.assertIn('href="../replay/?lang=zh-CN"', chinese)
            archive = (target / 'replay/data.json').read_bytes()
            self.assertEqual(archive, gzip.decompress((ROOT / 'examples/v2/replay.json.gz').read_bytes()))
            self.assertEqual(len(json.loads(archive)['cases']), 45)
            replay = (target / 'replay/index.html').read_text(encoding='utf-8')
            self.assertIn('<html lang="en">', replay)
            self.assertIn('href="?lang=zh-CN"', replay)
            self.assertIn('src="./replay.js"', replay)
            self.assertIn("fetch('./data.json')", (target / 'replay/replay.js').read_text(encoding='utf-8'))
            self.assertEqual(result['published_files'], len(files))
            self.assertEqual(result['FMU_calls'], 0)
            with self.assertRaises(FileExistsError):
                build(target)
        self.assertEqual((ROOT / 'workbench/v2/index.html').read_bytes(), frozen)
