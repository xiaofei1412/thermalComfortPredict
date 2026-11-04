"""Build an allowlisted GitHub Pages showcase from the frozen public replay."""

import argparse
import gzip
from pathlib import Path

from scripts.verify_v2_delivery import ROOT, verify


def build(destination, root=ROOT):
    result = verify(root)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'zh-CN').mkdir()
    (destination / 'replay').mkdir()
    (destination / 'assets').mkdir()
    (destination / 'index.html').write_bytes((root / 'site/index.html').read_bytes())
    (destination / 'zh-CN/index.html').write_bytes((root / 'site/zh-CN/index.html').read_bytes())
    (destination / 'assets/showcase.css').write_bytes((root / 'site/showcase.css').read_bytes())
    for name in ('index.html', 'copy.js', 'replay.js', 'replay.css'):
        (destination / 'replay' / name).write_bytes((root / 'site/replay' / name).read_bytes())
    (destination / 'replay/data.json').write_bytes(gzip.decompress((root / 'examples/v2/replay.json.gz').read_bytes()))
    (destination / 'assets/demo.png').write_bytes((root / 'site/replay-preview.png').read_bytes())
    (destination / '.nojekyll').write_bytes(b'')
    return {'cases': result['cases'], 'published_files': 10, 'FMU_calls': 0,
            'scope': 'bilingual public replay and project overview'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/public-site')
    args = parser.parse_args()
    print(build(args.output))
