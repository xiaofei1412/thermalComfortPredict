"""Build an allowlisted GitHub Pages showcase from the frozen public replay."""

import argparse
import gzip
from pathlib import Path

from scripts.verify_v2_delivery import ROOT, verify


def build(destination, root=ROOT):
    result = verify(root)
    source = (root / 'workbench/v2/index.html').read_text(encoding='utf-8')
    if source.count("fetch('/data.json')") != 1:
        raise ValueError('Archived replay entry changed; review the relative data route')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'replay').mkdir()
    (destination / 'assets').mkdir()
    (destination / 'index.html').write_bytes((root / 'site/index.html').read_bytes())
    (destination / 'replay/index.html').write_text(
        source.replace("fetch('/data.json')", "fetch('./data.json')"), encoding='utf-8', newline='\n')
    (destination / 'replay/data.json').write_bytes(gzip.decompress((root / 'examples/v2/replay.json.gz').read_bytes()))
    (destination / 'assets/demo.png').write_bytes((root / 'reports/v2_delivery/demo.png').read_bytes())
    (destination / '.nojekyll').write_bytes(b'')
    return {'cases': result['cases'], 'published_files': 5, 'FMU_calls': 0,
            'scope': 'public archived replay and project overview only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/public-site')
    args = parser.parse_args()
    print(build(args.output))
