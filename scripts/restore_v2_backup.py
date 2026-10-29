"""Restore the encrypted V2 source/evidence/prerequisites and exact offline wheels."""

import argparse
import json
from pathlib import Path

from scripts.restore_private_backup import restore,restore_zip


def restore_v2(archive,receipt,key,destination):
    result=restore(archive,receipt,key,destination)
    runtime=restore_zip(destination/'runtime-wheels.zip',
        json.loads((destination/'runtime-wheels.manifest.json').read_text(encoding='utf-8')),
        Path(result['restored_source']))
    result.update(runtime_members_restored=runtime,wheel_count=receipt['wheel_count'],
        scope='local independent restore only; actual remote download must be documented separately')
    (destination/'restore-verification.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(restore_v2(args.archive,json.loads(args.receipt.read_text(encoding='utf-8')),args.key_file,args.destination),ensure_ascii=False))
