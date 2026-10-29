"""Copy and read-back verify all closed V2 assets; never train or resume a study."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import zipfile

from scripts.package_delivery import sha_stream

ROOT=Path(__file__).resolve().parents[1]


def paths(root):
    config=json.loads((root/'configs/v2_delivery_archive.json').read_text(encoding='utf-8'))
    for name in config['closed_formal_folders']:
        folder=root/'outputs'/name/'live-registered'
        registration=json.loads((folder/'registration.json').read_text(encoding='utf-8'))
        budget=json.loads((folder/'budget.json').read_text(encoding='utf-8'))
        if registration['formal_training_authorized'] or any(v['status']=='running' for v in budget['seeds'].values()):
            raise ValueError('Refusing an active trial: '+name)
    result=[]
    for name in config['private_output_folders']:
        folder=root/'outputs'/name
        if not folder.is_dir():
            raise FileNotFoundError('Required V2 archive folder: '+name)
        for directory,dirs,names in os.walk(folder):
            dirs[:]=[d for d in dirs if d.lower() not in config['excluded_directory_names']]
            for name in names:
                path=Path(directory)/name
                if name.startswith('.env') or path.suffix=='.pyc':
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Archive input escapes workspace')
                result.append(path)
    return sorted(result)


def package(destination,root=ROOT):
    files=paths(root)
    destination.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with zipfile.ZipFile(destination,'x') as archive:
        for path in files:
            name=path.relative_to(root).as_posix()
            with path.open('rb') as stream:expected=sha_stream(stream)
            size=path.stat().st_size
            if path.suffix in ['.gz','.zip','.npz']:
                archive.write(path,name,compress_type=zipfile.ZIP_STORED)
            else:
                archive.write(path,name,compress_type=zipfile.ZIP_DEFLATED,compresslevel=1)
            rows.append({'path':name,'bytes':size,'sha256':expected})
    with zipfile.ZipFile(destination) as archive:
        assert len(archive.namelist())==len(rows)
        for row in rows:
            with archive.open(row['path']) as stream:
                if sha_stream(stream)!=row['sha256']:
                    raise ValueError('Archive read-back mismatch: '+row['path'])
    with destination.open('rb') as stream:checksum=sha_stream(stream)
    result={'kind':'private_v2_closed_evidence','archive':destination.name,'archive_sha256':checksum,
        'archive_bytes':destination.stat().st_size,'file_count':len(rows),'uncompressed_bytes':sum(r['bytes'] for r in rows),
        'source_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'files':rows,'read_back_verified':True,'new_model_experiments':0,'FMU_calls':0,
        'scope':'private closed V2 evidence; not an off-site verification, redistribution license or live simulator checkpoint'}
    destination.with_suffix('.manifest.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result=package(parser.parse_args().output)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))
