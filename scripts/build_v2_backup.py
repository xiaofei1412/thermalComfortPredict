"""Prepare an owner-private encrypted V2 backup with a separately stored key."""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def stored_zip(path,members):
    rows=[]
    with zipfile.ZipFile(path,'x',zipfile.ZIP_STORED) as archive:
        for source,name in members:
            archive.write(source,name)
            rows.append({'path':name,'bytes':source.stat().st_size,'sha256':sha(source)})
    result={'archive_sha256':sha(path),'files':rows,'scope':'owner-private restore prerequisites'}
    write(path.with_suffix('.manifest.json'),result)
    return result


def build(source,research,out):
    import py7zr  # Optional backup-only environment, never required for the demo.
    out=out.resolve()
    if not out.is_relative_to(ROOT/'outputs/v2-delivery'):
        raise ValueError('Backup output must stay in outputs/v2-delivery')
    payload,upload,key_dir=out/'payload',out/'upload',out/'key'
    payload.mkdir(parents=True,exist_ok=False)
    upload.mkdir(exist_ok=False)
    key_dir.mkdir(exist_ok=False)
    if os.name=='nt':
        subprocess.run(['icacls',str(key_dir),'/inheritance:r','/grant:r',getpass.getuser()+':(OI)(CI)F'],
            check=True,capture_output=True)
    for path,name in [(source,'source.zip'),(research,'research-evidence.zip')]:
        manifest=path.with_suffix('.manifest.json')
        value=json.loads(manifest.read_text(encoding='utf-8'))
        if sha(path)!=value['archive_sha256'] or not value['read_back_verified']:
            raise ValueError('Input archive not verified: '+path.name)
        # Same-volume staging links avoid duplicating the large verified ZIP.
        os.link(path,payload/name)
        (payload/Path(name).with_suffix('.manifest.json')).write_bytes(manifest.read_bytes())
    files=['outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu',
        'outputs/boptest-v0.9.0/license.md']
    baseline=json.loads((ROOT/'reports/v2_0/lock.json').read_text(encoding='utf-8'))['v1_baseline']
    assert sha(ROOT/files[0])==baseline['fmu_sha256']
    stored_zip(payload/'upstream-prerequisites.zip',[(ROOT/name,name) for name in files])
    wheels=sorted((ROOT/'outputs/v2-delivery/wheels').glob('*.whl'))
    assert len(wheels)==34
    stored_zip(payload/'runtime-wheels.zip',[(p,'wheels/'+p.name) for p in wheels]+
        [(ROOT/'requirements/v2-windows-py314.lock.txt','requirements/v2-windows-py314.lock.txt')])
    (payload/'README-RESTORE.txt').write_bytes(b'Private owner V2 backup. No public share or redistribution license.\n'
        b'Never upload the separate key directory. Verify with the trusted Git/local receipt.\n'
        b'Use scripts/restore_v2_backup.py, then the offline verification commands in docs/REPRODUCE_V2.md.\n'
        b'Local restoration alone does not establish off-site recovery.\n')
    contents={'role':'private_closed_V2_backup','files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(payload.iterdir())]}
    write(payload/'BACKUP_CONTENTS.json',contents)
    password=secrets.token_urlsafe(32)
    key=key_dir/'RECOVERY_KEY_V2.txt'
    with key.open('x',encoding='utf-8',newline='\n') as stream:stream.write(password+'\n')
    encrypted=upload/'thermalComfortPredict-V2-private-20261029.7z'
    filters=[{'id':py7zr.FILTER_COPY},{'id':py7zr.FILTER_CRYPTO_AES256_SHA256}]
    with py7zr.SevenZipFile(encrypted,'x',password=password,header_encryption=True,filters=filters) as archive:
        for p in sorted(payload.iterdir()):archive.write(p,p.name)
    try:
        with py7zr.SevenZipFile(encrypted) as archive:archive.getnames()
    except py7zr.exceptions.PasswordRequired:
        pass
    else:
        raise ValueError('Archive headers are not password protected')
    receipt={'schema':2,'archive':encrypted.name,'archive_bytes':encrypted.stat().st_size,'archive_sha256':sha(encrypted),
        'contents_sha256':sha(payload/'BACKUP_CONTENTS.json'),'encryption':'7z AES-256, encrypted filenames',
        'password_required_for_listing':True,'contains_recovery_key':False,'wheel_count':34,
        'source_archive_sha256':sha(source),'research_archive_sha256':sha(research),
        'transfer_status':'prepared_local_only_not_uploaded_or_downloaded',
        'destination_selected':'owner Baidu Netdisk private directory; no public share',
        'key_location_hint':key.relative_to(ROOT).as_posix()+'; local only, not uploaded'}
    write(upload/'backup-receipt.json',receipt)
    (upload/'UPLOAD_README.txt').write_bytes(b'Upload only these three files to your private Baidu Netdisk.\n'
        b'Do not share publicly or upload ../key. Keep the key separately in a reliable offline location.\n'
        b'Re-download the .7z from Baidu to ../download. A local copy is not a remote download.\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--research',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v2-delivery')
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.research,args.output),ensure_ascii=False))
