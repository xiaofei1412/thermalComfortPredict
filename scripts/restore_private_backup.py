"""Verify an owner-downloaded encrypted backup and restore into a new directory."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def member_path(root, name):
    parts = PurePosixPath(name).parts
    if not parts or name.startswith('/') or '\\' in name or ':' in name or '..' in parts:
        raise ValueError('Unsafe archive member')
    target = (root/name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Archive member escapes destination')
    return target


def check_files(folder, rows):
    names = [row['path'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest members')
    for row in rows:
        path = member_path(folder, row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Missing or changed backup member: '+row['path'])


def restore_zip(archive_path, manifest, destination):
    if sha256(archive_path) != manifest['archive_sha256']:
        raise ValueError('Inner archive checksum mismatch')
    rows = {row['path']: row for row in manifest['files']}
    if len(rows) != len(manifest['files']):
        raise ValueError('Duplicate inner manifest members')
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(rows) or set(archive.namelist()) != set(rows):
            raise ValueError('Inner archive/manifest member mismatch')
        for name, row in rows.items():
            path = member_path(destination, name)
            data = archive.read(name)
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Inner archive member checksum mismatch')
            if path.exists():
                if not path.is_file() or sha256(path) != row['sha256']:
                    raise ValueError('Refusing to overwrite existing different evidence')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(data)
    check_files(destination, manifest['files'])
    return len(rows)


def restore(archive_path, receipt, key_file, destination):
    if destination.exists():
        raise FileExistsError('Choose a new recovery directory')
    if (archive_path.stat().st_size != receipt['archive_bytes']
            or sha256(archive_path) != receipt['archive_sha256']):
        raise ValueError('Downloaded encrypted archive checksum mismatch')
    # Optional dependency, isolated from the lightweight workbench runtime.
    import py7zr
    key = key_file.read_text(encoding='utf-8').strip()
    if not key:
        raise ValueError('Empty recovery key')
    with py7zr.SevenZipFile(archive_path, password=key) as archive:
        names = archive.getnames()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate encrypted archive members')
        for name in names:
            member_path(destination, name)
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)
    contents = destination/'BACKUP_CONTENTS.json'
    if sha256(contents) != receipt['contents_sha256']:
        raise ValueError('Backup contents manifest checksum mismatch')
    rows = json.loads(contents.read_text(encoding='utf-8'))['files']
    if set(names) != {'BACKUP_CONTENTS.json'} | {row['path'] for row in rows}:
        raise ValueError('Unexpected backup contents')
    check_files(destination, rows)
    source = destination/'restored-source'
    count = restore_zip(destination/'source.zip', json.loads((destination/'source.manifest.json').read_text()), source)
    evidence = restore_zip(destination/'research-evidence.zip', json.loads((destination/'research-evidence.manifest.json').read_text()), source)
    return {'encrypted_sha256_verified': True, 'decrypted_files_verified': len(rows),
            'source_files_restored': count, 'research_files_restored': evidence,
            'restored_source': str(source), 'new_research_experiments': 0,
            'remote_provenance': 'caller must document actual remote download; local restore alone is not off-site verification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = restore(args.archive, json.loads(args.receipt.read_text(encoding='utf-8')), args.key_file, args.destination)
    (args.destination/'restore-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
