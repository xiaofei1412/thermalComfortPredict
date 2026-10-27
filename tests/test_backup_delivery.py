import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.restore_private_backup import check_files, member_path, restore, restore_zip


class BackupDeliveryTests(unittest.TestCase):
    def test_valid_unicode_evidence_restores_and_identical_history_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); archive = root/'evidence.zip'; data = b'preserved original evidence'
            name = 'records/原始记录.json'
            with zipfile.ZipFile(archive, 'w') as package:
                package.writestr(name, data)
            manifest = {'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                        'files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}
            destination = root/'restore'
            self.assertEqual(restore_zip(archive, manifest, destination), 1)
            timestamp = (destination/name).stat().st_mtime_ns
            self.assertEqual(restore_zip(archive, manifest, destination), 1)
            self.assertEqual((destination/name).read_bytes(), data)
            self.assertEqual((destination/name).stat().st_mtime_ns, timestamp)

    def test_untrusted_archive_names_cannot_escape_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('../key.txt', '/outside', 'C:/outside', 'a\\..\\outside'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    member_path(Path(folder), name)

    def test_modified_download_is_rejected_before_decryption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); archive = root/'download.7z'; archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                restore(archive, {'archive_bytes':7, 'archive_sha256':'0'*64}, root/'unused-key', root/'new')
            self.assertFalse((root/'new').exists())

    def test_existing_recovery_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                restore(Path('not-read'), {}, Path('not-read'), Path(folder))

    def test_manifest_duplicates_and_different_history_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root/'restore'; target.mkdir()
            original = b'recorded evidence'
            row = {'path':'record.json','bytes':len(original),'sha256':hashlib.sha256(original).hexdigest()}
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                check_files(target, [row,row])
            archive = root/'evidence.zip'
            with zipfile.ZipFile(archive,'w') as package:
                package.writestr(row['path'],original)
            manifest = {'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':[row]}
            (target/'record.json').write_bytes(b'other user history')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                restore_zip(archive,manifest,target)
            self.assertEqual((target/'record.json').read_bytes(),b'other user history')


if __name__ == '__main__':
    unittest.main()
