import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from scripts.package_v2_evidence import package,paths
from scripts.restore_private_backup import restore_zip
from scripts.serve_v2_demo import server
from scripts.verify_v2_delivery import ROOT,pair,read,verify


class PublicV2DeliveryTests(unittest.TestCase):
    def test_all_methods_seeds_dates_and_original_negative_decisions(self):
        result=verify()
        self.assertEqual(result['cases'],45)
        self.assertEqual(result['active_joint_passed'],{'SAC':2,'PPO':0,'TD3':2,'residual_SAC':0})
        self.assertFalse(any(result['all_dates_qualified_seeds'].values()))
        self.assertEqual(result['FMU_calls'],0)

    def test_tampered_display_is_rejected_before_loading_other_assets(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'examples/v2').mkdir(parents=True)
            (root/'examples/v2/manifest.json').write_bytes((ROOT/'examples/v2/manifest.json').read_bytes())
            (root/'examples/v2/replay.json.gz').write_bytes(b'altered')
            with self.assertRaises(AssertionError):verify(root)

    def test_separate_cold_hot_and_zero_load_gates_preserved(self):
        gates=read(ROOT/'configs/v2_0.json')['comparison_gates']
        base={'energy_kWh':10,**{f'{kind}_{metric}':0 for kind in ['cold','hot'] for metric in ['Kh','duration_s','max_K']}}
        candidate={**base,'energy_kWh':9,'hot_Kh':.01}
        self.assertFalse(pair(base,candidate,gates,True)['joint_gate_passed'])
        zero={**base,'energy_kWh':0}
        self.assertIsNone(pair(zero,zero,gates,False)['saving_fraction'])

    def test_demo_server_has_no_control_or_filesystem_api(self):
        http=server(0)
        worker=Thread(target=http.serve_forever,daemon=True)
        worker.start()
        try:
            base=f'http://127.0.0.1:{http.server_port}'
            with urllib.request.urlopen(base+'/health') as response:
                self.assertEqual(json.load(response)['mode'],'archived_readonly')
            with urllib.request.urlopen(base+'/data.json') as response:
                self.assertEqual(len(json.load(response)['cases']),45)
            for path,method in [('/api/step','POST'),('/../PROJECT.md','GET'),('/outputs/','GET')]:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(urllib.request.Request(base+path,method=method))
                self.assertEqual(error.exception.code,405 if method=='POST' else 404)
                error.exception.close()
        finally:
            http.shutdown()
            http.server_close()
            worker.join()


class V2ArchiveTests(unittest.TestCase):
    def fixture(self,root,running=False):
        (root/'configs').mkdir()
        config={'closed_formal_folders':['v2-formal'],'private_output_folders':['v2-formal'],
            'excluded_directory_names':['env','venv','.venv','__pycache__']}
        (root/'configs/v2_delivery_archive.json').write_text(json.dumps(config),encoding='utf-8')
        folder=root/'outputs/v2-formal/live-registered'
        folder.mkdir(parents=True)
        (folder/'registration.json').write_text(json.dumps({'formal_training_authorized':running}),encoding='utf-8')
        (folder/'budget.json').write_text(json.dumps({'seeds':{'SAC-11':{'status':'running' if running else 'failed'}}}),encoding='utf-8')
        (folder/'records.gz').write_bytes(b'closed trajectory')
        (folder/'.env').write_bytes(b'excluded credentials')
        (folder/'env').mkdir()
        (folder/'env/unused.bin').write_bytes(b'excluded runtime')
        return folder

    def test_active_trial_rejected_without_archive(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            self.fixture(root,True)
            with self.assertRaises(ValueError):package(root/'archive.zip',root)
            self.assertFalse((root/'archive.zip').exists())

    def test_closed_archive_excludes_environment_restores_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            self.fixture(root)
            self.assertEqual(len(paths(root)),3)
            with patch('scripts.package_v2_evidence.subprocess.check_output',return_value='fixture'):
                result=package(root/'archive.zip',root)
                self.assertTrue(result['read_back_verified'])
                self.assertEqual(restore_zip(root/'archive.zip',result,root/'restored'),3)
                old=(root/'archive.zip').read_bytes()
                with self.assertRaises(FileExistsError):package(root/'archive.zip',root)
                self.assertEqual((root/'archive.zip').read_bytes(),old)

    def test_escaping_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            folder=self.fixture(root)
            outside=root/'outside.txt'
            outside.write_bytes(b'not an archive input')
            try:
                (folder/'alias').symlink_to(outside)
            except OSError:
                self.skipTest('Symlink privilege unavailable')
            with self.assertRaises(ValueError):paths(root)
