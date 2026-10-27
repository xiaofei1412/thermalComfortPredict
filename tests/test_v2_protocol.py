import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from thermal_comfort.v2_protocol import CASE, ROOT, metadata_registry, validate_model_exposure, validate_split, verify_files


class V2ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT/'configs/v2_0.json').read_text(encoding='utf-8'))

    def test_registered_roles_include_forecast_tail_and_retain_legacy_guards(self):
        windows, guards = validate_split(self.plan)
        self.assertEqual(len(windows),13)
        self.assertEqual(guards,[[293,302],[322,331],[350,359]])
        e = next(e for e in windows if e['id']=='holdout342')
        self.assertEqual(e['envelope_days'],[335,349.5])
        self.assertEqual(sum(e['role']=='train' for e in windows),6)

    def test_day343_forecast_tail_reaches_protected_warmup(self):
        next(e for e in self.plan['episodes'] if e['id']=='holdout342')['start_day']=343
        with self.assertRaisesRegex(ValueError,'protected'):
            validate_split(self.plan)

    def test_old_day315_cannot_be_relaunched_through_protection(self):
        next(e for e in self.plan['episodes'] if e['id']=='historical238')['start_day']=315
        with self.assertRaisesRegex(ValueError,'protected'):
            validate_split(self.plan)

    def test_warmup_and_forecast_must_not_overlap_between_roles(self):
        next(e for e in self.plan['episodes'] if e['id']=='historical238')['start_day']=273
        with self.assertRaisesRegex(ValueError,'Cross-role'):
            validate_split(self.plan)

    def test_v2_holdout_must_not_be_v1_model_fitting_data(self):
        windows,_=validate_split(self.plan)
        lineage=[{'split':'train','warmup_start_s':252*86400,'end_s':266*86400}]
        with self.assertRaisesRegex(ValueError,'model fitting'):
            validate_model_exposure(windows,lineage)
        validate_model_exposure(windows,[{'split':'train','warmup_start_s':7*86400,'end_s':21*86400}])

    def test_this_phase_cannot_enable_training_or_change_actuator(self):
        bad=copy.deepcopy(self.plan);bad['learning']['authorized_in_this_phase']=True
        with self.assertRaisesRegex(ValueError,'authorize training'):validate_split(bad)
        bad=copy.deepcopy(self.plan);bad['action']['max_C']=26
        with self.assertRaisesRegex(ValueError,'actuator'):validate_split(bad)

    def test_metadata_projection_never_exports_metrics_or_other_domains(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent=root/'outputs/simulation/day014';parent.mkdir(parents=True)
            value={'start_time_s':14*86400,'end_time_s':21*86400,'config':{'testcase':CASE,'warmup_days':7,
                   'episode':{'split':'test'}},'controller':'rule','metrics':{'PRIVATE_SCORE':123456}}
            (parent/'summary.json').write_text(json.dumps(value),encoding='utf-8')
            other=parent.parent/'other_domain';other.mkdir()
            value['config']['testcase']='bestest_air'
            (other/'summary.json').write_text(json.dumps(value),encoding='utf-8')
            result=metadata_registry(root,self.plan)
            self.assertEqual(len(result['sources']),1)
            self.assertEqual(result['sources'][0]['config_role'],'test')
            self.assertNotIn('PRIVATE_SCORE',json.dumps(result))

    def test_protected_named_artifact_is_not_opened_even_if_invalid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent=root/'outputs/simulation/day329';parent.mkdir(parents=True)
            (parent/'summary.json').write_text('not JSON; protected labels must not be parsed',encoding='utf-8')
            result=metadata_registry(root,self.plan)
            self.assertEqual(result['sources'],[])
            self.assertEqual(result['protected_named_metadata_not_read'],['outputs/simulation/day329/summary.json'])

    def test_unknown_failed_attempt_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent=root/'outputs/simulation/unknown';parent.mkdir(parents=True)
            (parent/'api_metadata.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Unclassified'):
                metadata_registry(root,self.plan)

    def test_frozen_asset_tampering_is_rejected_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'model.joblib';p.write_bytes(b'original')
            expected={'model.joblib':hashlib.sha256(p.read_bytes()).hexdigest()}
            verify_files(root,expected)
            p.write_bytes(b'changed bytes')
            with self.assertRaisesRegex(ValueError,'Frozen asset changed'):verify_files(root,expected)


if __name__=='__main__':
    unittest.main()
