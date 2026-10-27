"""Read-only V2-0 metadata, baseline fingerprints and time-role validation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CASE = 'bestest_hydronic_heat_pump'
OMIT_ROOTS = ('office', 'native', 'terminal', 'baseline', 'boptest', 'storage',
              'delivery', 'deployment', 'release', 'v2-', 'interaction')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_path(value):
    return Path(value.replace('\\', '/'))


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def merged_windows(windows):
    merged = []
    for low, high in sorted(windows):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(high, merged[-1][1])
        else:
            merged.append([low,high])
    return merged


def validate_split(plan):
    env = plan['environment']
    if (env['case'] != CASE or env['step_s'] != 900 or env['warmup_days'] != 7
            or env['episode_days'] != 7 or env['forecast_steps'] != 48
            or env['common_rule_prefix_steps'] != 8 or env['history_steps'] != 16):
        raise ValueError('V2-0 fixed environment/time contract changed')
    action = plan['action']
    if (action['min_C'], action['max_C'], action['max_delta_C'], action['abort_temperature_C']) != (18.,23.25,.5,[18,35]):
        raise ValueError('V2-0 fixed actuator/abort contract changed')
    protected = [[p['start_day']-p['warmup_days'], p['start_day']+p['days']] for p in plan['legacy_protected']]
    windows = []
    for e in plan['episodes']:
        start = e['start_day']; end = start+env['episode_days']
        window = [start-env['warmup_days'], end+env['forecast_steps']*env['step_s']/86400]
        if e['role'] not in ('train', 'development', 'holdout', 'historical_regression'):
            raise ValueError('Unregistered role')
        if window[0] < 0 or window[1] > plan['data_rules']['year_days'] or any(overlap(window, p) for p in protected):
            raise ValueError('Episode including forecast/warmup intersects protected window or year boundary')
        windows.append({**e, 'evaluation_end_day':end, 'envelope_days':window})
    if len({e['id'] for e in windows}) != len(windows):
        raise ValueError('Duplicate episode IDs')
    if {e['role'] for e in windows} != {'train','development','holdout','historical_regression'}:
        raise ValueError('Missing comparison role')
    for i, a in enumerate(windows):
        for b in windows[i+1:]:
            if a['role'] != b['role'] and overlap(a['envelope_days'], b['envelope_days']):
                raise ValueError('Cross-role warmup/forecast overlap')
    if plan['learning']['authorized_in_this_phase'] or plan['learning']['fmustep_budget'] is not None:
        raise ValueError('V2-0 cannot authorize training')
    return windows, protected


def verify_files(root, expected):
    for name, value in expected.items():
        if digest(root/relative_path(name)) != value:
            raise ValueError('Frozen asset changed: '+name)


def baseline_snapshot(root, plan):
    freeze = read(root/plan['baseline_freeze'])
    verify_files(root, freeze['files_sha256'])
    model = digest(root/plan['baseline_model'])
    if model != plan['baseline_model_sha256'] or model != freeze['model_sha256']:
        raise ValueError('Frozen baseline model changed')
    config = {}
    for name in plan['baseline_config_chain']:
        config.update(read(root/name))
    if config != freeze['policy_config']:
        raise ValueError('Effective baseline configuration differs from V1 freeze')
    action = plan['action']
    if [config[k] for k in ('action_min_C','action_max_C','slew_K_per_step','abort_min_C','abort_max_C')] != [
            action['min_C'],action['max_C'],action['max_delta_C'],*action['abort_temperature_C']]:
        raise ValueError('Comparison actuator contract differs from V1')
    manifest = read(root/plan['baseline_manifest'])
    if model != manifest['artifacts_sha256']['model.joblib']:
        raise ValueError('Model manifest identity differs')
    verify_files(root, {str(relative_path(n)/'summary.json'):h for n,h in manifest['source_summaries'].items()})
    fmu = root/'outputs/boptest-v0.9.0/testcases'/CASE/'models/wrapped.fmu'
    if digest(fmu) != plan['environment']['fmu_sha256']:
        raise ValueError('Fixed FMU differs')
    for name in freeze['files_sha256']:
        data = subprocess.check_output(['git','show',plan['v1_delivery_commit']+':'+name.replace('\\','/')],cwd=root)
        if hashlib.sha256(data).hexdigest() != digest(root/relative_path(name)):
            raise ValueError('Policy differs from V1 delivery Git object')
    upstream = root/'outputs/boptest-v0.9.0'
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
    if revision != plan['environment']['source_revision']:
        raise ValueError('BOPTEST revision differs')
    if subprocess.check_output(['git','status','--porcelain'],cwd=upstream,text=True).strip():
        raise ValueError('BOPTEST checkout is not clean')
    lineage = [{k:e[k] for k in ('id','split','original_role','warmup_start_s','start_s','end_s','source') if k in e}
               for e in read(root/plan['baseline_split_manifest'])]
    return {'files_sha256':freeze['files_sha256'], 'model_sha256':model, 'fmu_sha256':digest(fmu),
            'source_revision':revision, 'effective_config':config, 'model_data_lineage':lineage,
            'manifest_sha256':digest(root/plan['baseline_manifest']), 'model_loaded_or_refitted':False}


def metadata_registry(root, plan):
    records, failures, blocked = [], [], []
    protected_days = '|'.join(str(p['start_day']) for p in plan['legacy_protected'])
    protected_name = re.compile(r'(?:day|test|calibration)('+protected_days+r')(?:\D|$)')
    def record(path, s):
        cfg = s.get('config', {}); domain = s.get('domain', {})
        if cfg.get('testcase', domain.get('testcase')) != CASE:
            return
        start = s.get('start_time_s', s.get('day', -999)*86400)/86400
        end = s.get('end_time_s', (start+s.get('days',cfg.get('days',0)))*86400)/86400
        if start < 0 or end <= start:
            raise ValueError('Unresolved historical interval: '+str(path))
        records.append({'source':path.relative_to(root).as_posix(),'sha256':digest(path),
            'start_day':start,'end_day':end,'warmup_start_day':start-cfg.get('warmup_days',7),
            'warmup_is_conservative':'warmup_days' not in cfg,
            'forecast_tail_end_day_conservative':end+plan['data_rules']['historical_forecast_tail_bound_days'],
            'config_role':cfg.get('episode',{}).get('split'), 'controller':s.get('controller'),
            'domain':domain.get('id', CASE)})
    for parent in sorted((root/'outputs').iterdir()):
        if not parent.is_dir() or parent.name.startswith(OMIT_ROOTS):
            continue
        for path in sorted(parent.glob('*/summary.json')):
            if protected_name.search(path.parent.name):
                blocked.append(path.relative_to(root).as_posix()); continue
            value = read(path)
            if isinstance(value, dict):
                record(path, value)
        for path in sorted(parent.glob('*/api_metadata.json')):
            if not (path.parent/'summary.json').exists():
                failures.append(path.parent.relative_to(root).as_posix())
    if any(p != 'outputs/mpc-smoke/day98__guarded_rule' for p in failures):
        raise ValueError('Unclassified failed initialization metadata: '+str(failures))
    if failures:
        records.append({'source':failures[0], 'sha256':None, 'start_day':98,'end_day':105,
                        'warmup_start_day':91,'warmup_is_conservative':True,
                        'forecast_tail_end_day_conservative':105.5,'config_role':'unknown_failed_attempt',
                        'controller':'guarded_rule','domain':CASE})
    # Restored archive copies are excluded; the one new F02 session is explicit.
    sessions = list((root/'outputs/workbench/sessions').glob('*.json'))
    extra = root/'outputs/deployment-f2/source/outputs/workbench/sessions/64ac6cbb6570.json'
    if extra.exists(): sessions.append(extra)
    for path in sorted(sessions):
        s = read(path); p = s['provenance']
        if p['case'] != CASE: continue
        start = p['start_s']/86400
        end = s['frames'][-1]['time_s']/86400 if s['frames'] else start
        end = max(end, start+900/86400)
        if s['status'] != 'completed': end = max(end,start+1)
        records.append({'source':path.relative_to(root).as_posix(),'sha256':digest(path),
            'start_day':start,'end_day':end,'warmup_start_day':start-p['warmup_s']/86400,
            'warmup_is_conservative':s['status']!='completed','forecast_tail_end_day_conservative':end+.5,
            'config_role':'engineering_only','controller':'workbench','domain':CASE})
    return {'sources':records,'failed_attempts':failures,'protected_named_metadata_not_read':blocked,
            'excluded_copy_roots':['delivery-f','deployment-f2 except declared F02 session','release-final'],
            'excluded_other_domain_roots':['office*','native*','terminal*','non-original domain summaries'],
            'read_fields_only':'time/config-role/domain/controller/provenance; no metrics, CSV, forecast or transition labels exported'}


def validate_model_exposure(windows, lineage):
    used = [[e['warmup_start_s']/86400,e['end_s']/86400+.5] for e in lineage if e['split'] in ('train','validation')]
    for e in windows:
        if e['role'] in ('holdout','historical_regression') and any(overlap(e['envelope_days'],w) for w in used):
            raise ValueError('Evaluation envelope intersects baseline model fitting/development exposure')


def audit(root, plan):
    windows, protected = validate_split(plan)
    legacy = read(root/'configs/residential_c.json')
    expected = [{'id':e['id'],'start_day':e['start_day'],'days':legacy['episode_days'],'warmup_days':legacy['warmup_days']}
                for e in legacy['reserved_episodes'] if e['role'] in ('calibration','test')]
    if plan['legacy_protected'] != expected:
        raise ValueError('V1 protected windows were removed or changed')
    old_c2 = read(root/'outputs/residential-c2/freeze.json')
    verify_files(root, old_c2['files_sha256'])
    if digest(root/'reports/residential_c/protocol_lock.json') != old_c2['c1_lock_sha256']:
        raise ValueError('V1 C1 protocol lock changed')
    baseline = baseline_snapshot(root, plan)
    registry = metadata_registry(root, plan)
    lineage_by_source = {relative_path(e['source']).as_posix():e for e in baseline['model_data_lineage'] if 'source' in e}
    for s in registry['sources']:
        parent = s['source'].split('/')[1]
        original = s['config_role']
        if parent in ('workbench','deployment-f2') or '-repeat' in parent or '-smoke' in parent:
            use = 'engineering_only'
        elif parent == 'residential-c2':
            use = 'C2_train' if '/train224/' in s['source'] else 'C2_development_or_response'
        elif 'response' in parent:
            use = 'development_intervention'
        elif 'validation' in parent:
            use = 'development_control'
        elif parent.endswith('-control') or parent == 'mpc':
            use = 'historical_test_control'
        elif parent == 'simulation':
            use = 'historical_control_comparison'
        else:
            use = original or 'historical_role_unspecified'
        s['historical_use'] = use
        model_use = lineage_by_source.get(str(Path(s['source']).parent).replace('\\','/'))
        s['selected_V1_model_use'] = model_use['split'] if model_use else None
    if registry['protected_named_metadata_not_read']:
        raise ValueError('Protected named artifacts found; investigate without reading labels')
    raw_overlap, forecast_overlap = [], []
    for s in registry['sources']:
        for p, guard in zip(plan['legacy_protected'], protected):
            if overlap([s['warmup_start_day'],s['end_day']],guard): raw_overlap.append([s['source'],p['id']])
            if overlap([s['end_day'],s['forecast_tail_end_day_conservative']],guard): forecast_overlap.append([s['source'],p['id']])
    if raw_overlap:
        raise ValueError('Historical observation/warmup conflicts with legacy protection: '+str(raw_overlap))
    initializations = [e['data'] for e in read(root/'outputs/residential-c2/operations.json') if e['path'].startswith('initialize/')]
    if len(initializations) != 4 or any(e['start_time']/86400 not in (224,287) for e in initializations):
        raise ValueError('C2 initialization ledger differs from the closed authorized round')
    closure = read(root/'outputs/residential-c2/closure.json')
    verify_files(root/'outputs/residential-c2', closure['data_sha256'])
    if closure['status'] != 'closed_negative' or any(closure[k] for k in ('refits','calibration_rounds','blind_rounds')):
        raise ValueError('V1 C2 closure has changed')
    validate_model_exposure(windows, baseline['model_data_lineage'])
    exposure = []
    for e in windows:
        hits = [s['source'] for s in registry['sources'] if overlap(e['envelope_days'],[s['warmup_start_day'],s['forecast_tail_end_day_conservative']])]
        exposure.append({**e,'historical_sources_overlapping':hits,'globally_unseen_weather':not hits})
    occupied = merged_windows([[s['warmup_start_day'],s['forecast_tail_end_day_conservative']] for s in registry['sources']]+protected)
    gaps, previous = [], 0
    for low, high in occupied:
        if low > previous: gaps.append([previous,low])
        previous = max(previous,high)
    if previous < plan['data_rules']['year_days']: gaps.append([previous,plan['data_rules']['year_days']])
    required = plan['environment']['warmup_days']+plan['environment']['episode_days']+.5
    return {'status':'protocol_ready_for_V2_1_not_training_qualified', 'created_UTC':datetime.now(timezone.utc).isoformat(),
        'source_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'baseline':baseline,'history':registry,'episodes':exposure,'legacy_protected_envelopes':protected,
        'historical_observation_protection_conflicts':raw_overlap,
        'historical_forecast_tail_possible_overlaps':forecast_overlap,
        'C2_initialization_start_days':[e['start_time']/86400 for e in initializations],
        'C2_closure_status':closure['status'],
        'history_and_protection_union_days':occupied,'unregistered_gaps_days':gaps,
        'globally_unseen_full_episode_candidates_in_registered_history':sum(b-a>=required for a,b in gaps),
        'full_episode_envelope_days_required':required,
        'forecast_overlap_interpretation':'conservative exogenous prefetch bound, not protected control/temperature labels',
        'training_authorized':False,'new_fmu_initializations':0,'new_model_fits':0,'protected_labels_read':False,
        'empirical_load_and_action_response_coverage':'pending V2-2; not inferred from this metadata audit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read(ROOT/'configs/v2_0.json')
    result = audit(ROOT, plan)
    files = ['configs/v2_0.json','thermal_comfort/v2_protocol.py','docs/experiments/v2_0.md','tests/test_v2_protocol.py']
    lock = {'protocol':plan['protocol'],'status':result['status'],
            'files_sha256':{n:digest(ROOT/n) for n in files},'v1_baseline':result['baseline'],
            'old_c2_freeze_sha256':digest(ROOT/'outputs/residential-c2/freeze.json'),
            'old_c2_closure_sha256':digest(ROOT/'outputs/residential-c2/closure.json')}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    lock['audit_sha256'] = digest(args.output/'audit.json')
    (args.output/'lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':result['status'],'historical_sources':len(result['history']['sources']),
        'episodes':len(result['episodes']),'protected_observation_conflicts':len(result['historical_observation_protection_conflicts']),
        'forecast_possible_overlaps':len(result['historical_forecast_tail_possible_overlaps']),
        'new_fmu_initializations':0,'new_model_fits':0}))


if __name__ == '__main__':
    main()
