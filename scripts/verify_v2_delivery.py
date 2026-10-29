"""Public, standard-library verification of the archived V1/V2 delivery demo."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATES=["development084","development168","development189"]
METRICS=["energy_kWh",*[f"{kind}_{metric}" for kind in ["cold","hot"] for metric in ["Kh","duration_s","max_K"]]]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pair(reference,candidate,gates,active):
    saving=reference['energy_kWh']-candidate['energy_kWh']
    energy=(saving>=gates['saving_absolute_kWh_min'] and saving/reference['energy_kWh']>=gates['saving_relative_min']
        if active else -saving<=gates['low_load_energy_increase_kWh_tolerance'])
    thermal=all(candidate[f'{kind}_{metric}']-reference[f'{kind}_{metric}']<=gates[key]
        for kind in ['cold','hot'] for metric,key in [('Kh','cold_and_hot_degree_hour_increase_tolerance'),
            ('duration_s','cold_and_hot_duration_increase_s_tolerance'),('max_K','cold_and_hot_max_deviation_increase_K_tolerance')])
    return {'energy_gate_passed':bool(energy),'thermal_nonincrease':bool(thermal),
        'joint_gate_passed':bool(energy and thermal),'saving_kWh':saving,
        'saving_fraction':saving/reference['energy_kWh'] if active else None}


def verify(root=ROOT):
    manifest=read(root/'examples/v2/manifest.json')
    compressed=(root/'examples/v2/replay.json.gz').read_bytes()
    assert hashlib.sha256(compressed).hexdigest()==manifest['data_sha256']
    raw=gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest()==manifest['uncompressed_sha256']
    data=json.loads(raw)
    assert data['mode']=='ARCHIVED_DEVELOPMENT_REPLAY_ONLY' and data['heldout_initializations']==0
    assert data['steps']==672 and data['step_s']==900 and data['prefix_steps']==8
    assert hashlib.sha256((root/manifest['source_lock']).read_bytes()).hexdigest()==manifest['source_lock_sha256']
    lock=read(root/'reports/v2_4_safe/lock.json')
    for name in [*lock['parent_locks_sha256'],'reports/v2_4_safe/lock.json']:
        for path,expected in read(root/name)['files_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
    baselines=read(root/'reports/v2_2/summary.json')['runs']
    original=read(root/'reports/v2_safe_formal/archive-audit.json')['seeds']
    residual=read(root/'reports/v2_4_safe/archive-audit.json')['base_audit']['seeds']
    gates=read(root/'configs/v2_0.json')['comparison_gates']
    roster={(date,kind,None) for date in DATES for kind in ['guarded_rule','solar_mpc','minimum_heat']}
    roster|={(date,family,seed) for date in DATES for family in ['SAC','PPO','TD3','residual_SAC'] for seed in [11,23,37]}
    assert len(data['cases'])==manifest['cases']==45
    assert {(c['episode'],c['family'],c['seed']) for c in data['cases']}==roster
    prefixes={}
    active_pass=dict.fromkeys(['SAC','PPO','TD3','residual_SAC'],0)
    qualified={k:[] for k in active_pass}
    per_seed={}
    rows=[]
    for case in data['cases']:
        date,family,seed=case['episode'],case['family'],case['seed']
        frozen=(baselines[date+'__'+family]['metrics'] if seed is None else
            (residual if family=='residual_SAC' else original)[f"{'SAC' if family=='residual_SAC' else family}-{seed}"]['development'][date]['metrics'])
        assert case['metrics']=={k:frozen[k] for k in METRICS}
        frames=case['frames']
        assert len(frames)==673 and all(len(f)==12 for f in frames)
        assert all(v is None or math.isfinite(v) for f in frames for v in f)
        assert all(18<=f[1]<=23.25 for f in frames)
        assert all(abs(b[1]-a[1])<=.500001 for a,b in zip(frames,frames[1:]))
        assert all(b[3]>=a[3]-1e-6 for a,b in zip(frames,frames[1:]))
        for column,key in [(3,'energy_kWh'),(4,'cold_Kh'),(5,'hot_Kh')]:
            assert abs(frames[-1][column]-case['metrics'][key])<=.500001e-6
        if family=='guarded_rule':
            prefixes[date]=frames[:9]
        active=baselines[date+'__guarded_rule']['metrics']['energy_kWh']>=gates['active_reference_energy_kWh_min']
        pairs={kind:pair(baselines[date+'__'+kind]['metrics'],case['metrics'],gates,active)
            for kind in ['guarded_rule','solar_mpc']}
        if seed is not None:
            expected=(residual if family=='residual_SAC' else original)[f"{'SAC' if family=='residual_SAC' else family}-{seed}"]['development'][date]
            assert case['comparison']==expected
            for kind,p in pairs.items():
                assert all(p[k]==expected['comparisons'][kind][k] for k in p)
            jointly=all(v['joint_gate_passed'] for v in pairs.values()) and case['fallback_fraction']<=gates['rl_fallback_fraction_max_excluding_common_prefix']
            if active:
                active_pass[family]+=bool(jointly)
            per_seed.setdefault((family,seed),[]).append(bool(jointly))
        rows.append({'episode':date,'family':family,'seed':seed,'metrics':case['metrics'],
            'comparisons':pairs,'fallback_fraction':case['fallback_fraction']})
    assert all(c['frames'][:9]==prefixes[c['episode']] for c in data['cases'])
    for (family,seed),values in per_seed.items():
        if all(values):qualified[family].append(seed)
    assert active_pass=={'SAC':2,'PPO':0,'TD3':2,'residual_SAC':0} and not any(qualified.values())
    return {'status':'public_delivery_verified','cases':45,'frames':45*673,'full_precision_metric_rows':45,
        'active_joint_passed':active_pass,'active_combinations_per_RL_method':6,'all_dates_qualified_seeds':qualified,
        'rows':rows,'new_model_experiments':0,'FMU_calls':0,'heldout_initializations':0,
        'scope':'public bundle bytes, all frozen development metrics/gates and display structure; raw native audit requires private archive'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=verify()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
