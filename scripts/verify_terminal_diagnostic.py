"""Read-only audit and independent recalculation of the bounded diagnostic round."""

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import terminal_diagnostic as d, native as n, terminal as t
from thermal_comfort.generalization import integrate
from thermal_comfort.simulation import digest


def frame(path):return pd.read_csv(path,float_precision='round_trip')


def main():
    d.check_baseline();meta=t.load(d.OUTPUT/'manifest.json');data_meta=t.load(d.DATA/'manifest.json')
    assert meta['code_sha256']=={name:digest(Path(name)) for name in d.CORE}
    assert meta['model_sha256']==digest(n.MODEL/'native_joint.joblib')
    assert data_meta['fmu_sha256']==digest(t.SOURCE/'testcases/bestest_air/models/wrapped.fmu')
    assert data_meta['collector_sha256']==digest(d.DATA/'collector_snapshot.py')==digest(Path(d.CORE[1]))
    assert data_meta['protocol_sha256']==digest(d.DATA/'protocol_snapshot.md')==digest(Path(d.CORE[-1]))
    summaries=t.load(d.DATA/'summary.json');checks=[]
    components=t.load(d.OUTPUT/'component_checks.json')
    for summary,c in zip(summaries,components):
        e,f,_=d.load_episode(summary);source=e['root'];native=frame(source/'native.csv.gz')
        assert e['role']=='development' and len(f)==len(native)==8641
        assert f.time.iloc[0]==summary['day']*86400 and np.all(np.diff(f.time)==30)
        for column in native:np.testing.assert_array_equal(f[column].to_numpy(),native[column].to_numpy())
        # All actual actions and unchanged non-heating overrides match the original development recording.
        np.testing.assert_array_equal(f.con_oveTSetHea_u.iloc[1:].to_numpy(),e['actions']+273.15)
        assert (f[['con_oveTSetCoo_activate','fcu_oveFan_activate','fcu_oveTSup_activate']]==0).all().all()
        forecast=frame(source/'exogenous.csv.gz');actual=integrate(native,forecast,t.IO,48)
        for key,value in actual.items():np.testing.assert_allclose(value,t.load(source/'summary.json')['metrics'][key],atol=1e-12,rtol=0)
        # Independently use exactly the external FMU streams specified by FanCoilUnit_T.mo.
        net=f.fcu_reaFloSup_y*(f['mod.fcu.supplyAir.h_outflow']-f['mod.zon.returnAir.h_outflow'])/1000
        observed=(.9*f.fcu_reaPHea_y-3*f.fcu_reaPCoo_y)/1000
        residual=float(np.max(abs(net-observed)));assert residual<1e-10
        assert c['source_filter_output_closure_max_fraction']<1e-12
        assert c['source_filter_dx1_closure_max']<1e-12 and c['source_filter_dx2_closure_max']<1e-12
        checks.append({'day':summary['day'],'kind':summary['kind'],'public_samples_exact':len(f),
            'actions_exact':len(e['actions'])//30,'enthalpy_balance_max_kW':residual,
            'source_summary_sha256':digest(source/'summary.json'),'native_sha256':digest(source/'native.csv.gz')})
    recompute=Path('outputs/terminal-diagnostic-recompute');d.main(recompute)
    for name in ['metrics.csv','paired_6h_errors.csv','component_checks.json']:
        assert digest(d.OUTPUT/name)==digest(recompute/name),(name,'independent recompute')
    metrics=frame(d.OUTPUT/'metrics.csv');old=frame(Path('reports/native_terminal/prediction_metrics.csv'))
    baseline=metrics[metrics.model=='baseline'];old=old[old.model=='native_joint']
    keys=['day','trajectory','target','seconds']
    joined=baseline.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(joined)==len(old)==len(baseline)==140
    for column in ['RMSE','bias','truth_RMS']:np.testing.assert_array_equal(joined[column+'_new'],joined[column+'_old'])
    pairs=frame(d.OUTPUT/'paired_6h_errors.csv')
    mapping={'temperature':'temperature_error_K','gas_energy':'gas_error_kWh',
        'cooling_electricity_energy':'cooling_error_kWh','fan_electricity_energy':'fan_error_kWh'}
    for (day,trajectory,variant),group in pairs.groupby(['day','trajectory','variant']):
        assert len(group)==63
        for target,column in mapping.items():
            reported=metrics[(metrics.day==day)&(metrics.trajectory==trajectory)&(metrics.model==variant)&(metrics.seconds==21600)&(metrics.target==target)].iloc[0]
            np.testing.assert_allclose(np.sqrt(np.mean(group[column]**2)),reported.RMSE,atol=1e-12,rtol=0)
    result={'created_UTC':datetime.now(timezone.utc).isoformat(),'integrity_passed':True,'diagnostic_only':True,
        'source_sha256':{name:digest(Path(name)) for name in [*d.CORE,'scripts/verify_terminal_diagnostic.py','tests/test_terminal_diagnostic.py']},
        'collection_manifest_sha256':digest(d.DATA/'manifest.json'),'analysis_manifest_sha256':digest(d.OUTPUT/'manifest.json'),
        'public_replay_audits':checks,'independent_recompute_byte_identical':True,'baseline_metric_rows_exact':140,
        'diagnostic_metric_rows':len(metrics),'paired_6h_rows':len(pairs),'refitted_parameters':False,
        'old_freezes_valid':True,'calibration_not_generated':not n.CALIBRATION.exists(),'new_scenario_run':False,
        'oracle_variants_not_deployment_candidates':d.VARIANTS[1:]}
    (d.OUTPUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
