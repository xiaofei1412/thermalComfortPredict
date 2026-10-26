"""Run in the pinned PyFMI worker image; replay existing development actions only."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT=Path('/workspace')
SOURCE=ROOT/'outputs/boptest-v0.9.0'
sys.path.insert(0,str(SOURCE))
from testcase import TestCase

INTERNAL=[
    'mod.con.heaPI.I.y','mod.con.cooPI.I.y','mod.con.heaPI.y','mod.con.cooPI.y',
    'mod.con.hysCoo.y','mod.con.yFan','mod.con.TSup','mod.con.oveTSetCoo.y',
    'mod.fcu.fan.filter.x[1]','mod.fcu.fan.filter.x[2]','der(mod.fcu.fan.filter.x[1])','der(mod.fcu.fan.filter.x[2])',
    'mod.fcu.fan.filter.u','mod.fcu.fan.filter.y','mod.fcu.fan.P','mod.fcu.fan.dpMachine',
    'mod.fcu.fan.eta','mod.fcu.fan.eff.rho','mod.fcu.coi.h_in_a','mod.fcu.coi.h_internal_b',
    'mod.fcu.coi.Xi_in_a[1]','mod.fcu.coi.Xi_internal_b[1]',
    'mod.zon.returnAir.h_outflow','mod.fcu.supplyAir.h_outflow']
PARAMETERS=['mod.con.heaPI.k','mod.con.heaPI.Ti','mod.con.heaPI.Ni','mod.con.heaPI.I.k',
    'mod.con.cooPI.k','mod.con.cooPI.Ti','mod.con.cooPI.Ni','mod.con.cooPI.I.k',
    'mod.fcu.fan.riseTime','mod.fcu.fan.filter.r[1]','mod.fcu.fan.filter.r[2]',
    'mod.fcu.fan.filter.cr[1]','mod.fcu.fan.filter.cr[2]','mod.fcu.fan.filter.f_cut']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def collect():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='outputs/terminal-diagnostic-replay')
    args=parser.parse_args();output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    fmu=SOURCE/'testcases/bestest_air/models/wrapped.fmu'
    shutil.copyfile(ROOT/'scripts/collect_terminal_diagnostic.py',output/'collector_snapshot.py')
    shutil.copyfile(ROOT/'docs/experiments/terminal_diagnostic.md',output/'protocol_snapshot.md')
    plans=[{'day':d,'kind':kind,'source':str(ROOT/(f'outputs/terminal-development/day{d}__excitation' if kind=='excitation' else f'outputs/native-rule-development/day{d}__guarded_rule'))}
        for d in [62,150] for kind in ['excitation','guarded_rule']]
    save(output/'manifest.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),'role':'development_diagnostic',
        'fmu_sha256':sha(fmu),'collector_sha256':sha(ROOT/'scripts/collect_terminal_diagnostic.py'),
        'protocol_sha256':sha(ROOT/'docs/experiments/terminal_diagnostic.md'),'plans':plans,'internal_variables':INTERNAL,
        'network_required':False,'changed_fmu':False,'changed_solver':False})
    summaries=[]
    for plan in plans:
        directory=output/f"day{plan['day']}__{plan['kind']}";directory.mkdir()
        shutil.copyfile(SOURCE/'version.txt',directory/'version.txt')
        os.chdir(directory)
        case=TestCase(fmupath=str(fmu),forecast_uncertainty_params_path=str(SOURCE/'forecast/forecast_uncertainty_params.json'))
        status,_,_=case.initialize((plan['day']-1)*86400,6*86400);assert status==200
        case.set_step(900)
        source=Path(plan['source']);prelude=load(source/'prelude.json')
        for r in prelude:
            status,_,_=case.advance(r['action']);assert status==200
        params={name:float(case.fmu.get(name)[0]) for name in PARAMETERS};save(directory/'parameters.json',params)
        # PyFMI uses shell-style patterns: literal Modelica array brackets need escaping.
        case.options['filter']=[name.replace('[','[[]') for name in case.output_names+case.input_names+INTERNAL]
        original=case._TestCase__simulation
        names=[*case.output_names,*case.input_names,*INTERNAL]
        # Preserve the left endpoint before applying the first new action, as BOPTEST y_store does.
        initial={'time':float(case.start_time),**{name:case.fmu.get(name)[0] for name in names}}
        parts=[pd.DataFrame([initial])]
        def capture(start,end,input_object=None):
            result=original(start,end,input_object)
            if isinstance(result,str):raise RuntimeError(result)
            part=pd.DataFrame({name:np.asarray(result[name]) for name in ['time',*case.output_names,*case.input_names,*INTERNAL]})
            parts.append(part.iloc[1:])
            return result
        case._TestCase__simulation=capture
        records=[json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()]
        for r in records:
            status,message,_=case.advance(r['actual_action'])
            if status!=200:raise RuntimeError(message)
        frame=pd.concat(parts,ignore_index=True)
        if len(frame)!=8641 or not (np.diff(frame.time)==30).all():raise ValueError('Unexpected diagnostic grid')
        frame.to_csv(directory/'internal.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        native=pd.read_csv(source/'native.csv.gz',float_precision='round_trip');common=[c for c in native if c in frame]
        differences={name:float(np.max(abs(native[name].to_numpy()-frame[name].to_numpy()))) for name in common}
        # Internal filter extension must not alter the public trajectory.
        exact=all(value==0 for value in differences.values())
        summary={**plan,'complete':True,'native_samples':len(frame),'decisions':len(records),'public_bit_identical':exact,
            'public_max_absolute_difference':differences,'parameters_sha256':sha(directory/'parameters.json'),
            'internal_sha256':sha(directory/'internal.csv.gz'),'source_summary_sha256':sha(source/'summary.json')}
        save(directory/'summary.json',summary);summaries.append(summary);save(output/'summary.json',summaries)
        print(json.dumps({'day':plan['day'],'kind':plan['kind'],'public_bit_identical':exact,'largest_difference':max(differences.values())}),flush=True)
        del case


if __name__=='__main__':collect()
