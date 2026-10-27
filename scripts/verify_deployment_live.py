"""Three-step deployment smoke on already-used day63; never a model experiment."""

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from thermal_comfort.simulation import integrate, transitions_frame, validate_kpis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8770")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def api(path, body=None, expected=200):
        req = Request(args.url+path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type":"application/json"})
        try:
            with urlopen(req, timeout=10) as response:
                if response.status != expected: raise ValueError("Unexpected HTTP response")
                return json.load(response)
        except HTTPError as error:
            if error.code != expected: raise
            return json.load(error)
    def wait():
        end = time.monotonic()+300
        while time.monotonic() < end:
            s = api('/api/state')
            if not s['busy']:
                if s['session']['status']=='error': raise ValueError(s['message'])
                return s['session']
            time.sleep(.2)
        raise TimeoutError('Deployment task timed out')
    def submit(person,kind,ttl=1):
        s=api('/api/state')['session']
        api('/api/demand',{'person':person,'kind':kind,'base_C':21.,'ttl_steps':ttl,'revision':s['demand_revision']},202)
    def advance(resolution='unresolved',expected=202):
        s=api('/api/state')['session']
        api('/api/advance',{'steps':1,'intent':{'mode':'needs','target_C':21.,'preferences':{'A':'hold','B':'hold'},'resolution':resolution,'demand_revision':s['demand_revision']}},expected)
        return wait() if expected==202 else s
    before=api('/api/state')
    if before['busy'] or before['can_end']: raise ValueError('Do not take over an active workbench session')
    api('/api/create',{'scene':'cool'},202);wait()
    submit('A','warmer');submit('B','cooler')
    advance(expected=409)
    assert len(api('/api/state')['session']['events'])==0
    first=advance('A')
    assert all(r['status']=='expired' for r in first['demands'].values())
    advance()
    submit('A','hold');advance()
    api('/api/end',{},202);s=wait()
    full=api('/api/download/'+s['id'])
    assert len(full['events'])==3 and full['status']=='completed'
    for e,i in zip(full['events'],full['intervals'],strict=True):
        raw,weather=pd.DataFrame(i['native']),pd.DataFrame(i['forecast'])
        transitions_frame([e],raw,weather,{'step_s':900,'area_m2':192})
        assert len(raw)==31 and np.allclose(np.diff(raw.time),30)
        assert all(np.isclose(v,e['metrics'][k],atol=1e-10,rtol=0) for k,v in integrate(raw,weather,192).items())
    sums={k:sum(e['metrics'][k] for e in full['events']) for k in full['events'][0]['metrics']}
    differences=validate_kpis(sums,full['api_kpis'])
    result={'role':'deployment_functional_only_existing_day63','source_scene':'cool','session_id':s['id'],'actions':3,'native_interval_points':93,
            'conflict_rejected_without_advance':True,'expiry_and_confirmed_override':True,'saved_with_kpi':True,'api_kpi_absolute_difference':differences,
            'new_model_experiments':0,'calibration_test_windows_used':False,'source_provenance':full['provenance']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_provenance'}),flush=True)


if __name__=='__main__':
    main()
