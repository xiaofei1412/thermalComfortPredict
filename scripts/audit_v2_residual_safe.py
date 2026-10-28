"""Closed-trial audit: native scores, residual action chain and same-trajectory rule replay."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.audit_v2_formal import lines
from scripts.audit_v2_safe_formal import audit
from thermal_comfort.rl_budget_safe import write_snapshot
from thermal_comfort.rl_interface import project_action
from thermal_comfort.rl_residual import mapped_action
from thermal_comfort.simulation import digest
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_protocol import ROOT, read


def check_control(record):
    c = record["control"]
    previous = record["before"]["oveTSet_u"]-273.15
    raw, anchor = c["residual_normalized"],c["rule_anchor_C"]
    assert np.isfinite([raw,anchor]).all() and abs(raw) <= 1
    assert 18-1e-7 <= anchor <= 23.25+1e-7 and abs(anchor-previous) <= .5+1e-7
    mapped, target = mapped_action([raw],previous,anchor,.5)
    action, projection = project_action(mapped,previous)
    expected = anchor if c["fallback"] else action["oveTSet_u"]-273.15
    actual = record["action"]["oveTSet_u"]-273.15
    for left,right in [(c["raw_actor_normalized"],raw),(c["residual_unprojected_target_C"],target),
            (c["mapped_increment_normalized"],mapped[0]),(c["actor_projection"]["raw_normalized"],mapped[0]),
            (c["actor_projection"]["applied_C"],projection["applied_C"]),
            (c["actor_target_C"],projection["applied_C"]),(actual,expected),
            (c["actual_command_C"],actual),(c["measured_setpoint_C"],actual),
            (record["after"]["oveTSet_u"]-273.15,actual),(c["anchor_deviation_after_limits_K"],actual-anchor)]:
        assert abs(left-right) < 1e-7,(left,right)
    assert c["mode"] == ("fallback" if c["fallback"] else "rule_residual_sac")
    return {"residual":raw,"effective_offset_K":actual-anchor,"at_bound":actual <= 18+1e-7 or actual >= 23.25-1e-7}


def check_journal(action,record,episode,training):
    assert action["episode"] == episode
    assert record["control"]["common_prefix"] is False and record["control"]["actor_transition"] is True
    # These two structural flags are stored by advance(), not returned in info.control.
    exported = {k:v for k,v in record["control"].items() if k not in ["common_prefix","actor_transition"]}
    assert action["control"] == exported
    assert abs(action["library_action"][0]-record["control"]["residual_normalized"]) < 1e-7
    if training:
        assert abs(action["buffer_action"][0]-record["control"]["residual_normalized"]) < 1e-7


def replay_rule(folder,model,config):
    records, forecasts = lines(folder/"steps.jsonl.gz"),lines(folder/"forecasts.jsonl.gz")
    native = pd.read_csv(folder/"native.csv.gz",float_precision="round_trip")
    controller = SolarController(model,config,"guarded_rule","perfect")
    controller.reset(records[0]["before"],{"oveTSet_u":{"Minimum":18+273.15,"Maximum":23.25+273.15}})
    maximum = 0.
    for index,(record,forecast) in enumerate(zip(records,forecasts,strict=True)):
        view = pd.DataFrame(forecast)
        action,info = controller.decide(index,record["before"],view)
        expected = action["oveTSet_u"]-273.15
        observed = record["control"]["rule_anchor_C"] if index >= 8 else record["action"]["oveTSet_u"]-273.15
        maximum = max(maximum,abs(expected-observed))
        if index >= 8:
            assert info["reason"] == record["control"]["rule_anchor_reason"]
        if not record["aborted"]:
            controller.update(record["before"],record["after"],native.iloc[index*30:index*30+31],view.iloc[0])
    assert maximum < 1e-7
    return {"steps":len(records),"max_anchor_difference_K":maximum,
        "scope":"frozen rule replayed on the residual policy's actual state, native intervals and archived forecasts"}


def audit_residual(directory):
    base = audit(directory)
    registered,auth = read(directory/"registration-authorized.json"),read(directory/"authorization.json")
    assert digest(directory/"entry-gate.json") == auth["entry_gate_sha256"]
    assert digest(directory/"diagnostic.json") == auth["diagnostic_sha256"]
    assert registered["entry_protocol"] == "residential_rl_v2_4_rule_residual_safe_1"
    assert base["ledger_counts"] == {"select":225,"initialize":225,"advance":151200}
    assert base["final_candidates"] == 3
    plan = registered["protocol"]
    baseline = read(ROOT/"reports/v2_0/lock.json")["v1_baseline"]
    model = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
    checks = {}
    for number in plan["seeds"]:
        key, root = f"SAC-{number}",directory/f"SAC-{number}"
        details = base["seeds"][key]
        assert details["status"] == "completed" and len(details["exports"]) == 75
        assert details["budgeted_advances_minus_archived_segments"] == 0
        manifest = read(root/"checkpoints/cycle-12/manifest.json")
        for name,expected in manifest["episodes_sha256"].items():
            assert digest(root/name) == expected
        callback = read(root/"checkpoints/cycle-12/callback.json")
        assert callback["rows"] == 47808
        import hashlib
        for name,size in [("actions.jsonl",callback["actions_bytes"]),("curve.jsonl",callback["curve_bytes"])]:
            with (root/name).open("rb") as stream:
                prefix = stream.read(size)
            assert len(prefix) == size and hashlib.sha256(prefix).hexdigest() == manifest["journal_prefix_sha256"][name]
        train,development = lines(root/"actions.jsonl"),lines(root/"development-actions.jsonl")
        assert len(train) == 47808 and len(development) == 1992
        cursor, dev_cursor, offsets,raws,bounds = 0,0,[],[],0
        folders = [root/"train"/f"cycle-{cycle:02}"/name for cycle in range(1,13) for name in plan["train_ids"]]
        folders += [root/"development"/name for name in plan["development_ids"]]
        replays, grouped = {},{}
        for folder in folders:
            records = lines(folder/"steps.jsonl.gz")
            development_role = folder.parent.name == "development"
            journal = development if development_role else train
            selected = []
            for record in records[8:]:
                checked = check_control(record)
                action = journal[dev_cursor if development_role else cursor]
                check_journal(action,record,folder.name,not development_role)
                if not development_role:
                    cursor += 1
                else:
                    dev_cursor += 1
                offsets.append(checked["effective_offset_K"])
                raws.append(checked["residual"])
                bounds += checked["at_bound"]
                selected.append(checked["effective_offset_K"])
            if development_role:
                grouped[folder.name] = {"mean_effective_offset_K":float(np.mean(selected)),
                    "mean_abs_effective_offset_K":float(np.mean(np.abs(selected))),
                    "nonzero_effective_offset_fraction":float(np.mean(np.abs(selected)>1e-7))}
                replays[folder.name] = replay_rule(folder,model,baseline["effective_config"])
        assert cursor == 47808 and dev_cursor == 1992
        checks[key] = {"actor_actions_checked":len(offsets),"training_actions":cursor,"development_actions":dev_cursor,
            "mean_abs_residual":float(np.mean(np.abs(raws))),"mean_abs_effective_offset_K":float(np.mean(np.abs(offsets))),
            "nonzero_effective_offset_fraction":float(np.mean(np.abs(offsets)>1e-7)),
            "at_physical_bound_fraction":bounds/len(offsets),"development_effective_actions":grouped,
            "development_rule_replay":replays,"complete_export_and_final_journal_binding_passed":True}
    return {"status":"closed_residual_action_and_native_audit_passed","base_audit":base,"action_checks":checks,
        "training_action_mapping_checked":143424,"development_action_mapping_checked":5976,
        "development_rule_replay_steps":6048,"additional_FMU_calls":0,"heldout_initializations":0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    write_snapshot(args.output,audit_residual(args.directory))
