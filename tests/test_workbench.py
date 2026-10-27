import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import numpy as np
import pandas as pd

from thermal_comfort.simulation import FORECAST, POWER, integrate
from thermal_comfort.workbench import (CASE, Conflict, Handler, LocalServer, Workbench, command,
                                       intent, ThreadingHTTPServer)


REQUEST = {"mode": "manual", "target_C": 23.25, "preferences": {"A": "warmer", "B": "cooler"}}


class FakeClient:
    """Deterministic 30-second physical-interface fixture, never used by the app."""
    def __init__(self):
        self.calls = []
        self.block = None
        self.on_advance = None
        self.fail_advance = False
        self.fail_stop = False

    def call(self, method, path, data=None):
        self.calls.append((method, path, copy.deepcopy(data)))
        endpoint = path.split("/")[0]
        if path == "version":
            return {"version": "0.9.0"}
        if path == f"testcases/{CASE}/select":
            return {"testid": "fixture"}
        if endpoint == "inputs":
            return {"oveTSet_u": {"Unit": "K", "Minimum": 278.15, "Maximum": 308.15}}
        if endpoint == "measurements":
            return {"reaTZon_y": {"Unit": "K"}, **{p: {"Unit": "W"} for p in POWER}}
        if endpoint == "scenario":
            return {}
        if endpoint == "initialize":
            self.raw = {"time": data["start_time"], "reaTZon_y": 294.15, "oveTSet_u": 294.15,
                        "oveTSet_activate": 0, "reaTSetHea_y": 293.15, "reaTSetCoo_y": 299.15,
                        "reaQFloHea_y": 4000., "reaTRet_y": 305., "reaTSup_y": 310.,
                        "reaPHeaPum_y": 1000., "reaPFan_y": 300., "reaPPumEmi_y": 100.}
            self.native = pd.DataFrame([self.raw])
            return self.raw.copy()
        if endpoint == "step":
            return 900
        if endpoint == "forecast":
            t = np.arange(self.raw["time"], self.raw["time"] + 901, 30).tolist()
            return self.weather(t)
        if endpoint == "advance":
            if self.block:
                self.block.wait(timeout=4)
            if self.fail_advance:
                raise TimeoutError("ambiguous advance")
            before = self.raw.copy()
            self.raw = {**before, **data, "time": before["time"] + 900,
                        "reaTZon_y": before["reaTZon_y"] + .05 * (data["oveTSet_u"] - before["reaTZon_y"])}
            rows = []
            for i in range(31):
                row = {**self.raw, "time": before["time"] + 30 * i,
                       "reaTZon_y": before["reaTZon_y"] + i / 30 * (self.raw["reaTZon_y"] - before["reaTZon_y"])}
                if not i:
                    row.update(oveTSet_u=before["oveTSet_u"], oveTSet_activate=before["oveTSet_activate"])
                rows.append(row)
            self.interval = pd.DataFrame(rows)
            self.native = pd.concat([self.native, self.interval.iloc[1:]], ignore_index=True)
            if self.on_advance:
                self.on_advance()
            return self.raw.copy()
        if endpoint == "results":
            return self.interval.to_dict("list")
        if endpoint == "kpi":
            metrics = integrate(self.native, pd.DataFrame(self.weather(self.native.time.tolist())), 192)
            return {"ener_tot": metrics["boptest_energy_kWh_m2"], "tdis_tot": metrics["discomfort_Kh"],
                    "cost_tot": metrics["cost_EUR_m2"], "emis_tot": metrics["emissions_kg_m2"]}
        if endpoint == "stop":
            if self.fail_stop:
                raise TimeoutError("stop unavailable")
            return None
        raise AssertionError(path)

    @staticmethod
    def weather(t):
        values = [293.15, 299.15, 1., 279.15, 100., .2, .1]
        return {"time": t, **{key: [value] * len(t) for key, value in zip(FORECAST, values)}}


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = FakeClient()
        self.app = Workbench(self.tmp.name, self.client)

    def wait(self):
        self.app.thread.join(timeout=5)
        self.assertFalse(self.app.busy)
        self.assertNotEqual((self.app.session or {}).get("status"), "error", self.app.message)

    def start(self):
        self.app.create("cool")
        self.wait()

    def submit(self, person="A", kind="warmer", ttl=4, **extra):
        self.app.submit_demand({"person": person, "kind": kind, "base_C": 21., "ttl_steps": ttl,
                                "revision": self.app.session["demand_revision"], **extra})

    def needs(self, resolution="unresolved"):
        return {**REQUEST, "mode": "needs", "target_C": 21., "resolution": resolution,
                "demand_revision": self.app.session["demand_revision"]}

    def test_demand_submit_revoke_and_stale_revision_never_advance_time(self):
        self.start()
        before = self.app.session["frames"][-1]["time_s"]
        self.submit()
        self.assertEqual(self.app.demand_preview({"base_C": 21., "resolution": "unresolved"})["target_C"], 21.5)
        with self.assertRaises(Conflict):
            self.app.revoke_demand({"person": "A", "revision": 0})
        self.app.revoke_demand({"person": "A", "revision": 1})
        self.assertEqual(self.app.session["frames"][-1]["time_s"], before)
        self.assertEqual(self.app.session["demands"]["A"]["status"], "withdrawn")
        self.assertEqual(self.app.session["demand_revision"], 2)

    def test_conflict_blocks_without_closing_or_advancing_and_choice_is_audited(self):
        self.start(); self.submit(); self.submit("B", "cooler")
        with self.assertRaises(Conflict):
            self.app.advance({"steps": 1, "intent": self.needs()})
        self.assertEqual(self.app.session["events"], [])
        self.assertEqual(self.app.session["status"], "paused")
        self.app.advance({"steps": 1, "intent": self.needs("A")}); self.wait()
        event = self.app.session["events"][0]
        self.assertEqual(event["decision"]["target_C"], 21.5)
        self.assertEqual(event["demand_evaluation"]["unmet_people"], ["B"])
        self.assertEqual(self.app.session["demand_summary"]["B"]["unmet_steps"], 1)

    def test_batch_expiry_is_not_refreshed_and_returns_to_base(self):
        self.start(); self.submit(ttl=1)
        self.app.advance({"steps": 3, "intent": self.needs()}); self.wait()
        self.assertEqual([e["decision"]["target_C"] for e in self.app.session["events"]], [21.5, 21., 21.])
        self.assertEqual(self.app.session["demands"]["A"]["status"], "expired")
        self.assertEqual([e["action"] for e in self.app.session["demand_events"]], ["submit", "expire"])
        self.assertEqual(self.app.session["demand_summary"]["A"]["active_steps"], 1)

    def test_old_confirmation_invalidated_when_selected_person_expires(self):
        self.start(); self.submit(kind="target", target_C=26.)
        self.submit("B", "target", ttl=1, target_C=21.)
        self.app.advance({"steps": 3, "intent": self.needs("B")}); self.wait()
        self.assertEqual(len(self.app.session["events"]), 1)
        self.assertEqual(self.app.session["status"], "paused")
        self.assertIsNotNone(self.app.testid)
        self.app.advance({"steps": 1, "intent": self.needs("base")}); self.wait()
        self.assertEqual(self.app.session["events"][-1]["demand_evaluation"]["unmet_people"], ["A"])

    def test_comparison_preserves_demand_timeline_and_rejects_equal_comfort_claim(self):
        self.start(); self.submit(ttl=1)
        self.app.advance({"steps": 3, "intent": self.needs()}); self.wait()
        self.app.end(); self.wait()
        sid = self.app.session["id"]; source = self.app.load(sid)
        self.app.compare(sid); self.wait()
        result = self.app.session
        self.assertTrue(result["demand_replay_valid"])
        self.assertTrue(result["comparison_valid"])
        self.assertEqual(result["demand_events"], source["demand_events"])
        self.assertEqual([e["demand_snapshot"] for e in result["events"]], [e["demand_snapshot"] for e in source["events"]])
        self.assertGreater(result["demand_summary"]["A"]["deviation_Kh"], source["demand_summary"]["A"]["deviation_Kh"])
        self.assertEqual(self.app.load(sid), source)

    def test_need_changes_rejected_while_action_in_flight(self):
        self.start(); blocker = threading.Event(); self.client.block = blocker
        self.app.advance({"steps": 1, "intent": REQUEST})
        try:
            with self.assertRaises(Conflict): self.submit()
            self.assertEqual(self.app.session["demand_events"], [])
        finally:
            blocker.set(); self.wait()

    def test_stale_demand_confirmation_and_malformed_intent_do_not_advance(self):
        self.start(); old = self.needs(); self.submit()
        with self.assertRaises(Conflict): self.app.advance({"steps": 1, "intent": old})
        with self.assertRaises(ValueError): self.app.advance({"steps": 1, "intent": []})
        self.assertEqual(self.app.session["events"], [])
        self.assertEqual(self.app.session["status"], "paused")

    def test_legacy_preferences_are_not_activated_as_new_demands(self):
        self.start(); self.app.advance({"steps": 1, "intent": REQUEST}); self.wait()
        self.assertEqual(self.app.session["demand_events"], [])
        self.assertFalse(any(self.app.session["demands"].values()))
        self.assertEqual(self.app.session["events"][0]["decision"]["demand"]["active_requests"], {})

    def test_new_session_starts_without_previous_person_requests(self):
        self.start(); self.submit(); self.app.end(); self.wait()
        old = self.app.load(self.app.session["id"])
        self.app.create("cool"); self.wait()
        self.assertEqual(self.app.session["demand_revision"], 0)
        self.assertFalse(any(self.app.session["demands"].values()))
        self.assertEqual(self.app.session["demand_events"], [])
        self.assertEqual(old["demands"]["A"]["status"], "active")

    def completed(self):
        self.start()
        self.app.advance({"steps": 2, "intent": REQUEST})
        self.wait()
        self.app.end()
        self.wait()
        return self.app.session["id"]

    def test_intent_rejects_invalid_values_and_preserves_semantics(self):
        self.assertEqual(intent(REQUEST), REQUEST)
        for value in [True, float("nan"), float("inf"), 17.9, 23.3, "21"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                intent({**REQUEST, "target_C": value})
        for change in [{"mode": "mpc"}, {"preferences": {"A": "warmer"}}, {"occupancy": 2}]:
            with self.assertRaises(ValueError):
                intent({**REQUEST, **change})

    def test_manual_rate_limit_and_preferences_do_not_change_action(self):
        before = {"reaTZon_y": 294.15, "oveTSet_u": 294.15}
        current = pd.DataFrame(FakeClient.weather([0])).iloc[0]
        action, detail = command(REQUEST, before, current)
        self.assertAlmostEqual(action["oveTSet_u"] - 273.15, 21.5)
        self.assertTrue(detail["rate_limited"])
        self.assertEqual(action, command({**REQUEST, "preferences": {"A": "cooler", "B": "warmer"}}, before, current)[0])
        with self.assertRaises(ValueError):
            command(REQUEST, {**before, "reaTZon_y": 310.}, current)

    def test_rule_guard_and_slew(self):
        current = pd.DataFrame(FakeClient.weather([0])).iloc[0]
        request = {**REQUEST, "mode": "rule"}
        _, normal = command(request, {"reaTZon_y": 294.15, "oveTSet_u": 294.15}, current)
        self.assertEqual(normal["reason"], "schedule_rule")
        _, low = command(request, {"reaTZon_y": 292.15, "oveTSet_u": 294.15}, current)
        self.assertEqual(low["reason"], "recover_heat")
        self.assertAlmostEqual(low["applied_C"], 21.5)

    def test_office_and_duplicate_session_blocked_at_backend(self):
        with self.assertRaises(ValueError):
            self.app.create("office")
        self.start()
        with self.assertRaises(Conflict):
            self.app.create("warm")

    def test_initializing_archive_never_has_negative_step_count(self):
        original = self.client.call
        entered, unblock = threading.Event(), threading.Event()
        def initializing(method, path, data=None):
            if path.startswith("initialize/"):
                entered.set()
                unblock.wait(timeout=4)
            return original(method, path, data)
        self.client.call = initializing
        self.app.create("cool")
        try:
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(self.app.sessions()[0]["steps"], 0)
        finally:
            unblock.set()
            self.wait()

    def test_native_metrics_save_and_replay_without_engine(self):
        sid = self.completed()
        session = self.app.load(sid)
        self.assertEqual(session["status"], "completed")
        self.assertEqual(len(session["intervals"]), 2)
        self.assertEqual(len(session["intervals"][0]["native"]["time"]), 31)
        self.assertAlmostEqual(session["frames"][-1]["energy_kWh"], .7)
        self.assertEqual(session["provenance"]["temperature_kind"], "operative")
        self.assertEqual(session["provenance"]["role_semantics"], "virtual_control_intent_only")
        self.assertTrue(all(v < 1e-8 for v in session["kpi_absolute_difference"].values()))
        replay = Workbench(self.tmp.name, None)
        self.assertIsNone(replay.state()["session"])
        self.assertEqual(replay.load(sid), session)
        self.assertNotIn("intervals", replay.public(session))
        with self.assertRaises(ValueError):
            replay.load("../secret")

    def test_pause_completes_current_step_and_duplicate_mutation_rejected(self):
        self.start()
        blocker = threading.Event()
        self.client.block = blocker
        self.client.on_advance = self.app.cancel.set
        self.app.advance({"steps": 16, "intent": REQUEST})
        with self.assertRaises(Conflict):
            self.app.advance({"steps": 1, "intent": REQUEST})
        blocker.set()
        self.wait()
        self.assertEqual(len(self.app.session["events"]), 1)
        self.assertEqual(self.app.session["status"], "paused")

    def test_uncertain_advance_is_never_retried_and_releases_worker(self):
        self.start()
        self.client.fail_advance = True
        self.app.advance({"steps": 2, "intent": REQUEST})
        self.app.thread.join(timeout=5)
        self.assertEqual(self.app.session["status"], "error")
        self.assertEqual(len([c for c in self.client.calls if c[1].startswith("advance/")]), 1)
        self.assertIsNone(self.app.testid)
        self.assertEqual(len(self.app.session["events"]), 0)

    def test_release_failure_retains_worker_id_for_retry(self):
        self.start()
        self.client.fail_stop = True
        self.app.end()
        self.wait()
        self.assertIsNotNone(self.app.testid)
        with self.assertRaises(Conflict):
            self.app.create("warm")
        self.client.fail_stop = False
        self.app.end()
        self.wait()
        self.assertIsNone(self.app.testid)

    def test_expired_engine_session_preserves_trace_and_allows_new_session(self):
        self.start()
        self.app.advance({"steps": 1, "intent": REQUEST})
        self.wait()
        sid = self.app.session["id"]
        saved = copy.deepcopy(self.app.session["events"])
        original = self.client.call
        def expired(method, path, data=None):
            if path.startswith(("kpi/", "stop/")):
                body = json.dumps({"errors": [{"msg": "Invalid testid: fixture", "param": "testid"}]}).encode()
                error = HTTPError("http://fixture/" + path, 400, "Bad Request", {}, io.BytesIO(body))
                self.addCleanup(error.close)
                raise error
            return original(method, path, data)
        self.client.call = expired
        self.app.end()
        self.app.thread.join(timeout=5)
        self.assertIsNone(self.app.testid)
        self.assertEqual(self.app.load(sid)["status"], "interrupted")
        self.assertEqual(self.app.load(sid)["events"], saved)
        self.client.call = original
        self.app.create("warm")
        self.wait()

    def test_unrelated_http400_does_not_forget_worker(self):
        self.start()
        original = self.client.call
        def bad_request(method, path, data=None):
            if path.startswith("stop/"):
                body = json.dumps({"errors": [{"msg": "Other error", "param": "testid"}]}).encode()
                raise HTTPError("http://fixture/" + path, 400, "Bad Request", {}, io.BytesIO(body))
            return original(method, path, data)
        self.client.call = bad_request
        self.app.end()
        self.wait()
        self.assertEqual(self.app.testid, "fixture")

    def test_same_start_rule_comparison_preserves_intents_and_checks_conditions(self):
        sid = self.completed()
        source = self.app.load(sid)
        self.app.compare(sid)
        self.wait()
        result = self.app.session
        self.assertTrue(result["comparison_valid"])
        self.assertEqual(result["comparison_of"], sid)
        self.assertEqual(len(result["events"]), 2)
        self.assertTrue(all(e["intent"]["mode"] == "rule" for e in result["events"]))
        self.assertEqual(result["events"][0]["intent"]["preferences"], REQUEST["preferences"])
        self.assertNotEqual(result["events"][0]["action"], source["events"][0]["action"])
        self.assertEqual(self.app.load(sid), source)

    def test_changed_weather_disqualifies_comparison(self):
        sid = self.completed()
        original = self.client.weather
        def changed(t):
            data = original(t)
            data["TDryBul"] = [280.] * len(t)
            return data
        self.client.weather = changed
        self.app.compare(sid)
        self.wait()
        self.assertFalse(self.app.session["comparison_valid"])

    def test_step_count_validation(self):
        self.start()
        for steps in [0, 17, True, 1.5]:
            with self.assertRaises(ValueError):
                self.app.advance({"steps": steps, "intent": REQUEST})

    def test_negative_native_measurements_cannot_be_reported_as_success(self):
        self.start()
        original = self.client.call
        def broken(method, path, data=None):
            result = original(method, path, data)
            if path.startswith("results/"):
                result["reaPHeaPum_y"] = [-1.] * len(result["time"])
            return result
        self.client.call = broken
        self.app.advance({"steps": 1, "intent": REQUEST})
        self.app.thread.join(timeout=5)
        self.assertEqual(self.app.session["status"], "error")
        self.assertEqual(len(self.app.session["events"]), 0)
        self.assertIsNone(self.app.testid)

    def test_unavailable_engine_does_not_rewrite_old_archive(self):
        sid = self.completed()
        old = self.app.load(sid)
        self.client.call = lambda *args: (_ for _ in ()).throw(ConnectionError("offline"))
        self.app.create("warm")
        self.app.thread.join(timeout=5)
        self.assertEqual(self.app.load(sid), old)
        self.assertIsNone(self.app.session)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.app = Workbench(self.tmp.name, FakeClient())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        if self.server.app.thread:
            self.server.app.thread.join(timeout=5)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def test_duplicate_listener_cannot_share_port(self):
        first = LocalServer(("127.0.0.1", 0), Handler)
        try:
            with self.assertRaises(OSError):
                LocalServer(("127.0.0.1", first.server_port), Handler)
        finally:
            first.server_close()

    def test_static_assets_and_readonly_evidence(self):
        for path in ["/", "/app.js", "/style.css", "/api/state", "/api/health", "/api/evidence/office"]:
            with self.subTest(path=path), urlopen(self.base + path) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell launcher requires pwsh")
    def test_repeated_launcher_reuses_same_workspace_without_python_or_simulator(self):
        result = subprocess.run(["pwsh", "-NoProfile", "-File", "scripts/start_workbench.ps1",
                                 "-Port", str(self.server.server_port)], capture_output=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already running", result.stdout)
        self.assertEqual(self.server.app.client.calls, [])

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell launcher requires pwsh")
    def test_launcher_does_not_take_over_other_workspace(self):
        with patch("thermal_comfort.workbench.ROOT", Path(self.tmp.name)):
            result = subprocess.run(["pwsh", "-NoProfile", "-File", "scripts/start_workbench.ps1",
                                     "-Port", str(self.server.server_port)], capture_output=True, encoding="utf-8", timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another workspace", result.stderr)
        with urlopen(self.base + "/api/health") as response:
            self.assertEqual(response.status, 200)

    def test_reject_office_path_traversal_foreign_origin_and_malformed_json(self):
        cases = [("/api/create", b'{"scene":"office"}', {}, 400),
                 ("/api/create", b'{"scene":"cool"}', {"Origin": "http://evil.example"}, 403),
                 ("/api/create", b'{}', {"Host": "evil.example"}, 403),
                 ("/api/create", b'not json', {}, 400),
                 ("/api/create", b'[]', {}, 400),
                 ("/api/create", b'{"scene":"cool","testcase":"bestest_air"}', {}, 400)]
        for path, data, extra, status in cases:
            req = Request(self.base + path, data=data, headers={"Content-Type": "application/json", **extra})
            with self.subTest(data=data), self.assertRaises(HTTPError) as error:
                urlopen(req)
            self.assertEqual(error.exception.code, status)
            error.exception.close()
        for path in ["/../AGENTS.md", "/api/evidence/../../pyproject.toml", "/api/download/../x"]:
            with self.assertRaises(HTTPError) as error:
                urlopen(self.base + path)
            error.exception.close()


if __name__ == "__main__":
    unittest.main()
