"""Local single-zone workbench. Demo sessions never enter research datasets."""

import argparse
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
from urllib.parse import urlsplit
from urllib.error import HTTPError
import uuid

import numpy as np
import pandas as pd

from .simulation import (Client, FORECAST, POINTS, POWER, digest, integrate,
                         rule_action, schedule_at, transitions_frame, validate_kpis)
from . import demands


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "workbench"
CASE = "bestest_hydronic_heat_pump"
SCENES = {"cool": {"name": "早春住宅", "start_day": 63},
          "warm": {"name": "暖季住宅", "start_day": 147},
          "cold": {"name": "晚秋住宅", "start_day": 315}}
LIMITS = {"min_C": 18., "max_C": 23.25, "slew_K": .5, "step_s": 900,
          "stop_min_C": 18., "stop_max_C": 35., "max_steps": 96}
RULE_CONFIG = {"occupied_offset_K": .5, "unoccupied_offset_K": 3.}
ID = re.compile(r"[a-f0-9]{12}")
EVIDENCE = {"delivery": "reports/DELIVERY_V1.md",
            "house": "reports/solar_mpc/REPORT.md",
            "office": "reports/office_final_round/BOUNDARY.md"}


class Conflict(ValueError):
    pass


class LocalServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR can let a second process share a listening port.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def intent(body):
    if not isinstance(body, dict):
        raise ValueError("intent必须是JSON对象")
    fields = {"mode", "target_C", "preferences"}
    if body.get("mode") == "needs":
        fields |= {"resolution", "demand_revision"}
    if set(body) != fields:
        raise ValueError("需要 mode、target_C、preferences，不能增加其他字段")
    if body["mode"] not in ("manual", "rule", "needs"):
        raise ValueError("控制方式只能是 manual、rule 或 needs")
    if body["mode"] == "needs" and (body["resolution"] not in demands.RESOLUTIONS or
            type(body["demand_revision"]) is not int or body["demand_revision"] < 0):
        raise ValueError("需求裁决或版本无效")
    target = body["target_C"]
    if type(target) not in (int, float) or not math.isfinite(target) or not 18 <= target <= 23.25:
        raise ValueError("供暖目标必须是 18～23.25 °C 的有限数值")
    prefs = body["preferences"]
    if not isinstance(prefs, dict) or set(prefs) != {"A", "B"} or any(
            p not in ("warmer", "hold", "cooler") for p in prefs.values()):
        raise ValueError("两位虚拟住户均需选择 warmer、hold 或 cooler")
    return copy.deepcopy(body)


def command(request, before, current):
    """Same actuator limits for both modes; no individual comfort guarantee."""
    temperature = before["reaTZon_y"] - 273.15
    if not math.isfinite(temperature) or not LIMITS["stop_min_C"] <= temperature <= LIMITS["stop_max_C"]:
        raise ValueError("操作温度越过演示停止带，终止仿真；这不是舒适保证")
    target, reason = request["target_C"], "manual_target"
    if request["mode"] == "rule":
        target = rule_action(current["LowerSetp[1]"], current["Occupancy[1]"],
                             {"Minimum": 291.15, "Maximum": 296.4}, RULE_CONFIG)["oveTSet_u"] - 273.15
        reason = "schedule_rule"
        if temperature < current["LowerSetp[1]"] - 273.15 + .2:
            target, reason = 23.25, "recover_heat"
        elif temperature > current["UpperSetp[1]"] - 273.15 - .2:
            target, reason = 18., "coast"
    previous = before["oveTSet_u"] - 273.15
    value = float(np.clip(np.clip(target, 18., 23.25), previous - .5, previous + .5))
    if not 18 - 1e-8 <= value <= 23.25 + 1e-8:
        raise ValueError("初始设定值与执行器范围/变化速率不兼容")
    return {"oveTSet_u": value + 273.15, "oveTSet_activate": 1}, {
        "reason": reason, "target_C": target, "applied_C": value,
        "rate_limited": abs(value - target) > 1e-7}


class Workbench:
    def __init__(self, store, client=None):
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.client = client or Client("http://127.0.0.1:8080")
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.busy = False
        self.message = "选择住宅场景，开始一次仿真"
        self.session = None
        self.testid = None
        self.thread = None

    def state(self):
        with self.lock:
            return {"busy": self.busy, "message": self.message,
                    "can_end": self.testid is not None,
                    "session": self.public(self.session) if self.session else None}

    @staticmethod
    def public(session):
        return {k: copy.deepcopy(v) for k, v in session.items() if k != "intervals"}

    def save(self):
        # Atomic replacement keeps the last complete step readable after interruption.
        path = self.store / (self.session["id"] + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.session, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def load(self, sid):
        if not isinstance(sid, str) or not ID.fullmatch(sid):
            raise ValueError("无效的会话编号")
        return json.loads((self.store / (sid + ".json")).read_text(encoding="utf-8"))

    def sessions(self):
        with self.lock:
            result = []
            for path in sorted(self.store.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                s = json.loads(path.read_text(encoding="utf-8"))
                result.append({k: s[k] for k in ("id", "scene", "created_at", "status", "comparison_of")}
                              | {"steps": max(0, len(s["frames"]) - 1)})
            return result

    def demand_access(self, revision):
        if self.busy or not self.testid or not self.session or self.session["status"] != "paused":
            raise Conflict("只可在住宅会话暂停时变更需求")
        if type(revision) is not int or revision != self.session["demand_revision"]:
            raise Conflict("需求版本已变化，请刷新后重新确认")

    def submit_demand(self, body):
        with self.lock:
            self.demand_access(body.get("revision"))
            now = self.session["frames"][-1]["time_s"]
            value = demands.submission(body, now, self.session["demand_revision"])
            self.session["demands"][value["person"]] = value
            self.session["demand_revision"] += 1
            self.session["demand_events"].append({"action": "submit", "time_s": now,
                "revision": self.session["demand_revision"], "request": copy.deepcopy(value)})
            self.message = "需求已提交；切到按需求模式并推进才会影响动作。"
            self.save()

    def revoke_demand(self, body):
        with self.lock:
            if set(body) != {"person", "revision"} or body["person"] not in ("A", "B"):
                raise ValueError("撤销需提供住户A/B及需求版本")
            self.demand_access(body["revision"])
            value = self.session["demands"][body["person"]]
            if not value or value["status"] != "active":
                raise Conflict("该住户没有可撤销的有效需求")
            value["status"] = "withdrawn"
            now = self.session["frames"][-1]["time_s"]
            value["withdrawn_s"] = now
            self.session["demand_revision"] += 1
            self.session["demand_events"].append({"action": "withdraw", "time_s": now,
                "revision": self.session["demand_revision"], "request": copy.deepcopy(value)})
            self.message = "需求已撤销，仿真时间未推进。"
            self.save()

    def expire_demands(self, now):
        for value in self.session["demands"].values():
            if value and value["status"] == "active" and value["expires_s"] <= now:
                value["status"] = "expired"
                self.session["demand_revision"] += 1
                self.session["demand_events"].append({"action": "expire", "time_s": value["expires_s"],
                    "revision": self.session["demand_revision"], "request": copy.deepcopy(value)})

    def demand_preview(self, body):
        with self.lock:
            if set(body) != {"base_C", "resolution"} or not self.session or not self.session["frames"]:
                raise ValueError("需求预览需提供基准和裁决，并先初始化住宅")
            frame = self.session["frames"][-1]
            return {**demands.resolve(self.session["demands"], body["base_C"], frame["time_s"],
                                     frame["temperature_C"], body["resolution"]),
                    "revision": self.session["demand_revision"], "time_s": frame["time_s"]}

    def launch(self, task, message):
        with self.lock:
            if self.busy:
                raise Conflict("正在执行任务，请等待或暂停；操作不会自动重试")
            self.busy, self.message = True, message
            self.cancel.clear()

            def run():
                try:
                    task()
                except Exception as error:
                    with self.lock:
                        self.message = f"任务停止：{error}。不自动重试；可回放已保存步骤。"
                        if self.session:
                            self.session["status"] = "error"
                            self.session["error"] = str(error)
                            if self.session["comparison_of"]:
                                self.session["comparison_valid"] = False
                            self.save()
                    self.release()
                finally:
                    with self.lock:
                        self.busy = False

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

    def release(self):
        if self.testid is not None:
            try:
                self.client.call("PUT", f"stop/{self.testid}")
            except Exception as error:
                # v0.9.0 explicitly rejects expired IDs. Other 400s/timeouts do
                # not establish that a worker was released and must retain it.
                invalid = False
                if isinstance(error, HTTPError):
                    try:
                        body = json.loads(error.read())
                        invalid = error.code == 400 and any(
                            e.get("param") == "testid" and e.get("msg") == f"Invalid testid: {self.testid}"
                            for e in body.get("errors", []))
                    except (ValueError, AttributeError, TypeError):
                        pass
                    finally:
                        error.close()
                if invalid:
                    with self.lock:
                        self.testid = None
                        self.message = "仿真端已确认会话失效。原轨迹仅供回放，可新建会话；最终KPI未补算。"
                        if self.session:
                            self.session["status"] = "interrupted"
                            self.session["error"] = "engine_session_expired"
                            if self.session["comparison_of"]:
                                self.session["comparison_valid"] = False
                            self.save()
                    return
                # Keep the ID so the user can retry release instead of leaking a worker.
                with self.lock:
                    self.message += f" 释放仿真失败：{error}，请再次结束会话。"
                return
            self.testid = None

    def create(self, scene):
        with self.lock:
            if scene not in SCENES:
                raise ValueError("只允许住宅场景；办公室仅开放只读证据")
            if self.testid is not None:
                raise Conflict("请先结束当前会话，释放唯一仿真工作进程")
            self.launch(lambda: self.initialize(scene), "正在初始化真实 BOPTEST，七天预热可能需要一些时间…")

    def initialize(self, scene, source=None):
        with self.lock:
            self.session = None  # An unavailable engine must not overwrite an older archive.
        version = self.client.call("GET", "version")["version"]
        if version != "0.9.0":
            raise ValueError("只支持固定的 BOPTEST 0.9.0")
        with self.lock:
            self.session = {"schema": 2, "id": uuid.uuid4().hex[:12], "scene": scene,
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "initializing",
                "comparison_of": source["id"] if source else None,
                "provenance": {"engine": "BOPTEST", "version": version, "case": CASE,
                    "start_s": SCENES[scene]["start_day"] * 86400 + 8 * 3600,
                    "warmup_s": 7 * 86400, "area_m2": 192, "step_s": 900,
                    "temperature_kind": "operative", "energy_kind": "electricity",
                    "role_semantics": "virtual_control_intent_only",
                    "demand_protocol": demands.PROTOCOL,
                    "rule": "demo_schedule_guard_v1; not frozen research MPC",
                    "code_sha256": {"workbench.py": digest(Path(__file__)),
                                    "demands.py": digest(ROOT / "thermal_comfort/demands.py"),
                                    "simulation.py": digest(ROOT / "thermal_comfort/simulation.py"),
                                    **{name: digest(ASSETS / name) for name in ("index.html", "app.js", "style.css")}}},
                "frames": [], "events": [], "intervals": [],
                "demands": {"A": None, "B": None}, "demand_events": [], "demand_revision": 0,
                "demand_summary": {p: {"active_steps": 0, "unmet_steps": 0, "deviation_Kh": 0.} for p in ("A", "B")}}
            self.save()
        self.testid = self.client.call("POST", f"testcases/{CASE}/select")["testid"]
        inputs = self.client.call("GET", f"inputs/{self.testid}")
        measurements = self.client.call("GET", f"measurements/{self.testid}")
        if (inputs["oveTSet_u"]["Unit"] != "K" or measurements["reaTZon_y"]["Unit"] != "K"
                or any(measurements[p]["Unit"] != "W" for p in POWER)
                or inputs["oveTSet_u"]["Minimum"] > 291.15
                or inputs["oveTSet_u"]["Maximum"] < 296.4):
            raise ValueError("BOPTEST 接口单位或供暖执行器范围不符")
        self.client.call("PUT", f"scenario/{self.testid}", {"electricity_price": "constant"})
        p = self.session["provenance"]
        before = self.client.call("PUT", f"initialize/{self.testid}",
                                  {"start_time": p["start_s"], "warmup_period": p["warmup_s"]})
        self.client.call("PUT", f"step/{self.testid}", {"step": 900})
        if self.client.call("GET", f"step/{self.testid}") != 900:
            raise ValueError("仿真时间步未成功设置")
        forecast = self.forecast(before["time"])
        zero = {"energy_kWh": 0., "discomfort_Kh": 0.}
        with self.lock:
            self.session["initial"] = before
            self.session["frames"].append(self.frame(before, forecast.iloc[0], zero))
            self.session["status"] = "paused"
            self.message = "房间已就绪。改变需求不会立即改变室温；推进仿真后查看真实响应。"
            self.save()
        if source:
            if not np.allclose([before[k] for k in POINTS], [source["initial"][k] for k in POINTS], atol=1e-7, rtol=0):
                raise ValueError("规则对照与原会话初始状态不同，不给出收益比较")
            requests = [{**e["intent"], "mode": "rule"} for e in source["events"]]
            self.advance_many(requests, source)
            with self.lock:
                n = len(self.session["events"])
                valid = n == len(source["events"]) and all(
                    a["forecast"] == b["forecast"] for a, b in zip(self.session["intervals"], source["intervals"]))
                demand_valid = all(a["demand_snapshot"] == b.get("demand_snapshot", a["demand_snapshot"])
                                   for a, b in zip(self.session["events"], source["events"]))
                valid = valid and demand_valid
                self.session["demand_replay_valid"] = demand_valid and n == len(source["events"]) if "demands" in source else None
                if n == len(source["events"]) and "demands" in source:
                    for key in ("demands", "demand_events", "demand_revision"):
                        self.session[key] = copy.deepcopy(source[key])
                self.session["comparison_valid"] = valid
                self.session["comparison_note"] = "相同初始化、时长、天气和日程；规则不采纳虚拟住户偏好。" if valid else "中断或外生条件不一致，不比较收益。"
                self.save()
            self.finish()

    def forecast(self, start):
        frame = pd.DataFrame(self.client.call("PUT", f"forecast/{self.testid}",
                        {"point_names": FORECAST, "horizon": 900, "interval": 30}))
        if frame.time.iloc[0] != start or frame.time.iloc[-1] != start + 900:
            raise ValueError("天气与日程未覆盖控制区间")
        return frame

    @staticmethod
    def frame(raw, current, metrics):
        return {"time_s": raw["time"], "temperature_C": raw["reaTZon_y"] - 273.15,
                "setpoint_C": raw["oveTSet_u"] - 273.15,
                "power_kW": sum(raw[p] for p in POWER) / 1000,
                "heat_kW": raw["reaQFloHea_y"] / 1000,
                "outdoor_C": float(current["TDryBul"] - 273.15),
                "solar_W_m2": float(current["HGloHor"]),
                "lower_C": float(current["LowerSetp[1]"] - 273.15),
                "upper_C": float(current["UpperSetp[1]"] - 273.15),
                "occupancy": float(current["Occupancy[1]"]), **metrics}

    def advance(self, body):
        if set(body) != {"steps", "intent"} or type(body["steps"]) is not int or not 1 <= body["steps"] <= 16:
            raise ValueError("每批推进 1～16 个十五分钟步，且必须提供 intent")
        request = intent(body["intent"])
        with self.lock:
            if self.testid is None or not self.session or self.session["status"] != "paused":
                raise Conflict("没有可推进的住宅会话")
            if len(self.session["events"]) + body["steps"] > 96:
                raise ValueError("每会话最多演示 24 小时，请结束后重建")
            if request["mode"] == "needs":
                self.demand_access(request["demand_revision"])
                preview = self.demand_preview({"base_C": request["target_C"], "resolution": request["resolution"]})
                if preview["blocked"]:
                    raise Conflict("需求冲突或目标不可执行：请撤销/修改需求，或显式选择A、B、基准后确认")
            self.launch(lambda: self.advance_many([request] * body["steps"]), "正在计算真实仿真反馈…")

    def advance_many(self, requests, source=None):
        for index, request in enumerate(requests):
            if self.cancel.is_set():
                break
            before = self.session["events"][-1]["after"] if self.session["events"] else self.session["initial"]
            with self.lock:
                if source and "demand_snapshot" in source["events"][index]:
                    snapshot = source["events"][index]["demand_snapshot"]
                    self.session["demands"] = copy.deepcopy(snapshot["demands"])
                    self.session["demand_revision"] = snapshot["revision"]
                    self.session["demand_events"] = copy.deepcopy([e for e in source["demand_events"] if e["time_s"] <= before["time"]])
                snapshot = {"demands": copy.deepcopy(self.session["demands"]),
                            "revision": self.session["demand_revision"], "time_s": before["time"]}
                resolution = request.get("resolution", "unresolved") if request.get("demand_revision") == snapshot["revision"] else "unresolved"
                resolved = demands.resolve(snapshot["demands"], request["target_C"], before["time"],
                                           before["reaTZon_y"] - 273.15, resolution)
                if request["mode"] == "needs" and resolved["blocked"]:
                    self.message = "需求已变化且存在冲突/不可执行目标，本步未下发，已暂停等待裁决。"
                    self.save()
                    break
            forecast = self.forecast(before["time"])
            control = {**request, "mode": "manual", "target_C": resolved["target_C"]} if request["mode"] == "needs" else request
            action, decision = command(control, before, forecast.iloc[0])
            if request["mode"] == "needs":
                decision["reason"] = "demand_target"
            decision.update(demand=resolved, demand_handling="adopted" if request["mode"] == "needs" else "bypassed_" + request["mode"])
            evaluation = demands.evaluate(resolved, decision["applied_C"])
            # Never retry a mutation: a failed response may follow an applied action.
            after = self.client.call("POST", f"advance/{self.testid}", action)
            native = pd.DataFrame(self.client.call("PUT", f"results/{self.testid}",
                                  {"point_names": POINTS, "start_time": before["time"], "final_time": after["time"]}))
            record = {"before": before, "action": action, "after": after}
            transitions_frame([record], native, forecast, {"step_s": 900, "area_m2": 192})
            if (len(native) != 31 or not np.allclose(np.diff(native.time), 30)
                    or (native[POWER].to_numpy() < -1e-6).any()):
                raise ValueError("缺少原生30秒采样，或出现负电功率")
            metrics = integrate(native, forecast, 192)
            if not np.isclose(native.reaTZon_y.iloc[-1], after["reaTZon_y"], atol=1e-7, rtol=0):
                raise ValueError("原生轨迹与步进温度不一致")
            with self.lock:
                previous = self.session["frames"][-1]
                cumulative = {k: previous[k] + metrics[k] for k in ("energy_kWh", "discomfort_Kh")}
                self.session["events"].append({**record, "intent": copy.deepcopy(request),
                    "decision": decision, "metrics": metrics, "demand_snapshot": snapshot,
                    "demand_evaluation": evaluation})
                self.session["intervals"].append({"native": native.to_dict("list"), "forecast": forecast.to_dict("list")})
                self.session["frames"].append(self.frame(after, forecast.iloc[-1], cumulative))
                for p, value in evaluation["setpoint_intent_deviation_Kh"].items():
                    totals = self.session["demand_summary"][p]
                    totals["active_steps"] += 1
                    totals["unmet_steps"] += int(p in evaluation["unmet_people"])
                    totals["deviation_Kh"] += value
                self.expire_demands(after["time"])
                self.session["status"] = "paused"
                self.message = "本步已保存。" + ("设定值受每步0.5 K变化限制。" if decision["rate_limited"] else "")
                self.save()
            temperature = after["reaTZon_y"] - 273.15
            if not LIMITS["stop_min_C"] <= temperature <= LIMITS["stop_max_C"]:
                raise ValueError("步进后温度越过停止带，已保存最后实际响应并终止")
        with self.lock:
            if self.cancel.is_set():
                self.message = "已在完整时间步后暂停，当前计算步不会被强行中断。"

    def finish(self):
        if self.session and self.testid:
            if self.session["events"] and self.session["status"] != "error":
                metrics = {key: sum(e["metrics"][key] for e in self.session["events"])
                           for key in self.session["events"][0]["metrics"]}
                kpis = self.client.call("GET", f"kpi/{self.testid}")
                differences = validate_kpis(metrics, kpis)
                with self.lock:
                    self.session.update(api_kpis=kpis, kpi_absolute_difference=differences)
            self.release()
            with self.lock:
                if self.testid is None and self.session["status"] not in ("error", "interrupted"):
                    self.session["status"] = "completed"
                    self.message = "会话已结束并保存，可回放或运行同起点规则对照。"
                self.save()

    def end(self):
        self.launch(self.finish, "正在核对累计指标、保存并释放仿真…")

    def compare(self, sid):
        with self.lock:
            source = self.load(sid)
            if self.testid is not None:
                raise Conflict("请先结束当前会话，再运行规则对照")
            if source["status"] != "completed" or not source["events"] or source["comparison_of"]:
                raise ValueError("请选择已结束且至少推进一步的原始会话")
            self.launch(lambda: self.initialize(source["scene"], source), "正在重新初始化，顺序运行同起点规则对照…")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, code, data, content_type="application/json; charset=utf-8", download=False):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if isinstance(data, (dict, list)) else data
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="thermal-session.json"')
        self.end_headers()
        self.wfile.write(body)

    def local_request(self):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in allowed or (origin and origin not in {f"http://{h}" for h in allowed}):
            raise PermissionError("仅接受本机工作台页面请求")

    def route(self, method):
        try:
            self.local_request()
            app, path = self.server.app, urlsplit(self.path).path
            if method == "GET":
                if path == "/api/health":
                    return self.send(200, {"service": "thermal-workbench", "schema": 1,
                                           "workspace": str(ROOT), "pid": os.getpid()})
                if path == "/api/state":
                    return self.send(200, app.state())
                if path == "/api/config":
                    return self.send(200, {"scenes": SCENES, "limits": LIMITS})
                if path == "/api/sessions":
                    return self.send(200, app.sessions())
                if path.startswith("/api/sessions/"):
                    return self.send(200, app.public(app.load(path.rsplit("/", 1)[-1])))
                if path.startswith("/api/download/"):
                    return self.send(200, app.load(path.rsplit("/", 1)[-1]), download=True)
                if path.startswith("/api/evidence/"):
                    key = path.rsplit("/", 1)[-1]
                    if key not in EVIDENCE:
                        raise FileNotFoundError()
                    return self.send(200, (ROOT / EVIDENCE[key]).read_bytes(), "text/plain; charset=utf-8")
                assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                          "/style.css": ("style.css", "text/css")}
                if path in assets:
                    name, mime = assets[path]
                    return self.send(200, (ASSETS / name).read_bytes(), mime + "; charset=utf-8")
                raise FileNotFoundError()
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384 or self.headers.get_content_type() != "application/json":
                raise ValueError("需要不超过16 KiB的JSON请求")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON请求必须是对象")
            if path == "/api/create" and set(body) == {"scene"}:
                app.create(body["scene"])
            elif path == "/api/advance":
                app.advance(body)
            elif path == "/api/demand":
                app.submit_demand(body)
            elif path == "/api/demand/revoke":
                app.revoke_demand(body)
            elif path == "/api/demand-preview":
                return self.send(200, app.demand_preview(body))
            elif path == "/api/pause" and not body:
                app.cancel.set()
            elif path == "/api/end" and not body:
                app.end()
            elif path == "/api/compare" and set(body) == {"session_id"}:
                app.compare(body["session_id"])
            else:
                raise ValueError("未知操作或请求字段不符")
            self.send(202, {"accepted": True})
        except Conflict as error:
            self.send(409, {"error": str(error)})
        except PermissionError as error:
            self.send(403, {"error": str(error)})
        except FileNotFoundError:
            self.send(404, {"error": "未找到文件或会话"})
        except (ValueError, TypeError, KeyError) as error:
            self.send(400, {"error": str(error)})

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = Workbench(ROOT / "outputs/workbench/sessions")
    try:
        server = LocalServer(("127.0.0.1", args.port), Handler)
    except OSError:
        parser.exit(2, f"Cannot listen on 127.0.0.1:{args.port}. Close the existing service or choose --port.\n")
    server.app = app
    print(f"Thermal workbench: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.cancel.set()
        if app.thread:
            app.thread.join(timeout=310)
        if not app.busy:
            app.release()
            if app.session and app.session["status"] in ("paused", "initializing"):
                app.session["status"] = "interrupted"
                app.session["error"] = "workbench_shutdown_without_final_kpi"
                app.save()
        server.server_close()


if __name__ == "__main__":
    main()
