"use strict";
const $ = id => document.getElementById(id);
const words = {warmer:"更暖", hold:"保持", cooler:"更凉"};
const reasons = {manual_target:"手动目标",schedule_rule:"日程规则",recover_heat:"温度偏低，规则恢复供暖",coast:"温度偏高，规则提前停热"};
let state = {busy:false,session:null}, mode="manual", person="A", preferences={A:"hold",B:"hold"};
let replay=null, replayIndex=0, replayTimer=null, pending=false, pairedSource=null;
let listKey="", sourceKey="", pollCount=0;
let uiError="";
let requestedTarget=21;
const format=(value,d=2)=>Number.isFinite(value)?value.toFixed(d):"--";

async function api(path, body) {
  const response = await fetch(path, body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw new Error(data.error||`HTTP ${response.status}`);
  return data;
}
async function action(path, body) {
  if(pending) return;
  uiError=""; pending=true; render();
  try { await api(path,body); state=await api("/api/state"); }
  catch(error) { uiError=error.message; }
  finally { pending=false; render(); }
}
function currentIntent(){return {mode,target_C:requestedTarget,preferences:{...preferences}};}
function selectPerson(value){person=value; renderPreferences();}
function renderPreferences(){
  const s=replay||state.session, readOnly=!!replay||!!s?.comparison_of||["completed","error"].includes(s?.status);
  const event=s?.events[(replay?replayIndex:s.frames.length-1)-1];
  const visible=readOnly?(event?.intent.preferences||{A:"hold",B:"hold"}):preferences;
  $("person-a").classList.toggle("active",person==="A"); $("person-b").classList.toggle("active",person==="B");
  $("avatar-a").classList.toggle("selected",person==="A"); $("avatar-b").classList.toggle("selected",person==="B");
  document.querySelectorAll("[data-pref]").forEach(b=>b.classList.toggle("active",visible[person]===b.dataset.pref));
  $("preference-summary").textContent=`A：${words[visible.A]}　B：${words[visible.B]}`;
  $("conflict").hidden=!(Object.values(visible).includes("warmer")&&Object.values(visible).includes("cooler"));
  $("conflict").textContent=readOnly?"当时存在需求冲突 · 偏好未自动合并":"需求冲突 · 请由你裁决供暖目标";
}
function setMode(value){mode=value; $("manual").classList.toggle("active",mode==="manual"); $("rule").classList.toggle("active",mode==="rule"); render();}
function drawTrend(frames,index){
  const svg=$("trend"), ns="http://www.w3.org/2000/svg";
  svg.replaceChildren();
  const add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e;};
  if(!frames.length){add("text",{x:330,y:70,"text-anchor":"middle",class:"chart-label"},"开始仿真后显示真实轨迹");return;}
  const values=frames.flatMap(f=>[f.temperature_C,f.setpoint_C,f.lower_C,f.upper_C]);
  const lo=Math.floor(Math.min(...values)-.5),hi=Math.ceil(Math.max(...values)+.5);
  const x=i=>35+i/Math.max(frames.length-1,1)*610, y=v=>112-(v-lo)/(hi-lo)*95;
  [lo,(lo+hi)/2,hi].forEach(v=>{add("line",{x1:35,x2:645,y1:y(v),y2:y(v),class:"chart-grid"});add("text",{x:2,y:y(v)+3,class:"chart-label"},format(v,1));});
  ["lower_C","upper_C","setpoint_C","temperature_C"].forEach((key,i)=>add("polyline",{points:frames.map((f,k)=>`${x(k)},${y(f[key])}`).join(" "),class:["band-line","band-line","target-line","temp-line"][i]}));
  add("circle",{cx:x(index),cy:y(frames[index].temperature_C),r:4,fill:"#355b49"});
}
function render(){
  const busy=state.busy||pending, s=replay||state.session, frames=s?.frames||[];
  const index=replay?Math.min(replayIndex,frames.length-1):frames.length-1, f=frames[index];
  const active=state.session?.status==="paused", historical=!!replay;
  const event=index>0?s.events[index-1]:null;
  const readOnly=historical||!!s?.comparison_of||["completed","error"].includes(s?.status);
  const viewMode=readOnly?(event?.intent.mode||(s?.comparison_of?"rule":"manual")):mode;
  $("create").disabled=busy||active; $("scene").disabled=busy||active;
  ["step","run"].forEach(id=>$(id).disabled=busy||!active||historical||(state.session?.events.length||0)+(id==="run"?16:1)>96);
  $("end").disabled=busy||!state.session||state.session.status==="completed";
  $("pause").disabled=!state.busy; $("target").disabled=viewMode==="rule"||readOnly||busy;
  ["manual","rule","person-a","person-b"].forEach(id=>$(id).disabled=busy||readOnly);
  document.querySelectorAll("[data-pref]").forEach(b=>b.disabled=busy||readOnly);
  $("manual").classList.toggle("active",viewMode==="manual");$("rule").classList.toggle("active",viewMode==="rule");
  $("target").value=readOnly&&event?event.intent.target_C:requestedTarget;
  $("target-value").innerHTML=`${format(Number($("target").value),2)}<span> °C</span>`;
  $("mode-note").textContent=viewMode==="manual"?"目标经每15分钟0.5 K限速后下发，室温会滞后响应。":"日程＋温度保护；不采纳虚拟偏好，也不是此前研究的 MPC。";
  $("status").textContent=uiError||(historical?`历史回放 · ${s.id} · 不连接或改变仿真状态`:state.message||"等待开始");
  $("status-dot").classList.toggle("busy",state.busy&&!historical);
  $("status").parentElement.classList.toggle("error",!historical&&s?.status==="error");
  $("view-badge").textContent=historical?"HISTORY / 历史回放":s?(s.status==="completed"?(s.comparison_of?"SAVED / 规则对照已保存":"SAVED / 本次已保存"):s.comparison_of?"SIMULATION / 规则对照":"SIMULATION / BOPTEST 实时仿真"):"尚未连接会话";
  const prefs=readOnly?(event?.intent.preferences||{A:"hold",B:"hold"}):preferences;
  $("pref-a").textContent=words[prefs.A]; $("pref-b").textContent=words[prefs.B];
  if(f){
    const hours=Math.floor(f.time_s/3600)%24,minutes=Math.floor(f.time_s/60)%60;
    $("clock").textContent=`${String(hours).padStart(2,"0")}:${String(minutes).padStart(2,"0")}`;
    $("sim-day").textContent=`第 ${Math.floor(f.time_s/86400)} 天 · ${s.scene==="warm"?"暖季":s.scene==="cold"?"晚秋":"早春"}`;
    [["temperature",f.temperature_C," °C"],["applied",f.setpoint_C," °C"],["power",f.power_kW," kW"],["energy",f.energy_kWh," kWh"],["violation",f.discomfort_Kh," K·h"]].forEach(([id,v,u])=>{const el=$(id);el.replaceChildren(document.createTextNode(format(v,id==="violation"?3:2)));const small=document.createElement("small");small.textContent=u;el.append(small);});
    $("outdoor").textContent=format(f.outdoor_C,1); $("solar").textContent=format(f.solar_W_m2,0);
    $("heat-waves").toggleAttribute("hidden",f.heat_kW<.05);
    $("elapsed").textContent=`已推进 ${format((f.time_s-frames[0].time_s)/3600,2)} 小时 / ${frames.length-1} 个记录步`;
    $("bound").textContent=`当前日程 ${format(f.lower_C,1)}～${format(f.upper_C,1)} °C`;
  }else{
    $("heat-waves").toggleAttribute("hidden",true);
    ["temperature","applied","power","energy","violation","outdoor","solar"].forEach(id=>$(id).textContent="--");
    $("clock").textContent="--:--";$("sim-day").textContent="等待初始化";
    $("elapsed").textContent="还没有推进时间";$("bound").textContent="温度指标：操作温度";
  }
  $("decision").textContent=event?`${reasons[event.decision.reason]||event.decision.reason}。目标 ${format(event.decision.target_C)} °C → 实际下发 ${format(event.decision.applied_C)} °C。${event.decision.rate_limited?"变化速率限制生效。":""}本步耗电 ${format(event.metrics.energy_kWh,3)} kWh；A ${words[event.intent.preferences.A]}，B ${words[event.intent.preferences.B]}。`:"操作后显示本步的目标、限速与实际下发结果。";
  $("scrubber").disabled=!historical||!frames.length; $("scrubber").max=Math.max(0,frames.length-1); $("scrubber").value=Math.max(0,index);
  $("frame-index").textContent=`${Math.max(0,index)} / ${Math.max(0,frames.length-1)}`;
  $("replay-play").disabled=!historical||frames.length<2;
  $("download").classList.toggle("disabled",!s); $("download").href=s?`/api/download/${s.id}`:"#";
  $("compare").disabled=busy||active||!s||s.status!=="completed"||!s.events.length||!!s.comparison_of;
  drawTrend(frames,Math.max(index,0)); renderComparison(s); renderPreferences();
}
function renderComparison(s){
  const box=$("comparison");box.hidden=!s?.comparison_of;
  if(box.hidden)return;
  if(!s.comparison_valid){box.textContent=s.comparison_note||"规则对照尚未完成，暂不计算收益。";return;}
  if(!pairedSource||pairedSource.id!==s.comparison_of){box.textContent="读取原会话…";return;}
  const a=pairedSource.frames.at(-1),b=s.frames.at(-1),delta=b.energy_kWh-a.energy_kWh;
  box.replaceChildren();const title=document.createElement("strong");title.textContent="同起点与外生条件已核对";box.append(title);
  const table=document.createElement("table");[["", "原会话", "演示规则"],["耗电 kWh",format(a.energy_kWh,3),format(b.energy_kWh,3)],["越界 K·h",format(a.discomfort_Kh,3),format(b.discomfort_Kh,3)]].forEach(row=>{const tr=document.createElement("tr");row.forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td);});table.append(tr);});box.append(table);
  const note=document.createElement("p");note.textContent=`规则相对原会话耗电${delta>=0?"增加":"减少"} ${format(Math.abs(delta),3)} kWh。仅比较官方温度日程，未评估偏好满足；不能据此认为两者舒适等效，也不能外推节能结论。`;box.append(note);
}
async function refreshList(){
  const sessions=await api("/api/sessions"),key=JSON.stringify(sessions);
  if(key===listKey)return;listKey=key;const selected=$("session-list").value;
  $("session-list").replaceChildren(new Option("当前实时会话",""));
  for(const s of sessions) $("session-list").add(new Option(`${{cool:"早春",warm:"暖季",cold:"晚秋"}[s.scene]} · ${s.steps}步 · ${s.id.slice(0,6)}${s.comparison_of?" · 规则对照":""}`,s.id));
  $("session-list").value=selected;
}
function stopReplay(){if(replayTimer)clearInterval(replayTimer);replayTimer=null;$("replay-play").textContent="▶ 回放";}
async function loadReplay(sid){
  stopReplay(); if(!sid){replay=null;render();return;}
  try{replay=await api(`/api/sessions/${sid}`);replayIndex=0;await loadPair(replay);render();}catch(e){$("status").textContent=e.message;}
}
async function loadPair(s){
  if(!s?.comparison_of||sourceKey===s.comparison_of)return;
  pairedSource=await api(`/api/sessions/${s.comparison_of}`);sourceKey=s.comparison_of;
}
async function evidence(key){
  $("dialog-title").textContent=key==="office"?"办公室 · 封存与适用边界":key==="house"?"住宅 · 既有研究结果":"第一版交付说明";
  try{const response=await fetch(`/api/evidence/${key}`);if(!response.ok)throw new Error("证据读取失败");$("dialog-text").textContent=await response.text();$("details").showModal();}catch(e){$("status").textContent=e.message;}
}
$("create").onclick=()=>{replay=null;stopReplay();$("session-list").value="";action("/api/create",{scene:$("scene").value});};
$("manual").onclick=()=>setMode("manual");$("rule").onclick=()=>setMode("rule");
$("target").oninput=()=>{requestedTarget=Number($("target").value);$("target-value").innerHTML=`${format(requestedTarget,2)}<span> °C</span>`;};
["a","b"].forEach(p=>{const select=()=>selectPerson(p.toUpperCase());$("person-"+p).onclick=select;$("avatar-"+p).onclick=select;$("avatar-"+p).onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select();}};});
document.querySelectorAll("[data-pref]").forEach(b=>b.onclick=()=>{preferences[person]=b.dataset.pref;renderPreferences();});
$("step").onclick=()=>action("/api/advance",{steps:1,intent:currentIntent()});$("run").onclick=()=>action("/api/advance",{steps:16,intent:currentIntent()});
$("pause").onclick=()=>action("/api/pause",{});$("end").onclick=()=>action("/api/end",{});
$("session-list").onchange=()=>loadReplay($("session-list").value);$("live").onclick=()=>{$("session-list").value="";loadReplay("");};
$("scrubber").oninput=()=>{stopReplay();replayIndex=Number($("scrubber").value);render();};
$("replay-play").onclick=()=>{if(replayTimer){stopReplay();return;}if(replayIndex>=replay.frames.length-1)replayIndex=0;$("replay-play").textContent="Ⅱ 暂停";replayTimer=setInterval(()=>{replayIndex++;render();if(replayIndex>=replay.frames.length-1)stopReplay();},700);};
$("compare").onclick=()=>{const s=replay||state.session;replay=null;mode="rule";stopReplay();$("session-list").value="";action("/api/compare",{session_id:s.id});};
$("heat-device").onclick=()=>{$("dialog-title").textContent="热泵供暖 · 当前执行器";const s=replay||state.session, f=s?.frames[replay?replayIndex:s.frames.length-1];$("dialog-text").textContent=`供暖设定值：${f?format(f.setpoint_C):"--"} °C\n供热功率：${f?format(f.heat_kW):"--"} kW（热量）\n电功率：${f?format(f.power_kW):"--"} kW（热泵＋风机＋水泵）\n\n可调供暖目标为18～23.25 °C，每15分钟变化不超过0.5 K。\n本版不开放制冷、风量、开窗等未经验证的动作。\n规则采用日程与温度保护，不是住宅研究中冻结的MPC。`;$("details").showModal();};
$("heat-device").onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();$("heat-device").onclick();}};
$("office").onclick=()=>evidence("office");document.querySelectorAll("[data-evidence]").forEach(b=>b.onclick=()=>evidence(b.dataset.evidence));$("close-dialog").onclick=()=>$("details").close();
async function poll(){
  try{state=await api("/api/state");await loadPair(replay||state.session);render();if(pollCount++%5===0)await refreshList();}
  catch(error){$("status").textContent=`本地服务连接中断：${error.message}。不会自动重发控制动作。`;["step","run","create","compare"].forEach(id=>$(id).disabled=true);}
  finally{setTimeout(poll,1000);}
}
render();poll();
