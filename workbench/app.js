"use strict";
const $ = id => document.getElementById(id);
const words = {warmer:"更暖", hold:"保持", cooler:"更凉"};
const reasons = {manual_target:"手动目标",demand_target:"需求映射",schedule_rule:"日程规则",recover_heat:"温度偏低，规则恢复供暖",coast:"温度偏高，规则提前停热"};
let state = {busy:false,session:null}, mode="needs", person="A", preferences={A:"hold",B:"hold"};
let replay=null, replayIndex=0, replayTimer=null, pending=false, pairedSource=null;
let listKey="", sourceKey="", pollCount=0;
let uiError="";
let connected=false;
let requestedTarget=21;
let demandPreview=null, seenRevision=null, seenSession=null, draftKind="hold";
let draftTTL=4, draftPoint="21", draftLow="20.5", draftHigh="21.5";
const format=(value,d=2)=>Number.isFinite(value)?value.toFixed(d):"--";
$("demand-ttl").replaceChildren(...Array.from({length:16},(_,i)=>new Option(`${(i+1)*15}分钟`,String(i+1))));
$("demand-ttl").value="4";

async function api(path, body) {
  const response = await fetch(path, body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw new Error(data.error||`HTTP ${response.status}`);
  return data;
}
async function action(path, body) {
  if(pending) return;
  uiError=""; pending=true; render();
  try { await api(path,body); state=await api("/api/state"); syncRevision(); await refreshPreview(); }
  catch(error) { uiError=error.message; }
  finally { pending=false; render(); }
}
function currentIntent(){return {mode,target_C:requestedTarget,preferences:{...preferences},...(mode==="needs"?{resolution:$("resolution").value,demand_revision:state.session?.demand_revision||0}:{})};}
function syncRevision(){
  if(seenSession!==state.session?.id||seenRevision!==state.session?.demand_revision){
    $("resolution").value="unresolved"; demandPreview=null;
    seenSession=state.session?.id;seenRevision=state.session?.demand_revision;
  }
}
async function refreshPreview(){
  if(replay||!state.session?.frames.length||state.session.status!=="paused"){demandPreview=null;return;}
  const result=await api("/api/demand-preview",{base_C:requestedTarget,resolution:$("resolution").value});
  if(result.revision===state.session.demand_revision&&result.base_C===requestedTarget&&result.resolution===$("resolution").value)demandPreview=result;
}
function displayedDemands(s,now){
  if(!s?.demands)return null;
  if(!replay)return s.demands;
  const result={A:null,B:null};for(const e of s.demand_events||[])if(e.time_s<=now)result[e.request.person]=e.request;
  return result;
}
function requestStatus(r,now){return !r?"none":r.status==="active"&&now>=r.expires_s?"expired":r.status;}
function requestText(r,now){
  const status=requestStatus(r,now);
  if(status!=="active")return {none:"无请求",withdrawn:"已撤销",expired:"已到期"}[status]||status;
  const interval=r.kind==="warmer"?`≥${format(r.lower_C)}°C`:r.kind==="cooler"?`≤${format(r.upper_C)}°C`:r.lower_C===r.upper_C?`${format(r.lower_C)}°C`:`${format(r.lower_C)}～${format(r.upper_C)}°C`;
  return `${interval}，余${Math.max(0,(r.expires_s-now)/60)}分钟`;
}
function selectPerson(value){
  person=value;
  if(!replay&&state.session?.status==="paused"&&!state.session.comparison_of){
    const r=state.session.demands?.[person];draftKind=r?.kind||preferences[person];
    if(r){draftPoint=String(r.lower_C);draftLow=String(r.lower_C);draftHigh=String(r.upper_C);draftTTL=(r.expires_s-r.submitted_s)/900;}
  }
  render();
}
function renderPreferences(){
  const s=replay||state.session, readOnly=!!replay||!!s?.comparison_of||["completed","error","interrupted"].includes(s?.status);
  const event=s?.events[(replay?replayIndex:s.frames.length-1)-1];
  const visible=readOnly?(event?.intent.preferences||{A:"hold",B:"hold"}):preferences;
  $("person-a").classList.toggle("active",person==="A"); $("person-b").classList.toggle("active",person==="B");
  $("avatar-a").classList.toggle("selected",person==="A"); $("avatar-b").classList.toggle("selected",person==="B");
  const now=s?.frames[replay?replayIndex:s.frames.length-1]?.time_s||0, records=displayedDemands(s,now);
  document.querySelectorAll("[data-pref]").forEach(b=>b.classList.toggle("active",(readOnly?records?.[person]?.kind||visible[person]:draftKind)===b.dataset.pref));
  $("preference-summary").textContent=records?`A：${requestText(records.A,now)}；B：${requestText(records.B,now)}`:`旧字段仅记录：A ${words[visible.A]}，B ${words[visible.B]}`;
  const preview=readOnly?event?.decision.demand:demandPreview;
  $("conflict").hidden=!preview||!["conflict","infeasible"].includes(preview.status);
  $("conflict").textContent=preview?.blocked?"冲突或目标不可执行 · 请修改、撤销或显式裁决":"已显式裁决 · 被覆盖需求仍计入未满足";
}
function setMode(value){mode=value;refreshPreview().then(render).catch(e=>{uiError=e.message;render();});render();}
function renderDemandEditor(s,event,f,readOnly,busy,active,viewMode){
  const records=displayedDemands(s,f?.time_s||0),selected=records?.[person];
  $("demand-events").replaceChildren();
  for(const e of (s?.demand_events||[]).filter(e=>e.time_s<=(f?.time_s||0)).slice(-8)){
    const item=document.createElement("li"),hours=Math.floor(e.time_s/3600)%24,minutes=Math.floor(e.time_s/60)%60;
    item.textContent=`${String(hours).padStart(2,"0")}:${String(minutes).padStart(2,"0")} v${e.revision} ${e.request.person} ${{submit:"提交/替换",withdraw:"撤销",expire:"过期"}[e.action]||e.action}`;$("demand-events").append(item);
  }
  $("demand-kind").value=readOnly?(selected?.kind||"hold"):draftKind;
  const kind=$("demand-kind").value;
  $("point-field").hidden=kind!=="target";$("range-fields").hidden=kind!=="range";
  $("demand-point").value=readOnly&&selected?selected.lower_C:draftPoint;
  $("demand-low").value=readOnly&&selected?selected.lower_C:draftLow;
  $("demand-high").value=readOnly&&selected?selected.upper_C:draftHigh;
  $("demand-ttl").value=readOnly&&selected?(selected.expires_s-selected.submitted_s)/900:draftTTL;
  ["demand-kind","demand-point","demand-low","demand-high","demand-ttl"].forEach(id=>$(id).disabled=busy||readOnly||!active);
  $("submit-demand").disabled=busy||readOnly||!active;
  $("revoke-demand").disabled=busy||readOnly||!active||requestStatus(selected,f?.time_s||0)!=="active";
  $("submit-demand").textContent=`提交${person} · ${Number($("demand-ttl").value)*15}分钟`;$("revoke-demand").textContent=`撤销${person}需求`;
  $("resolution").disabled=busy||readOnly||!active||viewMode!=="needs";
  if(readOnly)$("resolution").value=event?.intent.resolution||"unresolved";
  const p=readOnly?event?.decision.demand:demandPreview;
  const message=!p?(s?.frames.length?"此时间点尚无需求决策记录；旧偏好不追溯激活。":"先初始化住宅。旧记录不追溯激活需求。"):p.blocked?`版本${readOnly?event?.demand_snapshot.revision:p.revision}：${p.status==="conflict"?"需求没有交集":"目标超出执行器范围"}，按需求模式尚未允许下发。`:`基准 ${format(p.base_C)} → 映射供暖目标 ${format(p.target_C)}°C（${p.choice==="intersection"?"共同可行区间":p.choice==="operator_base"?"操作者基准":"操作者优先"+p.choice.slice(-1)}）。`;
  $("demand-preview").textContent=message+(p?.capability_notes.includes("heating_only_no_active_cooling")?" 只能减少供暖，不能主动制冷或保证室温恢复。":"")+(viewMode!=="needs"?" 当前手动/规则旁路不采纳需求。":"");
}
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
  const busy=state.busy||pending||!connected, s=replay||state.session, frames=s?.frames||[];
  const index=replay?Math.min(replayIndex,frames.length-1):frames.length-1, f=frames[index];
  const active=state.session?.status==="paused", historical=!!replay;
  const event=index>0?s.events[index-1]:null;
  const readOnly=historical||!!s?.comparison_of||["completed","error","interrupted"].includes(s?.status);
  const viewMode=readOnly?(event?.intent.mode||(s?.comparison_of?"rule":s?.demands?"needs":"manual")):mode;
  $("create").disabled=busy||active; $("scene").disabled=busy||active;
  ["step","run"].forEach(id=>$(id).disabled=busy||!active||historical||(state.session?.events.length||0)+(id==="run"?16:1)>96);
  if(viewMode==="needs"&&(!demandPreview||demandPreview.blocked))["step","run"].forEach(id=>$(id).disabled=true);
  $("end").disabled=busy||!state.can_end;
  $("pause").disabled=!connected||pending||!state.busy||!state.session?.frames.length; $("target").disabled=viewMode==="rule"||readOnly||busy;
  ["needs","manual","rule","person-a","person-b"].forEach(id=>$(id).disabled=busy||readOnly);
  document.querySelectorAll("[data-pref]").forEach(b=>b.disabled=busy||readOnly);
  ["needs","manual","rule"].forEach(id=>$(id).classList.toggle("active",viewMode===id));
  $("target").value=readOnly&&event?event.intent.target_C:requestedTarget;
  $("target-value").innerHTML=`${format(Number($("target").value),2)}<span> °C</span>`;
  $("mode-note").textContent=viewMode==="needs"?"先提交住户需求，按共同区间映射；无有效需求时使用基准。每步仍限速0.5 K。":viewMode==="manual"?"手动旁路使用操作者基准，不采纳已提交需求。":"日程＋温度保护；不采纳人员需求，也不是此前研究的 MPC。";
  $("status").textContent=uiError||(historical?`历史回放 · ${s.id} · 不连接或改变仿真状态`:state.message||"等待开始");
  $("status-dot").classList.toggle("busy",state.busy&&!historical);
  $("status").parentElement.classList.toggle("error",!historical&&s?.status==="error");
  $("view-badge").textContent=historical?"HISTORY / 历史回放":s?(s.status==="completed"?(s.comparison_of?"SAVED / 规则对照已保存":"SAVED / 本次已保存"):s.comparison_of?"SIMULATION / 规则对照":"SIMULATION / BOPTEST 实时仿真"):"尚未连接会话";
  if(historical&&s.status!=="completed")$("view-badge").textContent="HISTORY / 未正常结束 · 仅回放";
  if(!historical&&["interrupted","error"].includes(s?.status))$("view-badge").textContent="STOPPED / 仅保留最后有效记录";
  if(!connected&&!historical){$("view-badge").textContent="OFFLINE / 最后已知状态";$("status").textContent="本地服务连接中断，已禁用控制；不会自动重发动作。";}
  const prefs=readOnly?(event?.intent.preferences||{A:"hold",B:"hold"}):preferences;
  const records=displayedDemands(s,f?.time_s||0);
  ["A","B"].forEach(p=>{$("pref-"+p.toLowerCase()).textContent=records?(requestStatus(records[p],f?.time_s||0)==="active"?({target:"指定目标",range:"区间"}[records[p].kind]||words[records[p].kind]):requestText(records[p],f?.time_s||0)):words[prefs[p]];});
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
  if(event?.demand_evaluation)$("decision").textContent=`${reasons[event.decision.reason]||event.decision.reason}：基准${format(event.intent.target_C)} → 控制目标${format(event.decision.target_C)} → 下发${format(event.decision.applied_C)}°C。${event.decision.rate_limited?"限速生效。":""}供暖意图未满足：${event.demand_evaluation.unmet_people.join("、")||"无"}。${event.decision.demand_handling!=="adopted"?"本步旁路未采纳需求。":""}不是人员舒适评价。`;
  $("scrubber").disabled=!historical||!frames.length; $("scrubber").max=Math.max(0,frames.length-1); $("scrubber").value=Math.max(0,index);
  $("frame-index").textContent=`${Math.max(0,index)} / ${Math.max(0,frames.length-1)}`;
  $("replay-play").disabled=!historical||frames.length<2;
  $("download").classList.toggle("disabled",!s); $("download").href=s?`/api/download/${s.id}`:"#";
  $("compare").disabled=busy||active||!s||s.status!=="completed"||!s.events.length||!!s.comparison_of;
  drawTrend(frames,Math.max(index,0)); renderComparison(s); renderPreferences();renderDemandEditor(s,event,f,readOnly,busy,active,viewMode);
}
function renderComparison(s){
  const box=$("comparison");box.hidden=!s?.comparison_of;
  if(box.hidden)return;
  if(!s.comparison_valid){box.textContent=s.comparison_note||"规则对照尚未完成，暂不计算收益。";return;}
  if(!pairedSource||pairedSource.id!==s.comparison_of){box.textContent="读取原会话…";return;}
  const a=pairedSource.frames.at(-1),b=s.frames.at(-1),delta=b.energy_kWh-a.energy_kWh;
  box.replaceChildren();const title=document.createElement("strong");title.textContent="同起点与外生条件已核对";box.append(title);
  const table=document.createElement("table");[["", "原会话", "演示规则"],["耗电 kWh",format(a.energy_kWh,3),format(b.energy_kWh,3)],["越界 K·h",format(a.discomfort_Kh,3),format(b.discomfort_Kh,3)]].forEach(row=>{const tr=document.createElement("tr");row.forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td);});table.append(tr);});box.append(table);
  const note=document.createElement("p");note.textContent=`规则相对原会话耗电${delta>=0?"增加":"减少"} ${format(Math.abs(delta),3)} kWh。日程越界不代表真实热舒适；供暖意图偏差另列，不能据此认为两者舒适等效或外推节能结论。`;box.append(note);
  if(s.demand_summary){const extra=document.createElement("p");const total=v=>Object.values(v||{}).reduce((sum,r)=>sum+r.deviation_Kh,0);extra.textContent=`设定意图偏差积分：原会话${format(total(pairedSource.demand_summary),3)} / 规则${format(total(s.demand_summary),3)} K·h。${s.demand_replay_valid?"相同需求快照/期限已核对。":""}规则未采纳需求，不是等需求满足的节能对照。`;box.append(extra);}
}
async function refreshList(){
  const sessions=await api("/api/sessions"),key=JSON.stringify(sessions);
  if(key===listKey)return;listKey=key;const selected=$("session-list").value;
  $("session-list").replaceChildren(new Option("当前实时会话",""));
  for(const s of sessions) $("session-list").add(new Option(`${{cool:"早春",warm:"暖季",cold:"晚秋"}[s.scene]} · ${s.steps}步 · ${s.id.slice(0,6)}${s.comparison_of?" · 规则对照":""} · ${{initializing:"预热中",paused:"未结束",interrupted:"仅回放",error:"异常·仅回放",completed:"已保存"}[s.status]||s.status}`,s.id));
  $("session-list").value=selected;
}
function stopReplay(){if(replayTimer)clearInterval(replayTimer);replayTimer=null;$("replay-play").textContent="▶ 回放";}
async function loadReplay(sid){
  stopReplay();demandPreview=null;$("resolution").value="unresolved"; if(!sid){replay=null;refreshPreview().then(render);return;}
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
$("needs").onclick=()=>setMode("needs");$("manual").onclick=()=>setMode("manual");$("rule").onclick=()=>setMode("rule");
$("target").oninput=()=>{requestedTarget=Number($("target").value);demandPreview=null;refreshPreview().then(render).catch(e=>{uiError=e.message;render();});render();};
["a","b"].forEach(p=>{const select=()=>selectPerson(p.toUpperCase());$("person-"+p).onclick=select;$("avatar-"+p).onclick=select;$("avatar-"+p).onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select();}};});
document.querySelectorAll("[data-pref]").forEach(b=>b.onclick=()=>{preferences[person]=b.dataset.pref;draftKind=b.dataset.pref;render();});
$("demand-kind").onchange=()=>{draftKind=$("demand-kind").value;render();};
$("demand-ttl").onchange=()=>{draftTTL=Number($("demand-ttl").value);render();};
$("demand-point").oninput=()=>{draftPoint=$("demand-point").value;};
$("demand-low").oninput=()=>{draftLow=$("demand-low").value;};
$("demand-high").oninput=()=>{draftHigh=$("demand-high").value;};
$("resolution").onchange=()=>{demandPreview=null;refreshPreview().then(render).catch(e=>{uiError=e.message;render();});render();};
$("submit-demand").onclick=()=>{const body={person,kind:draftKind,base_C:requestedTarget,ttl_steps:Number($("demand-ttl").value),revision:state.session.demand_revision};if(draftKind==="target")body.target_C=Number($("demand-point").value);if(draftKind==="range")Object.assign(body,{lower_C:Number($("demand-low").value),upper_C:Number($("demand-high").value)});action("/api/demand",body);};
$("revoke-demand").onclick=()=>action("/api/demand/revoke",{person,revision:state.session.demand_revision});
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
  try{state=await api("/api/state");connected=true;syncRevision();await refreshPreview();await loadPair(replay||state.session);render();if(pollCount++%5===0)await refreshList();}
  catch(error){connected=false;render();}
  finally{setTimeout(poll,1000);}
}
render();poll();
