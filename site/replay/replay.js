import { copy } from './copy.js';

const el = id => document.getElementById(id);
const timeline = el('timeline');
const totalSteps = Number(timeline.max);
const languageLinks = document.querySelectorAll('.language-switch a');
const colors = ['#64836c', '#b48a44', '#557da0'];
const names = {
  guarded_rule: 'ruleName', solar_mpc: 'mpcName', SAC: 'sacName', PPO: 'ppoName',
  TD3: 'td3Name', residual_SAC: 'residualName', minimum_heat: 'minimumName',
};
const plot = { left: 55, top: 15, width: 1020, height: 157 };
const xAt = frame => plot.left + frame / totalSteps * plot.width;
const fmt = (value, precision = 2) => Number(value).toFixed(precision);

let language = new URLSearchParams(location.search).get('lang') === 'zh-CN' ? 'zh-CN' : 'en';
let archive;
let selected = [];
let step = 0;
let timer = null;
let intervalNote = '';
let loadFailed = false;

function t(key, values = {}) {
  return copy[language][key].replace(/\{(\w+)\}/g, (_, name) => values[name]);
}

function passes(candidate) {
  const { comparisons, fallback_gate_passed } = candidate.comparison;
  return comparisons.guarded_rule.joint_gate_passed && comparisons.solar_mpc.joint_gate_passed && fallback_gate_passed;
}

function renderCounts() {
  if (!archive) {
    el('counts').textContent = t(loadFailed ? 'loadError' : 'loading');
    return;
  }
  el('counts').textContent = ['SAC', 'PPO', 'TD3', 'residual_SAC'].map(family => {
    const count = archive.cases.filter(candidate => candidate.family === family
      && candidate.episode !== 'development189' && passes(candidate)).length;
    const name = family === 'residual_SAC' ? t('residualShort') : family;
    return `${name} ${count}/6`;
  }).join(' · ');
}

function renderLanguage() {
  document.documentElement.lang = language;
  document.title = t('title');
  el('home').href = language === 'zh-CN' ? '../zh-CN/' : '../';
  for (const link of languageLinks) {
    if (link.hreflang === language) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  for (const node of document.querySelectorAll('[data-copy]')) node.textContent = t(node.dataset.copy);
  for (const node of document.querySelectorAll('[data-aria]')) node.setAttribute('aria-label', t(node.dataset.aria));
  el('play').textContent = t(timer === null ? 'play' : 'pause');
  el('error').textContent = loadFailed ? t('loadError') : '';
  renderCounts();
  if (archive) render();
}

function stop() {
  clearInterval(timer);
  timer = null;
  el('play').textContent = t('play');
}

function choose() {
  stop();
  intervalNote = '';
  const family = el('family').value;
  const seed = Number(el('seed').value);
  el('seed').disabled = family === 'minimum_heat';
  selected = ['guarded_rule', 'solar_mpc', family].map(method => archive.cases.find(candidate =>
    candidate.episode === el('episode').value && candidate.family === method
    && (candidate.seed === null || candidate.seed === seed)));
  render();
}

function metric(key, value, unit) {
  return `<div class="metric"><small>${t(key)}</small><b>${value}</b> ${unit}</div>`;
}

function room(candidate, index) {
  const [temperature, setpoint, power, energy, coldTotal, hotTotal, lower, upper, , solar, library, anchor] = candidate.frames[step];
  const { family, seed, metrics } = candidate;
  const name = t(names[family]);
  const state = temperature > upper ? 'hot' : temperature < lower ? 'cold' : 'inside';
  const tint = { hot: '#f1d8bb', cold: '#d8e8ef', inside: '#e2e8d7' }[state];
  const settings = t('setpointBand', { setpoint: fmt(setpoint), lower: fmt(lower, 1), upper: fmt(upper, 1) });
  const result = seed === null
    ? t(family === 'minimum_heat' ? 'diagnosticResult' : 'referenceResult')
    : t('jointResult', { result: t(passes(candidate) ? 'passed' : 'failed'), fallback: fmt(candidate.fallback_fraction * 100, 1) });

  let action;
  if (step <= archive.prefix_steps) action = t('commonAction');
  else if (family === 'residual_SAC') action = t('residualAction', { residual: fmt(library, 3), anchor: fmt(anchor), setpoint: fmt(setpoint) });
  else if (seed === null) action = t('baselineAction', { setpoint: fmt(setpoint) });
  else action = t('deltaAction', { delta: fmt(library, 3), setpoint: fmt(setpoint) });

  return `<article class="room-card" style="--color:${colors[index]}">
    <div class="room-head">
      <h2>${name}${seed === null ? '' : ` · s${seed}`}</h2>
      <span class="state ${state}">${t(state)}</span>
    </div>
    <svg class="room" viewBox="0 0 340 200" role="img" aria-label="${t('roomAria', { name })} · ${t('operativeTemp')} ${fmt(temperature)}°C · ${settings}">
      <rect x="33" y="23" width="274" height="147" rx="12" fill="${tint}" stroke="#b9c3b0" stroke-width="5"/>
      <rect x="130" y="19" width="80" height="12" fill="#b9d2cc"/>
      <path d="M134 34L94 140H257L207 34Z" fill="#f4d88c" opacity="${Math.min(.38, solar / 1800)}"/>
      <rect x="51" y="48" width="60" height="63" rx="9" fill="#a9b9a0"/>
      <rect x="242" y="50" width="43" height="85" rx="7" fill="#cfb898"/>
      <text x="170" y="100" text-anchor="middle" class="room-temp">${fmt(temperature)}°</text>
      <text x="170" y="121" text-anchor="middle" class="room-unit">${t('operativeTemp')}</text>
      <rect x="133" y="144" width="74" height="19" rx="4" fill="#fffdf4" stroke="#879e86"/>
      <path d="M143 147V160M155 147V160M167 147V160M179 147V160M191 147V160" stroke="${power > .1 ? '#b66d3f' : '#a1b19d'}" stroke-width="3"/>
      <text x="170" y="190" text-anchor="middle" class="room-unit">${settings}</text>
    </svg>
    <div class="metrics">
      ${metric('power', fmt(power), 'kW')}${metric('energy', fmt(energy), 'kWh')}
      ${metric('coldTotal', fmt(coldTotal, 3), 'K·h')}${metric('hotTotal', fmt(hotTotal, 3), 'K·h')}
    </div>
    <div class="action hint">${action}</div>
    <div class="total hint"><strong>${t('weekTotals', {
      energy: fmt(metrics.energy_kWh), cold: fmt(metrics.cold_Kh, 3), hot: fmt(metrics.hot_Kh, 3),
    })}</strong>${result}</div>
  </article>`;
}

function renderChart() {
  const temperatures = selected.flatMap(candidate => candidate.frames.map(frame => frame[0]));
  const lower = selected[0].frames.map(frame => frame[6]);
  const upper = selected[0].frames.map(frame => frame[7]);
  const lo = Math.floor(Math.min(...temperatures, ...lower) - .5);
  const hi = Math.ceil(Math.max(...temperatures, ...upper) + .5);
  const yAt = value => plot.top + (hi - value) / (hi - lo) * plot.height;
  const points = values => values.map((value, index) => `${xAt(index)},${yAt(value)}`);
  const band = [...points(lower), ...points(upper).reverse()].join(' ');
  let svg = `<polygon points="${band}" fill="#e9eae2"/>`;
  for (let value = lo; value <= hi; value += Math.max(1, Math.ceil((hi - lo) / 5))) {
    const y = yAt(value);
    svg += `<path d="M${plot.left} ${y}H${plot.left + plot.width}" stroke="#e4e7dd"/>
      <text x="8" y="${y + 4}" font-size="11" fill="#697770">${value}°C</text>`;
  }
  for (const [index, candidate] of selected.entries()) {
    svg += `<polyline points="${points(candidate.frames.map(frame => frame[0])).join(' ')}" fill="none" stroke="${colors[index]}" stroke-width="1.7"/>`;
  }
  svg += `<path d="M${xAt(step)} ${plot.top}V${plot.top + plot.height}" stroke="#263e39" stroke-dasharray="4 3"/>`;
  const stepsPerDay = 86400 / archive.step_s;
  for (let day = 0; day <= totalSteps / stepsPerDay; day++) {
    svg += `<text x="${xAt(day * stepsPerDay)}" y="197" text-anchor="middle" font-size="11" fill="#697770">${t('hour', { hours: day * 24 })}</text>`;
  }
  el('chart').innerHTML = svg;
}

function render() {
  const frame = selected[0].frames[step];
  const minutes = step * archive.step_s / 60;
  const hour = String(Math.floor(minutes % 1440 / 60)).padStart(2, '0');
  const minute = String(minutes % 60).padStart(2, '0');
  el('rooms').innerHTML = selected.map(room).join('');
  el('clock').textContent = step === totalSteps ? t('endDay') : t('day', { day: Math.floor(minutes / 1440) + 1, time: `${hour}:${minute}` });
  el('phase').textContent = t(step === totalSteps ? 'phaseEnd' : step <= archive.prefix_steps ? 'phaseCommon' : 'phaseRecorded');
  el('weather').textContent = t('weather', { outdoor: fmt(frame[8], 1), solar: fmt(frame[9], 0) });
  el('start-hour').textContent = t('hour', { hours: 0 });
  el('end-hour').textContent = t('hour', { hours: totalSteps * archive.step_s / 3600 });
  el('position').textContent = t('position', { step, total: totalSteps });
  el('interval-note').textContent = intervalNote ? t(intervalNote) : '';
  timeline.value = step;
  renderChart();
}

function seek(frame) {
  stop();
  step = frame;
  intervalNote = '';
  render();
}

for (const link of languageLinks) {
  link.addEventListener('click', event => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    language = link.hreflang;
    const url = new URL(location.href);
    if (language === 'zh-CN') url.searchParams.set('lang', language);
    else url.searchParams.delete('lang');
    history.replaceState(history.state, '', url);
    renderLanguage();
  });
}

for (const id of ['episode', 'family', 'seed']) el(id).addEventListener('change', choose);
timeline.addEventListener('input', event => seek(Number(event.target.value)));
el('restart').addEventListener('click', () => seek(0));
el('play').addEventListener('click', () => {
  if (timer !== null) {
    stop();
    return;
  }
  if (step === totalSteps) step = 0;
  intervalNote = '';
  el('play').textContent = t('pause');
  render();
  timer = setInterval(() => {
    step = Math.min(totalSteps, step + Number(el('speed').value));
    render();
    if (step === totalSteps) stop();
  }, 250);
});
el('cold').addEventListener('click', () => {
  stop();
  const frames = selected[2].frames;
  const index = frames.findIndex((frame, i) => i > archive.prefix_steps && frame[4] > frames[i - 1][4] + 1e-9);
  if (index >= 0) step = index;
  intervalNote = index < 0 ? 'noCold' : 'coldNote';
  render();
});
el('chart').addEventListener('click', event => {
  const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(el('chart').getScreenCTM().inverse());
  const frame = Math.round((point.x - plot.left) / plot.width * totalSteps);
  seek(Math.max(0, Math.min(totalSteps, frame)));
});

renderLanguage();
try {
  const response = await fetch('./data.json');
  if (!response.ok) throw new Error(`Replay data returned HTTP ${response.status}`);
  archive = await response.json();
  choose();
  renderCounts();
  el('player').inert = false;
  el('player').setAttribute('aria-busy', 'false');
  el('player').classList.remove('loading');
} catch {
  archive = undefined;
  loadFailed = true;
  el('player').hidden = true;
  el('error').hidden = false;
  renderLanguage();
}
