/* Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0 */
'use strict';
const $ = id => document.getElementById(id);
const state = {directory: '.', parent: null, selected: new Set(), episodes: [], path: null,
  info: null, index: 0, requestedIndex: 0, frame: null, playing: false, timer: null,
  generation: 0, frameTicket: 0, seriesTicket: 0, points: []};
const node = (tag, text, cls) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (cls) element.className = cls;
  return element;
};
function status(message, error = false) {
  $('status').textContent = message;
  document.querySelector('footer').classList.toggle('error', error);
}
function pause() {
  state.playing = false;
  clearTimeout(state.timer);
  $('play').textContent = '▶ 播放';
}
function run(fn) {
  return async (...args) => {
    try { await fn(...args); }
    catch (error) { pause(); status(error.message, true); }
  };
}
async function api(route, params = {}) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    for (const item of Array.isArray(value) ? value : [value]) query.append(key, item);
  }
  const response = await fetch(`/api/${route}?${query}`);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `请求失败：${response.status}`);
  return result;
}
async function browse(path) {
  const data = await api('browse', {path});
  state.directory = data.path;
  state.parent = data.parent;
  $('directory').value = data.path;
  $('root').textContent = `工作区：${data.root}`;
  $('parent').disabled = data.parent === null;
  $('folders').replaceChildren();
  for (const directory of data.directories) {
    const button = node('button', directory.split('/').pop(), 'folder');
    button.title = directory;
    button.onclick = run(() => browse(directory));
    $('folders').append(button);
  }
  if (!data.directories.length) $('folders').append(node('p', '此目录没有子目录，可直接添加。', 'hint'));
  status('添加目录后点击「扫描轨迹」。数据保留在本机。');
}
function renderSelected() {
  $('selected').replaceChildren();
  $('selected-count').textContent = state.selected.size;
  for (const path of state.selected) {
    const row = node('div', undefined, 'selected-row');
    const remove = node('button', '×');
    remove.setAttribute('aria-label', `移除 ${path}`);
    remove.onclick = () => { state.selected.delete(path); renderSelected(); };
    row.append(node('span', path), remove);
    $('selected').append(row);
  }
}
async function scan() {
  if (!state.selected.size) throw new Error('请先添加至少一个目录。');
  $('scan').disabled = true;
  status('正在扫描所选目录…');
  try {
    const result = await api('scan', {path: [...state.selected]});
    state.episodes = result.trajectories;
    renderEpisodes();
    $('episode-count').textContent = state.episodes.length;
    status(result.warnings.length ? result.warnings.join('；') : `找到 ${state.episodes.length} 条轨迹。点击任意轨迹开始查看。`, !!result.warnings.length);
  } finally { $('scan').disabled = false; }
}
function renderEpisodes() {
  const filter = $('search').value.toLowerCase();
  $('episodes').replaceChildren();
  for (const episode of state.episodes.filter(item => item.path.toLowerCase().includes(filter))) {
    const button = node('button', undefined, `episode${episode.path === state.path ? ' active' : ''}`);
    const title = node('div', undefined, 'episode-title');
    title.append(node('span', `轨迹 ${episode.name.split('_')[1]}`), node('span', `${(episode.bytes / 1048576).toFixed(1)} MB`));
    button.append(title, node('small', episode.directory));
    button.title = episode.path;
    button.onclick = run(() => selectEpisode(episode.path));
    $('episodes').append(button);
  }
  if (!$('episodes').children.length) $('episodes').append(node('p', '没有匹配的 trajectory_*.pt 文件。', 'hint'));
}
function setControls(enabled) {
  for (const id of ['play', 'previous', 'next', 'timeline', 'batch', 'observation', 'plot-field', 'dimension']) $(id).disabled = !enabled;
}
async function selectEpisode(path) {
  pause();
  const generation = ++state.generation;
  ++state.frameTicket;
  ++state.seriesTicket;
  state.path = path;
  state.info = null;
  state.frame = null;
  state.points = [];
  state.index = state.requestedIndex = 0;
  setControls(false);
  $('values').replaceChildren();
  $('metadata-content').textContent = '';
  $('cameras').replaceChildren(node('div', '正在载入轨迹…', 'empty'));
  $('title').textContent = `轨迹 ${path.split('/').pop().split('_')[1]}`;
  $('trajectory-path').textContent = path;
  $('frame-badge').textContent = '载入中';
  drawPlot();
  renderEpisodes();
  status('正在按需读取轨迹…');
  let info;
  try { info = await api('info', {path}); }
  catch (error) {
    if (generation !== state.generation) return;
    $('cameras').replaceChildren(node('div', '无法读取这条轨迹，请选择其他文件。', 'empty'));
    $('frame-badge').textContent = '载入失败';
    throw error;
  }
  if (generation !== state.generation) return;
  state.info = info;
  $('batch').replaceChildren(...Array.from({length: info.batches}, (_, i) => new Option(String(i), String(i))));
  $('timeline').max = info.length - 1;
  $('timeline').value = 0;
  $('plot-field').replaceChildren(...info.plot_fields.map(field => new Option(field, field)));
  $('dimension').value = 0;
  setControls(true);
  await Promise.all([showFrame(0), loadSeries()]);
  if (generation === state.generation) status(`已载入 ${info.length} 帧 · ${info.batches} 个 batch · 空格播放，方向键逐帧查看。`);
}
async function showFrame(index) {
  if (!state.info) return;
  index = Math.max(0, Math.min(state.info.length - 1, index));
  state.requestedIndex = index;
  const ticket = ++state.frameTicket;
  const frame = await api('frame', {path: state.path, index, batch: $('batch').value, observation: $('observation').value});
  if (ticket !== state.frameTicket) return;
  // Decode every camera before swapping images and numbers together.
  const cameraNodes = await Promise.all(Object.entries(frame.images).map(async ([name, source]) => {
    const card = node('div', undefined, 'camera');
    const image = new Image();
    image.alt = name;
    image.src = source;
    await image.decode();
    const viewport = node('div', undefined, 'camera-viewport');
    viewport.append(image);
    card.append(viewport, node('div', `${name} · ${image.naturalWidth}×${image.naturalHeight} · 无损`, 'camera-label'));
    return card;
  }));
  if (ticket !== state.frameTicket) return;
  state.index = index;
  state.frame = frame;
  $('cameras').replaceChildren(...cameraNodes);
  if (!cameraNodes.length) $('cameras').append(node('div', '此观测没有可显示的 RGB 图像。非图像数据仍可查看。', 'empty'));
  $('timeline').value = index;
  $('frame-badge').textContent = `FRAME ${String(index).padStart(4, '0')}`;
  $('position').textContent = `帧 ${index} / ${state.info.length - 1}`;
  const fps = readFps();
  $('time').textContent = `${(index / fps).toFixed(2)} / ${((state.info.length - 1) / fps).toFixed(2)} s · 按 ${fps} FPS 估算`;
  renderValues();
  $('metadata-content').textContent = JSON.stringify({metadata: frame.metadata, fields: state.info.fields}, null, 2);
  drawPlot();
}
function formatValue(value) {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : Number(value.toFixed(6)).toString();
  if (Array.isArray(value)) {
    if (value.every(x => !Array.isArray(x) && typeof x !== 'object')) return value.map((x, i) => `[${i}] ${formatValue(x)}`).join('   ');
    return value.map(formatValue).join('\n');
  }
  return typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
}
function renderValues() {
  if (!state.frame) return;
  const filter = $('field-filter').value.toLowerCase();
  $('values').replaceChildren();
  const priority = name => name === 'actions' ? 0 : name.startsWith('curr_obs/') ? 1 : name === 'rewards' ? 2 : 3;
  const fields = Object.entries(state.frame.values).sort(([a], [b]) => priority(a) - priority(b));
  for (const [name, value] of fields) {
    if (!name.toLowerCase().includes(filter)) continue;
    const card = node('div', undefined, 'field');
    const schema = state.info.fields[name];
    card.append(node('div', name, 'field-name'));
    if (schema) card.append(node('div', `${schema.dtype} · 原始形状 [${schema.shape.join(', ')}]`, 'field-shape'));
    card.append(node('pre', formatValue(value)));
    $('values').append(card);
  }
}
function readFps() {
  const value = Number($('fps').value);
  return Number.isFinite(value) && value > 0 ? Math.min(120, value) : 10;
}
async function tick() {
  if (!state.playing || !state.info) return;
  const generation = state.generation;
  const start = performance.now();
  let next = state.index + 1;
  if (next >= state.info.length) {
    if ($('loop').checked) next = 0;
    else { pause(); return; }
  }
  await showFrame(next);
  if (state.playing && generation === state.generation) state.timer = setTimeout(run(tick), Math.max(0, 1000 / readFps() / Number($('speed').value) - (performance.now() - start)));
}
async function togglePlay() {
  if (!state.info) return;
  if (state.playing) { pause(); return; }
  const generation = state.generation;
  if (state.index >= state.info.length - 1) await showFrame(0);
  if (generation !== state.generation) return;
  state.playing = true;
  $('play').textContent = 'Ⅱ 暂停';
  state.timer = setTimeout(run(tick), 1000 / readFps() / Number($('speed').value));
}
async function seek(index) { pause(); await showFrame(index); }
async function loadSeries() {
  const ticket = ++state.seriesTicket;
  state.points = [];
  drawPlot();
  if (!state.info || !$('plot-field').value) return;
  const data = await api('series', {path: state.path, field: $('plot-field').value,
    batch: $('batch').value, dimension: $('dimension').value});
  if (ticket !== state.seriesTicket) return;
  state.points = data.points;
  $('dimension').max = data.dimensions - 1;
  $('plot-hint').textContent = `维度 0–${data.dimensions - 1} · ${data.points.length} 个采样点 · 点击曲线跳转到对应帧`;
  drawPlot();
}
function drawPlot() {
  const canvas = $('plot');
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, rect.width * scale);
  canvas.height = rect.height * scale;
  const ctx = canvas.getContext('2d');
  ctx.scale(scale, scale);
  const width = rect.width, height = rect.height, left = 60, top = 15, bottom = height - 22;
  ctx.font = '10px monospace';
  $('plot-value').textContent = '';
  const valid = state.points.filter(([, y]) => typeof y === 'number' || typeof y === 'boolean');
  if (!valid.length || !state.info) return;
  let low = Math.min(...valid.map(([, y]) => Number(y))), high = Math.max(...valid.map(([, y]) => Number(y)));
  if (high === low) { high += 0.5; low -= 0.5; }
  const x = i => left + (width - left - 12) * i / Math.max(1, state.info.length - 1);
  const y = v => bottom - (Number(v) - low) / (high - low) * (bottom - top);
  for (let i = 0; i <= 3; i++) {
    const value = low + (high - low) * i / 3;
    ctx.strokeStyle = '#263141'; ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(width, y(value)); ctx.stroke();
    ctx.fillStyle = '#8d9caf'; ctx.fillText(value.toPrecision(3), 0, y(value) + 3);
  }
  ctx.beginPath(); ctx.strokeStyle = '#63dac5'; ctx.lineWidth = 1.5;
  let connected = false;
  for (const [index, value] of state.points) {
    if (typeof value !== 'number' && typeof value !== 'boolean') { connected = false; continue; }
    if (connected) ctx.lineTo(x(index), y(value)); else ctx.moveTo(x(index), y(value));
    connected = true;
  }
  ctx.stroke();
  ctx.strokeStyle = '#e9c888'; ctx.beginPath(); ctx.moveTo(x(state.index), top); ctx.lineTo(x(state.index), bottom); ctx.stroke();
  ctx.fillStyle = '#8d9caf'; ctx.fillText('0', left, height - 3); ctx.fillText(String(state.info.length - 1), width - 35, height - 3);
  const current = state.points.find(([index]) => index === state.index);
  if (current) $('plot-value').textContent = `帧 ${state.index} · ${formatValue(current[1])}`;
}
$('browse-form').onsubmit = run(async event => { event.preventDefault(); await browse($('directory').value || '.'); });
$('parent').onclick = run(() => browse(state.parent));
$('add-directory').onclick = () => { state.selected.add(state.directory); renderSelected(); };
$('scan').onclick = run(scan);
$('search').oninput = renderEpisodes;
$('field-filter').oninput = renderValues;
$('play').onclick = run(togglePlay);
$('previous').onclick = run(() => seek(state.requestedIndex - 1));
$('next').onclick = run(() => seek(state.requestedIndex + 1));
$('timeline').oninput = run(() => seek(Number($('timeline').value)));
$('observation').onchange = run(() => seek(state.index));
$('image-scale').onchange = () => $('cameras').classList.toggle('original-size', $('image-scale').value === 'original');
$('batch').onchange = run(async () => { pause(); await Promise.all([showFrame(state.index), loadSeries()]); });
$('plot-field').onchange = run(async () => { $('dimension').value = 0; await loadSeries(); });
$('dimension').onchange = run(loadSeries);
$('fps').onchange = run(() => { $('fps').value = readFps(); return state.info ? showFrame(state.index) : undefined; });
$('plot').onclick = run(event => {
  if (!state.info) return;
  const rect = $('plot').getBoundingClientRect();
  return seek(Math.round((event.clientX - rect.left - 60) / (rect.width - 72) * (state.info.length - 1)));
});
window.addEventListener('resize', drawPlot);
document.addEventListener('keydown', run(async event => {
  if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(event.target.tagName) || !state.info) return;
  if (event.code === 'Space') { event.preventDefault(); await togglePlay(); }
  else if (event.code === 'ArrowLeft') { event.preventDefault(); await seek(state.requestedIndex - 1); }
  else if (event.code === 'ArrowRight') { event.preventDefault(); await seek(state.requestedIndex + 1); }
}));
setControls(false);
run(() => browse('.'))();
