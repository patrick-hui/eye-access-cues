// Eye Access Cues: webcam eye tracking mapped to the NLP eye accessing cue chart.
// Face and iris landmarks come from MediaPipe Face Landmarker, running fully in the browser.

const MP_VERSION = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// Scale factors so a deliberate glance reads as roughly 1.0 on each axis.
const IRIS_X_RANGE = 0.1;
const IRIS_Y_RANGE = 0.06;
const BLEND_X_RANGE = 0.4;
const BLEND_Y_RANGE = 0.6;
const BLINK_LIMIT = 0.5;
const CALIBRATION_MS = 1500;
const HISTORY_LIMIT = 40;

// Face mesh landmark indices (subject's anatomy: "right" is the subject's own right eye).
const LM = {
  rOuter: 33, rInner: 133, rIris: 468,
  lOuter: 263, lInner: 362, lIris: 473,
  forehead: 10, chin: 152,
};
const R_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const L_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];

const MODES = {
  recall: 'Recalling',
  construct: 'Constructing',
  selftalk: 'Self-talk',
  feeling: 'Feeling',
  either: 'Recall or construct',
  none: 'No access cue',
};

// Keys are row (U/M/D) + column (L/C/R) from the subject's own point of view,
// for the standard (right-handed) organisation.
const CUES = {
  UL: { code: 'Vr', system: 'visual', mode: 'recall', name: 'Visual remembered',
        desc: 'Recalling images seen before.' },
  UR: { code: 'Vc', system: 'visual', mode: 'construct', name: 'Visual constructed',
        desc: 'Creating images never seen before.' },
  ML: { code: 'Ar', system: 'auditory', mode: 'recall', name: 'Auditory remembered',
        desc: 'Recalling sounds and voices heard before.' },
  MR: { code: 'Ac', system: 'auditory', mode: 'construct', name: 'Auditory constructed',
        desc: 'Creating sounds never heard before.' },
  DL: { code: 'Ad', system: 'digital', mode: 'selftalk', name: 'Auditory digital',
        desc: 'Internal dialogue: talking things over with yourself.' },
  DR: { code: 'K', system: 'kinesthetic', mode: 'feeling', name: 'Kinesthetic',
        desc: 'Accessing feelings, emotions and body sensations.' },
  UC: { code: 'V', system: 'visual', mode: 'either', name: 'Visual', minor: true,
        desc: 'Looking straight up: visualising, recalled or constructed.' },
  MC: { code: '·', system: 'none', mode: 'none', name: 'Centred', minor: true,
        desc: 'Eyes forward. A fixed, defocused stare is also read as visualising.' },
  DC: { code: '↓', system: 'none', mode: 'none', name: 'Down, centre', minor: true,
        desc: 'Between feelings and self-talk. Watch which side the eyes settle on.' },
};

const PROMPTS = {
  Vr: ['What colour was the front door of your childhood home?',
       'Picture the face of your first teacher.',
       'What did your bedroom look like when you were ten?'],
  Vc: ['Imagine a purple lion with green spots.',
       'Picture your home with the roof painted gold.',
       'What would you look like with bright blue hair?'],
  Ar: ['Hear the ringtone of your phone.',
       'How does your best friend’s laugh sound?',
       'Hear the opening bars of your favourite song.'],
  Ac: ['Imagine a dog that meows like a cat.',
       'Hear your name sung by a choir of robots.',
       'What would your voice sound like at double speed?'],
  Ad: ['What do you say to yourself when something goes wrong?',
       'Recite the alphabet backwards in your head.',
       'Silently talk through your plans for tomorrow.'],
  K: ['How does it feel to step into a warm bath?',
      'Recall the feeling of cold wind on your face.',
      'Feel the texture of a soft wool jumper in your hands.'],
};

const ROW_WORDS = { U: 'Up', M: 'Level', D: 'Down' };
const COL_WORDS = { L: "subject's left", C: 'centre', R: "subject's right" };

const $ = (sel) => document.querySelector(sel);
const els = {
  status: $('#status'), video: $('#video'), overlay: $('#overlay'), mirror: $('#mirror'),
  badge: $('#badge'), placeholder: $('#placeholder'), secureHint: $('#secureHint'),
  startBtn: $('#startBtn'), stopBtn: $('#stopBtn'), frontBtn: $('#frontBtn'), backBtn: $('#backBtn'),
  cameraSelect: $('#cameraSelect'), mirrorToggle: $('#mirrorToggle'),
  calibrateBtn: $('#calibrateBtn'), calibNote: $('#calibNote'),
  readout: $('#readout'), roCode: $('#roCode'), roMode: $('#roMode'), roName: $('#roName'),
  roDesc: $('#roDesc'), roDir: $('#roDir'), grid: $('#grid'), gridPerspective: $('#gridPerspective'),
  pad: $('#pad'), numX: $('#numX'), numY: $('#numY'), numFps: $('#numFps'),
  txRange: $('#txRange'), tyRange: $('#tyRange'), smoothRange: $('#smoothRange'), holdRange: $('#holdRange'),
  txOut: $('#txOut'), tyOut: $('#tyOut'), smoothOut: $('#smoothOut'), holdOut: $('#holdOut'),
  reverseToggle: $('#reverseToggle'), history: $('#history'), historyEmpty: $('#historyEmpty'),
  clearBtn: $('#clearBtn'), promptBtns: $('#promptBtns'), promptText: $('#promptText'),
};

const state = {
  stream: null,
  devices: [],
  facing: 'user',
  mirrored: true,
  mirrorOverridden: false,
  landmarker: null,
  loadingModel: null,
  tracking: false,
  lastVideoTime: -1,
  lastTs: 0,
  fps: { frames: 0, since: performance.now() },
  baseline: { x: 0, y: 0 },
  calibration: null,
  gaze: { x: 0, y: 0 },
  trail: [],
  zone: { row: 'M', col: 'C' },
  candidate: 'MC',
  candidateSince: 0,
  current: 'MC',
  currentSince: performance.now(),
  history: [],
  faceVisible: false,
  settings: { tx: 0.35, ty: 0.35, smoothing: 0.5, holdMs: 150, reversed: false },
};

/* ---------- settings ---------- */

const SETTINGS_KEY = 'eye-access-cues:settings';

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || 'null');
    if (saved && typeof saved === 'object') Object.assign(state.settings, saved);
  } catch { /* storage unavailable: keep defaults */ }
}

function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings)); } catch { /* ignore */ }
}

function syncSettingsUI() {
  const s = state.settings;
  els.txRange.value = s.tx; els.txOut.textContent = s.tx.toFixed(2);
  els.tyRange.value = s.ty; els.tyOut.textContent = s.ty.toFixed(2);
  els.smoothRange.value = s.smoothing; els.smoothOut.textContent = s.smoothing.toFixed(2);
  els.holdRange.value = s.holdMs; els.holdOut.textContent = `${s.holdMs} ms`;
  els.reverseToggle.checked = s.reversed;
}

function bindSettings() {
  const bind = (input, key, parse) => input.addEventListener('input', () => {
    state.settings[key] = parse(input.value);
    syncSettingsUI();
    saveSettings();
    if (key === 'reversed') renderGrid();
  });
  bind(els.txRange, 'tx', Number);
  bind(els.tyRange, 'ty', Number);
  bind(els.smoothRange, 'smoothing', Number);
  bind(els.holdRange, 'holdMs', Number);
  els.reverseToggle.addEventListener('change', () => {
    state.settings.reversed = els.reverseToggle.checked;
    saveSettings();
    renderGrid();
    showCue(state.current);
  });
}

/* ---------- cue chart ---------- */

// Reversed organisation mirrors the chart: recall and construct swap sides.
function cueFor(key) {
  if (!state.settings.reversed) return CUES[key];
  const swap = { L: 'R', R: 'L', C: 'C' };
  return CUES[key[0] + swap[key[1]]];
}

// Screen columns follow the video: a mirrored selfie shows the subject's right on the right;
// an unmirrored (observer) view shows the subject's right on the left.
function screenColumns() {
  return state.mirrored ? ['L', 'C', 'R'] : ['R', 'C', 'L'];
}

function directionText(key) {
  const row = ROW_WORDS[key[0]];
  const col = COL_WORDS[key[1]];
  if (key === 'MC') return 'Eyes forward';
  if (key[1] === 'C') return `${row}, centre`;
  return `${row}, to the ${col}`;
}

function renderGrid() {
  const cols = screenColumns();
  els.grid.innerHTML = '';
  for (const row of ['U', 'M', 'D']) {
    for (const col of cols) {
      const key = row + col;
      const cue = cueFor(key);
      const cell = document.createElement('div');
      cell.className = 'cell' + (cue.minor ? ' minor' : '') + (key === state.current ? ' on' : '');
      cell.setAttribute('role', 'listitem');
      cell.dataset.key = key;
      cell.dataset.system = cue.system;
      cell.dataset.mode = cue.mode;
      cell.innerHTML = `
        <span class="c-code"></span>
        <span class="c-name"></span>
        <span class="c-mode"></span>
        <span class="c-pos"></span>`;
      cell.querySelector('.c-code').textContent = cue.code;
      cell.querySelector('.c-name').textContent = cue.name;
      cell.querySelector('.c-mode').textContent = MODES[cue.mode];
      cell.querySelector('.c-pos').textContent = directionText(key);
      els.grid.append(cell);
    }
  }
  els.gridPerspective.textContent = state.mirrored
    ? "Mirror view: boxes match the subject's own left and right."
    : "Observer view: the subject's left appears on your right, as when facing them.";
}

function showCue(key) {
  const cue = cueFor(key);
  for (const cell of els.grid.children) cell.classList.toggle('on', cell.dataset.key === key);
  els.readout.dataset.system = cue.system;
  els.roCode.textContent = cue.code;
  els.roMode.textContent = MODES[cue.mode];
  els.roName.textContent = cue.name;
  els.roDesc.textContent = cue.desc;
  els.roDir.textContent = directionText(key);
  els.badge.dataset.system = cue.system;
  els.badge.textContent = key === 'MC' ? 'Centred' : `${cue.code} · ${MODES[cue.mode]} · ${cue.name}`;
}

/* ---------- classification ---------- */

// Hysteresis: entering a side needs the full threshold, leaving it only 80%.
function axisZone(v, t, prev, pos, neg, mid) {
  const exit = t * 0.8;
  if (prev === pos && v > exit) return pos;
  if (prev === neg && v < -exit) return neg;
  if (v > t) return pos;
  if (v < -t) return neg;
  return mid;
}

function classify(now) {
  const { tx, ty, holdMs } = state.settings;
  const col = axisZone(state.gaze.x, tx, state.zone.col, 'R', 'L', 'C');
  const row = axisZone(state.gaze.y, ty, state.zone.row, 'U', 'D', 'M');
  state.zone = { row, col };
  const key = row + col;

  if (key !== state.candidate) {
    state.candidate = key;
    state.candidateSince = now;
  }
  if (key !== state.current && now - state.candidateSince >= holdMs) commit(key, now);
}

function commit(key, now) {
  const open = state.history[0];
  if (open && open.end == null) open.end = now;
  state.current = key;
  state.currentSince = now;
  if (key !== 'MC') {
    state.history.unshift({ key, cue: cueFor(key), start: now, end: null, at: new Date() });
    state.history.length = Math.min(state.history.length, HISTORY_LIMIT);
  }
  showCue(key);
  renderHistory();
}

function renderHistory() {
  els.historyEmpty.hidden = state.history.length > 0;
  els.history.innerHTML = '';
  for (const h of state.history) {
    const li = document.createElement('li');
    li.dataset.system = h.cue.system;
    const dur = h.end == null ? 'now' : `${((h.end - h.start) / 1000).toFixed(1)} s`;
    li.innerHTML = '<span class="h-time"></span><span class="h-code"></span><span class="h-name"></span><span class="h-dur"></span>';
    li.querySelector('.h-time').textContent = h.at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    li.querySelector('.h-code').textContent = h.cue.code;
    li.querySelector('.h-name').textContent = `${h.cue.name} · ${MODES[h.cue.mode]}`;
    li.querySelector('.h-dur').textContent = dur;
    els.history.append(li);
  }
}

/* ---------- gaze estimation ---------- */

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const dot = (a, b) => a.x * b.x + a.y * b.y;
const len = (a) => Math.hypot(a.x, a.y);
const unit = (a) => { const l = len(a) || 1; return { x: a.x / l, y: a.y / l }; };
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

// Returns gaze in face-relative units: x > 0 is the subject's right, y > 0 is up.
// Each axis blends the iris offset from the eye-corner midpoint (measured along the face's
// own axes, so head roll doesn't leak in) with the model's eyeLook* blendshapes.
function estimateGaze(landmarks, blendshapes, W, H) {
  const P = (i) => ({ x: landmarks[i].x * W, y: landmarks[i].y * H });
  const right = unit(sub(P(LM.rOuter), P(LM.lOuter)));
  let up = { x: right.y, y: -right.x };
  if (dot(up, sub(P(LM.forehead), P(LM.chin))) < 0) up = { x: -up.x, y: -up.y };

  const eye = (outer, inner, iris) => {
    const o = P(outer), n = P(inner);
    const w = len(sub(n, o)) || 1;
    const d = sub(P(iris), mid(o, n));
    return { h: dot(d, right) / w, v: dot(d, up) / w };
  };
  const re = eye(LM.rOuter, LM.rInner, LM.rIris);
  const le = eye(LM.lOuter, LM.lInner, LM.lIris);
  const irisX = (re.h + le.h) / 2;
  const irisY = (re.v + le.v) / 2;

  const bs = {};
  for (const c of blendshapes?.categories ?? []) bs[c.categoryName] = c.score;
  const lookUp = ((bs.eyeLookUpLeft ?? 0) + (bs.eyeLookUpRight ?? 0)) / 2;
  const lookDown = ((bs.eyeLookDownLeft ?? 0) + (bs.eyeLookDownRight ?? 0)) / 2;
  const blink = ((bs.eyeBlinkLeft ?? 0) + (bs.eyeBlinkRight ?? 0)) / 2;
  // Subject's right = left eye looking in + right eye looking out.
  const lookRight = ((bs.eyeLookInLeft ?? 0) + (bs.eyeLookOutRight ?? 0)) / 2;
  const lookLeft = ((bs.eyeLookOutLeft ?? 0) + (bs.eyeLookInRight ?? 0)) / 2;

  const x = blendshapes
    ? 0.5 * ((lookRight - lookLeft) / BLEND_X_RANGE) + 0.5 * (irisX / IRIS_X_RANGE)
    : irisX / IRIS_X_RANGE;
  const y = blendshapes
    ? 0.5 * ((lookUp - lookDown) / BLEND_Y_RANGE) + 0.5 * (irisY / IRIS_Y_RANGE)
    : irisY / IRIS_Y_RANGE;
  return { x, y, blink };
}

function updateGaze(raw, now) {
  if (state.calibration) {
    state.calibration.samples.push(raw);
    const left = Math.ceil((CALIBRATION_MS - (now - state.calibration.start)) / 1000);
    els.calibNote.textContent = `Hold still and look into the lens… ${Math.max(left, 1)}`;
    if (now - state.calibration.start >= CALIBRATION_MS) finishCalibration();
  }
  const a = state.settings.smoothing;
  const tx = raw.x - state.baseline.x;
  const ty = raw.y - state.baseline.y;
  state.gaze.x = a * state.gaze.x + (1 - a) * tx;
  state.gaze.y = a * state.gaze.y + (1 - a) * ty;
  pushTrail();
}

function pushTrail() {
  state.trail.push({ x: state.gaze.x, y: state.gaze.y });
  if (state.trail.length > 24) state.trail.shift();
}

function startCalibration() {
  if (!state.tracking) return;
  state.calibration = { start: performance.now(), samples: [] };
  els.calibNote.classList.remove('done');
  els.calibrateBtn.disabled = true;
}

function finishCalibration() {
  const { samples } = state.calibration;
  state.calibration = null;
  els.calibrateBtn.disabled = !state.tracking;
  if (samples.length < 5) {
    els.calibNote.textContent = 'Calibration needs a visible face. Move into frame and try again.';
    return;
  }
  const avg = (k) => samples.reduce((s, p) => s + p[k], 0) / samples.length;
  state.baseline = { x: avg('x'), y: avg('y') };
  state.gaze = { x: 0, y: 0 };
  state.trail = [];
  els.calibNote.textContent = 'Calibrated. Ask a question and watch the boxes.';
  els.calibNote.classList.add('done');
}

/* ---------- drawing ---------- */

let colors = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  colors = { line: v('--line'), muted: v('--muted'), fg: v('--fg'), accent: v('--accent'),
             visual: v('--vis'), auditory: v('--aud'), kinesthetic: v('--kin'), digital: v('--dig'), none: v('--none') };
}

function drawOverlay(landmarks) {
  const c = els.overlay;
  const W = els.video.videoWidth, H = els.video.videoHeight;
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  if (!landmarks) return;
  const color = colors[cueFor(state.current).system] || '#3db5bd';
  const P = (i) => [landmarks[i].x * W, landmarks[i].y * H];
  const lw = Math.max(1.5, W / 400);

  ctx.lineWidth = lw;
  ctx.strokeStyle = 'rgba(255,255,255,0.75)';
  for (const ring of [R_EYE_RING, L_EYE_RING]) {
    ctx.beginPath();
    ring.forEach((i, n) => { const [x, y] = P(i); n ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath();
    ctx.stroke();
  }
  for (const [center, edge] of [[468, 469], [473, 474]]) {
    const [cx, cy] = P(center);
    const [ex, ey] = P(edge);
    const r = Math.max(Math.hypot(ex - cx, ey - cy), 2);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = lw * 1.6;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, lw * 1.2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

const PAD_RANGE = 2; // pad edge = 2x a full glance
function drawPad() {
  const c = els.pad;
  const dpr = window.devicePixelRatio || 1;
  const size = Math.round(c.clientWidth * dpr) || 240;
  if (c.width !== size) { c.width = size; c.height = size; }
  const ctx = c.getContext('2d');
  const S = size, half = S / 2;
  const toPx = (gx, gy) => [half + screenX(gx) / PAD_RANGE * half, half - gy / PAD_RANGE * half];
  ctx.clearRect(0, 0, S, S);

  ctx.strokeStyle = colors.line;
  ctx.lineWidth = dpr;
  ctx.beginPath();
  ctx.moveTo(half, 0); ctx.lineTo(half, S); ctx.moveTo(0, half); ctx.lineTo(S, half);
  ctx.stroke();

  const { tx, ty } = state.settings;
  const bw = tx / PAD_RANGE * half, bh = ty / PAD_RANGE * half;
  ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.strokeStyle = colors.muted;
  ctx.strokeRect(half - bw, half - bh, bw * 2, bh * 2);
  ctx.setLineDash([]);

  ctx.fillStyle = colors.muted;
  ctx.font = `${11 * dpr}px "IBM Plex Mono", monospace`;
  ctx.textAlign = 'center';
  ctx.fillText('UP', half, 13 * dpr);
  ctx.fillText('DOWN', half, S - 6 * dpr);

  const color = colors[cueFor(state.current).system] || colors.accent;
  state.trail.forEach((p, i) => {
    const [x, y] = toPx(clamp(p.x), clamp(p.y));
    ctx.globalAlpha = (i + 1) / state.trail.length * 0.35;
    ctx.beginPath(); ctx.arc(x, y, 3 * dpr, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  });
  ctx.globalAlpha = 1;
  const [gx, gy] = toPx(clamp(state.gaze.x), clamp(state.gaze.y));
  ctx.beginPath(); ctx.arc(gx, gy, 8 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
  ctx.lineWidth = 2 * dpr; ctx.strokeStyle = colors.fg; ctx.stroke();

  els.numX.textContent = signed(state.gaze.x);
  els.numY.textContent = signed(state.gaze.y);
}

const clamp = (v) => Math.max(-PAD_RANGE, Math.min(PAD_RANGE, v));
const signed = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
// Subject's right is on screen right only in a mirrored view.
const screenX = (gx) => (state.mirrored ? gx : -gx);

/* ---------- pad simulation (camera off) ---------- */

function bindPad() {
  let dragging = false;
  let holdTimer = 0;
  const move = (e) => {
    if (state.tracking) return;
    const r = els.pad.getBoundingClientRect();
    const sx = ((e.clientX - r.left) / r.width - 0.5) * 2 * PAD_RANGE;
    const sy = -((e.clientY - r.top) / r.height - 0.5) * 2 * PAD_RANGE;
    state.gaze = { x: state.mirrored ? sx : -sx, y: sy };
    pushTrail();
    classify(performance.now());
    drawPad();
    // No frame loop runs while the camera is off, so re-check once the hold time has passed.
    clearTimeout(holdTimer);
    holdTimer = setTimeout(() => { classify(performance.now()); drawPad(); }, state.settings.holdMs + 20);
  };
  els.pad.addEventListener('pointerdown', (e) => {
    if (state.tracking) return;
    dragging = true;
    els.pad.setPointerCapture(e.pointerId);
    move(e);
  });
  els.pad.addEventListener('pointermove', (e) => { if (dragging) move(e); });
  els.pad.addEventListener('pointerup', () => { dragging = false; });
  els.pad.addEventListener('pointercancel', () => { dragging = false; });
}

/* ---------- camera ---------- */

function setStatus(text, tone = 'idle') {
  els.status.textContent = text;
  els.status.dataset.tone = tone;
}

const BACK_RE = /back|rear|environment|world/i;
const FRONT_RE = /front|user|face|selfie|facetime/i;

function stopStream() {
  if (state.stream) for (const t of state.stream.getTracks()) t.stop();
  state.stream = null;
}

async function openCamera({ deviceId, facing }) {
  stopStream();
  const base = { width: { ideal: 1280 }, height: { ideal: 720 } };
  const attempts = deviceId
    ? [{ ...base, deviceId: { exact: deviceId } }]
    : [{ ...base, facingMode: { exact: facing } }, { ...base, facingMode: facing }];
  let lastErr;
  for (const video of attempts) {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      break;
    } catch (err) {
      lastErr = err;
      if (err.name === 'NotAllowedError') break;
    }
  }
  if (!state.stream) throw lastErr;

  const track = state.stream.getVideoTracks()[0];
  const settings = track.getSettings ? track.getSettings() : {};
  state.facing = settings.facingMode
    || (BACK_RE.test(track.label) ? 'environment' : FRONT_RE.test(track.label) ? 'user' : facing || 'user');
  if (!state.mirrorOverridden) state.mirrored = state.facing !== 'environment';

  els.video.srcObject = state.stream;
  await els.video.play();
  await refreshDevices(settings.deviceId);
  applyMirror();
}

async function refreshDevices(activeId) {
  const all = await navigator.mediaDevices.enumerateDevices();
  state.devices = all.filter((d) => d.kind === 'videoinput');
  els.cameraSelect.innerHTML = '';
  state.devices.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera ${i + 1}`;
    if (d.deviceId === activeId) opt.selected = true;
    els.cameraSelect.append(opt);
  });
  els.cameraSelect.disabled = state.devices.length < 2;
  const hasBack = state.devices.length > 1 || state.devices.some((d) => BACK_RE.test(d.label));
  els.backBtn.disabled = !hasBack;
  els.frontBtn.disabled = false;
  els.frontBtn.setAttribute('aria-pressed', String(state.facing !== 'environment'));
  els.backBtn.setAttribute('aria-pressed', String(state.facing === 'environment'));
  els.backBtn.title = hasBack ? '' : 'No second camera found on this device';
}

// Pick a device for the requested side: labelled match first, then facingMode, then "the other camera".
async function switchFacing(facing) {
  const re = facing === 'environment' ? BACK_RE : FRONT_RE;
  const labelled = state.devices.find((d) => re.test(d.label));
  const currentId = state.stream?.getVideoTracks()[0]?.getSettings?.().deviceId;
  state.mirrorOverridden = false;
  try {
    if (labelled) await openCamera({ deviceId: labelled.deviceId });
    else await openCamera({ facing });
  } catch {
    const other = state.devices.find((d) => d.deviceId !== currentId);
    if (!other) throw new Error('No other camera available');
    await openCamera({ deviceId: other.deviceId });
    state.facing = facing;
    if (!state.mirrorOverridden) state.mirrored = facing !== 'environment';
    await refreshDevices(other.deviceId);
    applyMirror();
  }
  resetForNewView();
}

function applyMirror() {
  els.mirror.classList.toggle('flip', state.mirrored);
  els.mirrorToggle.checked = state.mirrored;
  renderGrid();
  showCue(state.current);
}

function resetForNewView() {
  state.baseline = { x: 0, y: 0 };
  state.gaze = { x: 0, y: 0 };
  state.trail = [];
  els.calibNote.classList.remove('done');
  els.calibNote.innerHTML = 'New view: look straight into the lens and press <strong>Calibrate centre</strong>.';
}

async function loadModel() {
  if (state.landmarker) return state.landmarker;
  if (!state.loadingModel) {
    state.loadingModel = (async () => {
      setStatus('Loading face model…', 'busy');
      const { FaceLandmarker, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      const make = (delegate) => FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
      });
      try {
        state.landmarker = await make('GPU');
      } catch {
        state.landmarker = await make('CPU');
      }
      return state.landmarker;
    })();
  }
  try {
    return await state.loadingModel;
  } catch (err) {
    state.loadingModel = null;
    throw err;
  }
}

async function start() {
  els.startBtn.disabled = true;
  try {
    setStatus('Waiting for camera permission…', 'busy');
    await openCamera({ facing: state.facing });
    els.placeholder.hidden = true;
    await loadModel();
    state.tracking = true;
    els.stopBtn.disabled = false;
    els.calibrateBtn.disabled = false;
    els.badge.hidden = false;
    setStatus('Looking for a face…', 'busy');
    resetForNewView();
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    stopStream();
    els.placeholder.hidden = false;
    els.startBtn.disabled = false;
    els.startBtn.textContent = 'Try again';
    setStatus(cameraErrorText(err), 'warn');
  }
}

function stop() {
  state.tracking = false;
  state.calibration = null;
  stopStream();
  els.video.srcObject = null;
  els.overlay.getContext('2d').clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.placeholder.hidden = false;
  els.startBtn.disabled = false;
  els.startBtn.textContent = 'Start camera';
  els.stopBtn.disabled = true;
  els.calibrateBtn.disabled = true;
  els.badge.hidden = true;
  els.numFps.textContent = '–';
  setStatus('Camera off');
}

function cameraErrorText(err) {
  switch (err?.name) {
    case 'NotAllowedError': return 'Camera blocked. Allow camera access in the browser settings.';
    case 'NotFoundError': case 'OverconstrainedError': return 'No matching camera found.';
    case 'NotReadableError': return 'Camera is in use by another app.';
    default: return err?.message?.includes('fetch') || err?.message?.includes('import')
      ? 'Could not download the face model. Check the internet connection.'
      : `Could not start: ${err?.message || err}`;
  }
}

/* ---------- main loop ---------- */

function loop() {
  if (!state.tracking) return;
  const v = els.video;
  if (v.readyState >= 2 && v.currentTime !== state.lastVideoTime && state.landmarker) {
    state.lastVideoTime = v.currentTime;
    let ts = performance.now();
    if (ts <= state.lastTs) ts = state.lastTs + 1;
    state.lastTs = ts;
    let res;
    try {
      res = state.landmarker.detectForVideo(v, ts);
    } catch (err) {
      console.warn(err);
    }
    handleResult(res, ts);
    tickFps(ts);
  }
  drawPad();
  requestAnimationFrame(loop);
}

function handleResult(res, now) {
  const landmarks = res?.faceLandmarks?.[0];
  if (!landmarks || landmarks.length < 478) {
    if (state.faceVisible || els.status.dataset.tone !== 'warn') setStatus('No face found. Move closer, face the light.', 'warn');
    state.faceVisible = false;
    drawOverlay(null);
    return;
  }
  if (!state.faceVisible) setStatus('Tracking', 'live');
  state.faceVisible = true;

  const raw = estimateGaze(landmarks, res.faceBlendshapes?.[0], els.video.videoWidth, els.video.videoHeight);
  if (raw.blink < BLINK_LIMIT) {
    updateGaze(raw, now);
    classify(now);
  }
  drawOverlay(landmarks);
}

function tickFps(now) {
  const f = state.fps;
  f.frames++;
  if (now - f.since >= 1000) {
    els.numFps.textContent = `${Math.round(f.frames * 1000 / (now - f.since))} fps`;
    f.frames = 0;
    f.since = now;
  }
}

/* ---------- prompts ---------- */

function renderPromptButtons() {
  const order = ['Vr', 'Vc', 'Ar', 'Ac', 'Ad', 'K'];
  const byCode = Object.fromEntries(Object.values(CUES).map((c) => [c.code, c]));
  for (const code of order) {
    const cue = byCode[code];
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.system = cue.system;
    b.textContent = `${code} · ${cue.name}`;
    b.addEventListener('click', () => {
      const list = PROMPTS[code];
      els.promptText.textContent = list[Math.floor(Math.random() * list.length)];
    });
    els.promptBtns.append(b);
  }
}

/* ---------- boot ---------- */

function init() {
  loadSettings();
  syncSettingsUI();
  bindSettings();
  readColors();
  renderGrid();
  showCue('MC');
  renderHistory();
  renderPromptButtons();
  bindPad();
  drawPad();

  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  mq.addEventListener?.('change', () => { readColors(); drawPad(); });
  new MutationObserver(() => { readColors(); drawPad(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  window.addEventListener('resize', () => { if (!state.tracking) drawPad(); });

  const canUseCamera = window.isSecureContext && navigator.mediaDevices?.getUserMedia;
  if (!canUseCamera) {
    els.secureHint.hidden = false;
    els.startBtn.disabled = true;
    setStatus('Camera unavailable on this page', 'warn');
  }

  els.startBtn.addEventListener('click', start);
  els.stopBtn.addEventListener('click', stop);
  els.calibrateBtn.addEventListener('click', startCalibration);
  els.clearBtn.addEventListener('click', () => { state.history = []; renderHistory(); });

  const pick = (facing) => async () => {
    state.facing = facing;
    els.frontBtn.setAttribute('aria-pressed', String(facing === 'user'));
    els.backBtn.setAttribute('aria-pressed', String(facing === 'environment'));
    if (!state.tracking) return; // used on the next Start
    try {
      setStatus('Switching camera…', 'busy');
      await switchFacing(facing);
      setStatus('Looking for a face…', 'busy');
      state.faceVisible = false;
    } catch (err) {
      setStatus(cameraErrorText(err), 'warn');
    }
  };
  els.frontBtn.addEventListener('click', pick('user'));
  els.backBtn.addEventListener('click', pick('environment'));

  els.cameraSelect.addEventListener('change', async () => {
    if (!state.tracking) return;
    try {
      state.mirrorOverridden = false;
      await openCamera({ deviceId: els.cameraSelect.value });
      resetForNewView();
      state.faceVisible = false;
    } catch (err) {
      setStatus(cameraErrorText(err), 'warn');
    }
  });

  els.mirrorToggle.addEventListener('change', () => {
    state.mirrored = els.mirrorToggle.checked;
    state.mirrorOverridden = true;
    applyMirror();
    drawPad();
  });
}

init();
