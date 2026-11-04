import { Chessground } from './vendor/chessground.js';
import { Chess } from './vendor/chess.js';

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

const DEFAULTS = { bot: 'maia3-79m', elo: 1550, color: 'white', tc: '180+2', delay: true, thoughts: true };
let settings = { ...DEFAULTS, ...store.get('doppel.settings', {}) };
let BOTS = [{ id: 'maia3-79m', name: 'Maia-3 79M', blurb: '', clock: false }];

const chess = new Chess();
let game = null;       // {gen, userColor, base, inc, moves, clock:{w,b} ms, over, bot, elo, delay}
let thinking = false;
let lastThoughts = null; // {label, top, chosen}
let lastTick = performance.now();

const cg = Chessground($('board'), {
  coordinates: true,
  animation: { enabled: true, duration: 160 },
  highlight: { lastMove: true, check: true },
  movable: { free: false, showDests: true, events: { after: onUserMove } },
  premovable: { enabled: true },
  draggable: { enabled: true, showGhost: true },
  drawable: {
    enabled: true, visible: true,
    brushes: {
      h: { key: 'h', color: '#7aa2ff', opacity: 0.85, lineWidth: 10 },
      h2: { key: 'h2', color: '#a07aff', opacity: 0.7, lineWidth: 10 },
      h3: { key: 'h3', color: '#9aa1b0', opacity: 0.6, lineWidth: 10 },
    },
  },
});

// ------------------------------------------------------------------ helpers
const colorName = (c) => (c === 'w' ? 'white' : 'black');
const botColor = () => (game.userColor === 'white' ? 'b' : 'w');
const timed = () => game && game.base > 0;
const botMeta = (id) => BOTS.find((b) => b.id === id) || { id, name: id, clock: false };
const vibrate = (ms) => { try { navigator.vibrate && navigator.vibrate(ms); } catch { /* unsupported */ } };

function dests() {
  const d = new Map();
  for (const m of chess.moves({ verbose: true })) {
    if (!d.has(m.from)) d.set(m.from, []);
    d.get(m.from).push(m.to);
  }
  return d;
}

function fmt(ms) {
  if (ms <= 0) return '0:00';
  const s = ms / 1000;
  if (s < 10) return `0:0${s.toFixed(1)}`.replace(/^0:0(\d\d)/, '0:$1');
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
}

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add('hidden'), 1800);
}

function save() {
  if (game) store.set('doppel.game', { ...game, gen: undefined });
}

// ------------------------------------------------------------------ rendering
function render() {
  const turn = colorName(chess.turn());
  const hist = chess.history({ verbose: true });
  const last = hist[hist.length - 1];
  const userTurn = game && !game.over && chess.turn() === game.userColor[0];
  cg.set({
    fen: chess.fen(),
    orientation: game ? cg.state.orientation : 'white',
    turnColor: turn,
    lastMove: last ? [last.from, last.to] : undefined,
    check: chess.inCheck() ? turn : false,
    movable: {
      color: game && !game.over ? game.userColor : undefined,
      dests: userTurn ? dests() : new Map(),
    },
  });
  renderClocks();
  renderMoves(hist);
  renderThoughts();
  const meta = botMeta(game?.bot || settings.bot);
  $('bot-name').textContent = meta.name;
  $('bot-avatar').textContent = meta.id.startsWith('maia') ? 'M' : 'C';
  $('bot-sub').textContent = `${game?.elo ?? settings.elo} · ${game?.delay ? 'human think time' : 'instant'}`;
  $('status').textContent = game?.over ? 'Game over' : userTurn ? (chess.inCheck() ? 'Check — your move' : 'Your move') : 'Waiting…';
  $('thinking').classList.toggle('hidden', !thinking);
  $('btn-back').disabled = !game || chess.history().length < (game.userColor === 'white' ? 1 : 2);
  $('btn-resign').disabled = !game || !!game.over;
}

function renderClocks() {
  if (!game) return;
  const bottomColor = cg.state.orientation[0];
  const topColor = bottomColor === 'w' ? 'b' : 'w';
  const running = !game.over && clockRunning() ? chess.turn() : null;
  for (const [el, c] of [[$('clock-top'), topColor], [$('clock-bottom'), bottomColor]]) {
    el.classList.toggle('off', !timed());
    el.textContent = fmt(game.clock[c]);
    el.classList.toggle('running', running === c);
    el.classList.toggle('low', game.clock[c] < 20000);
  }
}

function renderMoves(hist) {
  const box = $('moves');
  box.innerHTML = '';
  hist.forEach((m, i) => {
    if (i % 2 === 0) {
      const n = document.createElement('span');
      n.className = 'n';
      n.textContent = `${i / 2 + 1}.`;
      box.appendChild(n);
    }
    const s = document.createElement('span');
    s.className = 'm' + (i === hist.length - 1 ? ' last' : '');
    s.textContent = m.san;
    box.appendChild(s);
  });
  box.scrollLeft = box.scrollWidth;
}

function renderThoughts() {
  const box = $('thoughts');
  box.innerHTML = '';
  if (!lastThoughts || !settings.thoughts) return;
  const lab = document.createElement('span');
  lab.className = 'thoughts-label';
  lab.textContent = lastThoughts.label;
  box.appendChild(lab);
  for (const t of lastThoughts.top) {
    const c = document.createElement('div');
    c.className = 'thought' + (t.uci === lastThoughts.chosen ? ' chosen' : '');
    c.innerHTML = `<b>${t.san}</b><span>${Math.round(t.p * 100)}%</span><span class="bar-fill"><i style="width:${Math.max(4, t.p * 100)}%"></i></span>`;
    box.appendChild(c);
  }
}

// ------------------------------------------------------------------ clocks
function clockRunning() {
  // Lichess-style: clocks start once both sides have moved
  return timed() && chess.history().length >= 2;
}

function tick(now) {
  const dt = now - lastTick;
  lastTick = now;
  if (game && !game.over && clockRunning()) {
    const c = chess.turn();
    game.clock[c] -= dt;
    if (game.clock[c] <= 0) {
      game.clock[c] = 0;
      finish(c === game.userColor[0] ? 'loss' : 'win', 'on time');
    }
    renderClocks();
  }
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ moves
function onUserMove(orig, dest) {
  const piece = chess.get(orig);
  if (piece?.type === 'p' && (dest[1] === '8' || dest[1] === '1')) {
    askPromotion(piece.color, (role) => commit({ from: orig, to: dest, promotion: role }, true));
    return;
  }
  commit({ from: orig, to: dest }, true);
}

function askPromotion(color, done) {
  const box = $('promo');
  box.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'cg-wrap promo-row';
  for (const [role, name] of [['q', 'queen'], ['r', 'rook'], ['b', 'bishop'], ['n', 'knight']]) {
    const p = document.createElement('piece');
    p.className = `${name} ${colorName(color)}`;
    p.onclick = () => { box.classList.add('hidden'); done(role); };
    row.appendChild(p);
  }
  box.appendChild(row);
  box.onclick = (e) => { if (e.target === box) { box.classList.add('hidden'); render(); } };
  box.classList.remove('hidden');
}

function commit(move, byUser) {
  let m;
  try { m = chess.move(move); } catch { render(); return; }
  game.moves.push(m.from + m.to + (m.promotion || ''));
  if (timed() && chess.history().length > 2) game.clock[m.color] += game.inc * 1000;
  if (byUser) cg.setAutoShapes([]);
  vibrate(m.captured ? 18 : 8);
  save();
  if (!checkOver()) {
    render();
    if (chess.turn() === botColor()) botMove();
  }
}

function checkOver() {
  if (!chess.isGameOver()) return false;
  if (chess.isCheckmate()) finish(chess.turn() === game.userColor[0] ? 'loss' : 'win', 'by checkmate');
  else if (chess.isStalemate()) finish('draw', 'by stalemate');
  else if (chess.isInsufficientMaterial()) finish('draw', 'insufficient material');
  else if (chess.isThreefoldRepetition()) finish('draw', 'by repetition');
  else finish('draw', 'by the 50-move rule');
  return true;
}

async function api(body) {
  const r = await fetch('api/move', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`server ${r.status}`);
  return r.json();
}

function reqBody(elo) {
  const base = game.base || 600;
  return {
    bot: game.bot, elo, moves: game.moves, base, inc: game.inc,
    white_clock: timed() ? game.clock.w / 1000 : base, black_clock: timed() ? game.clock.b / 1000 : base,
  };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function botMove() {
  const gen = game.gen;
  thinking = true;
  render();
  const t0 = performance.now();
  let res;
  try {
    res = await api(reqBody(game.elo));
  } catch (e) {
    thinking = false;
    render();
    toast(`Bot unavailable (${e.message}) — tap Undo or New`);
    return;
  }
  if (gen !== game.gen) return; // takeback / new game while thinking
  let wait = 0;
  if (game.delay) {
    // ChessMimic predicts a human think time; Maia-3 has no clock model, so give it a short human-ish pause
    const want = res.think_s != null ? res.think_s * 1000 : 400 + Math.random() * 1400;
    wait = Math.max(0, want - (performance.now() - t0));
    if (timed()) wait = Math.min(wait, game.clock[botColor()] + 50); // flagging happens via the clock tick
  }
  if (wait) await sleep(wait);
  if (gen !== game.gen || game.over) return;
  thinking = false;
  lastThoughts = { label: 'Bot weighed', top: res.top, chosen: res.move };
  commit({ from: res.move.slice(0, 2), to: res.move.slice(2, 4), promotion: res.move[4] }, false);
  cg.playPremove();
}

async function showHumans() {
  if (!game || game.over || chess.turn() !== game.userColor[0]) { toast('Available on your move'); return; }
  try {
    const res = await api({ ...reqBody(game.elo), temperature: 1 });
    const brushes = ['h', 'h2', 'h3'];
    cg.setAutoShapes(res.top.slice(0, 3).map((t, i) => ({
      orig: t.uci.slice(0, 2), dest: t.uci.slice(2, 4), brush: brushes[i], modifiers: { lineWidth: 4 + Math.round(16 * t.p) },
    })));
    lastThoughts = { label: `Humans at ${game.elo}`, top: res.top, chosen: null };
    renderThoughts();
  } catch (e) { toast(`Server error (${e.message})`); }
}

// ------------------------------------------------------------------ game lifecycle
function startGame() {
  const [base, inc] = settings.tc.split('+').map(Number);
  const userColor = settings.color === 'random' ? (Math.random() < 0.5 ? 'white' : 'black') : settings.color;
  game = { gen: (game?.gen || 0) + 1, userColor, base, inc, moves: [], clock: { w: base * 1000, b: base * 1000 },
    over: null, bot: settings.bot, elo: settings.elo, delay: settings.delay };
  chess.reset();
  thinking = false;
  lastThoughts = null;
  cg.setAutoShapes([]);
  cg.set({ orientation: userColor });
  $('over').classList.add('hidden');
  save();
  render();
  if (userColor === 'black') botMove();
}

function finish(outcome, reason) {
  if (game.over) return;
  game.over = { outcome, reason };
  thinking = false;
  game.gen++;
  save();
  render();
  vibrate([20, 40, 20]);
  $('over-title').textContent = outcome === 'win' ? 'You win' : outcome === 'loss' ? `${botMeta(game.bot).name} wins` : 'Draw';
  $('over-sub').textContent = reason;
  setTimeout(() => $('over').classList.remove('hidden'), 350);
}

function takeback() {
  if (!game) return;
  game.gen++;
  thinking = false;
  if (game.over) { game.over = null; $('over').classList.add('hidden'); }
  const plies = chess.turn() === game.userColor[0] ? 2 : 1;
  for (let i = 0; i < plies && chess.history().length; i++) { chess.undo(); game.moves.pop(); }
  cg.setAutoShapes([]);
  lastThoughts = null;
  save();
  render();
  if (chess.turn() === botColor()) botMove();
}

function pgn() {
  const meta = botMeta(game.bot);
  const pg = new Chess();
  for (const m of game.moves) pg.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] });
  const botName = `${meta.name} (${game.elo})`;
  pg.setHeader('Event', 'Doppel demo');
  pg.setHeader('White', game.userColor === 'white' ? 'You' : botName);
  pg.setHeader('Black', game.userColor === 'black' ? 'You' : botName);
  pg.setHeader('TimeControl', game.base ? `${game.base}+${game.inc}` : '-');
  const res = !game.over ? '*' : game.over.outcome === 'draw' ? '1/2-1/2'
    : (game.over.outcome === 'win') === (game.userColor === 'white') ? '1-0' : '0-1';
  pg.setHeader('Result', res);
  return pg.pgn() + (pg.pgn().endsWith(res) ? '' : ` ${res}`);
}

async function share() {
  const text = pgn();
  try {
    if (navigator.share) await navigator.share({ title: 'Doppel game', text });
    else { await navigator.clipboard.writeText(text); toast('PGN copied'); }
  } catch { /* user cancelled */ }
}

// ------------------------------------------------------------------ settings sheet
function openSheet() {
  const list = $('bot-list');
  list.innerHTML = '';
  for (const b of BOTS) {
    const el = document.createElement('button');
    el.className = 'bot-opt' + (b.id === settings.bot ? ' sel' : '');
    el.innerHTML = `<div class="avatar bot">${b.id.startsWith('maia') ? 'M' : 'C'}</div><div><b>${b.name}</b><small>${b.blurb}</small></div>`;
    el.onclick = () => { settings.bot = b.id; openSheet(); };
    list.appendChild(el);
  }
  $('elo').value = settings.elo;
  $('elo-val').textContent = settings.elo;
  for (const [seg, key] of [['seg-color', 'color'], ['seg-time', 'tc']]) {
    for (const btn of $(seg).querySelectorAll('button')) {
      btn.classList.toggle('sel', btn.dataset.v === settings[key]);
      btn.onclick = () => { settings[key] = btn.dataset.v; openSheet(); };
    }
  }
  $('opt-delay').checked = settings.delay;
  $('opt-thoughts').checked = settings.thoughts;
  $('sheet').classList.remove('hidden');
  $('sheet-backdrop').classList.remove('hidden');
}

function closeSheet() {
  $('sheet').classList.add('hidden');
  $('sheet-backdrop').classList.add('hidden');
}

$('elo').oninput = (e) => { settings.elo = Number(e.target.value); $('elo-val').textContent = settings.elo; };
$('opt-delay').onchange = (e) => { settings.delay = e.target.checked; };
$('opt-thoughts').onchange = (e) => { settings.thoughts = e.target.checked; renderThoughts(); };
$('btn-start').onclick = () => { store.set('doppel.settings', settings); closeSheet(); startGame(); };
$('sheet-backdrop').onclick = closeSheet;
$('btn-new').onclick = openSheet;
$('btn-rematch').onclick = () => { $('over').classList.add('hidden'); startGame(); };
$('btn-share').onclick = share;
$('btn-flip').onclick = () => { cg.toggleOrientation(); renderClocks(); };
$('btn-back').onclick = takeback;
$('btn-humans').onclick = showHumans;
$('btn-resign').onclick = () => {
  if (!game || game.over) return;
  if (confirm('Resign this game?')) finish('loss', 'by resignation');
};
$('over').onclick = (e) => { if (e.target === $('over')) $('over').classList.add('hidden'); };

// ------------------------------------------------------------------ boot
async function boot() {
  try {
    const r = await fetch('api/bots');
    BOTS = (await r.json()).bots;
  } catch { /* offline: keep defaults */ }
  const saved = store.get('doppel.game', null);
  if (saved && !saved.over && saved.moves) {
    game = { ...saved, gen: 1 };
    for (const m of game.moves) chess.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] });
    cg.set({ orientation: game.userColor });
    render();
    if (chess.turn() === botColor()) botMove();
  } else {
    game = null;
    render();
    openSheet();
  }
  requestAnimationFrame(tick);
}

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
boot();
