'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const PALETTE = ['#2f6df6', '#e0742f', '#1a9c62', '#8b5cf6', '#d64545', '#0891b2'];

const state = { mode: 'live', jobs: [], selectedJob: null, run: null, timer: null,
  stocks: { q: '', position: 'all', sort: 'rank', model: null, selected: null, loaded: null } };

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = j.detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

/* ── formatting ─────────────────────────────────────────── */
const pct = v => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (v * 100).toFixed(1) + '%';
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const fmtCell = (k, v) => {
  if (v === null || v === undefined) return '—';
  if (/return|drawdown|vol$|hit_rate|ic_positive/.test(k)) return pct(v);
  if (/turnover/.test(k)) return num(v, 3);
  if (/n_days|years/.test(k)) return num(v, 0);
  return num(v, 3);
};

/* ── theme ──────────────────────────────────────────────── */
(function theme() {
  const saved = (() => { try { return localStorage.getItem('qsm-theme'); } catch (_) { return null; } })();
  if (saved) document.documentElement.setAttribute('data-theme', saved);
  $('#theme-toggle').onclick = () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('qsm-theme', next); } catch (_) {}
    if (state.run) renderResults(state.run);
  };
})();

/* ── tabs ───────────────────────────────────────────────── */
$$('.tab').forEach(t => t.onclick = () => {
  $$('.tab').forEach(x => x.classList.remove('active'));
  $$('.panel').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $('#tab-' + t.dataset.tab).classList.add('active');
  if (t.dataset.tab === 'jobs') startActivity(); else stopActivity();
});
$('#about-link').onclick = e => { e.preventDefault(); $$('.tab').find(t => t.dataset.tab === 'about').click(); };

/* ── environment ────────────────────────────────────────── */
async function loadStatus() {
  try {
    const s = await api('/api/status');
    const live = await api('/api/live/status').catch(() => ({ available: false }));
    const pill = $('#env-pill');
    pill.textContent = live.available ? 'live feed ready' : 'live feed unavailable';
    pill.className = 'pill ' + (live.available ? 'ok' : 'warn');

    if (!live.available) {
      $('#dataset-status').innerHTML =
        '<span style="color:var(--bad)">yfinance is not installed — run: uv pip install -e ".[live]"</span>';
    } else {
      const cached = (live.cached || []).map(c =>
        `<div class="ds-row"><div style="flex:1"><code>${c.file}</code></div>` +
        `<div class="small muted">${c.age_hours}h old · ${c.mb} MB</div></div>`).join('');
      $('#dataset-status').innerHTML =
        `<div class="ds-row"><div style="flex:1"><strong>Yahoo Finance</strong><br>` +
        `<code>daily adjusted OHLCV · no API key</code></div>` +
        `<div class="small" style="color:var(--good)">connected</div></div>` +
        (cached || '<p class="hint">Nothing cached yet — the first run downloads.</p>');
    }

    if (!s.torch_available) {
      $('#gru-check').disabled = true;
      $('#gru-hint').textContent = 'GRU needs PyTorch: uv pip install -e ".[nn]"';
    } else {
      $('#gru-hint').textContent = 'GRU is much slower alongside LightGBM. Run it alone for full speed.';
    }
    loadLiveStatus();
  } catch (e) { console.error(e); }
}

/* ── launching jobs ─────────────────────────────────────── */
function formPayload() {
  const f = $('#run-form');
  const models = $$('input[name=models]:checked', f).map(i => i.value);
  const g = n => f.elements[n] ? f.elements[n].value : '';
  return {
    mode: 'live',
    dataset: 'huge',
    max_tickers: +g('max_tickers') || 500,   // preset decides size; this is only a cap
    start: g('start') || null,
    end: g('end') || null,
    horizon: +g('horizon'),
    folds: +g('folds'),
    embargo: +g('embargo'),
    models,
    cost_bps: +g('cost_bps'),
    quantile: +g('quantile'),
    execution_lag: +g('execution_lag'),
    universe: g('universe') || 'sp100',
    tickers: g('tickers') || '',
    provider: g('provider') || 'yahoo',
    refresh: !!(f.elements['refresh'] && f.elements['refresh'].checked),
  };
}

$('#run-form').onsubmit = async e => {
  e.preventDefault();
  const payload = formPayload();
  if (!payload.models.length) { alert('Pick at least one model.'); return; }
  await launch('/api/runs', payload);
};
$('#null-btn').onclick = () => launch('/api/null-test', { seeds: [11, 12, 13], models: ['ridge', 'lgbm'] });

async function launch(url, body) {
  try {
    const job = await api(url, { method: 'POST', body: JSON.stringify(body) });
    state.selectedJob = job.id;
    $$('.tab').find(t => t.dataset.tab === 'jobs').click();
    pollJobs(true);
  } catch (e) {
    alert('Could not start: ' + e.message);
  }
}

/* ── job polling ────────────────────────────────────────── */
async function pollJobs(force) {
  let jobs;
  try { jobs = await api('/api/jobs'); } catch (_) { return; }
  const wasActive = state.jobs.some(j => j.status === 'running' || j.status === 'queued');
  state.jobs = jobs;

  const active = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
  const badge = $('#jobs-badge');
  badge.textContent = active;
  badge.classList.toggle('hidden', active === 0);

  if (!state.selectedJob && jobs.length) state.selectedJob = jobs[0].id;
  renderJobs();
  if (state.selectedJob) renderConsole(state.selectedJob);

  // A job just finished: refresh the run list and jump to the new result.
  if (wasActive && active === 0) {
    const done = jobs.find(j => j.id === state.selectedJob);
    await loadRuns(done && done.run_dir ? done.run_dir : undefined);
    if (done && done.kind === 'run' && done.status === 'done') {
      $$('.tab').find(t => t.dataset.tab === 'results').click();
    }
  }

  clearTimeout(state.timer);
  state.timer = setTimeout(pollJobs, active ? 1200 : 5000);
}

function renderJobs() {
  $('#job-list').innerHTML = state.jobs.length ? state.jobs.map(j => `
    <div class="job ${j.id === state.selectedJob ? 'sel' : ''}" data-id="${j.id}">
      <span class="dot ${j.status}"></span>
      <span class="lbl">${j.label}</span>
      <span class="t">${j.status === 'running' ? j.elapsed + 's' : (j.error ? 'failed' : j.elapsed + 's')}</span>
    </div>`).join('') : '<p class="muted">No jobs yet.</p>';
  $$('.job').forEach(el => el.onclick = () => {
    state.selectedJob = el.dataset.id; renderJobs(); renderConsole(el.dataset.id);
  });
}

async function renderConsole(id) {
  let job;
  try { job = await api('/api/jobs/' + id); } catch (_) { return; }
  $('#console-label').textContent = job.label;
  const box = $('#console');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  let text = (job.log || []).join('\n') || 'Waiting…';

  if (job.kind === 'null-test' && job.result) {
    const r = job.result;
    text += '\n\n' + (r.passed
      ? `PASS — largest |IC t-stat| was ${r.worst_abs_t}. No signal found in noise, so the pipeline is not leaking.`
      : `FAIL — largest |IC t-stat| was ${r.worst_abs_t}. That is too large for pure noise; investigate leakage.`);
    text += '\n\n' + r.rows.map(x =>
      `  seed ${x.seed}  ${x.model.padEnd(9)} IC ${String(x.ic_mean).padStart(8)}  t ${String(x.ic_t_stat).padStart(7)}`).join('\n');
  }
  if (job.error) text += '\n\n' + job.error;
  box.textContent = text;
  if (atBottom) box.scrollTop = box.scrollHeight;
}

/* ── runs ───────────────────────────────────────────────── */
async function loadRuns(select) {
  let runs;
  try { runs = await api('/api/runs'); } catch (_) { return; }
  const picker = $('#run-picker');
  picker.innerHTML = runs.length
    ? runs.map(r => `<option value="${r.name}">${r.name}</option>`).join('')
    : '<option value="">— no runs yet —</option>';
  if (runs.length) {
    picker.value = select && runs.some(r => r.name === select) ? select : runs[0].name;
    selectRun(picker.value);
  }
}
$('#run-picker').onchange = e => e.target.value && selectRun(e.target.value);

async function selectRun(name) {
  try {
    const data = await api(`/api/runs/${encodeURIComponent(name)}/results`);
    state.run = data;
    renderResults(data);
    state.stocks.selected = null;
    state.stocks.model = null;
    state.stocks.loaded = null;
    $('#stock-detail').classList.add('hidden');
    loadStocks();
  } catch (e) { console.error(e); }
}

/* ── results ────────────────────────────────────────────── */
function renderResults(d) {
  $('#results-empty').classList.add('hidden');
  $('#results-body').classList.remove('hidden');

  const rows = d.summary.rows;
  const models = rows.filter(r => r.model !== 'buy&hold universe');
  const best = models.reduce((a, b) => ((b.ic_mean ?? -9) > (a.ic_mean ?? -9) ? b : a), models[0]);

  const cfg = d.config || {};
  $('#run-meta').textContent = cfg.labels
    ? `horizon ${cfg.labels.horizon}d · ${cfg.data.max_tickers} names · ${cfg.backtest.cost_bps}bps · ${(cfg.backtest.quantile * 100).toFixed(0)}% quantile`
    : '';

  renderVerdict(best, d, rows);
  renderCards(best);
  renderTable(rows, d.summary.columns);

  lineChart($('#chart-equity'), d.equity, $('#equity-legend'));
  if (d.quantiles) barChart($('#chart-quantiles'), d.quantiles.bins, d.quantiles.ann_return);
  if (d.importance) hbarChart($('#chart-importance'), d.importance);

  $('#folds').textContent = (d.folds || []).join('\n') || '—';
  $('#downloads').innerHTML = ['summary.csv', 'metrics.json', 'equity_curves.csv',
    'quantile_returns.csv', 'feature_importance.csv', 'oos_predictions.parquet', 'config.json']
    .map(f => `<a href="/api/runs/${encodeURIComponent(d.name)}/download/${f}">${f}</a>`).join('');

  const img = $('#report-img'), link = $('#report-link');
  if (d.has_report) {
    const url = `/api/runs/${encodeURIComponent(d.name)}/report.png`;
    img.src = url; link.href = url; img.classList.remove('hidden');
  } else { img.classList.add('hidden'); }
}

function renderVerdict(best, d, rows) {
  const el = $('#verdict');
  const ic = best.ic_mean, t = best.ic_t_stat, sh = best.sharpe, shPre = best.sharpe_before_costs;
  const bench = (rows || []).find(r => r.model === 'buy&hold universe');
  const benchSh = bench ? bench.sharpe : null;
  const cost = d.config ? d.config.backtest.cost_bps : '?';
  let cls, title, body;

  const beatsBench = benchSh === null || sh === null ? null : sh > benchSh;

  if (ic === null || t === null) {
    cls = 'weak'; title = 'Not enough data to judge';
    body = 'The out-of-sample window was too short to produce a stable information coefficient.';
  } else if (Math.abs(ic) > 0.15) {
    cls = 'suspect'; title = `Implausibly strong (IC ${num(ic, 3)}) — suspect a bug`;
    body = 'Daily cross-sectional ICs above ~0.15 are not realistic on equity data. Run the null test before believing any of this.';
  } else if (Math.abs(t) < 2) {
    cls = 'weak'; title = 'No signal detected';
    body = `The best model (${best.model}) has IC ${num(ic, 4)} with t-stat ${num(t, 2)} — indistinguishable from zero. This is the most common honest outcome, and it is not a failed run.`;
  } else if (t < 0) {
    cls = 'weak'; title = 'Signal is negative';
    body = `${best.model} has IC ${num(ic, 4)} (t=${num(t, 2)}) — the forecast is anti-correlated with outcomes. Over one sample this is usually noise, not an inverted edge.`;
  } else if (beatsBench === false) {
    // The forecast is real but the strategy is not worth trading. This is the
    // common case, and calling it a plain success would flatter it: a strong IC
    // says the ranking works, not that the book makes money.
    cls = 'weak';
    title = 'Forecast is real — but the strategy loses to buy-and-hold';
    body = `${best.model} ranks stocks with IC ${num(ic, 4)} (t=${num(t, 2)}), which is genuine skill. ` +
      `But net Sharpe is ${num(sh)} after ${cost}bps of costs versus ${num(benchSh)} for simply holding ` +
      `the whole universe, and gross Sharpe was ${num(shPre)} — so trading costs consume the edge. ` +
      `Turnover is ${num(best.avg_daily_turnover, 3)}/day. Lower it, or lower costs, before this is tradable.`;
  } else if (beatsBench === true) {
    cls = 'good'; title = 'Signal detected, and it beats buy-and-hold';
    body = `${best.model} has IC ${num(ic, 4)} (t=${num(t, 2)}) with net Sharpe ${num(sh)} against ` +
      `${num(benchSh)} for holding the universe. Check the decile staircase is monotone and the null test ` +
      `passes before trusting it.`;
  } else {
    cls = 'good'; title = 'Signal detected';
    body = `${best.model} has IC ${num(ic, 4)} (t=${num(t, 2)}) with net Sharpe ${num(sh)}.`;
  }
  el.className = 'verdict ' + cls;
  el.innerHTML = `<b>${title}</b><span class="muted">${body}</span>`;
}

function renderCards(m) {
  const spec = [
    ['IC', num(m.ic_mean, 4), 'forecast quality', m.ic_mean > 0.005 ? 'good' : (m.ic_mean < -0.005 ? 'bad' : '')],
    ['IC t-stat', num(m.ic_t_stat, 2), 'vs. zero', Math.abs(m.ic_t_stat) > 2 ? 'good' : 'warn'],
    ['Sharpe (net)', num(m.sharpe, 2), 'after costs', m.sharpe > 0.5 ? 'good' : (m.sharpe < 0 ? 'bad' : 'warn')],
    ['Sharpe (gross)', num(m.sharpe_before_costs, 2), 'before costs', ''],
    ['Ann. return', pct(m.ann_return), 'net', m.ann_return > 0 ? 'good' : 'bad'],
    ['Max drawdown', pct(m.max_drawdown), 'peak to trough', 'bad'],
    ['Turnover', num(m.avg_daily_turnover, 3), 'per day', ''],
    ['OOS days', num(m.n_days, 0), `model: ${m.model}`, ''],
  ];
  $('#cards').innerHTML = spec.map(([k, v, n, c]) =>
    `<div class="metric"><div class="k">${k}</div><div class="v ${c}">${v}</div><div class="n">${n}</div></div>`).join('');
}

function renderTable(rows, cols) {
  const head = '<tr><th>Model</th>' + cols.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('') + '</tr>';
  const body = rows.map(r => {
    const bench = r.model === 'buy&hold universe';
    return `<tr class="${bench ? 'bench' : ''}"><td>${r.model}</td>` +
      cols.map(c => `<td>${fmtCell(c, r[c])}</td>`).join('') + '</tr>';
  }).join('');
  $('#summary-table').innerHTML = head + body;
}

/* ── charts ─────────────────────────────────────────────── */
function css(v) { return getComputedStyle(document.body).getPropertyValue(v).trim(); }

function lineChart(el, data, legendEl) {
  const W = 900, H = 330, m = { t: 14, r: 14, b: 26, l: 52 };
  const names = Object.keys(data.series);
  const all = names.flatMap(n => data.series[n]).filter(v => v !== null);
  if (!all.length) { el.innerHTML = ''; return; }
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 0.01; lo -= pad; hi += pad;

  const n = data.dates.length;
  const X = i => m.l + (i / Math.max(1, n - 1)) * (W - m.l - m.r);
  const Y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);

  const line = css('--line'), ink3 = css('--ink-3');
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Equity curves">`;

  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo), y = Y(v);
    svg += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="${line}"/>`;
    svg += `<text x="${m.l - 8}" y="${y + 3.5}" text-anchor="end" font-size="10" fill="${ink3}">${v.toFixed(2)}</text>`;
  }
  const ticks = Math.min(6, n);
  for (let k = 0; k < ticks; k++) {
    const i = Math.round(k * (n - 1) / Math.max(1, ticks - 1));
    svg += `<text x="${X(i)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="${ink3}">${data.dates[i].slice(0, 7)}</text>`;
  }
  svg += `<line x1="${m.l}" y1="${Y(1)}" x2="${W - m.r}" y2="${Y(1)}" stroke="${ink3}" stroke-dasharray="3 3"/>`;

  names.forEach((name, si) => {
    const pts = [];
    data.series[name].forEach((v, i) => { if (v !== null) pts.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`); });
    if (!pts.length) return;
    const bench = /buy&hold/.test(name);
    svg += `<polyline points="${pts.join(' ')}" fill="none" stroke="${PALETTE[si % PALETTE.length]}"
      stroke-width="${bench ? 1.3 : 1.8}" ${bench ? 'stroke-dasharray="4 3"' : ''} stroke-linejoin="round"/>`;
  });
  svg += '</svg>';
  el.innerHTML = svg;

  if (legendEl) legendEl.innerHTML = names.map((nm, i) =>
    `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${nm}</span>`).join('');
}

function barChart(el, labels, values) {
  const W = 460, H = 250, m = { t: 14, r: 12, b: 28, l: 52 };
  const lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  const Y = v => m.t + (1 - (v - lo) / ((hi - lo) || 1)) * (H - m.t - m.b);
  const bw = (W - m.l - m.r) / labels.length;
  const line = css('--line'), ink3 = css('--ink-3'), good = css('--good'), bad = css('--bad');

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Returns by signal decile">`;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo), y = Y(v);
    svg += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="${line}"/>`;
    svg += `<text x="${m.l - 8}" y="${y + 3.5}" text-anchor="end" font-size="10" fill="${ink3}">${(v * 100).toFixed(0)}%</text>`;
  }
  labels.forEach((lb, i) => {
    const v = values[i], y0 = Y(0), y1 = Y(v);
    const x = m.l + i * bw + bw * 0.18, w = bw * 0.64;
    svg += `<rect x="${x}" y="${Math.min(y0, y1)}" width="${w}" height="${Math.abs(y1 - y0) || 1}"
      rx="2" fill="${v >= 0 ? good : bad}" opacity="0.85"/>`;
    svg += `<text x="${x + w / 2}" y="${H - 9}" text-anchor="middle" font-size="10" fill="${ink3}">${lb}</text>`;
  });
  svg += '</svg>';
  el.innerHTML = svg;
}

function hbarChart(el, imp) {
  const feats = imp.features, model = imp.models[imp.models.length - 1];
  const vals = imp.values[model];
  const W = 460, rowH = 17, m = { t: 6, r: 46, b: 6, l: 132 };
  const H = m.t + m.b + feats.length * rowH;
  const hi = Math.max(...vals) || 1;
  const ink3 = css('--ink-3'), accent = css('--accent');

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Feature importance">`;
  feats.forEach((f, i) => {
    const y = m.t + i * rowH, w = (vals[i] / hi) * (W - m.l - m.r);
    svg += `<text x="${m.l - 7}" y="${y + 11}" text-anchor="end" font-size="10" fill="${ink3}">${f}</text>`;
    svg += `<rect x="${m.l}" y="${y + 3}" width="${Math.max(1, w)}" height="${rowH - 7}" rx="2" fill="${accent}" opacity="0.8"/>`;
    svg += `<text x="${m.l + w + 6}" y="${y + 11}" font-size="9" fill="${ink3}">${(vals[i] * 100).toFixed(1)}%</text>`;
  });
  svg += '</svg>';
  el.innerHTML = svg;
}

/* ── boot ───────────────────────────────────────────────── */
loadStatus();
loadRuns();
pollJobs();
try {
  if (localStorage.getItem('qsm-live') === '1') { $('#live-toggle').checked = true; startLive(); }
} catch (_) {}


/* ══ stock search ═══════════════════════════════════════ */
let stockTimer = null;
// Requests can come back out of order — a slow one launched earlier will
// otherwise land after a fast one launched later and overwrite the newer
// results. Every load takes a ticket and stale replies are dropped.
let stockSeq = 0;
let detailSeq = 0;

$('#stock-q').addEventListener('input', e => {
  state.stocks.q = e.target.value;
  clearTimeout(stockTimer);
  stockTimer = setTimeout(loadStocks, 180);   // debounce: one request per pause, not per keystroke
});
$('#stock-sort').onchange = e => { state.stocks.sort = e.target.value; loadStocks(); };
$('#stock-model').onchange = e => {
  state.stocks.model = e.target.value;
  state.stocks.selected = null;
  $('#stock-detail').classList.add('hidden');
  loadStocks();
};
$$('#stock-position .seg-btn').forEach(b => b.onclick = () => {
  $$('#stock-position .seg-btn').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  state.stocks.position = b.dataset.pos;
  loadStocks();
});

async function loadStocks() {
  if (!state.run) return;
  const s = state.stocks;
  const qs = new URLSearchParams({ q: s.q, position: s.position, sort: s.sort, limit: '80' });
  if (s.model) qs.set('model', s.model);

  const ticket = ++stockSeq;
  let data;
  try {
    data = await api(`/api/runs/${encodeURIComponent(state.run.name)}/stocks?${qs}`);
  } catch (e) {
    if (ticket !== stockSeq) return;
    $('#stock-unavailable').classList.remove('hidden');
    $('#stock-table-card').classList.add('hidden');
    $('#stock-detail').classList.add('hidden');
    $('#stock-unavailable-msg').textContent = e.message;
    $('#stock-meta').textContent = '';
    return;
  }
  if (ticket !== stockSeq) return;   // a newer search already answered
  $('#stock-unavailable').classList.add('hidden');
  $('#stock-table-card').classList.remove('hidden');
  s.loaded = data;

  const sel = $('#stock-model');
  if (sel.options.length !== data.models.length) {
    sel.innerHTML = data.models.map(m => `<option value="${m}">${m}</option>`).join('');
  }
  sel.value = data.model;

  const asOf = new Date(data.as_of + 'T00:00:00');
  const ageDays = Math.round((Date.now() - asOf.getTime()) / 86400000);
  const freshness = ageDays <= 5
    ? `<span class="live-badge">last close ${data.as_of}</span>`
    : `last bar in sample <strong>${data.as_of}</strong> <span class="sub">(${ageDays.toLocaleString()} days ago)</span>`;
  $('#stock-meta').innerHTML =
    `Showing <strong>${data.returned}</strong> of <strong>${data.total_matches}</strong> matching ` +
    `(universe ${data.universe}) · model <strong>${data.model}</strong> · ${freshness}`;
  $('#stock-footnote').textContent =
    'Contribution is this name’s share of the book’s gross P&L over the out-of-sample window. ' +
    'The per-stock figures sum to gross, not to the headline net return — trading costs are charged ' +
    'against the whole book and are not attributable to a single name.';

  renderStockTable(data.rows);
}

const fcHead = () => state.stocks.fc
  ? `<th>Target (${(state.run && state.run.config ? state.run.config.labels.horizon : 5)}d)</th><th>80% range</th>`
  : '';

function renderStockTable(rows) {
  const query = ($('#stock-q').value || '').trim().toUpperCase();
  if (!rows.length) {
    $('#stock-table').innerHTML =
      `<tr><td style="text-align:center;padding:2rem;color:var(--ink-3)">
         No matching tickers in this run${query ? ` (universe of ${state.stocks.loaded ? state.stocks.loaded.universe : '?'})` : ''}.
         ${query ? `<div id="other-runs-hint" class="hint" style="margin-top:.6rem">checking other runs…</div>` : ''}
       </td></tr>`;
    if (query) findInOtherRuns(query);
    return;
  }
  const head = `<tr>
    <th>Ticker</th><th>Rank now</th><th>Position</th><th>Weight</th>
    <th>Contribution</th><th>Days L/S</th><th>Hit rate</th><th>Sample close</th>
    <th>Live price</th>${fcHead()}</tr>`;
  const body = rows.map(r => {
    const c = r.contribution ?? 0;
    return `<tr data-t="${r.ticker}" class="${r.ticker === state.stocks.selected ? 'sel' : ''}">
      <td><strong>${r.ticker}</strong></td>
      <td><span class="rankbar"><span style="width:${(r.latest_rank ?? 0)}%"></span></span>
          &nbsp;${r.latest_rank === null ? '—' : r.latest_rank.toFixed(0)}</td>
      <td><span class="pos ${r.position}">${r.position}</span></td>
      <td>${r.latest_weight === null ? '—' : (r.latest_weight * 100).toFixed(2) + '%'}</td>
      <td class="num ${c > 0 ? 'pos-v' : (c < 0 ? 'neg-v' : '')}">${(c * 100).toFixed(2)}%</td>
      <td>${r.days_long}/${r.days_short}</td>
      <td>${r.hit_rate === null ? '—' : (r.hit_rate * 100).toFixed(0) + '%'}</td>
      <td>${r.last_price === null ? '—' : r.last_price.toFixed(2)}</td>
      <td class="live-px">—</td>
      ${state.stocks.fc ? forecastCell(r.ticker) : ''}
    </tr>`;
  }).join('');
  $('#stock-table').innerHTML = head + body;
  $$('#stock-table tbody tr, #stock-table tr[data-t]').forEach(tr => {
    if (tr.dataset.t) tr.onclick = () => showStock(tr.dataset.t);
  });
  if (query) findInOtherRuns(query);
  if ($('#live-toggle').checked) pollQuotes(true);   // fill the new rows immediately
}


/* ══ names the model scored in a different run ══════════
   The search is scoped to the selected run's universe, so a stock the model
   has actually scored reads as absent from the whole tool whenever the run you
   happen to be on does not cover it — a Dow-30 run has no Google, and the
   search box could only shrug. These rows keep such names findable without
   silently changing which run every other number on the page came from; that
   only happens when you open one, and it is labelled. */
async function findInOtherRuns(q) {
  const ticket = stockSeq;
  let f = null;
  try {
    f = await api(`/api/search-runs?q=${encodeURIComponent(q)}` +
                  `&exclude=${encodeURIComponent(state.run.name)}&limit=8`);
  } catch (_) { /* the in-run results stand on their own */ }
  if (ticket !== stockSeq) return;          // a newer search already answered

  const table = $('#stock-table'), hint = $('#other-runs-hint');
  const have = new Set(((state.stocks.loaded || {}).rows || []).map(r => r.ticker));
  let rows = ((f && f.rows) || []).filter(r => !have.has(r.ticker));

  // When this run already answered the search, only a ticker typed in full
  // earns a row from somewhere else. Otherwise a two-letter query trails a
  // dozen off-run names underneath perfectly good results.
  if (have.size) rows = rows.filter(r => r.ticker === q);

  if (!rows.length) {
    if (hint) {
      hint.textContent = f
        ? `${q} is not in any run yet — run a backtest on a universe that includes it.`
        : '';
    }
    return;
  }

  const head = table.querySelector('tr');
  const cols = have.size && head ? head.children.length : 2;
  if (!have.size) table.innerHTML = '';     // drop the "no matching tickers" row

  table.insertAdjacentHTML('beforeend',
    `<tr class="other-head"><td colspan="${cols}" class="sub" style="padding-top:.9rem">
       ${have.size ? 'Also scored in another run' : 'Not in this run — scored in another'}
     </td></tr>` +
    rows.map(r =>
      `<tr class="other-run" data-t="${r.ticker}" data-run="${r.run}">
         <td><strong>${r.ticker}</strong></td>
         <td colspan="${Math.max(1, cols - 1)}" class="sub">
           in <strong>${r.run}</strong> · ${r.universe} names${r.as_of ? ` · as of ${r.as_of}` : ''}
           <button type="button" class="tiny" style="margin-left:.5rem">Open</button>
         </td>
       </tr>`).join(''));

  $$('#stock-table tr.other-run').forEach(tr =>
    tr.onclick = () => openInRun(tr.dataset.run, tr.dataset.t));
}

// Opening one of those names means switching to the run that scored it —
// every other figure on the page belongs to a run, so they have to move together.
async function openInRun(run, ticker) {
  state.stocks.q = ticker;
  $('#stock-q').value = ticker;
  const picker = $('#run-picker');
  if (picker) picker.value = run;
  // selectRun already kicks off the list for the new run. The detail card does
  // not depend on that list, so it opens alongside rather than behind it.
  await selectRun(run);
  showStock(ticker);
}

async function showStock(ticker) {
  const s = state.stocks;
  const qs = s.model ? `?model=${encodeURIComponent(s.model)}` : '';
  const ticket = ++detailSeq;
  // No `project=1` here. drawPrice immediately asks for a projection at the
  // range's own horizon, so requesting one at the default horizon first bought
  // a second of simulation whose only use was being replaced. Without it the
  // card is on screen in about 30ms and the chart fills in behind it.
  let d;
  try {
    d = await api(`/api/runs/${encodeURIComponent(state.run.name)}/stocks/${encodeURIComponent(ticker)}${qs}`);
  } catch (e) { return; }
  if (ticket !== detailSeq) return;  // user clicked another name meanwhile

  s.selected = ticker;
  s.lastDetail = d;
  $$('#stock-table tr[data-t]').forEach(tr => tr.classList.toggle('sel', tr.dataset.t === ticker));

  const st = d.stats;
  const box = $('#stock-detail');
  box.classList.remove('hidden');

  const detailed = wantsDetail();
  box.innerHTML = `
    <div class="detail-head">
      <h3>${d.ticker}</h3>
      <span class="sub">${d.model} · ${st.days} days tested</span>
      <div class="seg detail-seg">
        <button type="button" class="seg-btn ${detailed ? '' : 'active'}" data-detail="0">Simple</button>
        <button type="button" class="seg-btn ${detailed ? 'active' : ''}" data-detail="1">Detailed</button>
      </div>
      <button class="close-x" id="stock-close" title="Close">✕</button>
    </div>
    <p class="plain">${plainSummary(d)}</p>
    ${detailed ? `<div class="detail-stats">
      ${stat('Contribution', pctSigned(st.contribution))}
      ${stat('Mean rank', st.mean_rank === null ? '—' : st.mean_rank.toFixed(0))}
      ${stat('Days traded', st.days_traded)}
      ${stat('Long / short', st.days_long + ' / ' + st.days_short)}
      ${stat('Hit rate', st.hit_rate === null ? '—' : (st.hit_rate * 100).toFixed(0) + '%')}
      ${stat('Best day', pctSigned(st.best_day, 3))}
      ${stat('Worst day', pctSigned(st.worst_day, 3))}
    </div>` : ''}
    <div class="mini-charts">
      <div class="mini" style="grid-column:1/-1">
        <div class="range-head">
          <h4 id="sc-price-title">Price and outlook</h4>
          <div class="readout" id="sc-readout">
            <span class="ro-lbl" id="ro-lbl">hover the chart</span>
            <span class="ro-px" id="ro-px">—</span>
            <span class="ro-sub" id="ro-sub"></span>
          </div>
          <div class="ranges" id="sc-ranges">
            ${['1d','3d','5d','1mo','3mo','6mo','ytd','1y','max'].map(r =>
              `<button type="button" class="rng ${r === currentRange() ? 'active' : ''}" data-r="${r}">${
                { '1mo':'1M','3mo':'3M','6mo':'6M','ytd':'YTD','1y':'1Y','max':'MAX' }[r] || r.toUpperCase()
              }</button>`).join('')}
          </div>
        </div>
        <div class="chart" id="sc-price"></div>
        <p class="hint" id="sc-price-note"></p>
      </div>
      ${detailed ? `
      <div class="mini"><h4>Signal rank (0–100)</h4><div class="chart" id="sc-rank"></div></div>
      <div class="mini"><h4>Cumulative contribution</h4><div class="chart" id="sc-cum"></div></div>
      <div class="mini"><h4>Position weight</h4><div class="chart" id="sc-w"></div></div>` : ''}
    </div>`;

  $('#stock-close').onclick = () => {
    stopPriceLive();
    box.classList.add('hidden'); s.selected = null;
    $$('#stock-table tr[data-t]').forEach(tr => tr.classList.remove('sel'));
  };
  $$('.detail-seg .seg-btn').forEach(b => b.onclick = () => {
    try { localStorage.setItem('qsm-detail', b.dataset.detail); } catch (_) {}
    showStock(ticker);
  });

  $$('#sc-ranges .rng').forEach(b => b.onclick = () => {
    try { localStorage.setItem('qsm-range', b.dataset.r); } catch (_) {}
    $$('#sc-ranges .rng').forEach(x => x.classList.toggle('active', x === b));
    drawPrice(d);
    startPriceLive(d);
  });
  drawPrice(d);
  startPriceLive(d);
  if ($('#sc-rank')) {
    miniLine($('#sc-rank'), d.dates, d.rank, css('--accent'), 0, 100, 50);
    miniLine($('#sc-cum'), d.dates, d.cum_contribution, css('--good'), null, null, 0);
    miniLine($('#sc-w'), d.dates, d.weight, css('--warn'), null, null, 0);
  }
  scrollCardIntoView(box);
}

const stat = (k, v) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`;
const pctSigned = (v, d = 2) => v === null || v === undefined ? '—'
  : (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';

function miniLine(el, dates, values, color, lo, hi, midline) {
  if (!el) return;
  const W = 420, H = 130, m = { t: 8, r: 8, b: 18, l: 40 };
  const clean = values.filter(v => v !== null && v !== undefined);
  if (!clean.length) { el.innerHTML = ''; return; }
  const autoLo = lo === null || lo === undefined;
  const autoHi = hi === null || hi === undefined;
  let mn = autoLo ? Math.min(...clean) : lo;
  let mx = autoHi ? Math.max(...clean) : hi;
  if (mn === mx) { mn -= 0.5; mx += 0.5; }
  // Only pad axes we chose ourselves. Padding an explicit 0-100 percentile
  // range would label the chart -8 to 108, which is nonsense.
  const pad = (mx - mn) * 0.08;
  if (autoLo) mn -= pad;
  if (autoHi) mx += pad;

  const n = values.length;
  const X = i => m.l + (i / Math.max(1, n - 1)) * (W - m.l - m.r);
  const Y = v => m.t + (1 - (v - mn) / (mx - mn)) * (H - m.t - m.b);
  const line = css('--line'), ink3 = css('--ink-3');

  let svg = `<svg viewBox="0 0 ${W} ${H}">`;
  for (let k = 0; k <= 2; k++) {
    const v = mn + (k / 2) * (mx - mn), y = Y(v);
    svg += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="${line}"/>`;
    svg += `<text x="${m.l - 6}" y="${y + 3}" text-anchor="end" font-size="9" fill="${ink3}">${fmtTick(v)}</text>`;
  }
  if (midline !== undefined && midline !== null && midline > mn && midline < mx) {
    svg += `<line x1="${m.l}" y1="${Y(midline)}" x2="${W - m.r}" y2="${Y(midline)}" stroke="${ink3}" stroke-dasharray="3 3"/>`;
  }
  [0, n - 1].forEach(i => {
    svg += `<text x="${X(i)}" y="${H - 5}" text-anchor="${i ? 'end' : 'start'}" font-size="9" fill="${ink3}">${dates[i].slice(0, 7)}</text>`;
  });

  let run = [];
  const flush = () => {
    if (run.length > 1) svg += `<polyline points="${run.join(' ')}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>`;
    run = [];
  };
  values.forEach((v, i) => {
    if (v === null || v === undefined) flush();
    else run.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
  });
  flush();
  el.innerHTML = svg + '</svg>';
}

function niceTicks(lo, hi, count = 5) {
  // Round the axis to human steps (1, 2, 5 x 10^n) instead of slicing the
  // range into arbitrary fractions like 245 / 410 / 575.
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / Math.max(1, count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    ticks.push(Math.abs(v) < step / 1e6 ? 0 : v);
  }
  return ticks;
}

function fmtMoney(v, sym) {
  // A global universe quotes in many currencies; a hardcoded "$" turns a
  // 3,116-yen stock into a $3,116 one.
  const s = sym === undefined ? currentSymbol() : (sym || '');
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  v = n;
  const a = Math.abs(v);
  if (a >= 1e9) return s + (v / 1e9).toFixed(1) + 'B';
  if (a >= 1e6) return s + (v / 1e6).toFixed(1) + 'M';
  if (a >= 1000) return s + (v / 1000).toFixed(a >= 10000 ? 0 : 1) + 'k';
  if (a >= 10) return s + v.toFixed(0);
  if (a >= 1) return s + v.toFixed(1);
  return s + v.toFixed(2);
}

function fmtPrice(v, sym) {
  // Always two decimals: the readout is a specific price, not an axis label.
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  const s = sym === undefined ? currentSymbol() : (sym || '');
  return s + n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function currentSymbol() {
  const d = state.stocks.lastDetail;
  return (d && d.projection && d.projection.symbol) || '$';
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a < 1e-9) return '0';
  if (a >= 1000) return (v / 1000).toFixed(1) + 'k';
  if (a >= 10) return v.toFixed(0);
  if (a >= 0.1) return v.toFixed(2);
  return v.toFixed(3);
}


async function loadLiveStatus() {
  try {
    const s = await api('/api/live/status');
    const el = $('#live-status');
    if (!s.available) {
      el.innerHTML = '<span style="color:var(--bad)">yfinance is not installed — ' +
        'run: uv pip install -e ".[live]"</span>';
      return;
    }
    const key = $('select[name=universe]').value;
    // sp500 / us_all are fetched on demand, so their size is not in the static map.
    const n = s.universes[key] || ({ sp500: 503, us_all: 5874 })[key] || 0;
    const fresh = s.cached.length
      ? ` · cached data ${s.cached[0].age_hours}h old`
      : ' · nothing cached yet, first run will download';
    const label = n ? `${n.toLocaleString()} tickers` : 'universe fetched on run';
    el.innerHTML = `<span style="color:var(--good)">✓ live feed ready</span> · ${label}${fresh}`;
  } catch (_) {}
}
if ($('select[name=universe]')) $('select[name=universe]').onchange = loadLiveStatus;


/* ══ live prices ════════════════════════════════════════ */
let liveTimer = null;
const LIVE_MS = 20000;

$('#live-toggle').onchange = e => {
  if (e.target.checked) startLive(); else stopLive();
};
$('#live-once').onclick = () => pollQuotes(true);

function startLive() {
  stopLive();
  pollQuotes(true);
  liveTimer = setInterval(pollQuotes, LIVE_MS);
  try { localStorage.setItem('qsm-live', '1'); } catch (_) {}
}
function stopLive() {
  clearInterval(liveTimer); liveTimer = null;
  $('#live-dot').className = 'dot-live';
  $('#live-label').textContent = 'Live prices off';
  try { localStorage.setItem('qsm-live', '0'); } catch (_) {}
}

// Don't keep hitting the network while the tab is in the background.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { clearInterval(liveTimer); liveTimer = null; }
  else if ($('#live-toggle').checked && !liveTimer) startLive();
});

async function pollQuotes(force) {
  if (!state.run) return;
  const rows = $$('#stock-table tr[data-t]');
  if (!rows.length) return;
  const tickers = rows.map(r => r.dataset.t).slice(0, 200);

  let q;
  try {
    q = await api(`/api/live/quotes?tickers=${encodeURIComponent(tickers.join(','))}`);
  } catch (e) {
    $('#live-dot').className = 'dot-live closed';
    $('#live-label').textContent = 'Live prices unavailable — ' + e.message;
    return;
  }

  const open = q.market.is_open;
  $('#live-dot').className = 'dot-live ' + (open ? 'on' : 'closed');
  $('#live-label').innerHTML = open
    ? `<strong>Market open</strong> · ${q.returned}/${q.requested} quotes · updated ${q.fetched_at}`
    : `<strong>Market ${q.market.state}</strong> · showing last close · ${q.market.exchange_time}`;

  applyQuotes(q.quotes);
}

function applyQuotes(quotes) {
  $$('#stock-table tr[data-t]').forEach(tr => {
    const qd = quotes[tr.dataset.t];
    const cell = tr.querySelector('td.live-px');
    if (!cell) return;
    if (!qd) { cell.textContent = '—'; return; }
    const prev = cell.dataset.px;
    const sign = qd.change_pct >= 0 ? 'pos-v' : 'neg-v';
    cell.innerHTML = `${qd.price.toFixed(2)} <span class="${sign}">` +
      `${qd.change_pct >= 0 ? '+' : ''}${qd.change_pct.toFixed(2)}%</span>`;
    if (prev && prev !== String(qd.price)) {
      cell.classList.remove('flash'); void cell.offsetWidth; cell.classList.add('flash');
    }
    cell.dataset.px = String(qd.price);
  });
}


/* ══ forward value estimate ═════════════════════════════ */
$('#fc-toggle').onchange = e => {
  state.stocks.forecast = e.target.checked ? (state.stocks.forecast || null) : null;
  if (e.target.checked) loadForecast(); else { state.stocks.fc = null; $('#fc-note').classList.add('hidden');
    $('#fc-summary').textContent = ''; renderStockTable(state.stocks.loaded ? state.stocks.loaded.rows : []); }
};

async function loadForecast() {
  if (!state.run) return;
  $('#fc-summary').textContent = 'estimating…';
  const qs = state.stocks.model ? `?model=${encodeURIComponent(state.stocks.model)}` : '';
  let d;
  try {
    d = await api(`/api/runs/${encodeURIComponent(state.run.name)}/forecast${qs}`);
  } catch (e) {
    $('#fc-summary').textContent = 'unavailable — ' + e.message;
    return;
  }
  state.stocks.fc = {};
  d.rows.forEach(r => { state.stocks.fc[r.ticker] = r; });

  const c = d.calibration;
  const top = c.bins[c.bins.length - 1], bot = c.bins[0];
  $('#fc-summary').innerHTML =
    `${d.horizon}-day horizon · calibrated on ${c.observations.toLocaleString()} out-of-sample observations` +
    ` · priced ${d.priced_at}`;

  const sn = Math.abs(top.mean) / top.std;
  $('#fc-note').classList.remove('hidden');
  $('#fc-note').innerHTML =
    `<strong>Read the band, not the number.</strong> Over ${d.horizon} trading days the top decile ` +
    `historically earned <strong>${(top.mean * 100).toFixed(2)}%</strong> versus the universe and the ` +
    `bottom decile ${(bot.mean * 100).toFixed(2)}% — a real, monotone edge ` +
    `(rank correlation ${c.monotonicity.toFixed(2)}). But individual outcomes scattered with a standard ` +
    `deviation of <strong>${(top.std * 100).toFixed(1)}%</strong>, roughly ` +
    `<strong>${(1 / sn).toFixed(0)}× the estimate itself</strong>. The target is where the centre of a very ` +
    `wide cloud sits, not where the price is going. Excess return vs the universe — no market direction is ` +
    `assumed, and this is not advice.`;

  renderStockTable(state.stocks.loaded ? state.stocks.loaded.rows : []);
}

function forecastCell(ticker) {
  const fc = state.stocks.fc && state.stocks.fc[ticker];
  if (!fc) return '<td class="fc-cell">—</td><td class="fc-cell">—</td>';
  const lo = fc.target_low, hi = fc.target_high, now = fc.price, tgt = fc.target;
  const pos = v => Math.max(0, Math.min(100, ((v - lo) / ((hi - lo) || 1)) * 100));
  const up = tgt >= now;
  return `<td class="fc-cell">${tgt.toFixed(2)} ` +
    `<span class="${up ? 'pos-v' : 'neg-v'}">${up ? '+' : ''}${(fc.expected_return * 100).toFixed(2)}%</span></td>` +
    `<td class="fc-cell"><span class="band" title="80% interval ${lo.toFixed(2)} – ${hi.toFixed(2)}">` +
    `<span class="rail"></span>` +
    `<span class="now" style="left:${pos(now)}%"></span>` +
    `<span class="tgt ${up ? 'up' : 'down'}" style="left:calc(${pos(tgt)}% - 3px)"></span>` +
    `</span> <span class="sub">${lo.toFixed(0)}–${hi.toFixed(0)}</span></td>`;
}


/* ══ price chart with live dot and projection cone ══════ */
function priceChart(el, d, noteEl) {
  const p = d.projection;
  const pts = d.close.map((v, i) => ({ v, i })).filter(x => x.v !== null);
  if (!pts.length) { el.innerHTML = ''; return; }

  const cone = p && p.cone ? p.cone : null;
  const labels = d.dates || [];

  // Everything left of the anchor is history as it stood when the forecast was
  // drawn. Everything right of it is price that has arrived since, and belongs
  // on top of the cone rather than pushing it forward — that overlay is the
  // only way to see whether the band was any good.
  const anchor = d.anchor || labels[pts[pts.length - 1].i];
  const ai = labels.indexOf(anchor);
  let anchorK = pts.length - 1;
  if (ai >= 0) {
    anchorK = 0;
    for (let k = 0; k < pts.length; k++) if (pts[k].i <= ai) anchorK = k;
  }
  const past = pts.slice(0, anchorK + 1);
  const since = pts.slice(anchorK);        // starts at the anchor so the lines join
  const nHist = past.length;

  const W = 880, H = 320, m = { t: 14, r: 84, b: 26, l: 56 };
  const plotW = W - m.l - m.r;
  // Give the forecast a real share of the canvas. Scaled by elapsed days it
  // would be a 5-day sliver against 120 bars of history — about 4% of the
  // width — which is why it read as a stub rather than a continuation.
  const FUT = 0.34;
  const histW = plotW * (1 - FUT), futW = plotW * FUT;

  const Xh = k => m.l + (nHist <= 1 ? 0 : (k / (nHist - 1)) * histW);
  const Xf = frac => m.l + histW + Math.max(0, Math.min(1, frac)) * futW;
  const nowX = Xh(nHist - 1);

  // One time axis for the forecast half, in trading days, so the cone, the
  // simulated path and the realised price all land on the same clock instead
  // of each being stretched across the width by its own point count.
  const span = cone && cone.days.length ? (cone.days[cone.days.length - 1] || 1) : 1;
  const elapsedAt = k => elapsedTradingDays(anchor, labels[pts[k].i]);
  const Xs = k => {
    const e = elapsedAt(k);
    return e === null ? nowX : Xf(e / span);
  };

  const vals = pts.map(x => x.v);
  if (cone) vals.push(...cone.low, ...cone.high);
  (d.pastForecasts || []).forEach(pf => {
    const c = pf.projection && pf.projection.cone;
    if (c) vals.push(...c.low, ...c.high);
  });
  if (d.track && d.track.rows) d.track.rows.forEach(r => vals.push(r.predicted));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const Y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);

  const line = css('--line'), ink3 = css('--ink-3'), ink2 = css('--ink-2');
  let svg = `<svg viewBox="0 0 ${W} ${H}">`;

  niceTicks(lo, hi, 6).forEach(v => {
    const y = Y(v);
    svg += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="${line}"/>`;
    svg += `<text x="${m.l - 7}" y="${y + 3}" text-anchor="end" font-size="10" fill="${ink3}">${fmtMoney(v)}</text>`;
  });
  svg += `<text x="12" y="${m.t + 4}" font-size="9" fill="${ink3}">price</text>`;

  // shade the forecast region so past and future are visually distinct
  svg += `<rect x="${nowX}" y="${m.t}" width="${W - m.r - nowX}" height="${H - m.t - m.b}"
    fill="${ink3}" opacity="0.05"/>`;

  // Forecasts whose window has already closed, behind the price that settled
  // them. Drawn before the history line so the actual data reads on top.
  const bpd = barsPerDay(d.rangeMeta && d.rangeMeta.interval);
  const retired = [];
  (d.pastForecasts || []).forEach(pf => {
    const pi = labels.indexOf(pf.anchor);
    if (pi < 0) return;
    let ak = -1;
    for (let k = 0; k < past.length; k++) if (past[k].i === pi) { ak = k; break; }
    if (ak < 0) return;
    svg += pastForecastSvg(pf, ak, bpd, Xh, Y, nHist);
    retired.push({ pf, ak, endK: ak + pinSpan(pf) * bpd });
  });

  // Every call the model made over this window, each plotted on the date it
  // was predicting *for*, so the gap to the price line is the error itself.
  let trackPts = null;
  if (d.track && d.track.rows && d.track.rows.length) {
    const at = new Map();
    labels.forEach((lab, i) => at.set(String(lab).slice(0, 10), i));
    const run = [];
    d.track.rows.forEach(r => {
      const i = at.get(r.for);
      if (i === undefined || i > (past[nHist - 1] || {}).i) return;
      let k = -1;
      for (let j = 0; j < past.length; j++) if (past[j].i === i) { k = j; break; }
      if (k < 0) return;
      run.push({ k, v: r.predicted, row: r });
    });
    if (run.length > 1) {
      svg += `<polyline points="${run.map(p => `${Xh(p.k).toFixed(1)},${Y(p.v).toFixed(1)}`).join(' ')}"
        fill="none" stroke="${css('--accent')}" stroke-width="1.2"
        stroke-dasharray="3 2" opacity="0.55" stroke-linejoin="round"/>`;
      trackPts = run;
    }
  }

  svg += `<polyline points="${past.map((x, k) => `${Xh(k).toFixed(1)},${Y(x.v).toFixed(1)}`).join(' ')}"
    fill="none" stroke="${ink2}" stroke-width="1.6" stroke-linejoin="round"/>`;

  if (cone) {
    const up = cone.mid[cone.mid.length - 1] >= cone.mid[0];
    const col = up ? css('--good') : css('--bad');
    const fx = i => Xf(cone.days[i] / span);

    const upper = cone.high.map((v, i) => `${fx(i).toFixed(1)},${Y(v).toFixed(1)}`);
    const lower = cone.low.map((v, i) => `${fx(i).toFixed(1)},${Y(v).toFixed(1)}`).reverse();
    svg += `<polygon points="${upper.concat(lower).join(' ')}" fill="${col}" opacity="0.16"/>`;
    svg += `<polyline points="${cone.mid.map((v, i) => `${fx(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}"
      fill="none" stroke="${col}" stroke-width="1.6" stroke-dasharray="6 3" opacity="0.75"/>`;

    // Scenario paths, drawn in the same weight as the history line so the
    // future reads as a continuation of the same chart rather than a diagram.
    const sc = p.scenarios;
    if (sc && sc.path) {
      const sd = scenarioDays(sc);
      const sx = i => Xf(sd[i] / span);
      svg += `<polyline points="${sc.path.map((v, i) => `${sx(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}"
        fill="none" stroke="${col}" stroke-width="1.6" stroke-linejoin="round"/>`;
      const good = css('--good'), bad = css('--bad');
      // Mark the extremes of the path actually drawn.
      svg += `<circle cx="${sx(sc.low.index).toFixed(1)}" cy="${Y(sc.low.price).toFixed(1)}"
        r="4.5" fill="${bad}" stroke="var(--panel)" stroke-width="1.4"/>`;
      svg += `<text x="${sx(sc.low.index).toFixed(1)}" y="${(Y(sc.low.price) + 16).toFixed(1)}"
        text-anchor="middle" font-size="9" fill="${bad}">low ${fmtMoney(sc.low.price)} · ${
          markerWhen(sc.low.day, span, anchor)}</text>`;
      svg += `<circle cx="${sx(sc.high.index).toFixed(1)}" cy="${Y(sc.high.price).toFixed(1)}"
        r="4.5" fill="${good}" stroke="var(--panel)" stroke-width="1.4"/>`;
      svg += `<text x="${sx(sc.high.index).toFixed(1)}" y="${(Y(sc.high.price) - 9).toFixed(1)}"
        text-anchor="middle" font-size="9" fill="${good}">high ${fmtMoney(sc.high.price)} · ${
          markerWhen(sc.high.day, span, anchor)}</text>`;
    }

    const last = cone.mid.length - 1;
    svg += `<text x="${fx(last) + 6}" y="${Y(cone.high[last]) + 3}" font-size="10" fill="${ink3}">${fmtMoney(cone.high[last])}</text>`;
    svg += `<text x="${fx(last) + 6}" y="${Y(cone.mid[last]) + 3}" font-size="10" fill="${col}">${fmtMoney(cone.mid[last])}</text>`;
    svg += `<text x="${fx(last) + 6}" y="${Y(cone.low[last]) + 3}" font-size="10" fill="${ink3}">${fmtMoney(cone.low[last])}</text>`;

    // forecast gridlines, in units that match the horizon
    forecastTicks(span, anchor).forEach(({ frac, text }) => {
      const x = Xf(frac);
      svg += `<line x1="${x}" y1="${m.t}" x2="${x}" y2="${H - m.b}" stroke="${line}" opacity="0.6"/>`;
      svg += `<text x="${x}" y="${H - 7}" text-anchor="middle" font-size="9" fill="${ink3}">${text}</text>`;
    });

    const px = p.price;
    svg += `<circle cx="${nowX}" cy="${Y(px)}" r="5" fill="${col}" stroke="var(--panel)" stroke-width="1.6"/>`;
    svg += `<text x="${nowX - 9}" y="${Y(px) - 9}" text-anchor="end" font-size="10" fill="${col}">${fmtMoney(px)}</text>`;
  }

  // Price since the forecast was drawn: the same ink as the rest of the
  // history, laid over the band it is being judged against.
  const running = since.length > 1;
  if (running) {
    svg += `<polyline points="${since.map((x, k) => `${Xs(anchorK + k).toFixed(1)},${Y(x.v).toFixed(1)}`).join(' ')}"
      fill="none" stroke="${ink2}" stroke-width="1.6" stroke-linejoin="round"/>`;
    const liveK = pts.length - 1, liveX = Xs(liveK), liveV = pts[liveK].v;
    svg += `<line x1="${liveX}" y1="${m.t}" x2="${liveX}" y2="${H - m.b}" stroke="${ink3}" stroke-dasharray="2 3" opacity="0.75"/>`;
    svg += `<circle cx="${liveX}" cy="${Y(liveV)}" r="4" fill="${ink2}" stroke="var(--panel)" stroke-width="1.4"/>`;
    svg += `<text x="${liveX + 7}" y="${Y(liveV) - 7}" font-size="10" fill="${ink2}">${fmtMoney(liveV)}</text>`;
    svg += `<text x="${liveX}" y="${m.t - 3}" text-anchor="middle" font-size="9" fill="${ink3}">now</text>`;
  }

  svg += `<line x1="${nowX}" y1="${m.t}" x2="${nowX}" y2="${H - m.b}" stroke="${ink3}" stroke-dasharray="3 3"/>`;
  // x-axis ticks, formatted for whatever range is on screen
  axisTicks(labels, d.range || '6mo', nHist).forEach(({ index, text }) => {
    const x = Xh(index);
    svg += `<line x1="${x}" y1="${H - m.b}" x2="${x}" y2="${H - m.b + 4}" stroke="${ink3}" opacity="0.5"/>`;
    svg += `<text x="${x}" y="${H - 7}" text-anchor="middle" font-size="9" fill="${ink3}">${text}</text>`;
  });
  if (!running) {
    svg += `<text x="${nowX}" y="${m.t - 3}" text-anchor="middle" font-size="9" fill="${ink3}">now</text>`;
  } else {
    svg += `<text x="${nowX}" y="${m.t - 3}" text-anchor="middle" font-size="9" fill="${ink3}">forecast</text>`;
  }
  el.innerHTML = svg + '</svg>';

  const midAt = cone ? day => interp(cone.days, cone.mid, day) : null;
  attachHover(el, {
    series: past.map(x => x.v), labels,
    // Hovering the history inside a closed forecast's window should say how
    // that forecast was doing at the time, not just repeat the price.
    track: trackPts,
    retired: retired.map(r => ({
      startK: r.ak, endK: r.endK,
      midAt: k => interp(r.pf.projection.cone.days, r.pf.projection.cone.mid,
                         (k - r.ak) / (bpd || 1)),
    })),
    xOf: Xh, yOf: Y, m, W, H, nHist,
    symbol: (p && p.symbol) || '$',
    cone: cone ? { days: cone.days, mid: cone.mid, low: cone.low, high: cone.high,
                   xOf: i => Xf(cone.days[i] / span) } : null,
    // The realised leg takes priority over the cone wherever the two overlap:
    // once a price exists for a moment, that is what the cursor should read.
    since: running ? {
      x: since.map((_, k) => Xs(anchorK + k)),
      v: since.map(x => x.v),
      days: since.map((_, k) => elapsedAt(anchorK + k) || 0),
      labels: since.map(x => labels[x.i] || ''),
      endX: Xs(pts.length - 1),
      midAt,
    } : null,
    // The drawn projection is the wavy simulated path, not the near-flat cone
    // centre — the cursor should ride the line the eye is actually following.
    horizon: span,
    lastLabel: anchor,
    scenario: (p && p.scenarios && p.scenarios.path) ? {
      path: p.scenarios.path,
      days: scenarioDays(p.scenarios),
      horizon: span,
      xOf: i => Xf(scenarioDays(p.scenarios)[i] / span),
    } : null,
  });

  if (noteEl && p && cone) {
    const width = cone.high[cone.high.length - 1] - cone.low[cone.low.length - 1];
    const move = Math.abs(cone.mid[cone.mid.length - 1] - p.price);
    const dirWord = cone.mid[cone.mid.length - 1] >= cone.mid[0] ? 'up' : 'down';
    const dirColor = dirWord === 'up' ? 'var(--good)' : 'var(--bad)';
    const horizonWord = p.horizon_span || `the next ${Math.round(span)} trading days`;

    // What the overlay is worth saying out loud: the forecast is frozen, the
    // grey line is not, and here is the gap between them right now.
    let live = '';
    if (running && midAt) {
      const liveV = pts[pts.length - 1].v;
      const e = elapsedAt(pts.length - 1) || 0;
      const ref = midAt(Math.min(e, span));
      const gap = ((liveV / ref) - 1) * 100;
      const word = Math.abs(gap) < 0.01 ? 'right on' : gap > 0 ? 'above' : 'below';
      live = ` The forecast is frozen at ${whenLabel(anchor)}; the grey line running through the band ` +
        `is what has happened since. It is now ${fmtPrice(liveV, p.symbol)}, ` +
        (word === 'right on' ? 'right on the forecast centre.'
          : `<strong>${Math.abs(gap).toFixed(2)}% ${word}</strong> the forecast centre for this moment.`);
    }

    const tk = d.track;
    const record = !tk || !tk.points ? '' :
      `<br><br><strong>Its record on this name.</strong> The dotted line running through the ` +
      `history is what the model predicted for each of those days, made ${tk.horizon} sessions ` +
      `earlier — the gap to the price line is the error. Over ${tk.points.toLocaleString()} ` +
      `out-of-sample calls it got the direction right ` +
      `<strong>${tk.directional_hit_rate === null ? '—' : (tk.directional_hit_rate * 100).toFixed(0) + '%'}</strong> ` +
      `of the time, missing by <strong>${tk.mean_abs_error_pct === null ? '—'
        : tk.mean_abs_error_pct.toFixed(2)}%</strong> on average.`;

    const n = (d.pastForecasts || []).length;
    const closed = !n ? '' :
      ` The ${n === 1 ? 'fainter band earlier on' : `${n} fainter bands earlier on`} ` +
      `${n === 1 ? 'is a forecast whose' : 'are forecasts whose'} window has already closed — ` +
      `the price line runs straight through ${n === 1 ? 'it' : 'them'}, which is the model's ` +
      `record rather than its promise.`;

    if (!wantsDetail()) {
      // Simple mode: what the picture shows, in one sentence.
      noteEl.innerHTML =
        `The solid line is the past. Past “now”, the wavy line is one plausible path and the ` +
        `shaded band covers the likely range — <strong>${(width / Math.max(move, 1e-9)).toFixed(0)}× wider</strong> ` +
        `than the expected move, which is why nobody can call a single day.` + live + closed + record;
    } else {
      noteEl.innerHTML =
        `<span style="color:${dirColor}">${dirWord === 'up' ? '▲' : '▼'}</span> ` +
        `Model points <strong style="color:${dirColor}">${dirWord}</strong>. ` +
        `The shaded cone is the 80% range over ${horizonWord}, from decile ` +
        `${p.decile}'s historical outcomes. It flares because uncertainty grows with time. Centre moves ` +
        `<strong>${move.toFixed(2)}</strong>; the range spans <strong>${width.toFixed(2)}</strong> — about ` +
        `<strong>${(width / Math.max(move, 1e-9)).toFixed(0)}× wider</strong>. Names in this decile finished ` +
        `higher <strong>${p.hit_rate === null ? '—' : (p.hit_rate * 100).toFixed(0) + '%'}</strong> of the time. ` +
        `This is a distribution, not a path — and not advice.` + live + closed + record +
        scenarioNote() + accuracyNote();
    }
  }
}


// Bars of history in one trading day, per the interval the range was fetched
// at. The history axis is index-based, so this is what turns "three trading
// days after the anchor" into a position on it.
const BARS_PER_DAY = {
  '1m': 390, '2m': 195, '5m': 78, '15m': 26, '30m': 13, '60m': 6.5, '1h': 6.5,
  '1d': 1, '1wk': 0.2, '1mo': 1 / 21,
};
const barsPerDay = interval => BARS_PER_DAY[interval] || 1;

// A retired forecast, drawn where it actually sat on the history axis so the
// real price line runs straight through it. Muted: it is a record, not the
// thing the eye should land on first.
function pastForecastSvg(pf, ak, bpd, Xh, Y, nHist) {
  const c = pf.projection && pf.projection.cone;
  if (!c || !c.days.length) return '';
  const at = j => Math.min(nHist - 1, ak + c.days[j] * bpd);
  const up = c.mid[c.mid.length - 1] >= c.mid[0];
  const col = up ? css('--good') : css('--bad');

  const upper = c.high.map((v, j) => `${Xh(at(j)).toFixed(1)},${Y(v).toFixed(1)}`);
  const lower = c.low.map((v, j) => `${Xh(at(j)).toFixed(1)},${Y(v).toFixed(1)}`).reverse();
  return `<polygon points="${upper.concat(lower).join(' ')}" fill="${col}" opacity="0.09"/>` +
    `<polyline points="${c.mid.map((v, j) => `${Xh(at(j)).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}"
      fill="none" stroke="${col}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.5"/>` +
    `<line x1="${Xh(at(0))}" y1="${Y(c.mid[0])}" x2="${Xh(at(0))}" y2="${Y(c.low[0])}"
      stroke="${col}" stroke-width="1" opacity="0.45"/>`;
}

// Scenario paths are emitted at their own resolution; fall back to even
// spacing over the horizon when the backend did not send a day axis.
function scenarioDays(sc) {
  if (sc._days) return sc._days;
  const n = sc.path.length - 1;
  const days = sc.days && sc.days.length === sc.path.length
    ? sc.days : sc.path.map((_, i) => i / n);
  try {
    Object.defineProperty(sc, '_days', { value: days, enumerable: false, configurable: true });
  } catch (_) {}
  return days;
}

// Linear read of a (days, values) curve at an arbitrary day.
function interp(days, values, at) {
  if (!days.length) return null;
  if (at <= days[0]) return values[0];
  const n = days.length - 1;
  if (at >= days[n]) return values[n];
  let i = 1;
  while (i < n && days[i] < at) i++;
  const t = (at - days[i - 1]) / ((days[i] - days[i - 1]) || 1);
  return values[i - 1] + t * (values[i] - values[i - 1]);
}


function scenarioNote() {
  const d = state.stocks.lastDetail;
  const sc = d && d.projection && d.projection.scenarios;
  if (!sc) return '';
  const a = sc.across_draws;
  return `<br><br><strong>The line past “now”</strong> is the median of ` +
    `${sc.n_draws.toLocaleString()} simulated futures — the one whose end price lands in the middle. ` +
    `Each is built by resampling this stock's own daily moves in ${sc.block}-day blocks from ` +
    `${sc.source_days.toLocaleString()} days of history, so volatility clustering and fat tails carry over ` +
    `and a simulated week looks like a real one. It runs ${fmtMoney(sc.low)}–${fmtMoney(sc.high)}, ending ` +
    `${fmtMoney(sc.end)}. It is the central scenario, not a schedule — the shaded band is where the other ` +
    `${(sc.n_draws - 1).toLocaleString()} went.`;
}

function accuracyNote() {
  // Numbers from experiments/forecast_race.py: 199,686 out-of-sample pairs.
  return `<br><br><strong>Why this curve and not a trend line.</strong> Scored on ` +
    `<strong>199,686</strong> out-of-sample forecasts from this universe, mean absolute error over ` +
    `5 days was <strong>3.08%</strong> for this curve, <strong>3.08%</strong> for assuming no change, and ` +
    `<strong>3.23%</strong> for extrapolating the 60-day trend — the trend line was the worst of the three. ` +
    `A near-flat centre is not the model giving up; it is the most accurate forecast available. ` +
    `The model's edge is in <em>ranking</em> stocks against each other, which the width of this band ` +
    `shows is far too small to see in any single price path.`;
}


/* ══ the model's activity log ═══════════════════════════
   Replaces the jobs list as the front of this tab. A job tells you a run
   happened; this tells you what the model actually did, when, and why —
   fills, passes, retrains and the names it adopted, on one clock. */
let actTimer = null;
const ACT_MS = 30000;

async function loadActivity() {
  const box = $('#act-log');
  if (!box) return;
  const kind = (($$('#act-filter .rng').find(b => b.classList.contains('active')) || {}).dataset || {}).k || 'all';
  let d;
  try { d = await api(`/api/activity?kind=${encodeURIComponent(kind)}`); }
  catch (_) { box.innerHTML = '<p class="hint">Could not read the log.</p>'; return; }

  $('#act-status').textContent =
    `${d.total.toLocaleString()} entries · trader ${d.autotrader ? 'running' : 'stopped'} · ` +
    `learner ${d.learner ? 'running' : 'stopped'}`;
  $('#act-note').innerHTML = !d.fund_run ? 'No fund running yet.' :
    `Following <strong>${d.fund_run}</strong>. Times are this machine's clock, as recorded ` +
    `when the model acted — not when you opened the page.`;

  if (!d.events.length) { box.innerHTML = '<p class="hint">Nothing recorded yet.</p>'; return; }
  box.innerHTML = d.events.map(e => {
    const t = String(e.at || '').replace('T', ' ');
    return `<div class="act-row act-${e.action}">
      <span class="act-time">${t}</span>
      <span class="act-kind">${e.action}</span>
      <span class="act-what">${e.detail}</span>
    </div>`;
  }).join('');
}

function startActivity() {
  stopActivity();
  loadActivity();
  actTimer = setInterval(() => { if (!document.hidden) loadActivity(); }, ACT_MS);
}
function stopActivity() { clearInterval(actTimer); actTimer = null; }

$$('#act-filter .rng').forEach(b => b.onclick = () => {
  $$('#act-filter .rng').forEach(x => x.classList.toggle('active', x === b));
  loadActivity();
});


/* ══ watchlist dashboard ════════════════════════════════ */
let wlTimer = null;
const WL_MS = 20000;

$('#wl-add-btn').onclick = addToWatchlist;
$('#wl-add').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); addToWatchlist(); } });
$('#wl-refresh').onclick = () => loadWatchlist();
$('#wl-auto').onchange = e => { if (e.target.checked) startWatchlist(); else stopWatchlist(); };

async function addToWatchlist() {
  const input = $('#wl-add');
  const t = input.value.trim();
  if (!t) return;
  try {
    await api('/api/watchlist', { method: 'POST', body: JSON.stringify({ ticker: t }) });
    input.value = '';
    await loadWatchlist();
  } catch (e) { alert('Could not add: ' + e.message); }
}

async function removeFromWatchlist(t) {
  try {
    await api('/api/watchlist/' + encodeURIComponent(t), { method: 'DELETE' });
    await loadWatchlist();
  } catch (_) {}
}

function startWatchlist() {
  stopWatchlist();
  loadWatchlist();
  wlTimer = setInterval(loadWatchlist, WL_MS);
}
function stopWatchlist() { clearInterval(wlTimer); wlTimer = null; }

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopWatchlist();
  else if ($('#wl-auto').checked) startWatchlist();
});

async function loadWatchlist() {
  let d;
  try { d = await api('/api/watchlist'); } catch (e) { return; }

  $('#wl-empty').classList.toggle('hidden', d.rows.length > 0);
  const open = d.market && d.market.is_open;
  $('#wl-dot').className = 'dot-live ' + (d.market ? (open ? 'on' : 'closed') : '');
  $('#wl-market').innerHTML = d.market
    ? (open ? `<strong>Market open</strong> · live prices`
            : `<strong>Market ${d.market.state}</strong> · showing last close`)
    : 'prices unavailable';

  renderFundStrip(d.fund);
  loadPlan();
  const held = new Set((d.fund && d.fund.holdings) || []);
  const wanted = new Set((d.fund && d.fund.orders) || []);

  $('#wl-cards').innerHTML = d.rows.map(r => {
    const chg = r.change_pct;
    const cls = chg === null ? '' : (chg >= 0 ? 'pos-v' : 'neg-v');
    const m = r.model;
    const meta = m
      ? `<span class="pos ${m.position}">${m.position}</span>
         <span class="wl-rankbar"><span style="width:${m.rank}%"></span></span>
         <span>rank ${m.rank.toFixed(0)}</span>`
      : `<span>not in any run yet</span>`;
    return `<div class="wl-card" data-t="${r.ticker}">
      <button class="wl-x" data-x="${r.ticker}" title="Remove">×</button>
      <div class="wl-top"><span class="wl-tick">${r.ticker}</span>
        <span class="sub">${r.currency}</span></div>
      <div class="wl-px">${r.price === null ? '—' : fmtMoney(r.price, r.symbol)}</div>
      <div class="wl-chg ${cls}">${chg === null ? '' :
        (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%'}</div>
      <div class="wl-meta">${meta}</div>
      ${held.has(r.ticker) ? '<div class="wl-owned">the model owns this</div>'
        : wanted.has(r.ticker) ? '<div class="wl-wanted">the model has an order resting here</div>'
        : ''}
      ${planLine(r, d.dip_pct)}
      <button class="wl-buy" data-buy="${r.ticker}"
        ${r.price === null ? 'disabled' : ''}>Buy now at ${r.price === null ? '—' :
          fmtPrice(r.price, r.symbol)}</button>
    </div>`;
  }).join('');

  $$('#wl-cards .wl-buy').forEach(b => b.title =
    'Buys now, at the live price. The line below is when a limit order would historically have filled.');
  $$('#wl-cards .wl-x').forEach(b => b.onclick = ev => {
    ev.stopPropagation(); removeFromWatchlist(b.dataset.x);
  });
  $$('#wl-cards .wl-buy').forEach(b => b.onclick = ev => { ev.stopPropagation(); buyPrompt(b.dataset.buy); });
  $$('#wl-cards .wl-card').forEach(c => c.onclick = () => openInStocks(c.dataset.t));

  const scored = d.rows.filter(r => r.model).length;
  $('#wl-note').innerHTML = scored
    ? `Rank and position come from the most recent backtest that covered each name — ` +
      `they are model output from the last day of that sample, not live recommendations.`
    : `No backtest has covered these tickers yet. Run one on the left and the model's view appears here.`;
}

async function openInStocks(ticker) {
  $$('.tab').find(t => t.dataset.tab === 'stocks').click();

  // The selected run may not contain this ticker — the newest run is often a
  // small universe. Switch to one that actually covers it.
  try {
    const inCurrent = state.stocks.loaded &&
      (await api(`/api/runs/${encodeURIComponent(state.run.name)}/stocks` +
                 `?q=${encodeURIComponent(ticker)}&limit=1`)).total_matches > 0;
    if (!inCurrent) {
      const f = await api(`/api/find-run?ticker=${encodeURIComponent(ticker)}`);
      if (f.runs.length) {
        const picker = $('#run-picker');
        picker.value = f.runs[0].run;
        await selectRun(f.runs[0].run);
      }
    }
  } catch (_) {}

  const q = $('#stock-q');
  q.value = ticker;
  q.dispatchEvent(new Event('input'));

  // Open the chart straight away — arriving from the watchlist, seeing the
  // stock is the whole point; making the user click the row again is friction.
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 200));
    const row = $$('#stock-table tr[data-t]').find(tr => tr.dataset.t === ticker.toUpperCase());
    if (row) {
      await showStock(row.dataset.t);
      // The detail renders well below the fold on this page; without an
      // explicit scroll you land on the search bar and think nothing happened.
      scrollCardIntoView($('#stock-detail'));
      return;
    }
  }
}

startWatchlist();


/* ══ plain-language stock summary ═══════════════════════ */
function wantsDetail() {
  try { return localStorage.getItem('qsm-detail') === '1'; } catch (_) { return false; }
}

function plainSummary(d) {
  const st = d.stats, p = d.projection;
  const rank = st.mean_rank;
  const where = rank === null ? 'somewhere mid-pack'
    : rank >= 70 ? `<strong>near the top</strong> of its universe`
    : rank >= 55 ? 'in the <strong>upper half</strong>'
    : rank >= 45 ? 'around the <strong>middle</strong>'
    : rank >= 30 ? 'in the <strong>lower half</strong>'
    : '<strong>near the bottom</strong>';

  const side = st.days_long > st.days_short * 1.3 ? 'mostly wanted to own it'
    : st.days_short > st.days_long * 1.3 ? 'mostly wanted to bet against it'
    : 'switched between owning it and betting against it';

  const money = st.contribution === null ? ''
    : st.contribution >= 0
      ? ` Over the test it <strong style="color:var(--good)">added ${(st.contribution * 100).toFixed(2)}%</strong> to the portfolio.`
      : ` Over the test it <strong style="color:var(--bad)">cost the portfolio ${Math.abs(st.contribution * 100).toFixed(2)}%</strong>.`;

  let outlook = '';
  if (p) {
    const up = p.expected_return >= 0;
    const lo = p.cone ? p.cone.low[p.cone.low.length - 1] : null;
    const hi = p.cone ? p.cone.high[p.cone.high.length - 1] : null;
    const span = p.horizon_span || (p.horizon <= 1 ? 'the rest of the session'
      : p.horizon <= 10 ? `${p.horizon} trading days`
      : p.horizon <= 63 ? `${Math.round(p.horizon / 5)} weeks`
      : `${Math.round(p.horizon / 21)} months`);
    const caveat = p.extrapolated
      ? ` <em>The model is fitted at ${p.calibrated_horizon} trading days; this horizon is ` +
        `scaled from that, so treat the direction as weaker still than usual.</em>`
      : '';
    outlook = `<br><br>Looking ahead ${span}, the model leans ` +
      `<strong style="color:${up ? 'var(--good)' : 'var(--bad)'}">${up ? 'up' : 'down'}</strong> ` +
      `on this name — but only slightly. From ${fmtMoney(p.price, p.symbol)} it expects ` +
      `${fmtMoney(p.target !== undefined ? p.target : p.price * (1 + p.expected_return), p.symbol)}, ` +
      `while the realistic range is ` +
      `${lo === null ? '—' : fmtMoney(lo, p.symbol) + ' to ' + fmtMoney(hi, p.symbol)}. ` +
      `<em>The range is what matters; the direction is a faint tilt inside it.</em>` + caveat;
  }

  return `The model ranked <strong>${d.ticker}</strong> ${where}, and ${side}.${money}${outlook}`;
}


/* ══ track record ═══════════════════════════════════════ */
async function loadLedger() {
  let d;
  try { d = await api('/api/ledger'); } catch (_) { return; }
  const c = d.scorecard;

  $('#ledger-cards').innerHTML = [
    ['Forecasts logged', c.logged || 0, 'written before the outcome'],
    ['Resolved', c.resolved || 0, `outcome known (${d.horizon}d later)`],
    ['Awaiting outcome', c.pending || 0, 'still open'],
  ].map(([k, v, n]) =>
    `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`
  ).join('');

  const models = Object.entries(c.models || {});
  $('#ledger-models').innerHTML = !models.length
    ? '<p class="hint">Nothing scored yet. Run an update, then again after the horizon has passed.</p>'
    : '<div class="table-wrap"><table><tr><th>Model</th><th>Live IC</th><th>t-stat</th>' +
      '<th>Scored days</th></tr>' + models.map(([m, v]) =>
        `<tr><td>${m}</td><td>${v.ic === null || v.ic === undefined ? '—' : v.ic.toFixed(4)}</td>` +
        `<td>${v.ic_t === null || v.ic_t === undefined ? '—' : v.ic_t.toFixed(2)}</td>` +
        `<td>${v.days || 0}</td></tr>`).join('') + '</table></div>';

  const h = d.history || [];
  $('#ledger-history').innerHTML = !h.length
    ? '<tr><td class="muted">No updates recorded yet.</td></tr>'
    : '<tr><th>When</th><th>Universe</th><th>Tickers</th><th>Last bar</th><th>Logged</th></tr>' +
      h.map(e => `<tr><td>${e.at.replace('T', ' ')}</td><td>${e.universe}</td>` +
        `<td>${e.tickers}</td><td>${e.last_bar}</td><td>${e.forecasts_logged}</td></tr>`).join('');
}

$$('.tab').forEach(t => {
  const prior = t.onclick;
  t.addEventListener('click', () => { if (t.dataset.tab === 'ledger') loadLedger(); });
});


/* ══ chart range ════════════════════════════════════════ */
// Trading days to project for each range, so the forecast covers a span
// comparable to the history on screen instead of always five days.
const RANGE_HORIZON = {
  '1d': 1, '3d': 3, '5d': 5, '1mo': 21, '3mo': 63,
  '6mo': 126, 'ytd': 126, '1y': 252, 'max': 252,
};

// How often the open chart re-reads the feed, matched to how often the range's
// bars actually print. Polling a five-minute chart every fifteen seconds is
// twenty wasted calls an hour into a feed that already rate-limits us.
const RANGE_POLL_MS = { '1d': 20000, '3d': 60000, '5d': 60000, '1mo': 120000 };
const POLL_MS_DEFAULT = 300000;

function currentRange() {
  try { return localStorage.getItem('qsm-range') || '6mo'; } catch (_) { return '6mo'; }
}

/* ══ the pinned forecast ════════════════════════════════
   The price line keeps updating; the forecast deliberately does not. Fetching
   a new projection on every poll would slide the green cone forward in step
   with the price, and there would never be anything to compare it against —
   the whole point is to watch the real line walk into a band that was drawn
   before it got there. Pins are kept per ticker+range for the life of the
   page, so stepping away to another name and back does not throw away a
   comparison that has been filling in all session; each is replaced only once
   its own horizon has fully played out. */
const pinKey = (ticker, range) => `${ticker}|${range}|${state.stocks.model || ''}`;

// A finished forecast is the only honest record of how the model actually did,
// so it is kept and drawn behind the price that arrived during it rather than
// thrown away when its window closes. They outlive the page: a forecast made
// this morning is worth seeing this afternoon, and reloading is not a reason
// to lose it.
const PIN_STORE = 'qsm-pins';
const PAST_PINS = 4;                 // per ticker+range; older ones fall off
const PIN_MAX_AGE_MS = 7 * 86400000;

function loadPins() {
  let all;
  try { all = JSON.parse(localStorage.getItem(PIN_STORE) || '{}'); } catch (_) { return {}; }
  if (!all || typeof all !== 'object') return {};
  const cutoff = Date.now() - PIN_MAX_AGE_MS;
  for (const [k, slot] of Object.entries(all)) {
    if (!slot || typeof slot !== 'object') { delete all[k]; continue; }
    slot.past = (slot.past || []).filter(p => p && p.at > cutoff);
    if (slot.live && !(slot.live.at > cutoff)) slot.live = null;
    if (!slot.live && !slot.past.length) delete all[k];
  }
  return all;
}

function savePins(all) {
  // Quota is the only realistic failure here, and a chart that cannot remember
  // is better than one that throws.
  try { localStorage.setItem(PIN_STORE, JSON.stringify(all)); } catch (_) {}
}

function pinSlot(key) {
  const s = state.stocks;
  if (!s.pins) s.pins = loadPins();
  if (!s.pins[key]) s.pins[key] = { live: null, past: [] };
  return s.pins[key];
}

async function drawPrice(d, opts = {}) {
  const el = $('#sc-price'), note = $('#sc-price-note');
  if (!el) return;
  const s = state.stocks;
  const range = currentRange();
  const horizon = RANGE_HORIZON[range] || 5;
  const key = pinKey(d.ticker, range);
  const slot = pinSlot(key);
  let pin = opts.repin ? null : slot.live;

  let hist = null, proj = null;
  try {
    const calls = [api(`/api/history/${encodeURIComponent(d.ticker)}?range=${range}`)];
    if (!pin) {
      calls.push(api(`/api/runs/${encodeURIComponent(state.run.name)}/stocks/` +
        `${encodeURIComponent(d.ticker)}?project=1&horizon=${horizon}` +
        (s.model ? `&model=${encodeURIComponent(s.model)}` : '')));
    }
    [hist, proj] = await Promise.all(calls);
  } catch (_) { /* fall back to whatever the detail call already returned */ }

  // A poll that lands after the user moved on must not repaint the card.
  if (s.selected !== d.ticker || currentRange() !== range) return;

  // The model scores stocks once a day, so its past calls only line up with a
  // chart drawn from daily bars or coarser. Fetched once per ticker+run, and
  // deliberately NOT awaited: on a 4,000-name run the first call spends ~40s
  // reading the panel, and blocking on it left the price chart blank that whole
  // time for an overlay that is a nice-to-have. Draw now, redraw when it lands.
  const dailyBars = !hist || !hist.intraday;
  const trackKey = `${d.ticker}|${state.run.name}|${s.model || ''}`;
  if (dailyBars && s.trackKey !== trackKey) {
    s.trackKey = trackKey;
    s.track = null;
    api(`/api/runs/${encodeURIComponent(state.run.name)}/stocks/` +
        `${encodeURIComponent(d.ticker)}/track` +
        (s.model ? `?model=${encodeURIComponent(s.model)}` : ''))
      .then(t => {
        if (s.trackKey !== trackKey) return;      // moved on while it loaded
        s.track = t;
        if (s.selected === d.ticker && currentRange() === range) drawPrice(d);
      })
      .catch(() => {});
  }

  const view = { ...d, range };
  if (hist && hist.points > 1) {
    view.close = hist.close;
    view.dates = hist.labels;
    view.rangeMeta = hist;
  }
  if (!view.close || !view.close.length) { el.innerHTML = ''; return; }
  const lastLabel = (view.dates || [])[view.close.length - 1];

  if (!pin) {
    const raw = (proj && proj.projection) || d.projection;
    if (raw) {
      slot.live = pin = { key, anchor: lastLabel, at: Date.now(),
                          projection: fitToRange(raw, range, lastLabel) };
      savePins(s.pins);
    }
  } else {
    // Horizon fully elapsed. The forecast is finished, not wrong to have made:
    // retire it into the record and start a new one from where the price is.
    const done = elapsedTradingDays(pin.anchor, lastLabel);
    const span = pinSpan(pin);
    if (done !== null && done > span + 1e-6) {
      slot.past = [...(slot.past || []), pin].slice(-PAST_PINS);
      slot.live = null;
      savePins(s.pins);
      return drawPrice(d, { ...opts, repin: true });
    }
  }
  if (pin) { view.projection = pin.projection; view.anchor = pin.anchor; }
  view.track = dailyBars ? s.track : null;

  // Only the retired forecasts whose anchor is still on screen can be drawn;
  // the rest have scrolled out of the window the user picked.
  view.pastForecasts = (slot.past || [])
    .filter(p => p.anchor !== (pin && pin.anchor) && (view.dates || []).indexOf(p.anchor) >= 0);

  priceChart(el, view, note);

  const title = $('#sc-price-title');
  if (title && view.projection) title.textContent = `Price and ${view.projection.horizon_label} outlook`;

  // Keep the plain-language summary on the same horizon as the chart, or the
  // page says "5 trading days" above a one-year projection.
  const plain = $('.plain');
  if (plain && view.projection) plain.innerHTML = plainSummary(view);
}

// A one-day chart should end when the session does. Left at a whole trading
// day the cone runs past the bell into tomorrow, so a chart labelled "1D"
// ends up showing clock times that already happened that same morning.
function fitToRange(p, range, anchorLabel) {
  const a = parseLabel(anchorLabel);
  const q = { ...p };
  if (range === '1d' && p.cone && a && a.h !== null) {
    const left = SESSION_OPEN_MIN + SESSION_MIN - clampSession(a.h * 60 + a.mi);
    // Inside the last few minutes there is nothing left to draw, so the
    // forecast falls back to the next session, as it does when markets shut.
    if (left >= 20) {
      const cut = truncate(p, Math.min(p.horizon, left / SESSION_MIN));
      if (cut) { Object.assign(q, cut); q.session_only = true; }
    }
  }
  // Two forms of the same span: one that reads as an adjective in the chart
  // heading, one that reads as a noun in the prose underneath it.
  q.horizon_label = q.session_only ? 'rest-of-session'
    : q.horizon <= 1 ? 'next-session'
    : q.horizon <= 10 ? `${Math.round(q.horizon)}-day`
    : q.horizon <= 63 ? `${Math.round(q.horizon / 5)}-week`
    : `${Math.round(q.horizon / 21)}-month`;
  q.horizon_span = q.session_only ? 'the rest of the session'
    : q.horizon <= 1 ? 'the next session'
    : q.horizon <= 10 ? `${Math.round(q.horizon)} trading days`
    : q.horizon <= 63 ? `${Math.round(q.horizon / 5)} weeks`
    : `${Math.round(q.horizon / 21)} months`;
  return q;
}

// How far a pinned forecast reaches, in trading days.
const pinSpan = pin => (pin.projection.cone
  ? pin.projection.cone.days[pin.projection.cone.days.length - 1]
  : pin.projection.horizon);

const clampSession = mm =>
  Math.min(Math.max(mm, SESSION_OPEN_MIN), SESSION_OPEN_MIN + SESSION_MIN);

// Keep the leading `keepDays` of a projection. The cut lands exactly on that
// instant rather than on the nearest grid point, so "the rest of the session"
// really does end at the bell instead of half an hour short of it.
function truncate(p, keepDays) {
  const c = p.cone;
  const full = c.days[c.days.length - 1];
  const span = Math.min(keepDays, full);
  if (!(span > 0) || span >= full - 1e-9) return null;

  const lastAtOrBefore = (days, limit) => {
    let n = 0;
    while (n + 1 < days.length && days[n + 1] <= limit + 1e-9) n++;
    return n;
  };
  const cutTo = (days, series, n) => {
    const out = series.slice(0, n + 1);
    if (span > days[n] + 1e-9) out.push(round4(interp(days, series, span)));
    return out;
  };
  const cutDays = (days, n) => {
    const out = days.slice(0, n + 1);
    if (span > days[n] + 1e-9) out.push(span);
    return out;
  };

  const nc = lastAtOrBefore(c.days, span);
  const out = {
    horizon: span,
    cone: { days: cutDays(c.days, nc), mid: cutTo(c.days, c.mid, nc),
            low: cutTo(c.days, c.low, nc), high: cutTo(c.days, c.high, nc) },
  };

  const sc = p.scenarios;
  if (sc && sc.path && sc.path.length > 1) {
    const days = scenarioDays(sc);
    const ns = lastAtOrBefore(days, span);
    const path = cutTo(days, sc.path, ns);
    const pd = cutDays(days, ns);
    let loI = 0, hiI = 0;
    path.forEach((v, i) => { if (v < path[loI]) loI = i; if (v > path[hiI]) hiI = i; });
    out.scenarios = { ...sc, days: pd, path,
      end: path[path.length - 1],
      low: { price: path[loI], day: pd[loI], index: loI },
      high: { price: path[hiI], day: pd[hiI], index: hiI } };
  }
  return out;
}

const round4 = v => Math.round(v * 1e4) / 1e4;

/* ══ keeping the price line live ════════════════════════ */
let priceTimer = null;

function stopPriceLive() { clearInterval(priceTimer); priceTimer = null; }

function startPriceLive(d) {
  stopPriceLive();
  const ms = RANGE_POLL_MS[currentRange()] || POLL_MS_DEFAULT;
  priceTimer = setInterval(() => {
    // A background tab burning quota on a chart nobody is looking at is
    // exactly the traffic Yahoo throttles.
    if (document.hidden) return;
    if (!$('#sc-price') || state.stocks.selected !== d.ticker) return stopPriceLive();
    drawPrice(d);
  }, ms);
}

document.addEventListener('visibilitychange', () => {
  const s = state.stocks;
  if (document.hidden || !priceTimer || !s.selected || !s.lastDetail) return;
  drawPrice(s.lastDetail);       // catch up on whatever printed while hidden
});

/* ══ chart hover readout ════════════════════════════════ */
function attachHover(el, g) {
  const svg = el.querySelector('svg');
  const lbl = $('#ro-lbl'), pxOut = $('#ro-px'), sub = $('#ro-sub');
  if (!svg || !lbl) return;

  const toViewBox = ev => {
    const r = svg.getBoundingClientRect();
    return (ev.clientX - r.left) * (g.W / r.width);
  };

  let marker = null;
  function ensureMarker() {
    if (marker) return marker;
    const ns = 'http://www.w3.org/2000/svg';
    marker = document.createElementNS(ns, 'g');
    marker.innerHTML =
      `<line y1="${g.m.t}" y2="${g.H - g.m.b}" stroke="currentColor" opacity="0.45" stroke-dasharray="3 3"/>` +
      `<circle r="4" fill="currentColor"/>`;
    marker.style.color = css('--ink-2');
    svg.appendChild(marker);
    return marker;
  }

  const idle = () => {
    lbl.textContent = 'hover the chart';
    pxOut.textContent = '—';
    pxOut.className = 'ro-px';
    sub.textContent = '';
  };

  svg.addEventListener('mousemove', ev => {
    const x = toViewBox(ev);
    const histEndX = g.xOf(g.nHist - 1);
    let px, py;

    if (x <= histEndX || (!g.cone && !g.since)) {
      const frac = (x - g.m.l) / Math.max(1e-9, histEndX - g.m.l);
      const i = Math.max(0, Math.min(g.series.length - 1, Math.round(frac * (g.series.length - 1))));
      const v = g.series[i];
      if (v === null || v === undefined) return;
      px = g.xOf(i); py = g.yOf(v);
      lbl.textContent = g.labels[i] || '';
      pxOut.textContent = fmtPrice(v, g.symbol);
      pxOut.className = 'ro-px';
      const call = (g.track || []).find(p => p.k === i);
      const win = (g.retired || []).find(r => i >= r.startK && i <= r.endK);
      const ref = call ? call.v : (win ? win.midAt(i) : null);
      sub.textContent = !ref ? 'actual'
        : call
          ? `actual · model called ${fmtPrice(call.v, g.symbol)} on ${call.row.from} ` +
            `(${(((v / ref) - 1) * 100).toFixed(2)}% out)`
          : `actual · ${(((v / ref) - 1) * 100).toFixed(2)}% vs the forecast standing then`;
    } else if (g.since && x <= g.since.endX + 1) {
      // Inside the forecast region but behind the live edge: a real price
      // exists for this moment, so read that, and say how the band did.
      let best = 0, bd = Infinity;
      g.since.x.forEach((xx, k) => { const dd = Math.abs(xx - x); if (dd < bd) { bd = dd; best = k; } });
      const v = g.since.v[best];
      px = g.since.x[best]; py = g.yOf(v);
      lbl.textContent = g.since.labels[best] || '';
      pxOut.textContent = fmtPrice(v, g.symbol);
      pxOut.className = 'ro-px';
      const ref = g.since.midAt ? g.since.midAt(g.since.days[best]) : null;
      sub.textContent = ref ? `actual · ${(((v / ref) - 1) * 100).toFixed(2)}% vs forecast` : 'actual';
    } else if (g.cone) {
      const cn = g.cone.days.length - 1;
      const span = g.cone.xOf(cn) - g.cone.xOf(0);
      const frac = Math.max(0, Math.min(1, (x - g.cone.xOf(0)) / Math.max(1e-9, span)));

      // Follow the projected path that is actually drawn; fall back to the
      // cone centre only when no scenario was generated.
      let value, day;
      if (g.scenario) {
        const sn = g.scenario.path.length - 1;
        const si = Math.round(frac * sn);
        px = g.scenario.xOf(si);
        value = g.scenario.path[si];
        day = g.scenario.days[si];
      } else {
        const i = Math.round(frac * cn);
        px = g.cone.xOf(i);
        value = g.cone.mid[i];
        day = g.cone.days[i];
      }
      py = g.yOf(value);

      const ci = Math.round(frac * cn);
      lbl.textContent = aheadLabel(day, g.horizon, g.lastLabel);
      pxOut.textContent = fmtPrice(value, g.symbol);
      pxOut.className = 'ro-px projected';
      sub.innerHTML = `estimate · could be ${fmtPrice(g.cone.low[ci], g.symbol)}–${fmtPrice(g.cone.high[ci], g.symbol)}`;
    } else return;

    const mk = ensureMarker();
    mk.querySelector('line').setAttribute('x1', px);
    mk.querySelector('line').setAttribute('x2', px);
    mk.querySelector('circle').setAttribute('cx', px);
    mk.querySelector('circle').setAttribute('cy', py);
    mk.removeAttribute('hidden');
  });

  svg.addEventListener('mouseleave', () => {
    idle();
    if (marker) marker.setAttribute('hidden', '');
  });
  idle();
}

/* ══ range-aware x-axis ═════════════════════════════════ */
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function parseLabel(s) {
  // Backend emits "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" — parse without letting
  // the browser reinterpret either as UTC and shift the clock.
  if (!s) return null;
  const m = String(s).match(/^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}))?/);
  if (!m) return null;
  return { y: +m[1], mo: +m[2], d: +m[3], h: m[4] === undefined ? null : +m[4], mi: +(m[5] || 0) };
}

function axisTicks(labels, range, n) {
  if (!labels.length || n < 2) return [];
  const count = n > 260 ? 6 : n > 60 ? 5 : 4;
  const idx = [];
  for (let k = 0; k < count; k++) {
    idx.push(Math.round(k * (n - 1) / (count - 1)));
  }

  const hhmm = p => hour12(p.h * 60 + p.mi);
  const dayOf = p => `${MONTHS[p.mo - 1]} ${p.d}`;

  return [...new Set(idx)].map(i => {
    const p = parseLabel(labels[i]);
    if (!p) return { index: i, text: '' };
    let text;
    if (range === '1d') {
      text = hhmm(p);                                   // one session: clock time
    } else if (range === '3d' || range === '5d') {
      // Across a few sessions the useful axis is which day; exact times are a
      // hover away. Mixing "Aug 26" with a bare "15:55" just reads as noise.
      text = dayOf(p);
    } else if (range === '1mo') {
      text = dayOf(p);
    } else if (range === 'max') {
      text = String(p.y);                               // decades: years only
    } else if (range === '1y' || range === 'ytd') {
      // "Jan 26" would read as the 26th; the apostrophe marks it as a year.
      text = `${MONTHS[p.mo - 1]} '${String(p.y).slice(2)}`;
    } else {
      text = dayOf(p);
    }
    return { index: i, text };
  });
}


/* ══ forecast axis: real dates and times ════════════════ */
const SESSION_OPEN_MIN = 9 * 60 + 30;     // 09:30
const SESSION_MIN = 6.5 * 60;             // 09:30 -> 16:00

function hour12(totalMin) {
  let h = Math.floor(totalMin / 60), mi = Math.round(totalMin % 60);
  if (mi === 60) { mi = 0; h += 1; }
  const suffix = h >= 12 ? 'PM' : 'AM';
  const h12 = h % 12 === 0 ? 12 : h % 12;
  return `${h12}:${String(mi).padStart(2, '0')} ${suffix}`;
}

// Whole weekdays from one calendar date to another. Exchange holidays are not
// modelled here any more than they are in nextBusinessDay.
function businessDaysBetween(a, b) {
  // Closed form, not a day-by-day walk. On a MAX-range chart the old loop ran
  // once per calendar day between two points decades apart, for every point on
  // screen — millions of iterations to answer a question that is arithmetic.
  const DAY = 86400000;
  const from = Date.UTC(a.y, a.mo - 1, a.d), to = Date.UTC(b.y, b.mo - 1, b.d);
  const sign = from <= to ? 1 : -1;
  const lo = Math.min(from, to), hi = Math.max(from, to);

  // Whole weeks contribute five days each; the remainder is counted directly.
  const days = Math.round((hi - lo) / DAY);
  const weeks = Math.floor(days / 7);
  let n = weeks * 5;
  let wd = new Date(lo).getUTCDay();
  for (let k = 0; k < days % 7; k++) {
    wd = (wd + 1) % 7;
    if (wd !== 0 && wd !== 6) n += 1;
  }
  return n * sign;
}

// Trading time between two chart labels, in trading days — fractional inside a
// session. This is what places a price that printed at 2:14 PM at the right
// spot under a cone whose x-axis is measured in days, not wall-clock hours.
function elapsedTradingDays(fromLabel, toLabel) {
  const a = parseLabel(fromLabel), b = parseLabel(toLabel);
  if (!a || !b) return null;
  const days = businessDaysBetween(a, b);
  if (a.h === null || b.h === null) return days;
  return days + (clampSession(b.h * 60 + b.mi) - clampSession(a.h * 60 + a.mi)) / SESSION_MIN;
}

function nextBusinessDay(y, mo, d, n) {
  const dt = new Date(y, mo - 1, d);
  let added = 0;
  while (added < n) {
    dt.setDate(dt.getDate() + 1);
    const wd = dt.getDay();
    if (wd !== 0 && wd !== 6) added++;      // skip weekends (holidays not modelled)
  }
  return dt;
}

const fmtDay = dt => `${MONTHS[dt.getMonth()]} ${dt.getDate()}`;

function forecastTicks(span, anchorLabel) {
  const p = parseLabel(anchorLabel);
  const out = [];

  if (p && p.h !== null && span <= 1.001) {
    // Inside a session, label the clock times that follow the anchor. The
    // forecast starts mid-morning, so relabelling it as a fresh 9:30 -> 4:00
    // day put times on the chart that had already happened that morning.
    const total = span * SESSION_MIN;               // minutes of session covered
    const step = total <= 150 ? 30 : 60;
    const start = clampSession(p.h * 60 + p.mi);
    for (let off = step - (start % step); off <= total + 1e-6; off += step) {
      out.push({ frac: off / total, text: futureLabel(off / SESSION_MIN, anchorLabel, span) });
    }
  } else if (span <= 1.001) {
    for (let k = 1; k <= 6; k++) {
      const frac = k / 6.5;
      out.push({ frac, text: hour12(SESSION_OPEN_MIN + frac * SESSION_MIN) });
    }
  } else if (span <= 10) {
    for (let dday = 1; dday <= span; dday++) {
      out.push({ frac: dday / span, text: futureLabel(dday, anchorLabel, span) });
    }
  } else if (span <= 63) {
    const weeks = Math.round(span / 5);
    for (let w = 1; w <= weeks; w++) {
      if (weeks > 8 && w % 2) continue;
      out.push({ frac: (w * 5) / span, text: futureLabel(w * 5, anchorLabel, span) });
    }
  } else {
    const months = Math.round(span / 21);
    const step = months > 6 ? Math.ceil(months / 6) : 1;
    for (let mo = step; mo <= months; mo += step) {
      const dt = p ? nextBusinessDay(p.y, p.mo, p.d, mo * 21) : null;
      const text = dt ? `${MONTHS[dt.getMonth()]} '${String(dt.getFullYear()).slice(2)}` : `+${mo}mo`;
      out.push({ frac: (mo * 21) / span, text });
    }
  }
  return out.filter(t => t.frac > 0.001 && t.frac <= 1.001);
}

// Where a point `day` trading days past the anchor actually falls on a clock
// or a calendar. Within a single session it walks the clock forward from the
// anchor and rolls over at the bell, rather than pretending the forecast
// starts at 9:30. Past a day the time of day is noise, so it is dropped even
// when the anchor carries one.
function futureLabel(day, anchorLabel, span = 0) {
  const p = parseLabel(anchorLabel);
  if (!p) return `+${day.toFixed(day % 1 ? 1 : 0)}d`;

  if (p.h !== null && span <= 1.001) {
    let mins = clampSession(p.h * 60 + p.mi) + day * SESSION_MIN;
    let ahead = 0;
    while (mins > SESSION_OPEN_MIN + SESSION_MIN + 1e-6 && ahead < 400) {
      mins -= SESSION_MIN; ahead++;
    }
    const clock = hour12(mins);
    if (!ahead) return clock;
    const dt = nextBusinessDay(p.y, p.mo, p.d, ahead);
    return `${fmtDay(dt)} ${clock}`;
  }

  const n = Math.max(1, Math.round(day));
  const dt = nextBusinessDay(p.y, p.mo, p.d, n);
  if (!dt) return `+${day.toFixed(day % 1 ? 1 : 0)} trading days`;
  return span > 63 ? `${MONTHS[dt.getMonth()]} ${dt.getDate()}, ${dt.getFullYear()}` : fmtDay(dt);
}

function aheadLabel(day, span, anchorLabel) {
  return futureLabel(day, anchorLabel, span);
}

// Where the anchor itself sits, for "the forecast was drawn at ...".
function whenLabel(anchorLabel) {
  const p = parseLabel(anchorLabel);
  if (!p) return 'the last close';
  return p.h === null ? `${MONTHS[p.mo - 1]} ${p.d}` : hour12(p.h * 60 + p.mi);
}



/* ══ settings ═══════════════════════════════════════════ */
state.settings = null;

async function loadSettings(applyOnly) {
  let d;
  try { d = await api('/api/settings'); } catch (_) { return; }
  state.settings = d.settings;
  applySettings(d.settings);
  if (applyOnly) return;

  $$('[data-setting]').forEach(el => {
    const v = d.settings[el.dataset.setting];
    if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
    el.onchange = () => saveSetting(el);
  });

  const e = d.environment;
  const row = (k, v, ok) =>
    `<div class="ds-row"><div style="flex:1">${k}</div>` +
    `<div class="small" style="color:${ok ? 'var(--good)' : 'var(--ink-3)'}">${v}</div></div>`;
  $('#settings-env').innerHTML =
    row('Account', 'none needed — runs locally', true) +
    row('Price feed (yfinance)', e.yfinance ? 'installed' : 'missing', e.yfinance) +
    row('Neural model (PyTorch)', e.torch ? 'installed' : 'not installed', e.torch) +
    row('Tiingo API key', e.tiingo_key ? 'set' : 'not set', e.tiingo_key) +
    row('Kaggle credentials', e.kaggle_configured ? 'configured' : 'not configured', e.kaggle_configured) +
    row('Market right now', e.market.state, e.market.is_open) +
    row('Data folder', e.data_dir, false) +
    row('Runs folder', e.runs_dir, false);
}

async function saveSetting(el) {
  const key = el.dataset.setting;
  let value = el.type === 'checkbox' ? el.checked : el.value;
  if (el.type === 'number') value = Number(value);
  try {
    const r = await api('/api/settings', { method: 'POST', body: JSON.stringify({ [key]: value }) });
    state.settings = r.settings;
    applySettings(r.settings);
    flashSaved('Saved');
  } catch (err) {
    flashSaved(err.message, true);
    // Put the control back to the stored value rather than leaving a rejected one on screen.
    if (state.settings) {
      const v = state.settings[key];
      if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
    }
  }
}

function flashSaved(msg, bad) {
  const el = $('#settings-saved');
  if (!el) return;
  el.textContent = msg;
  el.style.color = bad ? 'var(--bad)' : 'var(--good)';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.textContent = ''; }, 2500);
}

function applySettings(s) {
  // theme
  if (s.theme === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', s.theme);
  try { localStorage.setItem('qsm-theme', s.theme); } catch (_) {}
  // detail level + default chart range feed the stock views
  try {
    localStorage.setItem('qsm-detail', s.detail_level === 'detailed' ? '1' : '0');
    if (!localStorage.getItem('qsm-range')) localStorage.setItem('qsm-range', s.default_range);
  } catch (_) {}
  // pre-fill the New run panel
  const f = $('#run-form');
  if (f) {
    const set = (name, v) => { if (f.elements[name] && v !== undefined) f.elements[name].value = v; };
    set('universe', s.default_universe);
    set('horizon', s.default_horizon);
    set('folds', s.default_folds);
    set('cost_bps', s.default_cost_bps);
    set('quantile', s.default_quantile);
  }
  const wl = $('#wl-auto');
  if (wl && wl.checked !== !!s.watchlist_autoupdate) {
    wl.checked = !!s.watchlist_autoupdate;
    if (wl.checked) startWatchlist(); else stopWatchlist();
  }
}

if ($('#settings-reset')) {
  $('#settings-reset').onclick = async () => {
    try {
      const r = await api('/api/settings/reset', { method: 'POST', body: '{}' });
      state.settings = r.settings;
      await loadSettings();
      flashSaved('Reset to defaults');
    } catch (e) { flashSaved(e.message, true); }
  };
}

$$('.tab').forEach(t => t.addEventListener('click', () => {
  if (t.dataset.tab === 'settings') loadSettings();
}));

loadSettings(true);


/* ══ the model's own book ═══════════════════════════════
   What replaced the "$1,000 per name since the open" hypothetical. That number
   described a portfolio nobody held, at prices nobody paid; this one is the
   fund's actual cash, actual fills and actual profit. */
function renderFundStrip(f) {
  const box = $('#wl-fund');
  if (!f || !f.budget) {
    box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');
  const cls = f.pnl > 0 ? 'pos-v' : (f.pnl < 0 ? 'neg-v' : '');
  const bench = f.benchmark_result;
  const vs = (f.vs_benchmark === null || f.vs_benchmark === undefined) ? null : f.vs_benchmark;

  const orders = (f.resting || []).map(o =>
    `<li><strong>${o.ticker}</strong> at ${fmtPrice(o.limit, '$')}` +
    `<span class="sub"> · ${o.pct_below}% below the ${fmtPrice(o.reference, '$')} previous close` +
    (o.timing && o.timing.median_time
      ? ` · usually reached ${o.timing.typical_from}–${o.timing.typical_to} ET, ` +
        `${(o.timing.hit_rate * 100).toFixed(0)}% of days`
      : o.timing ? ` · has not reached this in ${o.timing.sessions} sessions` : '') +
    `</span></li>`).join('');

  box.innerHTML =
    `<div class="perf-head">
       <span class="perf-total ${cls}">${f.pnl >= 0 ? '+' : '−'}$${Math.abs(f.pnl).toFixed(2)}</span>
       <span class="perf-since">the model's own money · ${fmtPrice(f.value, '$')} of a
         ${fmtPrice(f.budget, '$')} budget · ${fmtPrice(f.cash, '$')} still in cash</span>
       <span class="perf-acc">${
         vs === null ? `${(f.holdings || []).length} holdings`
         : `${vs >= 0 ? 'ahead of' : 'behind'} ${bench.ticker} by
            <strong>$${Math.abs(vs).toFixed(2)}</strong>`}</span>
     </div>
     <p class="hint">
       Every figure here comes from an order this fund actually placed at a price the
       market actually traded. It buys only while the exchange is open
       (${f.checks_every ? `checked every ${f.checks_every}` : 'checked on a timer'}), so
       nothing here is filled at a stale close.
       ${orders ? `<br><br><strong>Waiting to buy:</strong><ul class="wl-orders">${orders}</ul>`
                : '<br><br>No orders resting right now.'}
     </p>`;
}


/* ══ portfolio ══════════════════════════════════════════ */
async function buyPrompt(ticker) {
  const qty = prompt(`How many shares of ${ticker}?`, '10');
  if (qty === null) return;
  const n = Number(qty);
  if (!Number.isFinite(n) || n <= 0) { alert('Quantity must be a positive number.'); return; }
  try {
    await api('/api/portfolio/buy', { method: 'POST', body: JSON.stringify({ ticker, quantity: n }) });
    $$('.tab').find(t => t.dataset.tab === 'portfolio').click();
    loadPortfolio();
  } catch (e) { alert('Could not record: ' + e.message); }
}

async function loadPortfolio() {
  let d;
  try { d = await api('/api/portfolio'); } catch (_) { return; }
  const rows = d.positions || [];
  $('#pf-empty').classList.toggle('hidden', rows.length > 0);

  if (!rows.length) {
    $('#pf-summary').innerHTML = '';
    $('#pf-table').innerHTML = '';
    return;
  }

  const t = d.total;
  const cls = t.pnl > 0 ? 'good' : (t.pnl < 0 ? 'bad' : '');
  $('#pf-summary').innerHTML = [
    ['Market value', `$${t.value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`,
      t.unpriced ? `${t.priced} of ${t.positions} priced` : `${t.positions} positions`, ''],
    ['Cost basis', `$${t.cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`,
      t.unpriced ? 'priced positions only' : 'what you paid', ''],
    ['Profit / loss', `${t.pnl >= 0 ? '+' : '−'}$${Math.abs(t.pnl).toFixed(2)}`, 'unrealised', cls],
    ['Return', `${t.pnl_pct >= 0 ? '+' : ''}${(t.pnl_pct ?? 0).toFixed(2)}%`, 'since entry', cls],
  ].map(([k, v, n, c]) =>
    `<div class="metric"><div class="k">${k}</div><div class="v ${c}">${v}</div><div class="n">${n}</div></div>`
  ).join('');

  $('#pf-table').innerHTML =
    `<tr><th>Ticker</th><th>Qty</th><th>Entry</th><th>Bought</th><th>Now</th>
     <th>Today</th><th>Value</th><th>P&L</th><th></th></tr>` +
    rows.map(r => {
      const p = r.pnl ?? 0;
      const pc = p > 0 ? 'pos-v' : (p < 0 ? 'neg-v' : '');
      const dc = (r.day_change_pct ?? 0) >= 0 ? 'pos-v' : 'neg-v';
      return `<tr>
        <td><strong>${r.ticker}</strong></td>
        <td class="num">${r.quantity}</td>
        <td class="num">${fmtPrice(r.entry_price, r.symbol)}</td>
        <td class="num" style="font-size:.72rem">${(r.entry_at || '').replace('T', ' ')}</td>
        <td class="num">${r.price === null ? '—' : fmtPrice(r.price, r.symbol)}</td>
        <td class="num ${dc}">${r.day_change_pct === null ? '—' : (r.day_change_pct >= 0 ? '+' : '') + r.day_change_pct.toFixed(2) + '%'}</td>
        <td class="num">${r.value === null ? '—' : '$' + r.value.toFixed(2)}</td>
        <td class="num ${pc}">${r.pnl === null ? '—' :
          (p >= 0 ? '+' : '−') + '$' + Math.abs(p).toFixed(2) +
          ` (${r.pnl_pct >= 0 ? '+' : ''}${(r.pnl_pct ?? 0).toFixed(1)}%)`}</td>
        <td><button class="pf-sell" data-sell="${r.ticker}" title="Remove">×</button></td>
      </tr>`;
    }).join('');

  $$('#pf-table .pf-sell').forEach(b => b.onclick = async () => {
    try {
      await api('/api/portfolio/sell', { method: 'POST',
        body: JSON.stringify({ ticker: b.dataset.sell, quantity: 0 }) });
      loadPortfolio();
    } catch (_) {}
  });
}

if ($('#pf-clear')) {
  $('#pf-clear').onclick = async () => {
    if (!confirm('Remove every position from the portfolio?')) return;
    try { await api('/api/portfolio/clear', { method: 'POST', body: '{}' }); loadPortfolio(); }
    catch (_) {}
  };
}

$$('.tab').forEach(t => t.addEventListener('click', () => {
  if (t.dataset.tab === 'portfolio') { loadFund(); loadPortfolio(); loadAnalytics(); }
}));

/* ══ portfolio analytics ════════════════════════════════ */
async function loadAnalytics() {
  const box = $('#pf-analytics');
  box.innerHTML = '<p class="hint">Loading…</p>';
  let d;
  try {
    d = await api('/api/portfolio/analytics?benchmarks=' +
                  encodeURIComponent($('#pf-bench').value || 'SPY,QQQ'));
  } catch (e) { box.innerHTML = `<p class="hint">Unavailable — ${e.message}</p>`; return; }

  if (!d.available) {
    box.innerHTML = `<p class="hint">Nothing to analyse yet — ${d.reason}.</p>`;
    return;
  }

  const b = d.benchmarks || {};
  const w = b._window;
  const rows = Object.entries(b).filter(([k]) => !k.startsWith('_'));
  const port = b._portfolio_same_window;

  const compare = !rows.length ? '<p class="hint">No benchmark data available.</p>' :
    `<div class="table-wrap"><table>
      <tr><th>${b._context_only ? 'Market context' : 'Over the same window'}</th><th>Return</th></tr>
      ${b._context_only ? '' :
        `<tr><td><strong>This portfolio</strong></td>
         <td class="num ${port >= 0 ? 'pos-v' : 'neg-v'}">${port >= 0 ? '+' : ''}${(port ?? 0).toFixed(2)}%</td></tr>`}
      ${rows.map(([k, v]) =>
        `<tr><td>${k} <span class="sub">${v.name}</span></td>
         <td class="num ${v.return_pct >= 0 ? 'pos-v' : 'neg-v'}">${v.return_pct >= 0 ? '+' : ''}${v.return_pct.toFixed(2)}%</td></tr>`).join('')}
     </table></div>
     <p class="hint">${w ? `${w.from} → ${w.to}, ${w.sessions} sessions. ` : ''}${
       b._context_only
         ? 'Your positions were opened too recently to compare — these are the funds\' recent returns for context, not a comparison against this portfolio.'
         : 'Measured over the same window as your holdings, so it is like for like.'}</p>`;

  const w1 = Object.entries(d.weights || {});
  box.innerHTML =
    `<div class="cards" style="margin-bottom:1rem">
       ${['Positions', d.positions, `effective ${d.effective_positions}`, '']
         .map(() => '').join('')}
       <div class="metric"><div class="k">Positions</div><div class="v">${d.positions}</div>
         <div class="n">effective ${d.effective_positions}</div></div>
       <div class="metric"><div class="k">Largest holding</div><div class="v">${d.largest_weight_pct}%</div>
         <div class="n">${w1.length ? w1[0][0] : '—'}</div></div>
       <div class="metric"><div class="k">Best</div><div class="v good">${d.best.ticker}</div>
         <div class="n">${d.best.pnl_pct === null ? '—' : (d.best.pnl_pct >= 0 ? '+' : '') + d.best.pnl_pct.toFixed(2) + '%'}</div></div>
       <div class="metric"><div class="k">Worst</div><div class="v bad">${d.worst.ticker}</div>
         <div class="n">${d.worst.pnl_pct === null ? '—' : (d.worst.pnl_pct >= 0 ? '+' : '') + d.worst.pnl_pct.toFixed(2) + '%'}</div></div>
     </div>
     <h4 style="font-size:.8rem;color:var(--ink-3);margin:.2rem 0 .5rem">Allocation</h4>
     <div class="table-wrap"><table>${w1.map(([t, pct]) =>
       `<tr><td>${t}</td><td style="width:60%">
         <span class="wl-rankbar" style="width:100%"><span style="width:${pct}%"></span></span></td>
        <td class="num">${pct.toFixed(1)}%</td></tr>`).join('')}</table></div>
     <h4 style="font-size:.8rem;color:var(--ink-3);margin:1rem 0 .5rem">Compared with funds</h4>
     ${compare}`;
}

$('#pf-bench').addEventListener('change', loadAnalytics);

/* ══ balance simulator ══════════════════════════════════ */
$$('#sim-presets .rng').forEach(b => b.onclick = () => {
  $$('#sim-presets .rng').forEach(x => x.classList.toggle('active', x === b));
  $('#sim-balance').value = b.dataset.bal;
  runSimulation();
});
$('#sim-run').onclick = runSimulation;

async function runSimulation() {
  const balance = Number($('#sim-balance').value);
  const out = $('#sim-out');
  if (!Number.isFinite(balance) || balance <= 0) { out.innerHTML = '<p class="hint">Enter a positive balance.</p>'; return; }
  $('#sim-amount').textContent = '$' + balance.toLocaleString();
  out.innerHTML = '<p class="hint">Working…</p>';

  let d;
  try {
    d = await api('/api/portfolio/allocate', { method: 'POST',
      body: JSON.stringify({ balance, top_n: 10 }) });
  } catch (e) { out.innerHTML = `<p class="hint">Unavailable — ${e.message}</p>`; return; }

  const s = d.strategy;
  const beats = s.net_sharpe !== null && s.benchmark_sharpe !== null && s.net_sharpe > s.benchmark_sharpe;
  out.innerHTML =
    `<div class="table-wrap"><table>
      <tr><th>Ticker</th><th>Rank</th><th>Weight</th><th>Price</th><th>Shares</th><th>Spend</th></tr>
      ${d.rows.map(r => `<tr><td><strong>${r.ticker}</strong></td>
        <td class="num">${r.rank === null ? '—' : r.rank.toFixed(0)}</td>
        <td class="num">${r.weight_pct.toFixed(1)}%</td>
        <td class="num">$${r.price.toFixed(2)}</td>
        <td class="num"><strong>${r.shares}</strong></td>
        <td class="num">$${r.spend.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td></tr>`).join('')}
      <tr><td colspan="5"><strong>Invested</strong></td>
        <td class="num"><strong>$${d.invested.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong></td></tr>
      <tr><td colspan="5">Cash left over <span class="sub">whole shares only</span></td>
        <td class="num">$${d.cash.toFixed(2)}</td></tr>
     </table></div>
     <div class="warn-box" style="margin-top:.8rem">
       <strong>This is a simulation of a strategy that underperformed.</strong>
       These are the ${d.rows.length} names the model ranked highest in run <code>${s.run}</code>
       as of ${s.as_of}, with the balance split by its own position weights.
       That strategy's measured net Sharpe is <strong>${s.net_sharpe ?? '—'}</strong> against
       <strong>${s.benchmark_sharpe ?? '—'}</strong> for simply holding the whole universe —
       ${beats ? 'it beat buy-and-hold in the backtest.' :
         'it <em>lost</em> to buy-and-hold in the backtest.'}
       Share counts show what the rule does with the money, not what you should do with yours.
     </div>`;
}



/* ══ the strategy's own entry / exit rule ═══════════════ */
let planLoaded = false;
async function loadPlan() {
  if (planLoaded) return;
  planLoaded = true;
  let p;
  try { p = await api('/api/plan'); } catch (_) { return; }

  const box = $('#wl-plan');
  box.classList.remove('hidden');
  const beats = p.net_sharpe > p.benchmark_sharpe;
  box.innerHTML =
    `<div class="perf-head">
       <span class="perf-total">${p.entry.weekday} ${p.entry.date}</span>
       <span class="perf-since">buy at the <strong>open</strong> · hold ${p.horizon} sessions ·
         sell ${p.exit.weekday} ${p.exit.date} at the <strong>close</strong></span>
       <span class="perf-acc">signal from the ${p.signal_date} close</span>
     </div>
     <p class="hint">
       This is the rule the backtest measured, not a chosen moment: enter ${p.execution_lag}
       session after the signal, hold ${p.horizon}. Those dates are the only timing this model
       produces — it scores stocks once a day and has no view on which hour is better.
       ${p.holidays_modelled ? '' :
         '<span style="color:var(--warn)">Market holidays are not modelled, so check the exit date lands on a trading day.</span>'}
     </p>
     <div class="warn-box">
       <strong>What this rule actually did:</strong> net Sharpe <strong>${p.net_sharpe}</strong>
       against <strong>${p.benchmark_sharpe}</strong> for simply holding the universe, winning on
       <strong>${(p.hit_rate * 100).toFixed(1)}%</strong> of days —
       ${beats ? 'it beat buy-and-hold.' : 'it <em>lost</em> to buy-and-hold over the tested period.'}
       Shown because you asked for the timing; the numbers are what it earned.
     </div>`;
}


/* ══ point at a run that has the ticker ═════════════════ */
function scrollCardIntoView(el) {
  if (!el) return;
  // Jump, do not animate. Smooth-scrolling a long page to a card near the
  // bottom scrolls the reader through every section in between, which looks
  // like the app is paging through screens on its way somewhere.
  const go = () => {
    const top = el.getBoundingClientRect().top + window.scrollY - 12;
    window.scrollTo({ top: Math.max(0, top), behavior: 'auto' });
  };
  go();
  // The chart lands after the first pass and shifts the layout under it; only
  // correct for that if the card has actually been pushed off the top.
  setTimeout(() => { if (el.getBoundingClientRect().top < 0) go(); }, 450);
}


/* ══ model fund ═════════════════════════════════════════ */
async function loadFund() {
  const box = $('#fund-body');
  box.innerHTML = '<p class="hint">Loading…</p>';
  let d;
  try { d = await api('/api/fund'); } catch (e) { box.innerHTML = `<p class="hint">${e.message}</p>`; return; }

  if (d.exists === false) {
    box.innerHTML =
      `<p class="hint">No fund yet.</p>
       <div class="search-bar">
         <input type="number" id="fund-budget" value="5000" min="100" step="100">
         <button type="button" class="primary" id="fund-start" style="flex:none;min-width:120px">Start fund</button>
         <button type="button" class="secondary" id="fund-auto" style="flex:none;min-width:120px">Autopilot</button>
       </div>
       <p class="hint">A fixed fund buys once and holds to the exit date.
         Autopilot re-checks every session and trades on the model's signal.</p>`;
    $('#fund-start').onclick = startFund;
    $('#fund-auto').onclick = () => startFund(true);
    return;
  }

  if (d.mode === 'autopilot') { renderAutopilot(d); return; }

  if (d.status === 'awaiting entry') {
    box.innerHTML =
      `<div class="perf-head">
         <span class="perf-total">$${d.budget.toLocaleString()}</span>
         <span class="perf-since">allocated, entering <strong>${d.entry_date}</strong> ·
           exiting ${d.exit_date}</span>
         <span class="perf-acc">waiting for the market</span>
       </div>
       <p class="hint">${d.message} Prices fill in from the real session — nothing is priced
         at today's close and back-dated.</p>
       <div class="table-wrap"><table>
         <tr><th>Ticker</th><th>Rank</th><th>Weight</th><th>Planned</th></tr>
         ${d.plan.map(p => `<tr><td><strong>${p.ticker}</strong></td>
           <td class="num">${p.rank === null ? '—' : p.rank.toFixed(0)}</td>
           <td class="num">${(p.weight * 100).toFixed(2)}%</td>
           <td class="num">$${(d.budget * p.weight).toFixed(2)}</td></tr>`).join('')}
       </table></div>`;
    return;
  }

  const b = d.benchmark_result;
  const cls = d.pnl > 0 ? 'pos-v' : (d.pnl < 0 ? 'neg-v' : '');
  const won = b && d.vs_benchmark > 0;
  box.innerHTML =
    `<div class="cards" style="margin-bottom:1rem">
      <div class="metric"><div class="k">Fund value</div>
        <div class="v ${cls}">$${d.value.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
        <div class="n">from $${d.budget.toLocaleString()}</div></div>
      <div class="metric"><div class="k">Profit / loss</div>
        <div class="v ${cls}">${d.pnl >= 0 ? '+' : '−'}$${Math.abs(d.pnl).toFixed(2)}</div>
        <div class="n">${d.pnl_pct >= 0 ? '+' : ''}${d.pnl_pct.toFixed(2)}%</div></div>
      ${b ? `<div class="metric"><div class="k">${b.ticker} instead</div>
        <div class="v">${b.pnl >= 0 ? '+' : '−'}$${Math.abs(b.pnl).toFixed(2)}</div>
        <div class="n">${b.pnl_pct >= 0 ? '+' : ''}${b.pnl_pct.toFixed(2)}%</div></div>
      <div class="metric"><div class="k">Model vs ${b.ticker}</div>
        <div class="v ${won ? 'good' : 'bad'}">${d.vs_benchmark >= 0 ? '+' : '−'}$${Math.abs(d.vs_benchmark).toFixed(2)}</div>
        <div class="n">${won ? 'model ahead' : 'model behind'}</div></div>` : ''}
     </div>
     <p class="hint">Entered ${d.entry_date} · ${d.status} · marked at the ${d.as_of} close ·
       exit ${d.exit_date}. Cash left unspent: $${(d.cash ?? 0).toFixed(2)}.</p>
     <div class="table-wrap"><table>
       <tr><th>Ticker</th><th>Shares</th><th>Entry</th><th>Now</th><th>Cost</th><th>Value</th><th>P&L</th></tr>
       ${d.marks.map(m => {
         const c = (m.pnl ?? 0) > 0 ? 'pos-v' : ((m.pnl ?? 0) < 0 ? 'neg-v' : '');
         return `<tr><td><strong>${m.ticker}</strong></td>
           <td class="num">${m.shares}</td>
           <td class="num">$${m.entry_price.toFixed(2)}</td>
           <td class="num">${m.price === null ? '—' : '$' + m.price.toFixed(2)}</td>
           <td class="num">$${m.cost.toFixed(2)}</td>
           <td class="num">${m.value === null ? '—' : '$' + m.value.toFixed(2)}</td>
           <td class="num ${c}">${m.pnl === null ? '—' :
             ((m.pnl >= 0 ? '+' : '−') + '$' + Math.abs(m.pnl).toFixed(2) +
              ` (${m.pnl_pct >= 0 ? '+' : ''}${m.pnl_pct.toFixed(1)}%)`)}</td></tr>`;
       }).join('')}
     </table></div>`;
}

async function startFund(auto) {
  const budget = Number($('#fund-budget').value);
  if (!Number.isFinite(budget) || budget <= 0) { alert('Enter a positive budget.'); return; }
  try {
    await api(auto ? '/api/fund/autopilot' : '/api/fund',
              { method: 'POST', body: JSON.stringify({ budget, top_n: 10 }) });
    loadFund();
  } catch (e) { alert('Could not start: ' + e.message); }
}

function renderAutopilot(d) {
  const b = d.benchmark_result;
  const mk = d.market || {};
  const waiting = !!d.blocked_reason;
  const cls = d.pnl > 0 ? 'pos-v' : (d.pnl < 0 ? 'neg-v' : '');
  const won = b && d.vs_benchmark > 0;
  const sells = (d.trades || []).filter(t => t.action === 'sell');
  const wins = sells.filter(t => t.pnl > 0).length;

  $('#fund-body').innerHTML =
    `<div class="cards" style="margin-bottom:1rem">
      <div class="metric"><div class="k">Fund value</div>
        <div class="v ${cls}">$${d.value.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
        <div class="n">from $${d.budget.toLocaleString()}</div></div>
      <div class="metric"><div class="k">Profit / loss</div>
        <div class="v ${cls}">${d.pnl >= 0 ? '+' : '−'}$${Math.abs(d.pnl).toFixed(2)}</div>
        <div class="n">${d.pnl_pct >= 0 ? '+' : ''}${d.pnl_pct.toFixed(2)}%</div></div>
      ${b ? `<div class="metric"><div class="k">${b.ticker} instead</div>
        <div class="v">${b.pnl >= 0 ? '+' : '−'}$${Math.abs(b.pnl).toFixed(2)}</div>
        <div class="n">${b.pnl_pct >= 0 ? '+' : ''}${b.pnl_pct.toFixed(2)}%</div></div>
      <div class="metric"><div class="k">vs ${b.ticker}</div>
        <div class="v ${won ? 'good' : 'bad'}">${d.vs_benchmark >= 0 ? '+' : '−'}$${Math.abs(d.vs_benchmark).toFixed(2)}</div>
        <div class="n">${won ? 'ahead' : 'behind'}</div></div>` : ''}
     </div>
     <div class="live-strip" style="margin-bottom:.8rem">
       <span class="dot-live ${mk.is_open ? 'on' : 'closed'}"></span>
       <span>${mk.is_open
         ? `<strong>Market open</strong> — trading live, checking every ${d.checks_every || '60s'}`
         : `<strong>Market ${mk.state || 'closed'}</strong> — holding until it opens`}</span>
       <span class="sub" style="margin-left:auto">${mk.exchange_time || ''}</span>
     </div>
     ${d.next_open ? `<div class="card" style="margin-bottom:.8rem;padding:.8rem 1rem">
        <div class="perf-head" style="margin:0">
          <span class="perf-total" style="font-size:1.15rem">First trade: ${d.next_open.exchange}</span>
          <span class="perf-since">your time <strong>${d.next_open.local}</strong> ·
            ${Math.floor(d.next_open.minutes_away / 60)}h ${d.next_open.minutes_away % 60}m away</span>
        </div>
        <p class="hint" style="margin-top:.4rem">
          ${d.autotrader_running
            ? `The server is watching the clock by itself (checking every ${d.checks_every || '60s'} once
               the bell rings), so it trades whether or not this page is open.`
            : `<span style="color:var(--warn)">The background trader is not running — it will only
               act while this page is open.</span>`} Market holidays are not modelled — if that day
          is a holiday it waits for the next real session.
        </p>
      </div>` : ''}
     ${waiting ? `<div class="warn-box" style="margin-bottom:.8rem">
       Sitting in cash. It places orders only while the exchange is open, at live prices —
       filling at a stale close would be a backfill, not a trade.</div>` : ''}
     <p class="hint">
       ${d.holdings.length} holdings · $${d.cash.toFixed(2)} cash ·
       ${(d.trades || []).length} trades${d.as_of ? ` · last acted ${d.as_of.replace('T', ' ')}` : ''}
       ${sells.length ? `· ${wins} of ${sells.length} closed trades profitable` : ''}.
       It does not buy at the open. For each name the model rates 90+ it rests a
       <strong>limit order ${((d.dip_pct ?? 0.01) * 100).toFixed(1)}% below the previous close</strong>
       and buys only if it trades there; it sells when the rank drops below 70. The anchor is
       yesterday's close, not the opening print — a name that gaps up still sits below its own
       inflated open, so anchoring there buys strength and calls it a discount. Measured over 133
       holds: <strong>1.54%</strong> per hold anchored to the previous close, <strong>1.44%</strong>
       anchored to the open, <strong>1.27%</strong> buying at the open outright.
     </p>
     ${(d.orders || []).length ? `<h4 style="font-size:.8rem;color:var(--ink-3);margin:1rem 0 .5rem">
        Resting orders — waiting for a dip</h4>
      <div class="table-wrap"><table>
        <tr><th>Ticker</th><th>Rank</th><th>Prev close</th><th>Now</th><th>Buy if it hits</th>
            <th>How often it gets there</th><th>Usual time of day</th></tr>
        ${d.orders.map(o => {
          const t = o.timing;
          const rate = t ? `${(t.hit_rate * 100).toFixed(0)}% of ${t.sessions} sessions` : '—';
          const when = t && t.median_time
            ? `${t.typical_from}–${t.typical_to} ET <span class="sub">(median ${t.median_time})</span>`
            : (t ? 'has not reached it recently' : '—');
          const cls = t && t.hit_rate >= 0.6 ? 'pos-v' : (t && t.hit_rate < 0.3 ? 'neg-v' : '');
          const gap = o.gap_pct;
          return `<tr><td><strong>${o.ticker}</strong></td>
            <td class="num">${o.rank}</td>
            <td class="num">$${o.reference.toFixed(2)}</td>
            <td class="num">${o.live_at_placement === undefined ? '—' :
              '$' + o.live_at_placement.toFixed(2) +
              (gap === null || gap === undefined ? '' :
                ` <span class="sub ${gap >= 0 ? 'pos-v' : 'neg-v'}">${gap >= 0 ? '+' : ''}${gap.toFixed(1)}%</span>`)}</td>
            <td class="num" style="color:var(--good)">$${o.limit.toFixed(2)}</td>
            <td class="num ${cls}">${rate}</td>
            <td>${when}</td></tr>`;
        }).join('')}
      </table></div>
      <p class="hint">
        “How often” and “usual time” are <strong>base rates from the last 60 sessions</strong> of
        5-minute bars — what each stock has actually done, not a forecast. A name that reached this
        price on three quarters of recent mornings will probably do so again, but which minute is
        not knowable in advance, and some days it simply never dips and the order does not fill.
      </p>` : ''}
     ${d.marks && d.marks.length ? `<div class="table-wrap"><table>
       <tr><th>Held</th><th>Shares</th><th>Entry</th><th>Now</th><th>Value</th><th>P&L</th></tr>
       ${d.marks.map(m => {
         const c = (m.pnl ?? 0) > 0 ? 'pos-v' : ((m.pnl ?? 0) < 0 ? 'neg-v' : '');
         return `<tr><td><strong>${m.ticker}</strong></td><td class="num">${m.shares}</td>
           <td class="num">$${m.entry_price.toFixed(2)}</td>
           <td class="num">${m.price === null ? '—' : '$' + m.price.toFixed(2)}</td>
           <td class="num">${m.value === null ? '—' : '$' + m.value.toFixed(2)}</td>
           <td class="num ${c}">${m.pnl === undefined || m.pnl === null ? '—' :
             (m.pnl >= 0 ? '+' : '−') + '$' + Math.abs(m.pnl).toFixed(2)}</td></tr>`;
       }).join('')}</table></div>` : '<p class="hint">No open positions.</p>'}
     <h4 style="font-size:.8rem;color:var(--ink-3);margin:1.1rem 0 .5rem">Trade log</h4>
     <div class="table-wrap" style="max-height:280px;overflow-y:auto"><table>
       ${(d.trades || []).slice().reverse().map(t => `<tr>
         <td class="num" style="font-size:.72rem">${t.date}</td>
         <td><span class="pos ${t.action === 'buy' ? 'long' : 'short'}">${t.action}</span></td>
         <td><strong>${t.ticker}</strong></td>
         <td class="num">${t.shares} @ $${t.price.toFixed(2)}</td>
         <td class="num ${t.pnl > 0 ? 'pos-v' : (t.pnl < 0 ? 'neg-v' : '')}">${
           t.pnl === undefined ? '' : (t.pnl >= 0 ? '+' : '−') + '$' + Math.abs(t.pnl).toFixed(2)}</td>
         <td class="sub" style="font-size:.7rem">${t.reason}</td></tr>`).join('')}
     </table></div>`;
}

if ($('#fund-reset')) {
  $('#fund-reset').onclick = async () => {
    if (!confirm('Restart the fund? The current run is discarded.')) return;
    try { await api('/api/fund', { method: 'DELETE' }); loadFund(); } catch (_) {}
  };
}


function markerWhen(day, span, anchorLabel) {
  // A marker inside one session must read as a clock time, not "+0.897d",
  // and it has to be measured from the anchor the forecast was drawn at.
  return futureLabel(day, anchorLabel, span);
}


/* ══ what the model plans to pay, and when ══════════════ */
function planLine(r, dipPct) {
  const t = r.dip;
  const pct = ((dipPct ?? 0.01) * 100).toFixed(1);
  if (r.price === null) return '';
  // The limit is anchored to the previous close, which is what r.price is
  // whenever the market is shut — the same number the backend uses.
  const limit = r.price * (1 - (dipPct ?? 0.01));

  const when = !t ? ''
    : !t.median_time
      ? `<span class="plan-note">has not reached this in ${t.sessions} sessions</span>`
      : `<span class="plan-when">${t.typical_from}–${t.typical_to} ET</span>
         <span class="plan-rate">${(t.hit_rate * 100).toFixed(0)}% of days</span>`;

  return `<div class="wl-plan">
      <div class="plan-head">Model plans to buy at</div>
      <div class="plan-px">${fmtPrice(limit, r.symbol)}</div>
      <div class="plan-sub">${pct}% under the ${fmtPrice(r.price, r.symbol)} previous close</div>
      <div class="plan-when-row">${when}</div>
    </div>`;
}

