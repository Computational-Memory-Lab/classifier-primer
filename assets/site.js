/* Shared chrome: theme toggle, nav highlighting, tooltip singleton, and small
   SVG helpers. No dependencies and no build step -- everything here runs from
   file:// or from GitHub Pages unchanged. */

export const CHAPTERS = [
  { href: 'index.html', short: 'Start', title: 'A primer on classifying memory from EEG' },
  { href: 'task.html', short: '1 Task', title: 'The memory task' },
  { href: 'why-classifiers.html', short: '2 Why', title: 'Why classifiers, not just averages' },
  { href: 'features.html', short: '3 Features', title: 'From brain waves to 280 numbers' },
  { href: 'how-it-learns.html', short: '4 Learning', title: 'How a classifier actually learns' },
  { href: 'try-it.html', short: '5 Try it', title: 'Train one yourself, on real data' },
  { href: 'results.html', short: '6 Results', title: 'What we have found so far' },
  { href: 'open-question.html', short: '7 Open', title: 'The question we are trying to answer' },
  { href: 'reading.html', short: '8 Reading', title: 'Where to go next' },
];

/* ---------- theme ---------- */

const THEME_KEY = 'cml-primer-theme';

export function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === 'light' || saved === 'dark') {
    document.documentElement.setAttribute('data-theme', saved);
  }
  const btn = document.querySelector('.theme-toggle');
  if (!btn) return;
  const paint = () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark'
      || (!document.documentElement.hasAttribute('data-theme')
          && matchMedia('(prefers-color-scheme: dark)').matches);
    btn.textContent = dark ? '☀' : '☾';
    btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  };
  paint();
  btn.addEventListener('click', () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark'
      || (!document.documentElement.hasAttribute('data-theme')
          && matchMedia('(prefers-color-scheme: dark)').matches);
    const next = dark ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem(THEME_KEY, next);
    paint();
    window.dispatchEvent(new CustomEvent('themechange'));
  });
}

/* ---------- chrome ---------- */

function currentPage() {
  const f = location.pathname.split('/').pop();
  return f === '' ? 'index.html' : f;
}

export function initChrome() {
  const here = currentPage();
  const bar = document.querySelector('.topbar');
  if (bar) {
    const nav = bar.querySelector('.topbar__nav');
    if (nav) {
      nav.innerHTML = CHAPTERS.map((c) => {
        const cur = c.href === here ? ' aria-current="page"' : '';
        return `<a href="${c.href}"${cur}>${c.short}</a>`;
      }).join('');
    }
  }
  const pager = document.querySelector('.pager');
  if (pager) {
    const i = CHAPTERS.findIndex((c) => c.href === here);
    const prev = CHAPTERS[i - 1];
    const next = CHAPTERS[i + 1];
    pager.innerHTML = [
      prev ? `<a href="${prev.href}"><div class="dir">← Previous</div><div class="ttl">${prev.title}</div></a>` : '<div></div>',
      next ? `<a class="next" href="${next.href}"><div class="dir">Next →</div><div class="ttl">${next.title}</div></a>` : '<div></div>',
    ].join('');
  }
  initTheme();
}

/* ---------- tooltip ---------- */

let tipEl = null;

export function tooltip() {
  if (!tipEl) {
    tipEl = document.createElement('div');
    tipEl.className = 'tooltip';
    document.body.appendChild(tipEl);
  }
  return {
    show(html, x, y) {
      tipEl.innerHTML = html;
      tipEl.setAttribute('data-show', '');
      const r = tipEl.getBoundingClientRect();
      let left = x + 14;
      let top = y - r.height - 10;
      if (left + r.width > innerWidth - 8) left = x - r.width - 14;
      if (top < 8) top = y + 16;
      tipEl.style.left = `${Math.max(8, left)}px`;
      tipEl.style.top = `${top}px`;
    },
    hide() { tipEl.removeAttribute('data-show'); },
  };
}

/* ---------- svg helpers ---------- */

export const NS = 'http://www.w3.org/2000/svg';

export function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

export function svgRoot(host, w, h) {
  host.innerHTML = '';
  const svg = el('svg', {
    class: 'chart',
    viewBox: `0 0 ${w} ${h}`,
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
  }, host);
  return svg;
}

export function scale(d0, d1, r0, r1) {
  const m = (r1 - r0) / (d1 - d0 || 1);
  const f = (v) => r0 + (v - d0) * m;
  f.invert = (p) => d0 + (p - r0) / m;
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
}

export function ticks(d0, d1, count = 5) {
  const span = d1 - d0;
  if (span === 0) return [d0];
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(d0 / step) * step; v <= d1 + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toFixed(10));
  }
  return out;
}

export function linePath(points) {
  return points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)} ${p[1].toFixed(2)}`).join('');
}

/** Draws grid + axes and returns the plot rect. */
export function frame(svg, w, h, pad, xs, ys, opts = {}) {
  const { xLabel, yLabel, xTicks, yTicks, xFmt = String, yFmt = String } = opts;
  const g = el('g', {}, svg);
  const xt = xTicks || ticks(xs.domain[0], xs.domain[1], 6);
  const yt = yTicks || ticks(ys.domain[0], ys.domain[1], 5);

  for (const v of yt) {
    const y = ys(v);
    el('line', { class: 'grid', x1: pad.l, x2: w - pad.r, y1: y, y2: y }, g);
    el('text', { class: 'tick', x: pad.l - 7, y: y + 3.5, 'text-anchor': 'end' }, g)
      .textContent = yFmt(v);
  }
  for (const v of xt) {
    const x = xs(v);
    el('text', { class: 'tick', x, y: h - pad.b + 14, 'text-anchor': 'middle' }, g)
      .textContent = xFmt(v);
  }
  el('line', { class: 'axis', x1: pad.l, x2: w - pad.r, y1: h - pad.b, y2: h - pad.b }, g);
  el('line', { class: 'axis', x1: pad.l, x2: pad.l, y1: pad.t, y2: h - pad.b }, g);

  if (xLabel) {
    el('text', { class: 'axis-label', x: (pad.l + w - pad.r) / 2, y: h - 2, 'text-anchor': 'middle' }, g)
      .textContent = xLabel;
  }
  if (yLabel) {
    const cy = (pad.t + h - pad.b) / 2;
    el('text', {
      class: 'axis-label', x: 11, y: cy, 'text-anchor': 'middle',
      transform: `rotate(-90 11 ${cy})`,
    }, g).textContent = yLabel;
  }
  return g;
}

/* ---------- stats ---------- */

export const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

export function sd(a) {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
}

/** Rank-based AUC (handles ties), identical in spirit to MATLAB's perfcurve. */
export function auc(labels, scores) {
  const n = labels.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[a] - scores[b]);
  const ranks = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[idx[j + 1]] === scores[idx[i]]) j++;
    const r = (i + j + 2) / 2; // average of 1-based ranks
    for (let k = i; k <= j; k++) ranks[idx[k]] = r;
    i = j + 1;
  }
  let n1 = 0, n0 = 0, sum = 0;
  for (let k = 0; k < n; k++) {
    if (labels[k] === 1) { n1++; sum += ranks[k]; } else n0++;
  }
  if (n1 === 0 || n0 === 0) return NaN;
  return (sum - (n1 * (n1 + 1)) / 2) / (n1 * n0);
}

/** ROC points, ordered by descending score. */
export function roc(labels, scores) {
  const n = labels.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const P = labels.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
  const N = n - P;
  const pts = [[0, 0]];
  let tp = 0, fp = 0;
  for (const i of idx) {
    if (labels[i] === 1) tp++; else fp++;
    pts.push([fp / N, tp / P]);
  }
  return pts;
}

export const fmtAuc = (v) => (Number.isFinite(v) ? v.toFixed(3) : '—');

/** Two-sided p-value for a one-sample t-test, via a continued fraction for the
    incomplete beta function. Enough precision for display. */
export function tTest(values, mu = 0.5) {
  const n = values.length;
  const m = mean(values);
  const s = sd(values);
  const se = s / Math.sqrt(n);
  const t = (m - mu) / se;
  const df = n - 1;
  const x = df / (df + t * t);
  const p = betaInc(x, df / 2, 0.5);
  return { t, df, p, mean: m, se, ci95: 1.96 * se };
}

function betaInc(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
  let f = 1, c = 1, d = 0;
  for (let i = 0; i <= 300; i++) {
    const m = Math.floor(i / 2);
    let num;
    if (i === 0) num = 1;
    else if (i % 2 === 0) num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    else num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + num * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    c = 1 + num / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    f *= c * d;
    if (Math.abs(1 - c * d) < 1e-10) break;
  }
  return front * (f - 1);
}

function lgamma(z) {
  const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < g.length; i++) x += g[i] / (z + i + 1);
  const t = z + g.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/* ---------- deterministic RNG (so demos are reproducible) ---------- */

export function rng(seed = 1) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

export function gauss(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ---------- data loading ---------- */

const cache = new Map();

export async function loadJSON(path) {
  if (!cache.has(path)) cache.set(path, fetch(path).then((r) => {
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
  }));
  return cache.get(path);
}

export async function loadBin(path) {
  if (!cache.has(path)) cache.set(path, fetch(path).then((r) => {
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.arrayBuffer();
  }));
  return cache.get(path);
}

/** Reads a CSS custom property off :root (so charts follow the theme). */
export function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Re-run a draw function whenever the theme or size changes. */
export function responsive(host, draw) {
  let raf = null;
  const go = () => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(draw);
  };
  window.addEventListener('themechange', go);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', go);
  new ResizeObserver(go).observe(host);
  draw();
}
