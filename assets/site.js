/* Shared chrome: theme toggle, nav highlighting, tooltip singleton, and small
   SVG helpers. No dependencies and no build step -- everything here runs from
   file:// or from GitHub Pages unchanged. */

/* The eight-chapter primer. This sequence is stable -- it is what a new lab
   member is pointed at, and the numbering is cited elsewhere. Do not add to it.
   Ongoing project write-ups go in EXTRAS below, which renders as a dropdown
   after the chapters rather than extending the primer. */
export const CHAPTERS = [
  { href: 'index.html', short: 'Start', title: 'A primer on classifying memory from EEG' },
  { href: 'task.html', short: '1 Task', title: 'The memory task' },
  { href: 'why-classifiers.html', short: '2 Why', title: 'Why classifiers, not just averages' },
  { href: 'features.html', short: '3 Features', title: 'From EEG to 280 numbers' },
  { href: 'how-it-learns.html', short: '4 Learning', title: 'How a classifier actually learns' },
  { href: 'try-it.html', short: '5 Try it', title: 'Train one yourself, on real data' },
  { href: 'results.html', short: '6 Results', title: 'What we have found so far' },
  { href: 'open-question.html', short: '7 Open', title: 'The question we are trying to answer' },
  { href: 'reading.html', short: '8 Reading', title: 'Where to go next' },
];

/* Live project write-ups. Unlike CHAPTERS these are expected to grow, and they
   are not a reading sequence -- each stands alone. */
export const EXTRAS = {
  label: 'Projects',
  items: [
    { href: 'simulation.html', short: 'How much data', title: 'How much data would it take?',
      blurb: 'The trial-count simulation: how many trials a memory classifier needs, and why our pipeline comparisons come out flat.' },
    { href: 'sereega.html', short: 'SEREEGA simulator', title: 'Simulating EEG you already know the answer to',
      blurb: 'A rotatable 3D head with a draggable dipole, signal classes, and a live decode — plus what our own runs found.' },
    { href: 'lda.html', short: 'LDA deep dive', title: 'LDA, from the ground up',
      blurb: 'Every weight, where its value comes from, and what the model assumes. Assumes no machine learning background.' },
    { href: 'svm.html', short: 'SVM deep dive', title: 'SVM, from the ground up',
      blurb: 'The same walkthrough for support vector machines, section by section, so the two can be compared.' },
  ],
};

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
  const inExtras = EXTRAS.items.some((c) => c.href === here);

  const bar = document.querySelector('.topbar');
  if (bar) {
    const nav = bar.querySelector('.topbar__nav');
    if (nav) {
      const links = CHAPTERS.map((c) => {
        const cur = c.href === here ? ' aria-current="page"' : '';
        return `<a href="${c.href}"${cur}>${c.short}</a>`;
      }).join('');

      nav.innerHTML = links;

      /* The dropdown is a details/summary rather than a custom widget: it gets
         keyboard support, Escape-to-close and focus handling from the browser,
         and it still works with no JS at all.

         It is inserted as a SIBLING of .topbar__nav, never inside it. That nav
         sets overflow-x:auto so the chapter row can scroll on a phone, and per
         spec an auto overflow-x forces overflow-y to compute to auto as well --
         making the nav a scroll container that clips in BOTH axes. An
         absolutely-positioned panel inside it is silently cropped to nothing,
         which is exactly what happened the first time this shipped. */
      const items = EXTRAS.items.map((c) => {
        const cur = c.href === here ? ' aria-current="page"' : '';
        return `<a href="${c.href}"${cur}>
            <span class="navmenu__ttl">${c.short}</span>
            <span class="navmenu__sub">${c.blurb}</span>
          </a>`;
      }).join('');

      const menu = document.createElement('details');
      menu.className = 'navmenu';
      if (inExtras) menu.setAttribute('data-current', '1');
      menu.innerHTML = `<summary>${EXTRAS.label}<span class="navmenu__caret" aria-hidden="true">▾</span></summary>`;
      nav.insertAdjacentElement('afterend', menu);

      /* The panel lives on <body>, not inside the <details>, and is positioned
         with JS. Two separate things in this topbar would otherwise break it:

           1. .topbar__nav sets overflow-x:auto, and per spec that forces
              overflow-y to auto too -- so it clips in BOTH axes and an absolutely
              positioned panel inside it is cropped to nothing. (This is the bug
              that shipped.) Being a sibling of the nav fixes that one.
           2. .topbar sets backdrop-filter, which makes it the containing block
              for position:fixed descendants. So even a fixed panel would be
              positioned against the topbar rather than the viewport.

         Hanging the panel off <body> sidesteps both, and cannot be clipped by
         any ancestor, so it survives future changes to the bar's styling. */
      const panel = document.createElement('div');
      panel.className = 'navmenu__panel';
      panel.innerHTML = items;
      document.body.appendChild(panel);

      const place = () => {
        const r = menu.getBoundingClientRect();
        panel.style.top = `${r.bottom + 6}px`;
        const w = panel.offsetWidth;
        const left = Math.min(Math.max(8, r.right - w), innerWidth - w - 8);
        panel.style.left = `${left}px`;
      };
      const setOpen = (on) => {
        menu.open = on;
        if (on) { panel.setAttribute('data-open', ''); place(); }
        else panel.removeAttribute('data-open');
      };

      menu.addEventListener('toggle', () => setOpen(menu.open));
      addEventListener('resize', () => { if (menu.open) place(); });
      addEventListener('scroll', () => { if (menu.open) place(); }, true);
      document.addEventListener('click', (e) => {
        if (!menu.contains(e.target) && !panel.contains(e.target)) setOpen(false);
      });
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && menu.open) { setOpen(false); menu.querySelector('summary').focus(); }
      });
    }
  }

  /* Prev/next walks whichever sequence the page belongs to. The primer runs
     index -> 8; a project page is not part of that sequence, so chapter 8 stays
     the end of the primer and a project page gets no pager links of its own. */
  const pager = document.querySelector('.pager');
  if (pager) {
    const seq = inExtras ? EXTRAS.items : CHAPTERS;
    const i = seq.findIndex((c) => c.href === here);
    const prev = seq[i - 1];
    const next = seq[i + 1];
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

/**
 * A real <canvas> laid over an SVG chart's plot area, for marks too numerous to
 * be SVG nodes (tens of thousands of points, a 450x280 heatmap).
 *
 * Do NOT do this by drawing to an offscreen canvas and handing toDataURL() to an
 * SVG <image>: decoding a data URL is asynchronous, so during a slider drag each
 * redraw replaces the image before the last one has decoded and the marks vanish
 * until you let go. A canvas in the page paints synchronously.
 *
 * `rect` is the plot area in viewBox units; the returned context is pre-scaled so
 * the caller draws in those same units, with the rect's top-left as the origin.
 * The canvas sits under the SVG, so axes and labels stay on top.
 */
export function canvasOverlay(host, svg, vbW, vbH, rect, ss = 2) {
  host.style.position = 'relative';
  svg.style.position = 'relative';
  svg.style.zIndex = '1';
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(rect.w * ss));
  c.height = Math.max(1, Math.round(rect.h * ss));
  Object.assign(c.style, {
    position: 'absolute',
    left: `${(rect.x / vbW) * 100}%`,
    top: `${(rect.y / vbH) * 100}%`,
    width: `${(rect.w / vbW) * 100}%`,
    height: `${(rect.h / vbH) * 100}%`,
    zIndex: '0',
    pointerEvents: 'none',
  });
  host.insertBefore(c, svg);
  const ctx = c.getContext('2d');
  ctx.scale(ss, ss);
  return ctx;
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
/**
 * Mouse coordinates -> viewBox units.
 *
 * The obvious version -- `(ev.clientX - rect.left) / rect.width * viewBoxWidth`
 * -- is wrong on this site, and silently. `svg.chart` caps height at
 * min(42vh, 24rem), so any viewBox tall enough to hit that cap gets scaled down
 * and centred by preserveAspectRatio="xMidYMid meet". The drawing then occupies
 * only part of the element box, with letterbox bars either side, and arithmetic
 * on the bounding box maps the bars onto real data: every reading comes out
 * displaced, by an amount that changes with the window size.
 *
 * getScreenCTM() is the transform the browser actually used, letterboxing
 * included, so inverting it is exact wherever the SVG ends up. Use this for
 * every pointer handler rather than measuring the element.
 */
export function clientToViewBox(svg, ev) {
  const ctm = svg.getScreenCTM();
  if (!ctm) return { x: NaN, y: NaN };
  const pt = svg.createSVGPoint();
  pt.x = ev.clientX;
  pt.y = ev.clientY;
  const p = pt.matrixTransform(ctm.inverse());
  return { x: p.x, y: p.y };
}

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
