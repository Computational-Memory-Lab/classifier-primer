/* Interactives for the trial-count simulation chapter.

   Four figures, in the order the argument runs:
     1  trialCurve    how achieved AUC depends on trials, features and effect size
     2  ldaMechanics  what LDA actually computes, against the naive rule
     3  svmMargin     what an SVM actually computes, step by step
     4  smallN        why the two behave differently when trials are scarce

   Closed forms are used where one exists and honest training where it does not,
   so the pictures cannot drift away from the claims in the prose. Everything
   runs from file:// with no build step, like the rest of the site. Charts use a
   fixed viewBox and are scaled by CSS, matching how-it-learns.html. */

import { el, svgRoot, scale, frame, linePath, token, responsive } from './site.js';

/* ============================== shared maths ============================== */

/** Normal CDF, Abramowitz & Stegun 26.2.17 -- accurate to ~7.5e-8. */
export function normCdf(z) {
  const s = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return 0.5 * (1 + s * y);
}

/** Inverse normal CDF (Acklam). */
export function normInv(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Separation that yields a given asymptotic AUC.  AUC = Phi(delta/sqrt2). */
export const deltaForAuc = (a) => Math.SQRT2 * normInv(a);

/**
 * Achieved AUC for sample LDA. The estimated direction carries noise of squared
 * norm ~4p/n, so the recovered separation is delta^2 / sqrt(delta^2 + 4p/n).
 * This matched the cluster simulation to a median of 0.007 AUC across the grid.
 */
export function achievedAuc(delta, p, n) {
  const d2 = delta * delta;
  return normCdf((d2 / Math.sqrt(d2 + 4 * p / n)) / Math.SQRT2);
}

/** Smallest n reaching a given fraction of the way from 0.5 to the ceiling. */
export function trialsForFraction(delta, p, frac) {
  const ceil = normCdf(delta / Math.SQRT2);
  const target = 0.5 + frac * (ceil - 0.5);
  let lo = 4, hi = 1e12;
  for (let i = 0; i < 160; i++) {
    const mid = Math.sqrt(lo * hi);
    if (achievedAuc(delta, p, mid) < target) lo = mid; else hi = mid;
  }
  return hi;
}

/* ------------------------------- 2-D data -------------------------------- */

/* A small deterministic generator, so every redraw of the same settings gives
   the same picture and sliders feel continuous rather than reshuffling. */
function makeRand(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}
function makeGauss(rand) {
  return () => {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

/** Two Gaussian clouds sharing a covariance set by the sds and correlation. */
export function twoClouds({ n = 90, sep = 2, sx = 1, sy = 1, rho = 0, seed = 7 }) {
  const g = makeGauss(makeRand(seed));
  const L21 = sy * rho, L22 = sy * Math.sqrt(Math.max(1e-9, 1 - rho * rho));
  const pts = [];
  for (let c = 0; c < 2; c++) {
    const mx = c === 0 ? -sep / 2 : sep / 2;
    for (let i = 0; i < n; i++) {
      const z1 = g(), z2 = g();
      pts.push({ x: mx + sx * z1, y: L21 * z1 + L22 * z2, c });
    }
  }
  return pts;
}

/** Class means and the pooled within-class covariance. */
export function moments(pts) {
  const m = [[0, 0], [0, 0]], k = [0, 0];
  for (const p of pts) { m[p.c][0] += p.x; m[p.c][1] += p.y; k[p.c]++; }
  for (const c of [0, 1]) { m[c][0] /= k[c] || 1; m[c][1] /= k[c] || 1; }
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) {
    const dx = p.x - m[p.c][0], dy = p.y - m[p.c][1];
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  const dof = Math.max(1, pts.length - 2);
  return { m, S: [[sxx / dof, sxy / dof], [sxy / dof, syy / dof]] };
}

/**
 * LDA weights with fitcdiscr-style shrinkage: the CORRELATION matrix is pulled
 * toward the identity by gamma and then rescaled by the standard deviations.
 * gamma = 0 trusts the measured covariance; gamma = 1 discards the correlations
 * entirely, which is what the published study-phase analysis does.
 */
export function ldaWeights(pts, gamma = 0) {
  const { m, S } = moments(pts);
  const s1 = Math.sqrt(Math.max(1e-12, S[0][0]));
  const s2 = Math.sqrt(Math.max(1e-12, S[1][1]));
  const r = S[0][1] / (s1 * s2);
  const rg = r * (1 - gamma);
  const A = [[s1 * s1, rg * s1 * s2], [rg * s1 * s2, s2 * s2]];
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0] || 1e-12;
  const inv = [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]];
  const d = [m[1][0] - m[0][0], m[1][1] - m[0][1]];
  const w = [inv[0][0] * d[0] + inv[0][1] * d[1], inv[1][0] * d[0] + inv[1][1] * d[1]];
  return { w, m, S, d, rho: r };
}

/**
 * Soft-margin linear SVM by subgradient descent on the hinge loss. Labels are
 * +/-1 and C is the box constraint. Slow and simple, which is fine for a few
 * dozen points, and it optimises the real objective rather than approximating
 * the picture.
 */
export function svmTrain(pts, C = 1, iters = 3000) {
  let w0 = 0, w1 = 0, b = 0;
  const N = pts.length || 1;
  for (let t = 1; t <= iters; t++) {
    const eta = 1 / (0.05 * t + 1);
    let g0 = w0, g1 = w1, gb = 0;                 // gradient of 0.5*||w||^2
    for (const p of pts) {
      const yi = p.c === 1 ? 1 : -1;
      if (yi * (w0 * p.x + w1 * p.y + b) < 1) {
        g0 -= C * yi * p.x; g1 -= C * yi * p.y; gb -= C * yi;
      }
    }
    w0 -= eta * g0 / N; w1 -= eta * g1 / N; b -= eta * gb / N;
  }
  const norm = Math.hypot(w0, w1) || 1e-9;
  const sv = pts.filter((p) => {
    const yi = p.c === 1 ? 1 : -1;
    return yi * (w0 * p.x + w1 * p.y + b) <= 1.02;
  });
  return { w: [w0, w1], b, margin: 1 / norm, sv };
}

/* --------------------------- drawing utilities --------------------------- */

const W = 470, H = 400;
const PAD = { l: 46, r: 18, t: 16, b: 42 };

/** A line through `mid` perpendicular to `vec`, with a label at one end. */
/**
 * A <g> clipped to the plot rectangle.
 *
 * Needed because svg.chart sets `overflow: visible` site-wide -- deliberately,
 * so direct labels near an axis are not cut off -- while boundary() draws a line
 * 12 data units long through a window of about +/-4.6. Without this the decision
 * lines ran out over the surrounding text. Labels stay OUTSIDE the clip so they
 * keep the benefit of the global overflow rule.
 */
let clipSeq = 0;
function clipped(svg, w, h, pad) {
  const id = `plotclip${++clipSeq}`;
  const cp = el('clipPath', { id }, el('defs', {}, svg));
  el('rect', {
    x: pad.l, y: pad.t, width: w - pad.l - pad.r, height: h - pad.t - pad.b,
  }, cp);
  return el('g', { 'clip-path': `url(#${id})` }, svg);
}

function boundary(svg, xs, ys, mid, vec, colour, dash, label, clip) {
  const n = Math.hypot(vec[0], vec[1]) || 1;
  const ux = -vec[1] / n, uy = vec[0] / n;
  const L = 12;
  el('line', {
    x1: xs(mid[0] - ux * L), y1: ys(mid[1] - uy * L),
    x2: xs(mid[0] + ux * L), y2: ys(mid[1] + uy * L),
    stroke: colour, 'stroke-width': 2.6, 'stroke-dasharray': dash || null,
    'stroke-linecap': 'round',
  }, clip || svg);
  if (label) {
    el('text', {
      x: xs(mid[0] + ux * 2.3) + 7, y: ys(mid[1] + uy * 2.3),
      class: 'axis-label', fill: colour,
    }, svg).textContent = label;
  }
}

const stat = (label, value) =>
  `<div class="stat"><div class="stat__label">${label}</div><div class="stat__value">${value}</div></div>`;

/* ============================== figure 1 ================================= */

export function trialCurve(host, out) {
  const state = { ceil: 0.531, p: 120, n: 225 };
  const LO = Math.log10(25), HI = Math.log10(20000);

  const draw = () => {
    const svg = svgRoot(host, W, H);
    const yTop = 0.85;
    const xs = scale(LO, HI, PAD.l, W - PAD.r);
    const ys = scale(0.48, yTop, H - PAD.b, PAD.t);
    frame(svg, W, H, PAD, xs, ys, {
      xTicks: [25, 100, 1000, 10000].map(Math.log10),
      xFmt: (v) => { const n = Math.round(10 ** v); return n >= 1000 ? `${n / 1000}k` : `${n}`; },
      yTicks: [0.5, 0.6, 0.7, 0.8],
      yFmt: (v) => v.toFixed(2),
      xLabel: 'study trials per participant',
      yLabel: 'achieved AUC',
    });

    const delta = deltaForAuc(state.ceil);
    const clamp = (v) => Math.min(yTop, Math.max(0.48, v));

    el('line', {
      x1: PAD.l, x2: W - PAD.r, y1: ys(clamp(state.ceil)), y2: ys(clamp(state.ceil)),
      stroke: token('--series-6'), 'stroke-width': 1.6, 'stroke-dasharray': '5 4',
    }, svg);
    el('text', {
      x: W - PAD.r - 3, y: ys(clamp(state.ceil)) - 6, 'text-anchor': 'end',
      class: 'axis-label', fill: token('--series-6'),
    }, svg).textContent = `ceiling ${state.ceil.toFixed(3)}`;

    for (const mk of [{ n: 225, t: 'our study phase' }, { n: 720, t: 'feedback task' }]) {
      const x = xs(Math.log10(mk.n));
      el('line', {
        x1: x, x2: x, y1: PAD.t, y2: H - PAD.b,
        stroke: token('--baseline'), 'stroke-width': 1, 'stroke-dasharray': '3 3',
      }, svg);
      el('text', {
        x: x + 4, y: PAD.t + 11, class: 'axis-label', fill: token('--text-muted'),
      }, svg).textContent = mk.t;
    }

    const pts = [];
    for (let i = 0; i <= 200; i++) {
      const lx = LO + (HI - LO) * i / 200;
      pts.push([xs(lx), ys(clamp(achievedAuc(delta, state.p, 10 ** lx)))]);
    }
    el('path', {
      d: linePath(pts), fill: 'none', stroke: token('--series-1'), 'stroke-width': 2.6,
    }, svg);

    const got = achievedAuc(delta, state.p, state.n);
    el('circle', {
      cx: xs(Math.log10(state.n)), cy: ys(clamp(got)), r: 7,
      fill: token('--surface-1'), stroke: token('--series-1'), 'stroke-width': 3,
    }, svg);

    const need = trialsForFraction(delta, state.p, 0.9);
    const frac = (got - 0.5) / Math.max(1e-9, state.ceil - 0.5);
    out.innerHTML = stat('you would measure', got.toFixed(3))
      + stat('of the ceiling', `${(100 * frac).toFixed(0)}%`)
      + stat('trials for 90% of it', need > 5e5 ? '500k+' : Math.round(need).toLocaleString());
  };

  responsive(host, draw);
  return { state, refresh: draw };
}

/* ============================== figure 2 ================================= */

export function ldaMechanics(host, out) {
  const state = { rho: 0.75, gamma: 0 };
  const R = 4.6;

  const draw = () => {
    const svg = svgRoot(host, W, H);
    const xs = scale(-R, R, PAD.l, W - PAD.r);
    const ys = scale(-R, R, H - PAD.b, PAD.t);
    frame(svg, W, H, PAD, xs, ys, {
      xTicks: [], yTicks: [],
      xLabel: 'a feature (say, Pz at 500 ms)', yLabel: 'another feature',
    });

    const pts = twoClouds({ n: 90, sep: 1.7, sx: 1, sy: 1, rho: state.rho, seed: 11 });
    for (const p of pts) {
      el('circle', {
        cx: xs(p.x), cy: ys(p.y), r: 3.4,
        fill: token(p.c === 1 ? '--c-hit' : '--c-miss'), opacity: 0.5,
      }, svg);
    }

    const { w, m, S, d } = ldaWeights(pts, state.gamma);

    {
      const tr = S[0][0] + S[1][1];
      const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
      const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      const l1 = Math.sqrt(Math.max(1e-9, tr / 2 + disc));
      const l2 = Math.sqrt(Math.max(1e-9, tr / 2 - disc));
      const th = 0.5 * Math.atan2(2 * S[0][1], S[0][0] - S[1][1]);
      const unit = xs(1) - xs(0);
      for (const c of [0, 1]) {
        el('ellipse', {
          cx: xs(m[c][0]), cy: ys(m[c][1]), rx: l1 * unit, ry: l2 * unit,
          transform: `rotate(${-th * 180 / Math.PI} ${xs(m[c][0])} ${ys(m[c][1])})`,
          fill: 'none', stroke: token('--text-muted'),
          'stroke-width': 1.6, 'stroke-dasharray': '4 3',
        }, svg);
      }
    }

    for (const c of [0, 1]) {
      el('circle', {
        cx: xs(m[c][0]), cy: ys(m[c][1]), r: 7,
        fill: token(c === 1 ? '--c-hit' : '--c-miss'),
        stroke: token('--surface-1'), 'stroke-width': 2.5,
      }, svg);
    }

    const mid = [(m[0][0] + m[1][0]) / 2, (m[0][1] + m[1][1]) / 2];
    const clip = clipped(svg, W, H, PAD);
    boundary(svg, xs, ys, mid, d, token('--series-4'), '5 4', 'difference of means', clip);
    boundary(svg, xs, ys, mid, w, token('--series-7'), null, 'LDA', clip);

    let ang = Math.abs(Math.atan2(w[1], w[0]) - Math.atan2(d[1], d[0])) * 180 / Math.PI;
    if (ang > 180) ang = 360 - ang;
    if (ang > 90) ang = 180 - ang;
    out.innerHTML = stat('correlation', state.rho.toFixed(2))
      + stat('shrinkage &Gamma;', state.gamma.toFixed(2))
      + stat('angle between the rules', `${ang.toFixed(0)}&deg;`);
  };

  responsive(host, draw);
  return { state, refresh: draw };
}

/* ============================== whitening ================================

   The claim this figure has to make good: in the whitened space the naive rule
   IS the right rule. So it does not draw a before and an after -- it drags the
   space continuously from one to the other and draws both rules the whole way,
   because the fact worth seeing is the two lines converging, not two pictures
   side by side.

   Nothing is precomputed. At every slider position the points are transformed,
   the moments are recomputed FROM the transformed points, and both rules are
   derived there. So the lines coinciding at the far end is an outcome, not a
   thing the code arranges.                                                   */

/** Sigma^(-t/2) for a 2x2 symmetric positive-definite matrix. t=0 identity. */
function invSqrtPow(S, t) {
  const [a, b, c] = [S[0][0], S[0][1], S[1][1]];
  const tr = a + c, det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = Math.max(1e-12, tr / 2 + disc);
  const l2 = Math.max(1e-12, tr / 2 - disc);
  // eigenvectors: for a symmetric 2x2, (b, l-a) unless b is ~0
  let v1, v2;
  if (Math.abs(b) > 1e-12) {
    const n1 = Math.hypot(b, l1 - a), n2 = Math.hypot(b, l2 - a);
    v1 = [b / n1, (l1 - a) / n1]; v2 = [b / n2, (l2 - a) / n2];
  } else { v1 = [1, 0]; v2 = [0, 1]; }
  const f1 = Math.pow(l1, -t / 2), f2 = Math.pow(l2, -t / 2);
  return [
    [f1 * v1[0] * v1[0] + f2 * v2[0] * v2[0], f1 * v1[0] * v1[1] + f2 * v2[0] * v2[1]],
    [f1 * v1[0] * v1[1] + f2 * v2[0] * v2[1], f1 * v1[1] * v1[1] + f2 * v2[1] * v2[1]],
  ];
}

export function whitening(host, out) {
  const state = { rho: 0.85, t: 0 };
  const R = 4.6;

  const draw = () => {
    const svg = svgRoot(host, W, H);
    const xs = scale(-R, R, PAD.l, W - PAD.r);
    const ys = scale(-R, R, H - PAD.b, PAD.t);

    const raw = twoClouds({ n: 80, sep: 1.9, sx: 1, sy: 1, rho: state.rho, seed: 17 });
    const A = invSqrtPow(moments(raw).S, state.t);
    let pts = raw.map((p) => ({
      x: A[0][0] * p.x + A[0][1] * p.y, y: A[1][0] * p.x + A[1][1] * p.y, c: p.c,
    }));
    // Hold the cloud at a constant apparent size so only its SHAPE changes.
    const S0 = moments(pts).S;
    const k = Math.sqrt(Math.max(1e-12, (S0[0][0] + S0[1][1]) / 2));
    pts = pts.map((p) => ({ x: p.x / k, y: p.y / k, c: p.c }));

    frame(svg, W, H, PAD, xs, ys, {
      xTicks: [], yTicks: [],
      xLabel: 'a feature', yLabel: 'another feature',
    });

    // everything below is measured in the CURRENT space
    const { m, S } = moments(pts);
    const d = [m[1][0] - m[0][0], m[1][1] - m[0][1]];
    const det = S[0][0] * S[1][1] - S[0][1] * S[1][0] || 1e-12;
    const inv = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]];
    const w = [inv[0][0] * d[0] + inv[0][1] * d[1], inv[1][0] * d[0] + inv[1][1] * d[1]];
    const rNow = S[0][1] / Math.sqrt(Math.max(1e-12, S[0][0] * S[1][1]));

    // the noise, as an ellipse: round exactly when the space is white
    const tr = S[0][0] + S[1][1];
    const dsc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const e1 = Math.sqrt(Math.max(1e-9, tr / 2 + dsc));
    const e2 = Math.sqrt(Math.max(1e-9, tr / 2 - dsc));
    const th = 0.5 * Math.atan2(2 * S[0][1], S[0][0] - S[1][1]);
    const unit = xs(1) - xs(0);
    for (const c of [0, 1]) {
      el('ellipse', {
        cx: xs(m[c][0]), cy: ys(m[c][1]), rx: e1 * unit, ry: e2 * unit,
        transform: `rotate(${-th * 180 / Math.PI} ${xs(m[c][0])} ${ys(m[c][1])})`,
        fill: 'none', stroke: token('--text-muted'), 'stroke-width': 1.5,
        'stroke-dasharray': '4 3',
      }, svg);
    }

    for (const p of pts) {
      el('circle', { cx: xs(p.x), cy: ys(p.y), r: 3.4,
        fill: token(p.c === 1 ? '--c-hit' : '--c-miss'), opacity: 0.5 }, svg);
    }
    for (const c of [0, 1]) {
      el('circle', { cx: xs(m[c][0]), cy: ys(m[c][1]), r: 7,
        fill: token(c === 1 ? '--c-hit' : '--c-miss'),
        stroke: token('--surface-1'), 'stroke-width': 2.5 }, svg);
    }

    let ang = Math.abs(Math.atan2(w[1], w[0]) - Math.atan2(d[1], d[0])) * 180 / Math.PI;
    if (ang > 180) ang = 360 - ang;
    if (ang > 90) ang = 180 - ang;

    const mid = [(m[0][0] + m[1][0]) / 2, (m[0][1] + m[1][1]) / 2];
    const clip = clipped(svg, W, H, PAD);
    // Once the rules agree the two lines sit on top of each other, so two labels
    // would too. Say so instead.
    const together = ang < 1.5;
    boundary(svg, xs, ys, mid, d, token('--series-4'), '5 4',
      together ? null : 'difference of means', clip);
    boundary(svg, xs, ys, mid, w, token('--series-7'), null,
      together ? 'both rules, now the same line' : 'LDA', clip);
    out.innerHTML = stat('whitening applied', `${(state.t * 100).toFixed(0)}%`)
      + stat('correlation left in the data', rNow.toFixed(2))
      + stat('angle between the two rules', `${ang.toFixed(1)}&deg;`);
  };

  responsive(host, draw);
  return { state, refresh: draw };
}

/* ========================= the weights, by hand ==========================

   The one figure on the site where the reader sets the weights rather than
   watching a solver set them. Its whole job is the two sentences it sits under:
   the weights turn the line, the bias slides it without turning it. Nothing is
   fitted here -- w1, w2 and b are exactly what the sliders say, so the picture
   can be checked against the arithmetic by hand.

   The half-planes are tinted with the colour of the class they predict, which
   makes an error read as a dot sitting on the wrong background rather than as a
   number in a readout.                                                        */

export function weightGeometry(host, out) {
  const state = { w1: 0.9, w2: 0.45, b: 0 };
  const R = 4.6;
  // blue/red rather than the Hit/Miss pair: this figure shows no real condition,
  // and green at low opacity swamps a dark background while blue vanishes into it.
  const CLS = ['--series-8', '--series-1'];
  const pts = twoClouds({ n: 70, sep: 2.2, sx: 1, sy: 1, rho: 0.2, seed: 5 });

  const draw = () => {
    const svg = svgRoot(host, W, H);
    const xs = scale(-R, R, PAD.l, W - PAD.r);
    const ys = scale(-R, R, H - PAD.b, PAD.t);
    const { w1, w2, b } = state;

    // Shade each side with the colour of the class it predicts. Drawn first, so
    // the dots and the line sit on top of it.
    const clip = clipped(svg, W, H, PAD);
    const score = (x, y) => w1 * x + w2 * y + b;
    const corner = [[-R, -R], [R, -R], [R, R], [-R, R]];
    const pos = [], neg = [];
    for (let i = 0; i < 4; i++) {
      const a = corner[i], c = corner[(i + 1) % 4];
      (score(...a) >= 0 ? pos : neg).push(a);
      const sa = score(...a), sc = score(...c);
      if ((sa >= 0) !== (sc >= 0)) {                 // edge crosses the boundary
        const t = sa / (sa - sc);
        const p = [a[0] + t * (c[0] - a[0]), a[1] + t * (c[1] - a[1])];
        pos.push(p); neg.push(p);
      }
    }
    for (const [poly, c] of [[pos, CLS[1]], [neg, CLS[0]]]) {
      if (poly.length < 3) continue;
      el('polygon', {
        points: poly.map(([x, y]) => `${xs(x)},${ys(y)}`).join(' '),
        fill: token(c), opacity: 0.13,
      }, clip);
    }

    frame(svg, W, H, PAD, xs, ys, {
      xTicks: [], yTicks: [],
      xLabel: 'feature 1 — its value on this trial',
      yLabel: 'feature 2',
    });

    let right = 0;
    for (const p of pts) {
      const predicted = score(p.x, p.y) >= 0 ? 1 : 0;
      if (predicted === p.c) right++;
      el('circle', {
        cx: xs(p.x), cy: ys(p.y), r: 3.6,
        fill: token(CLS[p.c]),
        stroke: predicted === p.c ? 'none' : token('--text-primary'),
        'stroke-width': predicted === p.c ? 0 : 1.2,
        opacity: predicted === p.c ? 0.55 : 0.95,
      }, svg);
    }

    // the boundary: every point where the score is exactly zero
    if (Math.hypot(w1, w2) > 1e-6) {
      const n2 = w1 * w1 + w2 * w2;
      const foot = [-b * w1 / n2, -b * w2 / n2];      // closest point to the origin
      boundary(svg, xs, ys, foot, [w1, w2], token('--text-primary'), null, null, clip);

      // w itself, drawn from that point -- perpendicular, which is the callout below
      const L = 1.9 / Math.hypot(w1, w2);
      const tipx = foot[0] + w1 * L, tipy = foot[1] + w2 * L;
      const ah = el('defs', {}, svg);
      const mk = el('marker', { id: 'wg-arrow', viewBox: '0 0 10 10', refX: '8', refY: '5',
        markerWidth: '5', markerHeight: '5', orient: 'auto-start-reverse' }, ah);
      el('path', { d: 'M 0 1 L 9 5 L 0 9 z', fill: token('--text-primary') }, mk);
      el('line', {
        x1: xs(foot[0]), y1: ys(foot[1]), x2: xs(tipx), y2: ys(tipy),
        stroke: token('--text-primary'), 'stroke-width': 2, 'marker-end': 'url(#wg-arrow)',
      }, clip);
      el('text', {
        x: xs(tipx) + 7, y: ys(tipy) - 4, class: 'axis-label', fill: token('--text-primary'),
      }, svg).textContent = 'w';
    }

    const sign = (v) => (v < 0 ? '−' : '+');
    out.innerHTML = stat('the rule you have built',
      `<span style="font-size:0.62em">${w1.toFixed(2)}·x₁ ${sign(w2)} ${Math.abs(w2).toFixed(2)}·x₂ `
      + `${sign(b)} ${Math.abs(b).toFixed(2)}</span>`)
      + stat('dots on the right side', `${(100 * right / pts.length).toFixed(0)}%`);
  };

  responsive(host, draw);
  return { state, refresh: draw };
}

/* ============================== figure 3 ================================= */

export function svmMargin(host, out) {
  const state = { C: 1, sep: 1.9, n: 26 };
  const R = 4.2;

  const draw = () => {
    const svg = svgRoot(host, W, H);
    const xs = scale(-R, R, PAD.l, W - PAD.r);
    const ys = scale(-R, R, H - PAD.b, PAD.t);
    frame(svg, W, H, PAD, xs, ys, {
      xTicks: [], yTicks: [],
      xLabel: 'a feature', yLabel: 'another feature',
    });

    const pts = twoClouds({ n: state.n, sep: state.sep, sx: 0.85, sy: 0.85, rho: 0.1, seed: 23 });
    const { w, b, margin, sv } = svmTrain(pts, state.C);
    const svSet = new Set(sv);

    const n = Math.hypot(w[0], w[1]) || 1e-9;
    const ux = -w[1] / n, uy = w[0] / n;          // along the boundary
    const px = w[0] / n, py = w[1] / n;           // across it
    const cx = -b * w[0] / (n * n), cy = -b * w[1] / (n * n);
    const L = 12;
    const mg = Math.min(margin, 6);               // keep the band on screen

    const corner = (s, t) => [xs(cx + ux * s * L + px * t * mg), ys(cy + uy * s * L + py * t * mg)];
    const clip = clipped(svg, W, H, PAD);
    el('polygon', {
      points: [corner(-1, 1), corner(1, 1), corner(1, -1), corner(-1, -1)]
        .map((q) => q.join(',')).join(' '),
      fill: token('--series-1'), 'fill-opacity': 0.10,
    }, clip);
    for (const s of [1, -1]) {
      el('line', {
        x1: corner(-1, s)[0], y1: corner(-1, s)[1],
        x2: corner(1, s)[0], y2: corner(1, s)[1],
        stroke: token('--series-1'), 'stroke-width': 1.2, 'stroke-dasharray': '4 4',
      }, clip);
    }
    el('line', {
      x1: xs(cx - ux * L), y1: ys(cy - uy * L),
      x2: xs(cx + ux * L), y2: ys(cy + uy * L),
      stroke: token('--series-1'), 'stroke-width': 2.8, 'stroke-linecap': 'round',
    }, clip);

    /* Name the two things the figure exists to show. Without these the boundary
       and the margin band are distinguished only by line weight, which is not
       something a first-time reader is going to decode. */
    const lbl = (sx, tx, text, anchor) => {
      const [px2, py2] = corner(sx, tx);
      el('text', {
        x: px2, y: py2 - 6, class: 'axis-label',
        fill: token('--series-1'), 'text-anchor': anchor,
      }, svg).textContent = text;
    };
    lbl(0.72, 1, 'margin', 'middle');
    el('text', {
      x: xs(cx + ux * L * 0.72), y: ys(cy + uy * L * 0.72) - 6,
      class: 'axis-label', fill: token('--series-1'), 'text-anchor': 'middle',
    }, svg).textContent = 'boundary';

    for (const p of pts) {
      const isSv = svSet.has(p);
      el('circle', {
        cx: xs(p.x), cy: ys(p.y), r: isSv ? 6 : 4,
        fill: token(p.c === 1 ? '--c-hit' : '--c-miss'),
        opacity: isSv ? 1 : 0.38,
        stroke: isSv ? token('--text-primary') : 'none',
        'stroke-width': isSv ? 2 : 0,
      }, svg);
    }

    out.innerHTML = stat('box constraint C', state.C < 1 ? state.C.toFixed(2) : state.C.toFixed(1))
      + stat('margin width', (2 * margin).toFixed(2))
      + stat('points holding the line', `${sv.length} of ${pts.length}`);
  };

  responsive(host, draw);
  return { state, refresh: draw };
}

/* ============================== figure 4 ================================= */

export function smallN(host, out) {
  const state = { n: 12, rho: 0.8 };
  const R = 4.6;

  const draw = () => {
    const svg = svgRoot(host, W, H);
    const xs = scale(-R, R, PAD.l, W - PAD.r);
    const ys = scale(-R, R, H - PAD.b, PAD.t);
    frame(svg, W, H, PAD, xs, ys, {
      xTicks: [], yTicks: [],
      xLabel: 'a feature', yLabel: 'another feature',
    });

    // The answer unlimited data would give. Nobody in a real study sees this.
    const truth = twoClouds({ n: 400, sep: 1.5, rho: state.rho, seed: 5 });
    const wTrue = ldaWeights(truth, 0).w;
    const origin = [0, 0];

    /* LDA solid, SVM dotted, truth long-dashed. The dash patterns matter: this is
       the one figure whose whole job is telling two models apart, and with five
       lines each a per-line label would be unreadable, so identity has to survive
       without relying on hue. A legend follows below. */
    const clip = clipped(svg, W, H, PAD);
    for (let s = 0; s < 5; s++) {
      const samp = twoClouds({ n: state.n, sep: 1.5, rho: state.rho, seed: 101 + s * 7717 });
      boundary(svg, xs, ys, origin, ldaWeights(samp, 0).w, token('--series-7'), null, null, clip);
      boundary(svg, xs, ys, origin, svmTrain(samp, 1, 1200).w, token('--series-6'), '2 3', null, clip);
    }
    // Redraw the truth on top so it stays readable.
    boundary(svg, xs, ys, origin, wTrue, token('--text-primary'), '7 4', null, clip);

    const key = [
      ['LDA — 5 samples', token('--series-7'), null],
      ['SVM — 5 samples', token('--series-6'), '2 3'],
      ['the truth', token('--text-primary'), '7 4'],
    ];
    key.forEach(([name, colour, dash], i) => {
      const y = PAD.t + 12 + i * 15;
      el('line', {
        x1: PAD.l + 8, x2: PAD.l + 30, y1: y, y2: y,
        stroke: colour, 'stroke-width': 2.6, 'stroke-dasharray': dash || null,
        'stroke-linecap': 'round',
      }, svg);
      el('text', { x: PAD.l + 36, y: y + 3.5, class: 'axis-label', fill: token('--text-secondary') }, svg)
        .textContent = name;
    });

    const a0 = Math.atan2(wTrue[1], wTrue[0]);
    const fold = (a) => { let x = (a - a0) * 180 / Math.PI; while (x > 90) x -= 180; while (x < -90) x += 180; return x; };
    const errL = [], errS = [];
    for (let s = 0; s < 20; s++) {
      const samp = twoClouds({ n: state.n, sep: 1.5, rho: state.rho, seed: 4001 + s * 3313 });
      const wl = ldaWeights(samp, 0).w;
      const ws = svmTrain(samp, 1, 1000).w;
      errL.push(fold(Math.atan2(wl[1], wl[0])));
      errS.push(fold(Math.atan2(ws[1], ws[0])));
    }
    const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
    out.innerHTML = stat('trials per class', state.n)
      + stat('LDA error', `${rms(errL).toFixed(0)}&deg;`)
      + stat('SVM error', `${rms(errS).toFixed(0)}&deg;`);
  };

  responsive(host, draw);
  return { state, refresh: draw };
}
