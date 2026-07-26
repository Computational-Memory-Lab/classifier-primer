/* SEREEGA demo: a forward model, signal classes, and a decode, all in the page.
 *
 * WHAT THIS IS AND IS NOT. SEREEGA solves x = As + e -- source activity s,
 * projected by a lead field A, plus noise e. This file reimplements that shape
 * so it can be played with, but it is NOT SEREEGA and it is NOT its lead field:
 *
 *   - The real toolbox uses pre-computed lead fields (the New York Head has
 *     ~75,000 sources onto up to 228 channels) built from segmented MRI with
 *     separate brain/skull/scalp conductivities.
 *   - Here A is the analytic potential of a current dipole in an INFINITE
 *     HOMOGENEOUS conductor. It has no skull, no boundary condition at the
 *     scalp, no realistic geometry.
 *
 * That approximation gets the qualitative behaviour right -- dipolar
 * topographies, the depth/spread relationship, the radial-versus-tangential
 * difference -- which is what the page is teaching. It gets absolute amplitudes
 * and any fine spatial detail wrong. Every figure says so.
 *
 * No dependencies and no build step, matching the rest of the site. The 3D head
 * is a lat/long mesh, rotated by hand, depth-sorted, and filled on a canvas.
 */

import { el, svgRoot, canvasOverlay, responsive, token, rng, gauss, auc, frame, scale, linePath } from './site.js';

/* ============================ head geometry ============================== */

/* Semi-axes in arbitrary units, roughly adult-head proportioned:
   x = left-right, y = back-front, z = down-up. */
export const HEAD = { rx: 0.80, ry: 0.94, rz: 0.88 };

const onHead = (az, el_) => {
  // az: radians around z from +y (front). el_: radians up from the equator.
  const ce = Math.cos(el_);
  return [
    HEAD.rx * ce * Math.sin(az),
    HEAD.ry * ce * Math.cos(az),
    HEAD.rz * Math.sin(el_),
  ];
};

/* Approximate 10-20 positions, given as EEGLAB-style polar coordinates:
   theta = degrees clockwise from the nose, radius = 0 at the vertex and 0.5 at
   the equator. Converted to the ellipsoid below. Approximate on purpose -- the
   point is the sampling geometry, not electrode-accurate placement. */
const TEN_TWENTY = [
  ['Fp1', -18, 0.511], ['Fp2', 18, 0.511],
  ['F7', -54, 0.511], ['F3', -39, 0.333], ['Fz', 0, 0.256], ['F4', 39, 0.333], ['F8', 54, 0.511],
  ['T7', -90, 0.511], ['C3', -90, 0.256], ['Cz', 0, 0.001], ['C4', 90, 0.256], ['T8', 90, 0.511],
  ['P7', -126, 0.511], ['P3', -141, 0.333], ['Pz', 180, 0.256], ['P4', 141, 0.333], ['P8', 126, 0.511],
  ['O1', -162, 0.511], ['O2', 162, 0.511],
];

function polarToHead(thetaDeg, r) {
  const az = (thetaDeg * Math.PI) / 180;
  const el_ = (0.5 - Math.min(r, 0.5) / 0.5) * (Math.PI / 2); // r=0 -> vertex
  return onHead(az, el_);
}

/** Electrode montages. 19 is the named 10-20 set; larger counts are a
 *  quasi-uniform spiral over the upper head, standing in for a dense net. */
export function montage(n) {
  if (n <= 19) return TEN_TWENTY.map(([name, t, r]) => ({ name, pos: polarToHead(t, r) }));
  const out = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    // Sample the cap from the vertex down to ~20 deg below the equator.
    const f = i / (n - 1);
    const el_ = (Math.PI / 2) * (1 - f * 1.22);
    const az = i * golden;
    out.push({ name: `E${i + 1}`, pos: onHead(az, el_) });
  }
  return out;
}

/* ========================== the forward model =========================== */

/**
 * Scalp potential at `sensor` from a current dipole at `pos` with moment `mom`.
 * Infinite homogeneous medium: V = k * (q . r) / |r|^3, r = sensor - pos.
 * See the file header for what this omits.
 */
export function dipolePotential(sensor, pos, mom) {
  const rx = sensor[0] - pos[0], ry = sensor[1] - pos[1], rz = sensor[2] - pos[2];
  const d2 = rx * rx + ry * ry + rz * rz;
  const d = Math.sqrt(d2);
  if (d < 1e-4) return 0;
  return (mom[0] * rx + mom[1] * ry + mom[2] * rz) / (d2 * d);
}

/** Unit moment from azimuth/elevation in degrees. */
export function moment(azDeg, elDeg) {
  const a = (azDeg * Math.PI) / 180, e = (elDeg * Math.PI) / 180;
  return [Math.cos(e) * Math.sin(a), Math.cos(e) * Math.cos(a), Math.sin(e)];
}

/* ============================== signals ================================= */

/** SEREEGA's ERP class: a sum of Gaussian-windowed peaks. */
export function erpWave(t, peaks) {
  return t.map((tt) => peaks.reduce((s, p) => {
    if (!p.amp) return s;
    const z = (tt - p.lat) / (p.width / 2.355); // width given as FWHM
    return s + p.amp * Math.exp(-0.5 * z * z);
  }, 0));
}

/**
 * SEREEGA's noise class: 1/f^n. Built by summing sinusoids with amplitude
 * f^(-n/2) and random phase, which is transparent and good enough here.
 * n = 0 white, 1 pink, 2 brown.
 */
export function noiseWave(t, n, amp, rand) {
  const out = new Array(t.length).fill(0);
  const T = t[t.length - 1] - t[0];
  for (let k = 1; k <= 60; k++) {
    const f = k / (T / 1000);
    const a = Math.pow(f, -n / 2);
    const ph = rand() * Math.PI * 2;
    for (let i = 0; i < t.length; i++) out[i] += a * Math.sin(2 * Math.PI * f * (t[i] / 1000) + ph);
  }
  let mx = 0;
  for (const v of out) mx = Math.max(mx, Math.abs(v));
  return mx ? out.map((v) => (v / mx) * amp) : out;
}

/**
 * One epoch of source activity, with SEREEGA's per-epoch deviations applied:
 * amplitude scaled by (1 + dev*z) and latency shifted by jitter*z.
 */
export function epochSource(t, peaks, dev, jitter, rand) {
  const shifted = peaks.map((p) => ({
    amp: p.amp * (1 + dev * gauss(rand)),
    lat: p.lat + jitter * gauss(rand),
    width: p.width,
  }));
  return erpWave(t, shifted);
}

/* ============================ 3D plumbing =============================== */

function rotate(p, yaw, pitch) {
  const [x, y, z] = p;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  return [x1, y1 * cp - z * sp, y1 * sp + z * cp];
}

/* Weak perspective: far enough that the head does not distort, near enough to
   read as solid. */
const CAM = 4.2;
function project(p, cx, cy, s) {
  const zz = p[2];                    // +z points at the viewer after rotation
  const k = CAM / (CAM - zz);
  return [cx + p[0] * s * k, cy - p[1] * s * k, zz];
}

/* Diverging map: blue (negative) -> neutral -> orange (positive). Two hues with
   a neutral midpoint, never a rainbow. Returns components so that lighting can be
   applied to the COLOUR afterwards rather than to the value. */
function divergingRGB(v, dark) {
  const c = Math.max(-1, Math.min(1, v));
  const neg = [42, 120, 214];
  const pos = [235, 104, 52];
  const mid = dark ? [58, 58, 62] : [238, 238, 234];
  const target = c < 0 ? neg : pos;
  const g = Math.pow(Math.abs(c), 0.75);
  return [0, 1, 2].map((i) => mid[i] + (target[i] - mid[i]) * g);
}

const rgbStr = (c) => `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;

/**
 * Lambert shading, applied to the rendered colour and NOT to the value.
 *
 * It used to be folded into the value (`divergingColor(v * shade)`), which
 * quietly lied: a quad angled away from the viewer was drawn as though it
 * carried a weaker potential than it does, so the colour no longer meant what
 * the figure's caption says it means. Lighting is a property of the picture;
 * the value is the data. Keep them apart.
 */
function litColor(v, shade, dark) {
  return rgbStr(divergingRGB(v, dark).map((ch) => ch * shade));
}

/* ============================== head view =============================== */

/**
 * Rotatable 3D head with the dipole inside and its scalp projection painted on
 * the surface. Drag to rotate.
 */
export function headView(host, out) {
  const state = {
    dx: 0.0, dy: -0.35, dz: 0.30,   // dipole position
    az: 0, elv: 90,                  // dipole orientation (deg); 90 = radial-ish
    nCh: 19,
    yaw: -0.5, pitch: 0.32,
    showElec: true,
  };

  const W = 520, H = 440;
  let ctx = null;

  // Drag to rotate.
  let drag = null;
  host.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, yaw: state.yaw, pitch: state.pitch };
    host.setPointerCapture(e.pointerId);
  });
  host.addEventListener('pointermove', (e) => {
    if (!drag) return;
    state.yaw = drag.yaw + (e.clientX - drag.x) * 0.011;
    state.pitch = Math.max(-1.2, Math.min(1.2, drag.pitch + (e.clientY - drag.y) * 0.009));
    draw();
  });
  const end = () => { drag = null; };
  host.addEventListener('pointerup', end);
  host.addEventListener('pointercancel', end);
  host.style.touchAction = 'none';
  host.style.cursor = 'grab';

  function draw() {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark'
      || (!document.documentElement.hasAttribute('data-theme')
          && matchMedia('(prefers-color-scheme: dark)').matches);
    const svg = svgRoot(host, W, H);
    ctx = canvasOverlay(host, svg, W, H, { x: 0, y: 0, w: W, h: H }, 2);
    ctx.clearRect(0, 0, W, H);

    /* Keep the source inside the skull. The sliders reach a normalised radius of
       1.20 at their corners -- a dipole 20% outside the head, which is not a
       thing, and which made the depth readout go negative. Clamped to 0.92 so it
       always sits in tissue, and the readout says when clamping is active rather
       than silently ignoring the slider. */
    const rNorm = Math.hypot(state.dx / HEAD.rx, state.dy / HEAD.ry, state.dz / HEAD.rz);
    const clamped = rNorm > 0.92;
    const k = clamped ? 0.92 / rNorm : 1;
    const pos = [state.dx * k, state.dy * k, state.dz * k];
    const mom = moment(state.az, state.elv);
    const cx = W / 2, cy = H / 2 + 8, s = 150;

    // --- mesh, with the potential evaluated per vertex --------------------
    const NU = 52, NV = 30;
    const grid = [];
    let maxAbs = 1e-9;
    for (let j = 0; j <= NV; j++) {
      const row = [];
      const elv = -Math.PI / 2 + (j / NV) * Math.PI;
      for (let i = 0; i <= NU; i++) {
        const az = (i / NU) * Math.PI * 2;
        const p = onHead(az, elv);
        const v = dipolePotential(p, pos, mom);
        maxAbs = Math.max(maxAbs, Math.abs(v));
        row.push({ p, r: rotate(p, state.yaw, state.pitch), v });
      }
      grid.push(row);
    }

    // --- quads, painter's algorithm ---------------------------------------
    const quads = [];
    for (let j = 0; j < NV; j++) {
      for (let i = 0; i < NU; i++) {
        const a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i + 1], d = grid[j + 1][i];
        const depth = (a.r[2] + b.r[2] + c.r[2] + d.r[2]) / 4;
        const v = (a.v + b.v + c.v + d.v) / 4;
        /* Crude Lambert term so the ellipsoid reads as solid. The `|| 1` guard
           belongs OUTSIDE hypot -- inside it, a legitimate zero z-component was
           being replaced by 1, tilting the shading of the equator. */
        const rad = Math.hypot(a.r[0], a.r[1], a.r[2]) || 1;
        const nz = depth / rad;
        quads.push({ pts: [a.r, b.r, c.r, d.r], depth, v, shade: 0.62 + 0.38 * Math.max(0, nz) });
      }
    }
    quads.sort((p, q) => p.depth - q.depth);

    for (const q of quads) {
      const scr = q.pts.map((p) => project(p, cx, cy, s));
      ctx.beginPath();
      ctx.moveTo(scr[0][0], scr[0][1]);
      for (let k = 1; k < 4; k++) ctx.lineTo(scr[k][0], scr[k][1]);
      ctx.closePath();
      ctx.globalAlpha = q.depth > 0 ? 0.93 : 1;
      ctx.fillStyle = litColor(q.v / maxAbs, q.shade, dark);
      ctx.fill();
      ctx.lineWidth = 0.35;
      ctx.strokeStyle = ctx.fillStyle;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // --- the dipole itself, drawn over the back half ----------------------
    const dRot = rotate(pos, state.yaw, state.pitch);
    const tip = rotate([pos[0] + mom[0] * 0.34, pos[1] + mom[1] * 0.34, pos[2] + mom[2] * 0.34],
      state.yaw, state.pitch);
    const tail = rotate([pos[0] - mom[0] * 0.14, pos[1] - mom[1] * 0.14, pos[2] - mom[2] * 0.14],
      state.yaw, state.pitch);
    const P = (p) => project(p, cx, cy, s);
    const [tx, ty] = P(tip), [bx, by] = P(tail), [mx, my] = P(dRot);

    const ink = dark ? '#f4f4f2' : '#111';
    ctx.strokeStyle = ink; ctx.lineWidth = 3.2; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(tx, ty); ctx.stroke();
    // arrow head
    const ang = Math.atan2(ty - by, tx - bx);
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - 11 * Math.cos(ang - 0.4), ty - 11 * Math.sin(ang - 0.4));
    ctx.lineTo(tx - 11 * Math.cos(ang + 0.4), ty - 11 * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fillStyle = ink; ctx.fill();
    ctx.beginPath(); ctx.arc(mx, my, 4.5, 0, 7); ctx.fillStyle = ink; ctx.fill();

    // --- electrodes, front-facing only ------------------------------------
    const chans = montage(state.nCh);
    const readings = chans.map((c) => ({
      ...c,
      v: dipolePotential(c.pos, pos, mom),
      r: rotate(c.pos, state.yaw, state.pitch),
    }));
    if (state.showElec) {
      for (const c of readings) {
        if (c.r[2] < -0.05) continue;
        const [ex, ey] = P(c.r);
        ctx.beginPath();
        ctx.arc(ex, ey, state.nCh > 40 ? 2.6 : 5, 0, 7);
        ctx.fillStyle = rgbStr(divergingRGB(c.v / maxAbs, dark));  // unshaded: this is the readout
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = dark ? 'rgba(255,255,255,.75)' : 'rgba(0,0,0,.55)';
        ctx.stroke();
      }
    }

    // --- nose, so orientation is never ambiguous --------------------------
    const noseBase = rotate(onHead(0, 0.12), state.yaw, state.pitch);
    const noseTip = rotate([0, HEAD.ry * 1.13, HEAD.rz * 0.10], state.yaw, state.pitch);
    if (noseTip[2] > -0.2) {
      const [nx1, ny1] = P(noseBase), [nx2, ny2] = P(noseTip);
      ctx.beginPath(); ctx.moveTo(nx1, ny1); ctx.lineTo(nx2, ny2);
      ctx.strokeStyle = dark ? 'rgba(255,255,255,.5)' : 'rgba(0,0,0,.4)';
      ctx.lineWidth = 2; ctx.stroke();
    }

    el('text', {
      x: 12, y: H - 12, fill: token('--text-muted'), 'font-size': 11,
    }, svg).textContent = 'drag to rotate';

    // --- readout ----------------------------------------------------------
    const vals = readings.map((c) => Math.abs(c.v));
    const peak = Math.max(...vals);
    const spread = vals.filter((v) => v > peak * 0.5).length;
    const r = Math.hypot(pos[0] / HEAD.rx, pos[1] / HEAD.ry, pos[2] / HEAD.rz);
    out.innerHTML = [
      ['position', `${(r * 100).toFixed(0)}%${clamped ? ' (at the limit)' : ''}`,
        'centre → scalp'],
      ['electrodes', state.nCh, 'sampling the surface'],
      ['above half-peak', `${spread} of ${state.nCh}`, 'how spread out it is'],
    ].map(([lab, v, sub]) => `<div class="stat"><div class="stat__value">${v}</div><div class="stat__label">${lab}<br><span class="muted">${sub}</span></div></div>`).join('');
  }

  responsive(host, draw);
  return { state, refresh: draw };
}

/* ============================= signal view ============================== */

/** Source waveform, single epochs, and the average that emerges from them. */
export function signalView(host, out) {
  const state = {
    p1: 6, l1: 120, w1: 60,
    p2: -4, l2: 200, w2: 80,
    p3: 8, l3: 400, w3: 160,
    noise: 1, noiseAmp: 12, nTrials: 40, dev: 0.35, jitter: 25,
    show: 'trials',
  };

  function draw() {
    const W = 640, H = 300, PAD = { l: 52, r: 16, t: 14, b: 40 };
    const svg = svgRoot(host, W, H);
    const t = [];
    for (let i = 0; i <= 200; i++) t.push(-100 + (i / 200) * 800);
    const peaks = [
      { amp: state.p1, lat: state.l1, width: state.w1 },
      { amp: state.p2, lat: state.l2, width: state.w2 },
      { amp: state.p3, lat: state.l3, width: state.w3 },
    ];
    const clean = erpWave(t, peaks);
    const rand = rng(7);
    const trials = [];
    for (let k = 0; k < state.nTrials; k++) {
      const src = epochSource(t, peaks, state.dev, state.jitter, rand);
      const nz = noiseWave(t, state.noise, state.noiseAmp, rand);
      trials.push(src.map((v, i) => v + nz[i]));
    }
    const avg = t.map((_, i) => trials.reduce((s, tr) => s + tr[i], 0) / trials.length);

    let lo = 1e9, hi = -1e9;
    const scan = state.show === 'trials' ? trials.flat() : [...clean, ...avg];
    for (const v of scan) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    lo = Math.min(lo, -8); hi = Math.max(hi, 8);

    const xs = scale(t[0], t[t.length - 1], PAD.l, W - PAD.r);
    const ys = scale(lo, hi, H - PAD.b, PAD.t);
    frame(svg, W, H, PAD, xs, ys, { xLabel: 'time from word onset (ms)', yLabel: 'µV' });

    const pts = (arr) => arr.map((v, i) => [xs(t[i]), ys(v)]);
    if (state.show === 'trials') {
      for (const tr of trials.slice(0, 30)) {
        el('path', { d: linePath(pts(tr)), fill: 'none', stroke: token('--text-muted'),
          'stroke-width': 0.7, opacity: 0.28 }, svg);
      }
    }
    el('path', { d: linePath(pts(clean)), fill: 'none', stroke: token('--series-7') || '#2a78d6',
      'stroke-width': 2, 'stroke-dasharray': '6 4' }, svg);
    el('path', { d: linePath(pts(avg)), fill: 'none', stroke: token('--series-6') || '#eb6834',
      'stroke-width': 2.4 }, svg);

    const snr = Math.max(...clean.map(Math.abs)) / (state.noiseAmp || 1);
    out.innerHTML = [
      ['trials averaged', state.nTrials],
      ['peak signal / noise', snr.toFixed(2)],
      ['after averaging', (snr * Math.sqrt(state.nTrials)).toFixed(2)],
    ].map(([k, v]) => `<div class="stat"><div class="stat__value">${v}</div><div class="stat__label">${k}</div></div>`).join('');
  }

  responsive(host, draw);
  return { state, refresh: draw };
}

/* ============================= decode view ============================== */

/**
 * The payoff: simulate two conditions differing only in one peak's amplitude,
 * project through the forward model to a montage, take windowed means as
 * features, and actually classify them. Diagonal-covariance LDA, which is what
 * Gamma = 1 reduces to -- the setting our own pipeline runs.
 */
/**
 * Simulate two conditions that differ ONLY in the amplitude of the late
 * component, project through the forward model to a montage, and reduce each
 * channel to windowed means -- the same feature construction the real pipeline
 * uses. Exported so it can be tested outside a browser.
 */
export function simulateTrials(cfg) {
  const { nTrials, nCh, nBins, effect, noiseAmp, dev, jitter, seed = 11 } = cfg;
  const rand = rng(seed);
  const t = [];
  for (let i = 0; i < 120; i++) t.push(-100 + (i / 119) * 800);
  const chans = montage(nCh);
  const pos = [0, -0.30, 0.34];
  const mom = moment(0, 90);
  const gain = chans.map((c) => dipolePotential(c.pos, pos, mom));
  const gmax = Math.max(...gain.map(Math.abs)) || 1;

  /* A fresh 1/f draw per channel per trial costs O(harmonics x samples) each,
     which measured 4.5 s at 64 channels x 600 trials -- unusable behind a
     slider. Draw a pool once and sample it with a random sign instead. With 96
     traces the chance two channels in a trial share one is small, and the
     spectral character is identical either way. */
  const POOL = 96;
  const pool = [];
  for (let i = 0; i < POOL; i++) pool.push(noiseWave(t, 1, noiseAmp, rand));

  /* Labels are balanced and then SHUFFLED with an independent generator.
     They used to be `k % 2`, which ties the class to the trial's position in
     the generation stream -- and the xorshift32 here has a small but consistent
     even/odd bias (even draws average ~0.005 above odd, same sign at every seed
     tested). That leaked a real class difference into a zero-effect condition
     and pushed the null check to 0.519, t = 2.03 against chance. Decoupling the
     label from the stream removes it. A null check that is quietly off is worse
     than no null check. */
  const lab = [];
  for (let k = 0; k < nTrials; k++) lab.push(k % 2);
  const shuf = rng((seed * 7919 + 13) >>> 0);
  for (let i = lab.length - 1; i > 0; i--) {
    const j = (shuf() * (i + 1)) | 0;
    [lab[i], lab[j]] = [lab[j], lab[i]];
  }

  const per = Math.floor(t.length / nBins);
  const X = [], y = [];
  for (let k = 0; k < nTrials; k++) {
    const cls = lab[k];
    const peaks = [
      { amp: 6, lat: 120, width: 60 },
      { amp: 4 + (cls ? effect : 0), lat: 400, width: 160 },
    ];
    const src = epochSource(t, peaks, dev, jitter, rand);
    const row = [];
    for (let c = 0; c < chans.length; c++) {
      const g = gain[c] / gmax;
      const nz = pool[(rand() * POOL) | 0];
      const sgn = rand() < 0.5 ? -1 : 1;
      for (let b = 0; b < nBins; b++) {
        let sm = 0;
        for (let i = b * per; i < (b + 1) * per; i++) sm += src[i] * g + sgn * nz[i];
        row.push(sm / per);
      }
    }
    X.push(row); y.push(cls);
  }
  return { X, y };
}

/**
 * Diagonal-covariance LDA -- what Gamma = 1 reduces to, and what our study-phase
 * pipeline runs. Fitted on the first half of the trials, scored on the second.
 * Returns the held-out AUC. Exported for testing.
 */
export function diagLdaAuc(X, y) {
  const n = X.length, p = X[0].length, half = Math.floor(n / 2);
  const tr = [], te = [];
  for (let i = 0; i < n; i++) (i < half ? tr : te).push(i);
  const m = [new Array(p).fill(0), new Array(p).fill(0)];
  const c = [0, 0];
  for (const i of tr) { for (let j = 0; j < p; j++) m[y[i]][j] += X[i][j]; c[y[i]]++; }
  for (let k = 0; k < 2; k++) for (let j = 0; j < p; j++) m[k][j] /= (c[k] || 1);
  const va = new Array(p).fill(0);
  for (const i of tr) for (let j = 0; j < p; j++) { const d = X[i][j] - m[y[i]][j]; va[j] += d * d; }
  for (let j = 0; j < p; j++) va[j] = va[j] / Math.max(1, tr.length - 2) || 1;
  const w = m[1].map((v, j) => (v - m[0][j]) / va[j]);
  const scores = te.map((i) => X[i].reduce((sm, v, j) => sm + v * w[j], 0));
  return auc(te.map((i) => y[i]), scores);
}

export function decodeView(host, out) {
  const state = { nTrials: 200, nCh: 19, nBins: 12, effect: 1.5, noiseAmp: 12, dev: 0.35, jitter: 25 };

  const simulate = () => simulateTrials(state);
  const fitScore = diagLdaAuc;

  function draw() {
    const W = 640, H = 300, PAD = { l: 58, r: 18, t: 14, b: 42 };
    const svg = svgRoot(host, W, H);

    /* Simulate ONCE at the largest trial count on the ladder, then take
       prefixes -- so a point drawn at 640 trials really was fitted on 640
       trials. Slicing a smaller fixed pool made every point past its size
       reuse the same data and plot an identical AUC at different x, which read
       as the curve flattening when nothing of the sort had happened. */
    const NS = [40, 80, 160, 320, 640, 1280];
    const { X, y } = simulateTrials({ ...state, nTrials: Math.max(...NS) });
    const p = X[0].length;

    const measured = NS.map((n) => ({ n, a: fitScore(X.slice(0, n), y.slice(0, n)) }));
    const observed = fitScore(X.slice(0, state.nTrials), y.slice(0, state.nTrials));

    const xs = scale(Math.log10(30), Math.log10(2000), PAD.l, W - PAD.r);
    const ys = scale(0.45, 1.0, H - PAD.b, PAD.t);
    frame(svg, W, H, PAD, (v) => xs(Math.log10(v)), ys,
      { xLabel: 'trials', yLabel: 'AUC', xTicks: [50, 100, 250, 500, 1000, 2000] });

    el('line', { x1: PAD.l, x2: W - PAD.r, y1: ys(0.5), y2: ys(0.5),
      stroke: token('--border'), 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }, svg);

    el('path', {
      d: linePath(measured.map((d) => [xs(Math.log10(d.n)), ys(d.a)])),
      fill: 'none', stroke: token('--series-6') || '#eb6834', 'stroke-width': 2.4,
    }, svg);
    for (const d of measured) {
      el('circle', { cx: xs(Math.log10(d.n)), cy: ys(d.a), r: 4.5,
        fill: token('--page'), stroke: token('--series-6') || '#eb6834', 'stroke-width': 2 }, svg);
    }

    // where the trials slider currently sits
    el('circle', { cx: xs(Math.log10(state.nTrials)), cy: ys(observed), r: 7,
      fill: token('--series-7') || '#2a78d6', stroke: token('--page'), 'stroke-width': 2 }, svg);
    el('text', { x: xs(Math.log10(state.nTrials)), y: ys(observed) - 14,
      'text-anchor': 'middle', fill: token('--text-secondary'), 'font-size': 11 }, svg)
      .textContent = `your setting: ${observed.toFixed(3)}`;

    out.innerHTML = [
      ['features', p],
      ['trials', state.nTrials],
      ['trials per feature', (state.nTrials / p).toFixed(2)],
      ['AUC', observed.toFixed(3)],
    ].map(([k, v]) => `<div class="stat"><div class="stat__value">${v}</div><div class="stat__label">${k}</div></div>`).join('');
  }

  responsive(host, draw);
  return { state, refresh: draw };
}
