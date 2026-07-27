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

import { el, svgRoot, canvasOverlay, responsive, token, rng, gauss, auc, frame, scale, linePath,
         loadJSON, loadBin } from './site.js';
import { ELECTRODES, MONTAGES } from './electrodes.js';

/* ============================ head geometry ============================== */

/* Real electrode coordinates, generated from EEGLAB by build_head.py. See the
   header of assets/electrodes.js for provenance and for what they replaced. */
export { ELECTRODES, MONTAGES } from './electrodes.js';

/** Positions for a montage of n electrodes, as [x,y,z] triples. */
export function montage(n) {
  const key = [19, 32, 64, 128].reduce((a, b) => (Math.abs(b - n) < Math.abs(a - n) ? b : a));
  return MONTAGES[key].map((i) => ({ name: ELECTRODES[i].name, pos: ELECTRODES[i].pos }));
}

/* The scalp mesh is bigger than a browser wants inline, so it loads as binary.
   Cached: several redraws a second happen while dragging. */
let meshPromise = null;
export function loadHeadMesh() {
  if (!meshPromise) {
    meshPromise = Promise.all([
      loadJSON('data/head_mesh.json'),
      loadBin('data/head_mesh.bin'),
    ]).then(([meta, buf]) => {
      /* Float blocks FIRST. A uint16 face block is not necessarily a multiple
         of 4 bytes long -- 11,871 triangles is 71,226 bytes -- so putting it
         between the two float32 blocks lands the normals on an odd byte offset,
         where Float32Array refuses to construct at all and the whole mesh fails
         to load. build_head.py writes them in this order for that reason. */
      const nV = meta.nVerts, nF = meta.nFaces;
      const V = new Float32Array(buf, 0, nV * 3);
      const N = new Float32Array(buf, nV * 3 * 4, nV * 3);
      const F = new Uint16Array(buf, nV * 3 * 4 * 2, nF * 3);
      return { V, F, N, nV, nF, meta };
    });
  }
  return meshPromise;
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

/**
 * Head coordinates -> VIEW coordinates: [right, up, toward-viewer].
 *
 * The mesh is RAS -- x right, y ANTERIOR, z SUPERIOR. The renderer wants the
 * superior axis pointing up the screen and the anterior axis pointing at the
 * camera. This used to return them the other way round, so the vertical axis of
 * the picture was front-to-back and the camera looked straight down the top of
 * the head: Fpz projected to the top of the frame, Oz to the bottom, and Cz to
 * the middle. Everything below the widest point was hidden behind the skull,
 * which read as the bottom half of the head being cut off.
 *
 * A near-spherical ellipsoid hid this completely -- from any angle a ball looks
 * like a ball. It only became visible once the mesh was a real head.
 */
function rotate(p, yaw, pitch) {
  const [x, y, z] = p;
  const cw = Math.cos(yaw), sw = Math.sin(yaw);
  const xr = x * cw - y * sw;
  const ya = x * sw + y * cw;              // anterior, after spinning about the vertical
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const yt = ya * cp - z * sp;             // anterior, after tilting
  const zt = ya * sp + z * cp;             // superior, after tilting
  return [xr, zt, yt];
}

/* Weak perspective: far enough that the head does not distort, near enough to
   read as solid. */
const CAM = 4.2;
function project(p, cx, cy, s) {
  const zz = p[2];                    // view coords: [2] points at the viewer
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
  const g = Math.pow(Math.abs(c), 0.6);   // display curve: lifts mid-range detail
  return [0, 1, 2].map((i) => mid[i] + (target[i] - mid[i]) * g);
}

const rgbStr = (c) => `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;

/* A <g> clipped to the plot rectangle. svg.chart sets overflow:visible site-wide
   so axis labels are not cut off, which also means a series that leaves its axes
   draws over the page instead of being trimmed. */
let clipSeq = 0;
function clipped(svg, w, h, pad) {
  const id = `seclip${++clipSeq}`;
  const cp = el('clipPath', { id }, el('defs', {}, svg));
  el('rect', { x: pad.l, y: pad.t, width: w - pad.l - pad.r, height: h - pad.t - pad.b }, cp);
  return el('g', { 'clip-path': `url(#${id})` }, svg);
}

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
    dx: 0.0, dy: 0.30, dz: 0.30,     // dipole position, head coords
    az: 0, elv: 90,                   // dipole orientation, degrees
    nCh: 19,
    yaw: -0.62, pitch: 0.10,
  };

  const W = 520, H = 430;
  let mesh = null;

  let drag = null;
  /* Suppressing selection during a drag needs more than one guard: the pointer
     leaves the element while dragging, so whatever is underneath ends up owning
     the selection unless <body> refuses one for the duration. */
  const noSelect = (e) => e.preventDefault();
  host.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    drag = { x: e.clientX, y: e.clientY, yaw: state.yaw, pitch: state.pitch };
    host.setPointerCapture(e.pointerId);
    document.addEventListener('selectstart', noSelect);
    document.body.style.userSelect = 'none';
    host.style.cursor = 'grabbing';
  });
  host.addEventListener('pointermove', (e) => {
    if (!drag) return;
    state.yaw = drag.yaw + (e.clientX - drag.x) * 0.011;
    state.pitch = Math.max(-1.2, Math.min(1.2, drag.pitch + (e.clientY - drag.y) * 0.009));
    draw();
  });
  const end = () => {
    drag = null;
    document.removeEventListener('selectstart', noSelect);
    document.body.style.userSelect = '';
    host.style.cursor = 'grab';
  };
  host.addEventListener('pointerup', end);
  host.addEventListener('pointercancel', end);
  host.style.touchAction = 'none';
  host.style.cursor = 'grab';
  host.style.userSelect = 'none';
  host.style.webkitUserSelect = 'none';

  /* Half-extents of the mesh, used to keep the dipole inside the skull. */
  let half = [0.8, 1, 0.85];
  function measure(V) {
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (let i = 0; i < V.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], V[i + k]); hi[k] = Math.max(hi[k], V[i + k]);
      }
    }
    half = [0, 1, 2].map((k) => Math.max(Math.abs(lo[k]), Math.abs(hi[k])) || 1);
  }

  function draw() {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark'
      || (!document.documentElement.hasAttribute('data-theme')
          && matchMedia('(prefers-color-scheme: dark)').matches);
    const svg = svgRoot(host, W, H);

    if (!mesh) {
      el('text', {
        x: W / 2, y: H / 2, 'text-anchor': 'middle', fill: token('--text-muted'), 'font-size': 13,
      }, svg).textContent = 'loading head model…';
      return;
    }
    const ctx = canvasOverlay(host, svg, W, H, { x: 0, y: 0, w: W, h: H }, 2);
    ctx.clearRect(0, 0, W, H);

    /* Keep the source inside the skull: clamp to 88% of the head's half-extent
       along its own direction. Outside that it is not a physical configuration
       and the topography is meaningless. */
    const rNorm = Math.hypot(state.dx / half[0], state.dy / half[1], state.dz / half[2]);
    const clamped = rNorm > 0.88;
    const kk = clamped ? 0.88 / rNorm : 1;
    const pos = [state.dx * kk, state.dy * kk, state.dz * kk];
    const mom = moment(state.az, state.elv);

    const { V, F, N, nF } = mesh;

    // Rotate every vertex and normal once, and evaluate the potential.
    const RV = new Float32Array(V.length);
    const RN = new Float32Array(V.length);
    const PV = new Float32Array(V.length / 3);
    let maxAbs = 1e-9;
    for (let i = 0, j = 0; i < V.length; i += 3, j++) {
      const r = rotate([V[i], V[i + 1], V[i + 2]], state.yaw, state.pitch);
      RV[i] = r[0]; RV[i + 1] = r[1]; RV[i + 2] = r[2];
      const q = rotate([N[i], N[i + 1], N[i + 2]], state.yaw, state.pitch);
      RN[i] = q[0]; RN[i + 1] = q[1]; RN[i + 2] = q[2];
      PV[j] = dipolePotential([V[i], V[i + 1], V[i + 2]], pos, mom);
    }

    /* Scale the colour to a high percentile rather than the maximum. The field
       falls off as one over distance squared, so a handful of vertices right
       above the dipole take the peak and everything else collapses to neutral --
       the topography, which is the whole point of the figure, becomes invisible.
       Clipping the top few percent is what topographic plots normally do
       (EEGLAB's maplimits does the same). The mapping stays monotonic in the
       true value; only the top of the range is compressed. */
    const mags = Float32Array.from(PV, Math.abs).sort();
    maxAbs = Math.max(mags[Math.floor(mags.length * 0.97)], 1e-9);

    /* Auto-fit rather than a fixed scale. The mesh is a whole head including
       neck and shoulders, so its projected extent changes a lot with rotation
       and any constant would either crop it or waste half the frame. Measure the
       projection at unit scale, then scale and centre to fill the canvas. */
    let lox = 1e9, hix = -1e9, loy = 1e9, hiy = -1e9;
    for (let i = 0; i < RV.length; i += 3) {
      const k = CAM / (CAM - RV[i + 2]);
      const px = RV[i] * k, py = RV[i + 1] * k;
      if (px < lox) lox = px; if (px > hix) hix = px;
      if (py < loy) loy = py; if (py > hiy) hiy = py;
    }
    const PADPX = 14;
    const sc = Math.min((W - 2 * PADPX) / (hix - lox), (H - 2 * PADPX) / (hiy - loy));
    const cx = W / 2 - ((lox + hix) / 2) * sc;
    const cy = H / 2 + ((loy + hiy) / 2) * sc;

    /* Painter's algorithm over real triangles. Lighting uses the true face
       normal -- with an actual scalp this matters, because the surface is not a
       sphere and a radial approximation would shade the temples wrongly. */
    const order = new Array(nF);
    const depth = new Float32Array(nF);
    for (let f = 0; f < nF; f++) {
      const a = F[f * 3] * 3, b = F[f * 3 + 1] * 3, c = F[f * 3 + 2] * 3;
      depth[f] = (RV[a + 2] + RV[b + 2] + RV[c + 2]) / 3;
      order[f] = f;
    }
    order.sort((p, q) => depth[p] - depth[q]);

    for (const f of order) {
      const ia = F[f * 3], ib = F[f * 3 + 1], ic = F[f * 3 + 2];
      const a = ia * 3, b = ib * 3, c = ic * 3;
      const ax = RV[a], ay = RV[a + 1], az2 = RV[a + 2];

      /* Averaged VERTEX normals, so curvature reads smooth instead of faceted,
         and back faces are skipped entirely. Culling is not just a speed win:
         drawing the inside of the far wall and then painting over it left the
         head looking like it had a second surface inside it. */
      let nx = (RN[a] + RN[b] + RN[c]) / 3;
      let ny = (RN[a + 1] + RN[b + 1] + RN[c + 1]) / 3;
      let nz = (RN[a + 2] + RN[b + 2] + RN[c + 2]) / 3;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      if (nz <= 0.02) continue;                       // faces away from the camera
      // Light slightly above and to the left of the camera.
      const lam = Math.max(0, nx * -0.28 + ny * 0.34 + nz * 0.90);
      const shade = 0.46 + 0.54 * lam;

      const pa = project([ax, ay, az2], cx, cy, sc);
      const pb = project([RV[b], RV[b + 1], RV[b + 2]], cx, cy, sc);
      const pc = project([RV[c], RV[c + 1], RV[c + 2]], cx, cy, sc);
      const v = (PV[ia] + PV[ib] + PV[ic]) / 3;

      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.lineTo(pc[0], pc[1]);
      ctx.closePath();
      ctx.fillStyle = litColor(v / maxAbs, shade, dark);
      ctx.fill();
      // Hairline stroke in the same colour closes the seams between triangles.
      ctx.lineWidth = 0.4; ctx.strokeStyle = ctx.fillStyle; ctx.stroke();
    }

    // --- the dipole, drawn over the head so it is always visible ------------
    const P = (q) => project(rotate(q, state.yaw, state.pitch), cx, cy, sc);
    const tip = P([pos[0] + mom[0] * 0.34, pos[1] + mom[1] * 0.34, pos[2] + mom[2] * 0.34]);
    const tail = P([pos[0] - mom[0] * 0.14, pos[1] - mom[1] * 0.14, pos[2] - mom[2] * 0.14]);
    const mid = P(pos);
    const ink = dark ? '#f6f6f4' : '#101010';
    const halo = dark ? 'rgba(0,0,0,.55)' : 'rgba(255,255,255,.75)';
    for (const [col, wid] of [[halo, 6.5], [ink, 3.2]]) {
      ctx.strokeStyle = col; ctx.lineWidth = wid; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(tail[0], tail[1]); ctx.lineTo(tip[0], tip[1]); ctx.stroke();
    }
    const ang = Math.atan2(tip[1] - tail[1], tip[0] - tail[0]);
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(tip[0] - 11 * Math.cos(ang - 0.4), tip[1] - 11 * Math.sin(ang - 0.4));
    ctx.lineTo(tip[0] - 11 * Math.cos(ang + 0.4), tip[1] - 11 * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fillStyle = ink; ctx.fill();
    ctx.beginPath(); ctx.arc(mid[0], mid[1], 4.5, 0, 7); ctx.fillStyle = ink; ctx.fill();

    // --- electrodes, front-facing only -------------------------------------
    const chans = montage(state.nCh);
    const readings = chans.map((c) => ({
      ...c,
      v: dipolePotential(c.pos, pos, mom),
      r: rotate(c.pos, state.yaw, state.pitch),
    }));
    const rad = state.nCh > 64 ? 2.8 : state.nCh > 32 ? 3.6 : 5;
    for (const c of readings) {
      if (c.r[2] < -0.02) continue;                    // on the far side
      const [ex, ey] = project(c.r, cx, cy, sc);
      ctx.beginPath(); ctx.arc(ex, ey, rad, 0, 7);
      ctx.fillStyle = rgbStr(divergingRGB(c.v / maxAbs, dark));  // unshaded: the readout
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = dark ? 'rgba(255,255,255,.8)' : 'rgba(0,0,0,.6)';
      ctx.stroke();
    }

    /* pointer-events:none so a drag starting on the hint does not try to select
       it, and user-select:none so it cannot be highlighted at all. SVG <text> is
       selectable like any other text, and this label sits right where people put
       the pointer down. */
    el('text', {
      x: 12, y: H - 12, fill: token('--text-muted'), 'font-size': 11,
      style: 'pointer-events:none;user-select:none;-webkit-user-select:none',
    }, svg).textContent = 'drag to rotate';

    const vals = readings.map((c) => Math.abs(c.v));
    const peak = Math.max(...vals);
    const spread = vals.filter((v) => v > peak * 0.5).length;
    const r = Math.hypot(pos[0] / half[0], pos[1] / half[1], pos[2] / half[2]);
    out.innerHTML = [
      ['position', `${(r * 100).toFixed(0)}%${clamped ? ' (at the limit)' : ''}`, 'centre → scalp'],
      ['electrodes', state.nCh, 'sampling the surface'],
      ['above half-peak', `${spread} of ${state.nCh}`, 'how spread out it is'],
    ].map(([lab, v, sub]) => `<div class="stat"><div class="stat__value">${v}</div><div class="stat__label">${lab}<br><span class="muted">${sub}</span></div></div>`).join('');
  }

  loadHeadMesh().then((m) => { mesh = m; measure(m.V); draw(); })
    .catch((e) => {
      host.innerHTML = `<p class="small muted" style="padding:2rem;text-align:center">`
        + `could not load the head model (${e.message})</p>`;
    });
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

    /* The y-axis follows the data. It was pinned at 0.45 at the bottom, but a
       weak effect at 40 trials lands near 0.38 -- genuinely below chance, because
       a classifier fitted on 20 trials per class is noise -- and those points
       were being drawn underneath the plot. */
    const seen = measured.map((d) => d.a).concat(observed).filter(Number.isFinite);
    const lo = Math.max(0.2, Math.min(0.45, Math.floor((Math.min(...seen) - 0.04) * 20) / 20));
    const xs = scale(Math.log10(30), Math.log10(2000), PAD.l, W - PAD.r);
    const ys = scale(lo, 1.0, H - PAD.b, PAD.t);
    frame(svg, W, H, PAD, (v) => xs(Math.log10(v)), ys,
      { xLabel: 'trials', yLabel: 'AUC', xTicks: [50, 100, 250, 500, 1000, 2000] });

    el('line', { x1: PAD.l, x2: W - PAD.r, y1: ys(0.5), y2: ys(0.5),
      stroke: token('--border'), 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }, svg);

    const clip = clipped(svg, W, H, PAD);
    el('path', {
      d: linePath(measured.map((d) => [xs(Math.log10(d.n)), ys(d.a)])),
      fill: 'none', stroke: token('--series-6') || '#eb6834', 'stroke-width': 2.4,
    }, clip);
    for (const d of measured) {
      el('circle', { cx: xs(Math.log10(d.n)), cy: ys(d.a), r: 4.5,
        fill: token('--page'), stroke: token('--series-6') || '#eb6834', 'stroke-width': 2 }, clip);
    }

    // where the trials slider currently sits
    el('circle', { cx: xs(Math.log10(state.nTrials)), cy: ys(observed), r: 7,
      fill: token('--series-7') || '#2a78d6', stroke: token('--page'), 'stroke-width': 2 }, clip);
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
