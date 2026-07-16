/* Regularized linear discriminant analysis, SMOTE, and stratified k-fold
   cross-validation -- a JavaScript port of what the MATLAB pipeline does in
   scripts/classifiers/run_lda_auc.m and scripts/classifiers/applySMOTE.m.

   Kept in its own module so it can be exercised directly by site/_selftest.html
   rather than only through the page that uses it. If you change the MATLAB, this
   is the file that has to change with it. */

import { auc, rng } from './site.js';

/** Squared euclidean distance. */
export function dist2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s;
}

/**
 * Regularized LDA. Mirrors fitcdiscr(..., 'Gamma', g): the *correlation* matrix
 * is shrunk toward the identity and then rescaled by the standard deviations.
 * Returns a scoring function; only the ordering of its output matters for AUC.
 */
export function trainLDA(X, y, gamma) {
  const d = X[0].length;
  const m = [new Float64Array(d), new Float64Array(d)];
  const n = [0, 0];
  for (let i = 0; i < X.length; i++) {
    const c = y[i]; n[c]++;
    for (let j = 0; j < d; j++) m[c][j] += X[i][j];
  }
  for (const c of [0, 1]) for (let j = 0; j < d; j++) m[c][j] /= n[c] || 1;

  // pooled within-class covariance
  const S = Array.from({ length: d }, () => new Float64Array(d));
  for (let i = 0; i < X.length; i++) {
    const mu = m[y[i]];
    for (let a = 0; a < d; a++) {
      const va = X[i][a] - mu[a];
      if (va === 0) continue;
      for (let b = a; b < d; b++) S[a][b] += va * (X[i][b] - mu[b]);
    }
  }
  const df = Math.max(1, X.length - 2);
  for (let a = 0; a < d; a++) for (let b = a; b < d; b++) { S[a][b] /= df; S[b][a] = S[a][b]; }

  const sdv = new Float64Array(d);
  for (let a = 0; a < d; a++) sdv[a] = Math.sqrt(S[a][a]) || 1e-12;
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < d; b++) {
      const corr = S[a][b] / (sdv[a] * sdv[b]);
      const reg = a === b ? 1 : (1 - gamma) * corr;
      S[a][b] = reg * sdv[a] * sdv[b];
    }
  }

  // solve S w = (m1 - m0)
  const A = S.map((r) => Float64Array.from(r));
  const b = new Float64Array(d);
  for (let j = 0; j < d; j++) b[j] = m[1][j] - m[0][j];
  for (let i = 0; i < d; i++) A[i][i] += 1e-9;
  for (let i = 0; i < d; i++) {
    let piv = i;
    for (let k = i + 1; k < d; k++) if (Math.abs(A[k][i]) > Math.abs(A[piv][i])) piv = k;
    if (piv !== i) {
      const t = A[i]; A[i] = A[piv]; A[piv] = t;
      const tb = b[i]; b[i] = b[piv]; b[piv] = tb;
    }
    const p = A[i][i];
    if (Math.abs(p) < 1e-14) continue;
    for (let k = i + 1; k < d; k++) {
      const f = A[k][i] / p;
      if (f === 0) continue;
      for (let j = i; j < d; j++) A[k][j] -= f * A[i][j];
      b[k] -= f * b[i];
    }
  }
  const w = new Float64Array(d);
  for (let i = d - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < d; j++) s -= A[i][j] * w[j];
    w[i] = Math.abs(A[i][i]) < 1e-14 ? 0 : s / A[i][i];
  }
  const score = (row) => { let s = 0; for (let j = 0; j < d; j++) s += row[j] * w[j]; return s; };
  score.weights = w;
  return score;
}

/**
 * SMOTE: oversample the minority class to parity by interpolating between a
 * minority point and one of its k nearest minority neighbours. Note the gap is
 * drawn per feature, matching applySMOTE.m rather than canonical SMOTE.
 * Must only ever be given a training fold.
 */
export function smote(X, y, rand, k = 4) {
  const idx = [[], []];
  y.forEach((c, i) => idx[c].push(i));
  const minc = idx[0].length < idx[1].length ? 0 : 1;
  const need = idx[1 - minc].length - idx[minc].length;
  if (need <= 0 || idx[minc].length < 2) return { X, y };
  const Xm = idx[minc].map((i) => X[i]);
  const kk = Math.min(k, Xm.length - 1);
  const nn = Xm.map((a, i) => Xm
    .map((b, j) => ({ j, d: j === i ? Infinity : dist2(a, b) }))
    .sort((u, v) => u.d - v.d).slice(0, kk).map((o) => o.j));
  const Xo = X.slice(), yo = y.slice();
  for (let t = 0; t < need; t++) {
    const i = Math.floor(rand() * Xm.length);
    const j = nn[i][Math.floor(rand() * kk)];
    const row = new Float64Array(Xm[i].length);
    for (let c = 0; c < row.length; c++) row[c] = Xm[i][c] + rand() * (Xm[j][c] - Xm[i][c]);
    Xo.push(row); yo.push(minc);
  }
  return { X: Xo, y: yo };
}

/** Stratified fold assignment: each class is shuffled and dealt round-robin. */
export function stratifiedFolds(y, k, rand) {
  const folds = new Array(y.length);
  for (const c of [0, 1]) {
    const idx = y.map((v, i) => (v === c ? i : -1)).filter((i) => i >= 0);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    idx.forEach((i, n) => { folds[i] = n % k; });
  }
  return folds;
}

/**
 * k-fold cross-validated LDA.
 * opts: { gamma, smote, cheat, folds }
 * `cheat` scores each fold's model on its own training trials -- deliberately
 * wrong, and used on the site to show what overfitting looks like.
 */
export function crossValidate(X, y, opts = {}, seed = 1) {
  const { gamma = 0.5, smote: useSmote = true, cheat = false } = opts;
  const rand = rng(seed);
  const counts = [y.filter((v) => v === 0).length, y.filter((v) => v === 1).length];
  let k = opts.folds || 10;
  k = Math.min(k, Math.max(2, Math.min(...counts)));
  const folds = stratifiedFolds(y, k, rand);

  const foldAucs = [];
  const allLab = [], allScore = [];
  for (let f = 0; f < k; f++) {
    const trI = [], teI = [];
    folds.forEach((v, i) => (v === f ? teI : trI).push(i));
    if (!teI.length || !trI.length) continue;
    let Xtr = trI.map((i) => X[i]);
    let ytr = trI.map((i) => y[i]);
    if (useSmote) ({ X: Xtr, y: ytr } = smote(Xtr, ytr, rand));
    if (new Set(ytr).size < 2) continue;
    const score = trainLDA(Xtr, ytr, gamma);
    const evalI = cheat ? trI : teI;
    const labs = evalI.map((i) => y[i]);
    const scs = evalI.map((i) => score(X[i]));
    if (new Set(labs).size < 2) continue;
    foldAucs.push(auc(labs, scs));
    allLab.push(...labs);
    allScore.push(...scs);
  }
  return { foldAucs, allLab, allScore, k, counts, nFeat: X[0].length };
}
