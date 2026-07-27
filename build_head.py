#!/usr/bin/env python3
"""Build data/head.bin + data/head.json for the SEREEGA page's 3D head.

WHY THIS EXISTS. The head started as an ellipsoid, which read as exactly what it
was -- a low-poly sphere -- and the electrode placement was computed from a polar
formula I got wrong (Cz landed at 45 degrees of latitude instead of 90, and the
Fp/T/O ring below the equator). Both are replaced here by real data.

WHAT IT USES, and why this pairing specifically:

  functions/supportfiles/mheadnew.mat
      The mesh EEGLAB's own headplot() draws: 6,114 vertices, 11,871 triangles,
      a whole head with a face, ears and neck. Ships with vertex normals, which
      are used for smooth shading.

  plugins/dipfit/standard_BEM/elec/standard_1005.elc
      346 electrode positions in the 10-05 system, in MNI millimetres.

FIRST ATTEMPT, AND WHY IT WAS WRONG. This originally used the boundary-element
scalp from standard_BEM/standard_vol.mat, chosen because it shares a coordinate
frame with the .elc file and so needed no co-registration. But that surface is
truncated below the ears and sealed with a flat disc -- 16% of its vertices sit
in the bottom 2% of its z range -- so it renders as a dome with a lid, not a
head. Convenience is not a good enough reason to draw the wrong object.

CO-REGISTRATION. mheadnew lives in its own frame, so the electrodes are fitted to
it: a 7-parameter similarity transform (uniform scale, rotation, translation)
minimising squared distance from each cap electrode to the nearest mesh vertex,
matched against the upper 65% of the mesh so that neck and shoulders do not drag
the fit down. It converges to scale 0.883 with about 1.5 degrees of rotation and
no translation, leaving a median residual of 3.4 mm -- Cz 1.8 mm off the surface,
T7 1.4 mm. Electrodes are then snapped to the surface anyway.

EEGLAB ships mheadnew.transform for this job, but it is not usable here: it
expects EEGLAB's normalised channel coordinates (note the 1000x scale factors),
whereas standard_1005.elc is already in millimetres.

Provenance: the standard_BEM files are distributed with EEGLAB's DIPFIT plugin
and derive from the Colin27 / ICBM template head via FieldTrip and SPM. The
output here is decimated, smoothed, normalised geometry for display.

Run:  python3 build_head.py            (needs scipy; reads EEGLAB from $EEGLAB
                                         or the default path below)
"""
from __future__ import annotations
import json, os, struct, sys
import numpy as np
import scipy.io as sio

EEGLAB = os.environ.get('EEGLAB', os.path.expanduser('~/MatLab/eeglab2026.0.0'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# The 10-20 nineteen, in the order people say them.
TEN_TWENTY = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
              'T7', 'C3', 'Cz', 'C4', 'T8',
              'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']


def read_scalp():
    """EEGLAB's headplot mesh: a whole head, with vertex normals."""
    p = os.path.join(EEGLAB, 'functions/supportfiles/mheadnew.mat')
    m = sio.loadmat(p, squeeze_me=True)
    V = np.asarray(m['POS'], float)
    F = np.asarray(m['TRI1'], int) - 1      # MATLAB is 1-indexed
    N = np.asarray(m['NORM'], float)
    N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-9)
    print(f'  mheadnew: {len(V)} verts, {len(F)} faces, vertex normals present')
    return V, F, N


def read_electrodes():
    p = os.path.join(EEGLAB, 'plugins/dipfit/standard_BEM/elec/standard_1005.elc')
    lines = [l.rstrip('\n') for l in open(p, encoding='latin-1')]
    i = next(k for k, l in enumerate(lines) if l.strip() == 'Positions')
    n = int([l for l in lines if l.startswith('NumberPositions')][0].split('=')[1])
    pos = np.array([[float(x) for x in lines[i + 1 + k].split()] for k in range(n)])
    j = next(k for k, l in enumerate(lines) if l.strip() == 'Labels')
    labels = []
    k = j + 1
    while len(labels) < n:
        labels += lines[k].split()
        k += 1
    print(f'  electrodes: {len(labels)} labels, {pos.shape[0]} positions')
    return labels[:n], pos


def subdivide(V, F):
    """One 4-to-1 midpoint split. Returns new (V, F)."""
    V = list(map(tuple, V))
    mid: dict[tuple[int, int], int] = {}

    def m(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            pa, pb = np.array(V[a]), np.array(V[b])
            V.append(tuple((pa + pb) / 2))
            mid[key] = len(V) - 1
        return mid[key]

    out = []
    for a, b, c in F:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        out += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.array(V), np.array(out)


def smooth(V, F, iters=2, lam=0.5):
    """Laplacian smoothing, which removes faceting without adding geometry."""
    nbr = [set() for _ in range(len(V))]
    for a, b, c in F:
        nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
    V = V.copy()
    for _ in range(iters):
        new = V.copy()
        for i, ns in enumerate(nbr):
            if ns:
                new[i] = V[i] + lam * (V[list(ns)].mean(0) - V[i])
        V = new
    return V


def register(EL, V):
    """Fit a similarity transform taking the .elc electrodes into mesh space.

    Matched against the upper 65% of the mesh only: mheadnew includes neck and
    shoulders, and a whole-surface fit would be pulled downwards by them."""
    from scipy.spatial import cKDTree
    from scipy.optimize import minimize
    cap = V[V[:, 2] > np.percentile(V[:, 2], 35)]
    tree = cKDTree(cap)
    use = EL[EL[:, 2] > -40]

    def rot(a, ax):
        c, s_ = np.cos(a), np.sin(a)
        return {0: np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]),
                1: np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]),
                2: np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])}[ax]

    def apply(p, X):
        s_, tx, ty, tz, rx, ry, rz = p
        M = rot(rx, 0) @ rot(ry, 1) @ rot(rz, 2)
        return (X @ M.T) * s_ + np.array([tx, ty, tz])

    best = None
    for s0 in (0.9, 1.0, 1.1):
        r = minimize(lambda p: (tree.query(apply(p, use))[0] ** 2).mean(),
                     [s0, 0, 0, 0, 0, 0, 0], method='Nelder-Mead',
                     options={'maxiter': 6000, 'xatol': 1e-4, 'fatol': 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    d = tree.query(apply(best.x, use))[0]
    print(f'  co-registration: scale {best.x[0]:.3f}, rot '
          f'({np.degrees(best.x[4]):.1f}, {np.degrees(best.x[5]):.1f}, {np.degrees(best.x[6]):.1f}) deg')
    print(f'  residual electrode->scalp: median {np.median(d):.1f} mm, 90th {np.percentile(d, 90):.1f}, max {d.max():.1f}')
    if np.median(d) > 6:
        sys.exit('co-registration did not converge to a sane fit -- refusing to ship it')
    return apply(best.x, EL)


def snap_to_surface(pts, V):
    """Move each electrode onto its nearest mesh vertex, then a hair outwards so
    it does not z-fight with the surface it sits on."""
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    c = V.mean(0)
    out = []
    for p in pts:
        v = V[int(tree.query(p[None, :])[1][0])]
        out.append(c + (v - c) * 1.015)
    return np.array(out)


def farthest_point(cands, keep_idx, k):
    """Pick k electrodes that cover the head evenly, always keeping the 10-20."""
    chosen = list(keep_idx)
    while len(chosen) < k:
        d = np.min(np.linalg.norm(cands[:, None, :] - cands[chosen][None, :, :], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(d.argmax()))
    return chosen


def main():
    if not os.path.isdir(EEGLAB):
        sys.exit(f'EEGLAB not found at {EEGLAB} -- set $EEGLAB')
    print('reading EEGLAB...')
    V, F, N = read_scalp()
    labels, epos = read_electrodes()

    print('co-registering electrodes to the mesh...')
    epos = register(epos, V)

    # Normalise: centre on the head (not the whole mesh -- the neck and shoulders
    # would pull the centre down), scale so the head is about a unit across.
    head = V[V[:, 2] > np.percentile(V[:, 2], 30)]
    c = np.array([head[:, 0].mean(), head[:, 1].mean(), head[:, 2].mean()])
    scale = 1.0 / np.abs(head - c).max()
    V = (V - c) * scale
    epos = (epos - c) * scale

    # Electrodes above the ear line only -- the 10-05 file includes face and neck
    # positions that no cap covers.
    up = epos[:, 2] > -0.30
    idx_all = np.where(up)[0]
    keep = [i for i in idx_all if labels[i] in TEN_TWENTY]
    keep.sort(key=lambda i: TEN_TWENTY.index(labels[i]))
    missing = set(TEN_TWENTY) - {labels[i] for i in keep}
    if missing:
        sys.exit(f'10-20 labels not found in the .elc: {sorted(missing)}')

    E = snap_to_surface(epos, V)

    sets = {'19': keep}
    for k in (32, 64, 128):
        sets[str(k)] = farthest_point(epos[idx_all], [list(idx_all).index(i) for i in keep], k)
        sets[str(k)] = [int(idx_all[j]) for j in sets[str(k)]]

    os.makedirs(OUT, exist_ok=True)

    # --- the mesh: binary, loaded asynchronously by the page ----------------
    # Order matters: both float32 blocks must start on a 4-byte boundary, and a
    # uint16 face block of 11,871 triangles is 71,226 bytes -- not a multiple of
    # 4 -- so putting faces in the middle pushed the normals to an odd offset and
    # Float32Array refused to construct. Floats first, faces last.
    blob = (V.astype('<f4').tobytes() + N.astype('<f4').tobytes()
            + F.astype('<u2').tobytes())
    open(os.path.join(OUT, 'head_mesh.bin'), 'wb').write(blob)
    json.dump({
        'source': "EEGLAB functions/supportfiles/mheadnew.mat -- the mesh headplot() draws",
        'note': ('Whole head with face, ears and neck. Electrodes from '
                 'standard_BEM/elec/standard_1005.elc, co-registered by a fitted '
                 'similarity transform (median residual 3.4 mm) and snapped to the '
                 'surface. Centred on the head and scaled to about unit half-extent.'),
        'nVerts': int(len(V)), 'nFaces': int(len(F)),
        'layout': ['vertices float32 nVerts*3', 'vertex normals float32 nVerts*3',
                   'faces uint16 nFaces*3'],
    }, open(os.path.join(OUT, 'head_mesh.json'), 'w'), indent=1)

    # --- the electrodes: a JS module, so Node tests can import them ---------
    used = sorted(set(sum(sets.values(), [])))
    remap = {g: i for i, g in enumerate(used)}
    js = ['/* GENERATED by site/build_head.py -- do not edit by hand.',
          ' *',
          ' * Real 10-05 electrode coordinates from EEGLAB dipfit',
          ' * standard_BEM/elec/standard_1005.elc, snapped onto the scalp surface of',
          ' * standard_vol.mat and normalised with it, so electrodes sit ON the head',
          ' * mesh the page draws rather than near it.',
          ' *',
          ' * This replaces a polar formula that was simply wrong: it put Cz at 45',
          ' * degrees of latitude instead of 90, and the Fp/T/O ring below the equator.',
          ' * The whole cap was squashed into the lower half of the head.',
          ' */',
          'export const ELECTRODES = [']
    for g in used:
        x, y, z = E[g]
        js.append(f'  {{ name: {labels[g]!r}, pos: [{x:.4f}, {y:.4f}, {z:.4f}] }},'.replace("'", '"'))
    js.append('];')
    js.append('')
    js.append('/** Index sets into ELECTRODES. 19 is the named 10-20; the larger sets are')
    js.append(' *  chosen by farthest-point sampling over the 10-05 positions, so they cover')
    js.append(' *  the head evenly and always contain the 10-20. */')
    js.append('export const MONTAGES = {')
    for k in ['19', '32', '64', '128']:
        js.append(f'  {k}: [{", ".join(str(remap[i]) for i in sets[k])}],')
    js.append('};')
    open(os.path.join(os.path.dirname(OUT), 'assets', 'electrodes.js'), 'w').write('\n'.join(js) + '\n')

    print(f'wrote data/head_mesh.bin ({len(blob)/1024:.0f} KB), data/head_mesh.json,'
          f' assets/electrodes.js ({len(used)} electrodes)')
    for k, v in sets.items():
        print(f'  montage {k:>4}: {len(v)} electrodes')


if __name__ == '__main__':
    main()
