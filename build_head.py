#!/usr/bin/env python3
"""Build data/head.bin + data/head.json for the SEREEGA page's 3D head.

WHY THIS EXISTS. The head started as an ellipsoid, which read as exactly what it
was -- a low-poly sphere -- and the electrode placement was computed from a polar
formula I got wrong (Cz landed at 45 degrees of latitude instead of 90, and the
Fp/T/O ring below the equator). Both are replaced here by real data.

WHAT IT USES, and why this pairing specifically:

  plugins/dipfit/standard_BEM/standard_vol.mat -> vol.bnd[0]
      The outermost boundary-element surface, conductivity 0.33 -- the SCALP.
      This is the surface electrodes actually sit on.

  plugins/dipfit/standard_BEM/elec/standard_1005.elc
      346 electrode positions in the 10-05 system.

They come from the same directory and the same template head, so they share a
coordinate frame. That matters: it means electrodes land on the scalp by
construction rather than by a co-registration fudge. mheadnew.mat is a prettier
mesh (it has a face) but lives in its own frame and includes neck and shoulders,
which drags the centroid down and makes radial projection unreliable near the
vertex. Correctness beat prettiness.

The BEM scalp ships at 996 triangles, which still looks faceted, so it is
subdivided once (4x) and Laplacian-smoothed. That adds no anatomical information
-- it is interpolation, not detail -- but it removes the flat-shaded polygon look
without inventing geometry.

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
    p = os.path.join(EEGLAB, 'plugins/dipfit/standard_BEM/standard_vol.mat')
    vol = sio.loadmat(p, squeeze_me=True, struct_as_record=False)['vol']
    bnds = np.atleast_1d(vol.bnd)
    conds = np.atleast_1d(vol.cond)
    # The scalp is the outermost surface: pick the one with the largest extent.
    idx = int(np.argmax([np.ptp(np.asarray(b.pnt, float), axis=0).sum() for b in bnds]))
    b = bnds[idx]
    V = np.asarray(b.pnt, float)
    F = np.asarray(b.tri, int) - 1          # MATLAB is 1-indexed
    print(f'  scalp = bnd[{idx}] (cond {conds[idx]:.4f}): {len(V)} verts, {len(F)} faces')
    return V, F


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


def snap_to_surface(pts, V):
    """Move each electrode to its nearest mesh vertex, then a hair outwards so it
    is not z-fighting with the surface it sits on."""
    out = []
    c = V.mean(0)
    for p in pts:
        d = np.linalg.norm(V - p, axis=1)
        v = V[int(d.argmin())]
        out.append(c + (v - c) * 1.012)
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
    V, F = read_scalp()
    labels, epos = read_electrodes()

    print('subdividing + smoothing...')
    V, F = subdivide(V, F)
    V = smooth(V, F, iters=2, lam=0.5)
    print(f'  -> {len(V)} verts, {len(F)} faces')

    # Normalise: centre on the mesh centroid, scale so the head is ~1 unit tall.
    c = V.mean(0)
    scale = 1.0 / np.abs(V - c).max()
    V = (V - c) * scale
    epos = (epos - c) * scale

    # Electrodes above the ear line only -- the 10-05 file includes face and neck
    # positions that no cap covers.
    up = epos[:, 2] > -0.28
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
    blob = V.astype('<f4').tobytes() + F.astype('<u2').tobytes()
    open(os.path.join(OUT, 'head_mesh.bin'), 'wb').write(blob)
    json.dump({
        'source': 'EEGLAB dipfit standard_BEM/standard_vol.mat, outermost (scalp) boundary',
        'note': ('Colin27/ICBM template head via FieldTrip/SPM. Subdivided once and '
                 'Laplacian-smoothed for display; centred and scaled to unit half-extent.'),
        'nVerts': int(len(V)), 'nFaces': int(len(F)),
        'layout': ['vertices float32 nVerts*3', 'faces uint16 nFaces*3'],
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
