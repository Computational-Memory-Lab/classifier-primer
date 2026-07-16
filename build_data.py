#!/usr/bin/env python3
"""
Precompute the small derived data files the website runs on.

The website is static, so it cannot read the 8.7 GB of EEG on the lab Drive.
This script reads the real data and writes a few hundred KB of derived
summaries into site/data/. Run it from the repo root:

    python3 site/build_data.py

It needs scipy and numpy, and it needs the Drive mounted. Nothing here is
required to *view* the site -- the outputs are committed. Re-run it only if the
underlying data or the committed results change.

What it reads
-------------
  <DRIVE>/events_NN.mat        all 67, ~2.7 KB each (behavioural codes)
  <DRIVE>/test_45.mat          one participant's EEG, 257 x 325 x 450
  outputs/{LDA,SVM}/*_raw.mat  the committed per-participant AUCs

Why participant 45: it is representative (all three AUCs modest and above
chance) and it is deliberately NOT participant 06, whose numbers are the answer
key to the test_project/ onboarding exercise.

The feature extraction below is a faithful port of
scripts/features/feature_label_moving_bin.m -- same 10 electrodes, same 100 ms
bins at a 40 ms step, same electrode-major column order. Keep them in sync.
"""

import json
import os
import sys

import numpy as np
import scipy.io as sio
from scipy import stats

DRIVE = ("/Users/devon7y/Library/CloudStorage/GoogleDrive-dyanitsk@ualberta.ca/"
         "Shared drives/CML documents/Classifier Project/data")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "site", "data")

DEMO = 45

# --- constants that mirror the MATLAB pipeline -----------------------------
FS = 250                     # Hz
N_SAMPLES = 325              # -100 ms .. +1200 ms
BASELINE_SAMPLES = 25        # first 25 samples are the pre-stimulus baseline
ELECTRODES = [64, 194, 21, 41, 214, 8, 101, 87, 153, 137]  # 1-based, as in MATLAB
BIN_SIZE = 25                # 100 ms
BIN_STEP = 10                # 40 ms
FZ, P3 = 21, 87              # the univariate electrodes (1-based)

CODES = {1: "Hit", 2: "Miss", 3: "CR", 4: "FA"}


SFP = os.path.join(REPO, "AdultAverageNet256_v1 (1).sfp")


def sample_to_ms(idx0):
    """0-based sample index -> ms relative to stimulus onset."""
    return (idx0 - BASELINE_SAMPLES) * 1000.0 / FS


def read_montage():
    """Electrode positions from the .sfp, projected flat for a scalp map.

    The file holds 3 fiducials, E1..E256, and Cz -- i.e. 256 electrodes plus the
    vertex reference, which is why the recordings have 257 channels.

    Projection is azimuthal equidistant, the standard for scalp maps: distance
    from the vertex is proportional to the angle from it, so the equator (ear
    level) lands on the unit circle and everything below it falls outside.
    """
    rows = []
    with open(SFP) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 4:
                rows.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    fids = {r[0]: r[1:] for r in rows if r[0].startswith("Fid")}
    elecs = [r for r in rows if not r[0].startswith("Fid")]

    out = {}
    for label, x, y, z in elecs:
        r = (x * x + y * y + z * z) ** 0.5
        phi = np.arccos(np.clip(z / r, -1, 1))       # 0 at the vertex
        az = np.arctan2(y, x)
        rad = phi / (np.pi / 2)                      # 1 at the equator
        # +Y is anterior (towards the nose), +X is the right ear
        out[label] = [round(float(rad * np.cos(az)), 4),
                      round(float(rad * np.sin(az)), 4)]
    return out, fids


def bin_starts():
    """0-based bin start indices, matching startBins = 26:10:301 in MATLAB."""
    start = BASELINE_SAMPLES              # 0-based 25 == MATLAB 26
    return list(range(start, N_SAMPLES - BIN_SIZE + 1, BIN_STEP))


def extract_features(eeg):
    """eeg: [257 x 325 x 450] -> X: [450 x 280], electrode-major."""
    starts = bin_starts()
    chans = [e - 1 for e in ELECTRODES]
    n_trials = eeg.shape[2]
    X = np.zeros((n_trials, len(chans) * len(starts)), dtype=np.float64)
    col = 0
    for ch in chans:                       # electrode-major, as in the MATLAB loop
        for s in starts:
            X[:, col] = eeg[ch, s:s + BIN_SIZE, :].mean(axis=0)
            col += 1
    return X


def dprime(counts):
    """Log-linear corrected d' and criterion, as in preprocessing/d_prime.m."""
    hits, misses, cr, fa = (counts["Hit"], counts["Miss"], counts["CR"], counts["FA"])
    old, new = hits + misses, cr + fa
    hr = (hits + 0.5) / (old + 1.0)
    fr = (fa + 0.5) / (new + 1.0)
    d = stats.norm.ppf(hr) - stats.norm.ppf(fr)
    c = -0.5 * (stats.norm.ppf(hr) + stats.norm.ppf(fr))
    return float(d), float(c), float(hr), float(fr)


def load_events(pid):
    return sio.loadmat(os.path.join(DRIVE, f"events_{pid:02d}.mat"))["test"][:, 1]


def participant_ids():
    ids = []
    for name in sorted(os.listdir(DRIVE)):
        if name.startswith("events_") and name.endswith(".mat"):
            ids.append(int(name[len("events_"):-len(".mat")]))
    return sorted(ids)


def read_auc(path):
    a = sio.loadmat(path)["AUC_all"][0, 0]
    out = {}
    for name in a.dtype.names:
        rec = a[name][0, 0]
        out[int(name[1:])] = {t: float(rec[t].squeeze()) for t in
                              ("OldNew", "HitMiss", "FAvsCR")}
    return out


def jsonable(obj):
    """NaN is not valid JSON -- a missing AUC becomes null.

    Two participants have too few false alarms to score FAvsCR at all, which is
    why the group N is 65 there and 67 elsewhere. Keep them as null rather than
    dropping them, so the site can show the gap honestly.
    """
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def summarize(values):
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    n = len(v)
    mean = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1) * sem)
    t = stats.ttest_1samp(v, 0.5)
    return {"mean": mean, "ci": [mean - half, mean + half], "n": n,
            "t": float(t.statistic), "df": n - 1, "p": float(t.pvalue)}


def write(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, separators=(",", ":"), allow_nan=False)
    print(f"  {name:28s} {os.path.getsize(path)/1024:8.1f} KB")


def main():
    if not os.path.isdir(DRIVE):
        sys.exit(f"Drive not mounted at:\n  {DRIVE}")
    os.makedirs(OUT, exist_ok=True)
    print("writing site/data/")

    ids = participant_ids()

    # -- 1. behaviour for every participant --------------------------------
    behaviour = {}
    for pid in ids:
        codes = load_events(pid)
        counts = {label: int((codes == code).sum()) for code, label in CODES.items()}
        d, c, hr, fr = dprime(counts)
        behaviour[pid] = {"counts": counts, "dprime": d, "criterion": c,
                          "hitRate": hr, "faRate": fr,
                          "accuracy": (counts["Hit"] + counts["CR"]) / len(codes)}

    # -- 2. real per-participant AUCs from the committed results -----------
    lda = read_auc(os.path.join(REPO, "outputs", "LDA", "LDA_results_raw.mat"))
    svm = read_auc(os.path.join(REPO, "outputs", "SVM", "SVM_results_raw.mat"))

    per_participant = []
    for pid in ids:
        per_participant.append({
            "id": pid,
            "lda": lda.get(pid),
            "svm": svm.get(pid),
            "dprime": behaviour[pid]["dprime"],
            "counts": behaviour[pid]["counts"],
        })

    group = {}
    for key, table in (("lda", lda), ("svm", svm)):
        group[key] = {t: summarize([table[p][t] for p in table])
                      for t in ("OldNew", "HitMiss", "FAvsCR")}

    write("participants.json", {
        "participants": per_participant,
        "group": group,
        "source": "outputs/{LDA,SVM}/*_results_raw.mat + events_NN.mat on the lab Drive",
    })

    # -- 2b. the montage, for the scalp map -------------------------------
    # The ten feature electrodes are not scattered: four sit on the midline and
    # the other six form three exact left/right mirror pairs. The map is how you
    # see that, and it is the real answer to "why these ten".
    pos, fids = read_montage()
    named = {21: "Fz", 87: "P3", 101: "Pz", 153: "P4"}
    write("montage.json", {
        "source": os.path.basename(SFP),
        "projection": "azimuthal equidistant; +Y anterior, +X right ear, 1.0 = equator",
        "note": ("E1..E256 are the electrodes; Cz is the vertex reference, which is "
                 "why the recordings have 257 channels but only 256 electrodes."),
        "all": {k: v for k, v in pos.items() if k != "Cz"},
        "cz": pos.get("Cz"),
        "electrodes": ELECTRODES,
        "names": {str(k): v for k, v in named.items()},
        "fiducials": {k: list(v) for k, v in fids.items()},
    })

    # -- 3. the demo participant's EEG ------------------------------------
    print(f"  reading test_{DEMO}.mat (~140 MB) ...")
    eeg = sio.loadmat(os.path.join(DRIVE, f"test_{DEMO}.mat"))["data"]
    codes = load_events(DEMO)
    assert eeg.shape == (257, N_SAMPLES, len(codes)), eeg.shape

    times = [round(sample_to_ms(i), 1) for i in range(N_SAMPLES)]

    # Condition-averaged ERPs at the two univariate electrodes, plus the 10
    # feature electrodes. Downsampled 2x in time to halve the payload; the
    # data is bandpass-filtered to 30 Hz so this loses nothing visible.
    keep = list(range(0, N_SAMPLES, 2))
    erp_channels = {}
    for ch in sorted(set(ELECTRODES + [FZ, P3])):
        per_cond = {}
        for code, label in CODES.items():
            sel = codes == code
            if sel.sum() == 0:
                continue
            wave = eeg[ch - 1, :, sel].mean(axis=0)
            per_cond[label] = [round(float(wave[i]), 3) for i in keep]
        erp_channels[str(ch)] = per_cond

    write("erp.json", {
        "participant": DEMO,
        "times": [times[i] for i in keep],
        "channels": erp_channels,
        "fz": FZ, "p3": P3,
        "electrodes": ELECTRODES,
        "counts": behaviour[DEMO]["counts"],
        "windows": {"FN400": [300, 500], "LPP": [500, 800]},
    })

    # Every single trial at Fz, so the site can show an average being built up
    # one real trial at a time. int16 + scale keeps it small.
    fz = eeg[FZ - 1, :, :][keep, :].T          # [trials x samples]
    fz_scale = float(np.abs(fz).max()) / 32000.0
    with open(os.path.join(OUT, "fz_trials.bin"), "wb") as fh:
        fh.write(np.round(fz / fz_scale).astype(np.int16).tobytes(order="C"))
    print(f"  fz_trials.bin                "
          f"{os.path.getsize(os.path.join(OUT, 'fz_trials.bin'))/1024:8.1f} KB")
    write("fz_trials.json", {
        "participant": DEMO, "channel": FZ,
        "shape": [int(fz.shape[0]), int(fz.shape[1])],
        "scale": fz_scale, "dtype": "int16",
        "times": [times[i] for i in keep],
        "labels": [int(c) for c in codes],
        "note": "row-major [trials x samples]; reconstruct with int16 * scale",
    })

    # -- 4. the feature matrix, for the in-browser classifier --------------
    X = extract_features(eeg)
    starts = bin_starts()
    assert X.shape == (450, 280), X.shape

    # int16 + per-matrix scale keeps this ~250 KB instead of ~2 MB of JSON.
    scale = float(np.abs(X).max()) / 32000.0
    q = np.round(X / scale).astype(np.int16)
    with open(os.path.join(OUT, "features_45.bin"), "wb") as fh:
        fh.write(q.tobytes(order="C"))
    print(f"  features_45.bin              "
          f"{os.path.getsize(os.path.join(OUT, 'features_45.bin'))/1024:8.1f} KB")

    write("features_45.json", {
        "participant": DEMO,
        "shape": list(X.shape),
        "scale": scale,
        "dtype": "int16",
        "order": "row-major, electrode-major columns",
        "labels": [int(c) for c in codes],
        "electrodes": ELECTRODES,
        "binStartMs": [round(sample_to_ms(s), 1) for s in starts],
        "binWidthMs": BIN_SIZE * 1000 / FS,
        "binStepMs": BIN_STEP * 1000 / FS,
        "note": "reconstruct with X = int16 * scale",
    })

    print("done")


if __name__ == "__main__":
    main()
