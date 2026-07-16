#!/usr/bin/env python3
"""
Build the grand-average ERP across all 67 participants.

Why this is separate from build_data.py: it has to read all 67 test_NN.mat files
(~9.3 GB). The Drive keeps them as cloud stubs, so this pulls each one down,
averages it, caches the (tiny) result, and evicts the local copy again. Disk use
stays flat at roughly one file, and a re-run costs nothing because of the cache.

    python3 site/build_grand_average.py            # uses the cache where it can
    python3 site/build_grand_average.py --refresh  # ignore the cache, re-read everything

Output: site/data/grand_erp.json (~130 KB).
Cache:  site/data/.erp_cache/P<id>.npz (~30 KB each, gitignored)

The point of all this: a single participant's ERP is far too noisy to show the
classic old/new effect -- for participant 45 the Hit/CR difference at Fz is about
0.02 uV, i.e. nothing at all. The effect is a group-level phenomenon, and the
honest way to show it is to average over people the way the literature does.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import (  # noqa: E402
    DRIVE, OUT, CODES, ELECTRODES, FZ, P3, N_SAMPLES,
    participant_ids, load_events, sample_to_ms, jsonable,
)

CHANNELS = sorted(set(ELECTRODES + [FZ, P3]))
KEEP = list(range(0, N_SAMPLES, 2))          # 2x decimation, as in build_data
WINDOWS = {"FN400": (300, 500), "LPP": (500, 800)}
LABELS = list(CODES.values())                # Hit, Miss, CR, FA
CACHE = os.path.join(OUT, ".erp_cache")
RETRIES = 4


def evict(path):
    """Drop the local cache of a Drive file, keeping disk usage flat."""
    subprocess.run(["brctl", "evict", path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_eeg(path):
    """Read one participant's EEG, retrying: the Drive times out under load."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return sio.loadmat(path)["data"]
        except Exception as exc:                          # noqa: BLE001
            last = exc
            if attempt < RETRIES:
                wait = 5 * attempt
                print(f"        attempt {attempt} failed ({exc}); retrying in {wait}s",
                      flush=True)
                time.sleep(wait)
    raise last


def participant_average(pid, refresh=False):
    """[channels x conditions x samples] of condition means, cached on disk.

    Returns (array, present) where present[c] is False if the participant had
    too few trials of condition c to average.
    """
    cpath = os.path.join(CACHE, f"P{pid:02d}.npz")
    if os.path.exists(cpath) and not refresh:
        z = np.load(cpath)
        return z["avg"], z["present"]

    path = os.path.join(DRIVE, f"test_{pid:02d}.mat")
    eeg = read_eeg(path)
    codes = load_events(pid)
    if eeg.shape != (257, N_SAMPLES, len(codes)):
        evict(path)
        raise ValueError(f"unexpected shape {eeg.shape}")

    avg = np.zeros((len(CHANNELS), len(LABELS), len(KEEP)), dtype=np.float32)
    present = np.zeros(len(LABELS), dtype=bool)
    for ci, ch in enumerate(CHANNELS):
        trace = eeg[ch - 1]                                # [samples x trials]
        for li, (code, _) in enumerate(CODES.items()):
            sel = codes == code
            if sel.sum() < 2:
                continue
            present[li] = True
            avg[ci, li] = trace[:, sel].mean(axis=1)[KEEP]
    evict(path)

    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(cpath, avg=avg, present=present)
    return avg, present


def main():
    refresh = "--refresh" in sys.argv
    ids = participant_ids()
    times = np.array([sample_to_ms(i) for i in KEEP])

    per = {}          # pid -> [channels x conditions x samples]
    presence = {}
    failed = []
    for n, pid in enumerate(ids, 1):
        cached = os.path.exists(os.path.join(CACHE, f"P{pid:02d}.npz")) and not refresh
        try:
            avg, present = participant_average(pid, refresh=refresh)
        except Exception as exc:                            # noqa: BLE001
            print(f"  [{n:2d}/{len(ids)}] P{pid:02d} FAILED after {RETRIES} attempts: {exc}",
                  flush=True)
            failed.append(pid)
            continue
        per[pid] = avg
        presence[pid] = present
        print(f"  [{n:2d}/{len(ids)}] P{pid:02d} ok{' (cached)' if cached else ''}",
              flush=True)

    if failed:
        print(f"\n  {len(failed)} participant(s) could not be read: {failed}")
        print("  Re-run to retry just those -- everything else comes from the cache.")
        sys.exit(1)

    kept = sorted(per)
    stack = np.stack([per[p] for p in kept])               # [P x ch x cond x samples]

    channels = {}
    for ci, ch in enumerate(CHANNELS):
        per_cond = {}
        for li, lab in enumerate(LABELS):
            ok = [i for i, p in enumerate(kept) if presence[p][li]]
            if not ok:
                continue
            arr = stack[ok, ci, li, :]
            m = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(len(ok))
            per_cond[lab] = {
                "mean": [round(float(v), 4) for v in m],
                "sem": [round(float(v), 4) for v in sem],
                "n": len(ok),
            }
        channels[str(ch)] = per_cond

    # per-participant mean voltage in each ERP window, for the univariate story
    window_means = []
    for p in kept:
        row = {"id": int(p)}
        for ch, name in ((FZ, "Fz"), (P3, "P3")):
            ci = CHANNELS.index(ch)
            for wname, (lo, hi) in WINDOWS.items():
                sl = (times >= lo) & (times <= hi)
                for li, lab in enumerate(LABELS):
                    row[f"{name}_{wname}_{lab}"] = (
                        float(per[p][ci, li, sl].mean()) if presence[p][li] else None)
        window_means.append(row)

    out = {
        "participants": [int(p) for p in kept],
        "nParticipants": len(kept),
        "times": [round(float(t), 1) for t in times],
        "channels": channels,
        "fz": FZ, "p3": P3,
        "electrodes": ELECTRODES,
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "windowMeans": window_means,
        "note": "grand average across participants of each participant's condition average",
    }
    path = os.path.join(OUT, "grand_erp.json")
    with open(path, "w") as fh:
        json.dump(jsonable(out), fh, separators=(",", ":"), allow_nan=False)
    print(f"\n  grand_erp.json  {os.path.getsize(path)/1024:.1f} KB "
          f"({len(kept)} participants)")


if __name__ == "__main__":
    main()
