"""
Step 4: Turn per-record IBI series into a windowed regular/irregular dataset.

Method: slide a window of WINDOW_BEATS consecutive, valid, contiguous IBIs
across each record. For each window compute:
    CV = std(IBI) / mean(IBI)
Label irregular (1) if CV > CV_THRESHOLD, else regular (0).

This is unsupervised/rule-based since MIMIC waveform records don't ship
per-beat rhythm labels -- CV of IBI is a standard, well-established proxy
for rhythm irregularity (used in AF-screening literature).

First pass: prints the CV distribution across your whole dataset so you can
sanity-check the threshold before committing to final labels. Second pass
(after you confirm/adjust CV_THRESHOLD below) builds and saves the dataset.

Run locally:
    python build_dataset.py
"""

import os
import json
import numpy as np

IBI_DIR = 'ibi_data'
OUT_DIR = 'dataset'
WINDOW_BEATS = 10      # beats per window
STEP_BEATS = 5         # slide step (50% overlap)
CV_THRESHOLD = 0.08    # tune after looking at the printed histogram

os.makedirs(OUT_DIR, exist_ok=True)


def sliding_windows(ibi, valid_mask, window, step):
    """Yield (start_idx, window_ibis) for contiguous, fully-valid windows."""
    n = len(ibi)
    for start in range(0, n - window + 1, step):
        seg_valid = valid_mask[start:start + window]
        if not seg_valid.all():
            continue
        yield start, ibi[start:start + window]


def main():
    with open('keep_list.json') as f:
        keep_names = json.load(f)

    all_cv = []
    per_record_windows = {}

    for name in keep_names:
        path = os.path.join(IBI_DIR, f'{name}_ibi.npz')
        if not os.path.exists(path):
            continue
        data = np.load(path)
        ibi = data['ibi']
        valid = data['ibi_valid_mask']

        windows = list(sliding_windows(ibi, valid, WINDOW_BEATS, STEP_BEATS))
        cvs = [np.std(w) / np.mean(w) for _, w in windows]
        all_cv.extend(cvs)
        per_record_windows[name] = (windows, cvs)

    all_cv = np.array(all_cv)
    print(f"Total candidate windows across {len(keep_names)} records: {len(all_cv)}\n")
    print("CV distribution (std/mean of IBI per window):")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:3d}th percentile: {np.percentile(all_cv, p):.4f}")

    n_irregular = (all_cv > CV_THRESHOLD).sum()
    print(f"\nWith CV_THRESHOLD = {CV_THRESHOLD}:")
    print(f"  Regular:   {len(all_cv) - n_irregular}  ({100*(1 - n_irregular/len(all_cv)):.1f}%)")
    print(f"  Irregular: {n_irregular}  ({100*n_irregular/len(all_cv):.1f}%)")

    # Build and save the actual dataset using current threshold
    X, y, record_ids = [], [], []
    for name, (windows, cvs) in per_record_windows.items():
        for (start, w), cv in zip(windows, cvs):
            X.append(w)
            y.append(1 if cv > CV_THRESHOLD else 0)
            record_ids.append(name)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    record_ids = np.array(record_ids)

    np.savez(os.path.join(OUT_DIR, 'ibi_windows_dataset.npz'),
             X=X, y=y, record_ids=record_ids,
             window_beats=WINDOW_BEATS, step_beats=STEP_BEATS, cv_threshold=CV_THRESHOLD)

    print(f"\nSaved dataset: X shape {X.shape}, y shape {y.shape}")
    print(f"-> {os.path.join(OUT_DIR, 'ibi_windows_dataset.npz')}")
    print("\nIf class balance above looks off, adjust CV_THRESHOLD at the top of this "
          "script and rerun -- it's cheap, no need to recompute peaks/IBIs.")


if __name__ == '__main__':
    main()