"""
Quick pass over all downloaded records: apply the same clip/BPM checks as
qc_peaks.py but just print a final KEEP / REJECT tally and write the keep
list to keep_list.json. No plots -- fast, and gives you one clean summary
instead of scrolling through console output.

Run locally:
    python qc_summary.py
"""

import os
import glob
import json
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

IN_DIR = 'mimic_pleth_records'
PLOT_SECONDS = 30
LOWCUT, HIGHCUT = 0.5, 10
FILTER_ORDER = 3


def bandpass(sig, fs, low=LOWCUT, high=HIGHCUT, order=FILTER_ORDER):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def detect_peaks(sig, fs, max_hr=150):
    min_distance = int(fs * 60 / max_hr)
    prominence = np.std(sig) * 0.5
    peaks, _ = find_peaks(sig, distance=min_distance, prominence=prominence)
    return peaks


def clipping_fraction(sig, tol=1e-6):
    lo, hi = np.min(sig), np.max(sig)
    pinned = np.isclose(sig, lo, atol=tol) | np.isclose(sig, hi, atol=tol)
    return np.mean(pinned)


def main():
    npy_files = sorted(glob.glob(os.path.join(IN_DIR, '*.npy')))
    keep, reject = [], []

    for npy_path in npy_files:
        name = os.path.basename(npy_path).replace('.npy', '')
        json_path = npy_path.replace('.npy', '.json')
        try:
            raw = np.load(npy_path)
            with open(json_path) as f:
                meta = json.load(f)
            fs = meta['fs']

            valid = ~np.isnan(raw)
            raw = raw[valid]

            n_plot = min(int(PLOT_SECONDS * fs), len(raw))
            if n_plot < fs * 5:
                reject.append((name, 'too short'))
                continue

            segment = raw[:n_plot]
            clip_frac = clipping_fraction(segment)
            filtered = bandpass(segment, fs)
            peaks = detect_peaks(filtered, fs)
            implied_bpm = len(peaks) * 60 / (len(segment) / fs)

            if clip_frac > 0.05:
                reject.append((name, f'{clip_frac*100:.0f}% clipped'))
            elif implied_bpm < 40 or implied_bpm > 180:
                reject.append((name, f'{implied_bpm:.0f} bpm non-physiological'))
            else:
                keep.append((name, fs, meta['n_samples'], round(implied_bpm)))
        except Exception as e:
            reject.append((name, f'error: {e}'))

    print(f"\n=== KEEP ({len(keep)}) ===")
    for name, fs, n, bpm in keep:
        hours = n / fs / 3600
        print(f"  {name}  fs={fs}Hz  {hours:.2f}h  ~{bpm}bpm")

    print(f"\n=== REJECT ({len(reject)}) ===")
    for name, reason in reject:
        print(f"  {name}: {reason}")

    with open('keep_list.json', 'w') as f:
        json.dump([k[0] for k in keep], f, indent=2)
    print(f"\nSaved {len(keep)} names to keep_list.json")


if __name__ == '__main__':
    main()
    