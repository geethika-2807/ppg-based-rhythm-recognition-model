"""
Step 2: Bandpass filter each downloaded PLETH record, run peak detection,
and plot results for visual QC. Reject any record where peaks clearly
don't track the real pulse (flatline, motion artifact, sensor dropout).

Reads the .npy/.json pairs produced by fetch_mimic_pleth.py.

Run locally:
    pip install scipy matplotlib numpy
    python qc_peaks.py
"""

import os
import glob
import json
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

IN_DIR = 'mimic_pleth_records'
OUT_DIR = 'qc_plots'
PLOT_SECONDS = 30
LOWCUT, HIGHCUT = 0.5, 10
FILTER_ORDER = 3

os.makedirs(OUT_DIR, exist_ok=True)


def bandpass(sig, fs, low=LOWCUT, high=HIGHCUT, order=FILTER_ORDER):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def detect_peaks(sig, fs, max_hr=150):
    min_distance = int(fs * 60 / max_hr)
    prominence = np.std(sig) * 0.5
    peaks, props = find_peaks(sig, distance=min_distance, prominence=prominence)
    return peaks, props


def clipping_fraction(sig, tol=1e-6):
    lo, hi = np.min(sig), np.max(sig)
    pinned = np.isclose(sig, lo, atol=tol) | np.isclose(sig, hi, atol=tol)
    return np.mean(pinned)


def qc_record(npy_path):
    name = os.path.basename(npy_path).replace('.npy', '')
    json_path = npy_path.replace('.npy', '.json')

    raw = np.load(npy_path)
    with open(json_path) as f:
        meta = json.load(f)
    fs = meta['fs']

    valid = ~np.isnan(raw)
    if not valid.all():
        raw = raw[valid]

    n_plot = min(int(PLOT_SECONDS * fs), len(raw))
    if n_plot < fs * 5:  # less than 5s of usable signal
        print(f"  {name}: too short after NaN removal ({n_plot} samples) -- SKIP")
        return

    segment = raw[:n_plot]
    clip_frac = clipping_fraction(segment)

    filtered = bandpass(segment, fs)
    peaks, _ = detect_peaks(filtered, fs)

    implied_bpm = len(peaks) * 60 / (len(segment) / fs)
    flag = ""
    if clip_frac > 0.05:
        flag = f"  [REJECT candidate] {clip_frac*100:.1f}% samples clipped/saturated"
    elif implied_bpm < 40 or implied_bpm > 180:
        flag = f"  [REJECT candidate] implied HR {implied_bpm:.0f} bpm is non-physiological"

    t = np.arange(len(filtered)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(t, segment, linewidth=0.8)
    axes[0].set_title(f'{name} -- raw PLETH')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t, filtered, linewidth=0.8)
    axes[1].plot(t[peaks], filtered[peaks], 'rx', markersize=8, label='detected peaks')
    axes[1].set_title(f'{name} -- filtered + peaks (~{implied_bpm:.0f} bpm)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{name}_qc.png')
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"  {name}: fs={fs}Hz, {len(peaks)} peaks in {PLOT_SECONDS}s (~{implied_bpm:.0f} bpm) "
          f"-> {out_path}{flag}")


def main():
    npy_files = glob.glob(os.path.join(IN_DIR, '*.npy'))
    print(f"Found {len(npy_files)} records to QC.\n")

    for npy_path in sorted(npy_files):
        try:
            qc_record(npy_path)
        except Exception as e:
            print(f"  FAILED {npy_path}: {e}")

    print(f"\nDone. Review plots in ./{OUT_DIR}/ and note which records to keep or reject.")


if __name__ == '__main__':
    main()