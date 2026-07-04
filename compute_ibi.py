"""
Step 3: For every record in keep_list.json, run peak detection across the
FULL signal, compute inter-beat intervals (IBI), and save per-record IBI
series + timestamps. Also does a rolling QC pass across the whole record
(not just first 30s) since long recordings can have clean starts but bad
stretches later.

Output per record: ibi_data/<name>_ibi.npz containing:
    peak_times   -- seconds, timestamp of each detected beat
    ibi          -- seconds, interval to the NEXT beat (len = len(peak_times)-1)
    fs, n_samples, duration_sec

Run locally:
    python compute_ibi.py
"""

import os
import json
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

IN_DIR = 'mimic_pleth_records'
OUT_DIR = 'ibi_data'
LOWCUT, HIGHCUT = 0.5, 10
FILTER_ORDER = 3
CHUNK_SECONDS = 3600  # process in 1-hour chunks to keep filtfilt memory sane on long records

os.makedirs(OUT_DIR, exist_ok=True)


def bandpass(sig, fs, low=LOWCUT, high=HIGHCUT, order=FILTER_ORDER):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def detect_peaks(sig, fs, max_hr=150):
    min_distance = int(fs * 60 / max_hr)
    prominence = np.std(sig) * 0.5
    peaks, _ = find_peaks(sig, distance=min_distance, prominence=prominence)
    return peaks


def process_record(name):
    raw = np.load(os.path.join(IN_DIR, f'{name}.npy'))
    with open(os.path.join(IN_DIR, f'{name}.json')) as f:
        meta = json.load(f)
    fs = meta['fs']

    chunk_len = int(CHUNK_SECONDS * fs)
    all_peak_times = []

    for start in range(0, len(raw), chunk_len):
        end = min(start + chunk_len, len(raw))
        chunk = raw[start:end]

        if np.isnan(chunk).all():
            continue
        # Interpolate small NaN gaps within the chunk; skip chunk if too much missing
        nan_frac = np.isnan(chunk).mean()
        if nan_frac > 0.3:
            continue
        if nan_frac > 0:
            idx = np.arange(len(chunk))
            good = ~np.isnan(chunk)
            chunk = np.interp(idx, idx[good], chunk[good])

        if len(chunk) < fs * 2:
            continue

        filtered = bandpass(chunk, fs)
        peaks = detect_peaks(filtered, fs)
        peak_times = (peaks + start) / fs
        all_peak_times.append(peak_times)

    if not all_peak_times:
        return None

    peak_times = np.concatenate(all_peak_times)
    peak_times.sort()

    if len(peak_times) < 2:
        return None

    ibi = np.diff(peak_times)

    # Drop physiologically impossible IBIs (chunk-boundary artifacts, missed beats)
    valid = (ibi > 0.33) & (ibi < 2.0)  # 30-180 bpm equivalent
    # Keep peak_times aligned: peak_times has len(ibi)+1, so mask applies to ibi only
    # but we should also drop the corresponding peak if its IBI is invalid on both sides
    # -- simplest robust approach: just keep valid IBIs, accept some peak_times become orphaned

    return {
        'peak_times': peak_times,
        'ibi': ibi,
        'ibi_valid_mask': valid,
        'fs': fs,
        'n_samples': len(raw),
        'duration_sec': len(raw) / fs,
    }


def main():
    with open('keep_list.json') as f:
        keep_names = json.load(f)

    print(f"Processing {len(keep_names)} records...\n")

    summary = []
    for name in keep_names:
        result = process_record(name)
        if result is None:
            print(f"  {name}: no usable beats found -- SKIP")
            continue

        out_path = os.path.join(OUT_DIR, f'{name}_ibi.npz')
        np.savez(out_path, **result)

        n_beats = len(result['peak_times'])
        pct_valid = result['ibi_valid_mask'].mean() * 100
        mean_bpm = 60 / np.median(result['ibi'][result['ibi_valid_mask']]) if result['ibi_valid_mask'].any() else float('nan')
        print(f"  {name}: {n_beats} beats, {result['duration_sec']/3600:.2f}h, "
              f"{pct_valid:.0f}% IBIs valid, median ~{mean_bpm:.0f} bpm -> {out_path}")
        summary.append(name)

    print(f"\nDone. {len(summary)}/{len(keep_names)} records processed successfully.")
    print(f"IBI data saved in ./{OUT_DIR}/")


if __name__ == '__main__':
    main()
    