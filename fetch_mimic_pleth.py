"""
Step 1: Find MIMIC-III matched-subset records that contain a PLETH (PPG) channel,
then download a diverse batch (default 20, capped per patient) for peak-detection QC.

Saves each record as a plain .npy (signal) + .json (metadata) pair -- avoids
wfdb.wrsamp(), which chokes on merged multi-segment records missing gain/baseline
metadata.

Run locally (needs internet access to physionet.org):
    pip install wfdb numpy
    python fetch_mimic_pleth.py
"""

import wfdb
import numpy as np
import os
import json

DB = 'mimic3wdb-matched/1.0'
N_TARGET = 40             # how many good records to collect
N_SCAN = 3000              # how many candidate master records to check headers for
MAX_PER_PATIENT = 2       # cap so one patient can't fill the whole batch
OUT_DIR = 'mimic_pleth_records'

os.makedirs(OUT_DIR, exist_ok=True)


def get_patient_dirs():
    return wfdb.io.get_record_list(DB, records='all')


def get_master_records(patient_dir):
    try:
        return wfdb.io.get_record_list(f'{DB}/{patient_dir}')
    except Exception:
        return []


def find_pleth_segment(patient_dir, master_name):
    """Drill into a (possibly multi-segment) master record to find a leaf
    segment containing a PLETH channel. Returns leaf segment name or None."""
    pn_dir = f'{DB}/{patient_dir}'
    try:
        master = wfdb.rdheader(master_name, pn_dir=pn_dir)
    except Exception:
        return None

    if not hasattr(master, 'seg_name'):
        if master.sig_name and 'PLETH' in master.sig_name and master.sig_len >= 1250:
            return master_name
        return None

    for seg_name, seg_len in zip(master.seg_name, master.seg_len):
        if seg_name == '~' or seg_len < 1250:  # skip placeholders / <10s segments
            continue
        try:
            seg_hdr = wfdb.rdheader(seg_name, pn_dir=pn_dir)
        except Exception:
            continue
        if seg_hdr.sig_name and 'PLETH' in seg_hdr.sig_name:
            return seg_name
    return None


def main():
    print("Fetching patient directory list (this can take a minute)...")
    patients = get_patient_dirs()
    print(f"Total patient directories available: {len(patients)}")

    found = []          # (patient_dir, leaf_segment_name)
    seen = set()
    per_patient_count = {}
    scanned = 0

    for patient_dir in patients:
        if scanned >= N_SCAN or len(found) >= N_TARGET:
            break
        if per_patient_count.get(patient_dir, 0) >= MAX_PER_PATIENT:
            continue

        for master_name in get_master_records(patient_dir):
            if scanned >= N_SCAN or len(found) >= N_TARGET:
                break
            if per_patient_count.get(patient_dir, 0) >= MAX_PER_PATIENT:
                break
            scanned += 1
            seg = find_pleth_segment(patient_dir, master_name)
            key = (patient_dir, seg)
            if seg and key not in seen:
                seen.add(key)
                found.append((patient_dir, seg))
                per_patient_count[patient_dir] = per_patient_count.get(patient_dir, 0) + 1
                print(f"  [{len(found)}/{N_TARGET}] PLETH found: {patient_dir}{seg}")

    print(f"\nScanned {scanned} master records across {len(per_patient_count)} patients, "
          f"found {len(found)} usable PLETH segments.")

    print("\nDownloading signals for verified segments...")
    saved = 0
    for patient_dir, seg_name in found:
        pn_dir = f'{DB}/{patient_dir}'
        try:
            record = wfdb.rdrecord(seg_name, pn_dir=pn_dir)
            if 'PLETH' not in record.sig_name:
                print(f"  SKIP {patient_dir}{seg_name}: PLETH missing after full read")
                continue

            ch = record.sig_name.index('PLETH')
            sig = record.p_signal[:, ch].astype(np.float32)

            safe_name = seg_name.replace('/', '_')
            np.save(os.path.join(OUT_DIR, f'{safe_name}.npy'), sig)
            with open(os.path.join(OUT_DIR, f'{safe_name}.json'), 'w') as f:
                json.dump({'fs': record.fs, 'patient_dir': patient_dir,
                           'record_name': seg_name, 'n_samples': len(sig)}, f)

            print(f"  Saved: {safe_name}.npy  (fs={record.fs}Hz, {len(sig)} samples)")
            saved += 1
        except Exception as e:
            print(f"  FAILED {patient_dir}{seg_name}: {e}")

    print(f"\nDone. {saved}/{len(found)} records saved to ./{OUT_DIR}/")


if __name__ == '__main__':
    main()