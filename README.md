# PPG-Based Cardiac Rhythm Regularity Classification

A machine learning pipeline that classifies cardiac rhythm as **regular** or **irregular** using photoplethysmography (PPG) waveform data from the **MIMIC-III** clinical database. The system downloads PPG signals, performs quality control, extracts inter-beat intervals (IBIs), builds a labeled dataset using the coefficient of variation (CV) of IBI as a regularity proxy, trains a 1D convolutional neural network (CNN), and serves predictions through an interactive Streamlit web application.

---

## Pipeline Overview

The project is organized as a sequential 7-step pipeline. Each step has its own script and can be run independently once its prerequisites are met.

```
┌─────────────────────────────────────────────────────────┐
│  1. fetch_mimic_pleth.py    Download PPG signals from   │
│                            MIMIC-III database           │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  2. qc_peaks.py           Visual quality control with   │
│                            peak detection plots         │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  3. qc_summary.py         Automated QC summary —        │
│                            generates keep_list.json     │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  4. compute_ibi.py        Full-record peak detection →  │
│                            inter-beat intervals         │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  5. build_dataset.py      Sliding-window dataset with   │
│                            CV-based labels              │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  6. train_cnn.py          Patient-aware split → train   │
│                            1D CNN classifier            │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  7. app.py               Streamlit GUI for interactive  │
│                           rhythm classification         │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start (Run the GUI with Demo Data)

The repository includes pre-processed demo data so you can launch the classifier immediately — no downloads required.

### 1. Clone the Repository

```bash
git clone https://github.com/geethika-2807/ppg-based-rhythm-recognition-model
cd ppg-based-rhythm-recognition-model
```

### 2. Install Dependencies

**⚠️ Important:** TensorFlow requires **Python 3.9 – 3.12**. If your default `python` is 3.13 or 3.14, use the full path to Python 3.12.

```bash
# Option A: Using default Python (if 3.9–3.12)
pip install -r requirements.txt

# Option B: Using Python 3.12 explicitly (recommended on Windows with Python 3.14)
C:\Users\<YOUR_USERNAME>\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

### 3. Launch the App

```bash
# If using Option A above
streamlit run app.py

# If using Option B above
C:\Users\<YOUR_USERNAME>\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app.py
```

Open **http://localhost:8501** in your browser. Click **"🎲 Pick another random beat"** to classify different 10-beat windows.

---

## Full Pipeline (End-to-End)

Run the steps below **in order** to reproduce the entire pipeline from raw data to trained model.

### Step 1: Download PPG (PLETH) Waveforms from MIMIC-III

```bash
python fetch_mimic_pleth.py
```

- Searches up to 3,000 master records across the MIMIC-III matched subset.
- Saves up to 40 diverse PLETH segments (max 2 per patient) to `mimic_pleth_records/`.
- Each record is saved as `<name>.npy` (signal array) and `<name>.json` (metadata: sample rate, duration).

### Step 2: Visual Quality Control

```bash
python qc_peaks.py
```

- Applies a 0.5–10 Hz bandpass filter to the first 30 seconds of each record.
- Runs peak detection and generates a 2-panel plot (raw signal + filtered signal with detected peaks).
- Flags records with >5% clipping or non-physiological heart rate (<40 or >180 bpm).
- Outputs: `qc_plots/<name>_qc.png` — review these to decide which records pass QC.

### Step 3: Automated QC Summary

```bash
python qc_summary.py
```

- Repeats the QC checks programmatically (no plots).
- Prints a KEEP / REJECT tally.
- Writes accepted record names to `keep_list.json`, consumed by all downstream steps.

### Step 4: Compute Inter-Beat Intervals (IBI)

```bash
python compute_ibi.py
```

- Processes the **entire** signal in 1-hour chunks to manage memory efficiently.
- Applies bandpass filtering → peak detection → computes IBI = `diff(peak_timestamps)`.
- Flags physiologically impossible intervals (<0.33s or >2.0s, i.e., 30–180 bpm) via `ibi_valid_mask`.
- Outputs per record to `ibi_data/<name>_ibi.npz` containing:
  - `peak_times` — beat timestamps in seconds
  - `ibi` — inter-beat intervals in seconds
  - `ibi_valid_mask` — boolean array marking physiologically valid intervals
  - `fs`, `n_samples`, `duration_sec`

### Step 5: Build Labeled Dataset

```bash
python build_dataset.py
```

- Slides a window of 10 consecutive, fully-valid IBIs across each record (step = 5, 50% overlap).
- Computes **CV = σ(IBI) / μ(IBI)** for each window.
- Assigns labels:
  - **CV ≤ 0.08** → Regular (0)
  - **CV > 0.08** → Irregular (1)
- Prints the CV distribution percentile table so you can verify/adjust the threshold.
- Output: `dataset/ibi_windows_dataset.npz` (X, y, record_ids).

> **Why CV?** MIMIC waveform records lack per-beat rhythm labels. The coefficient of variation of IBI is a standard, well-established proxy for rhythm irregularity widely used in atrial fibrillation screening literature.

### Step 6: Train the 1D CNN

```bash
python train_cnn.py
```

- **Patient-aware split:** 70% train / 15% validation / 15% test using `GroupShuffleSplit` — prevents data leakage by ensuring windows from the same patient stay in the same split.
- **Model architecture:**

```
Input: (batch, 10, 1) — 10 consecutive z-score-normalized IBI values
│
├─ Conv1D(16 filters, kernel=3, ReLU, padding='same')
├─ BatchNormalization
├─ Conv1D(32 filters, kernel=3, ReLU, padding='same')
├─ BatchNormalization
├─ GlobalAveragePooling1D
├─ Dense(16, ReLU)
├─ Dropout(0.3)
└─ Dense(1, sigmoid) → P(irregular)
```

- **Loss:** Binary crossentropy
- **Optimizer:** Adam (1e-3, with ReduceLROnPlateau)
- **Callbacks:** Early stopping (patience=10 on validation AUC, restore best weights)
- **Output:**
  - `ppg_regularity_cnn.keras` — trained model
  - `normalization_stats.npz` — z-score mean and std (required by the GUI)

### Step 7: Launch the Interactive Streamlit GUI

```bash
streamlit run app.py
```

The GUI loads the trained model, randomly samples a 10-beat window from the held-out records, and displays:

- Raw PPG waveform segment
- Classification result (Regular / Irregular) with confidence score
- Probability of irregularity
- IBI values for the window
- Coefficient of variation (CV)
- Mean heart rate in BPM

---

## Project Structure

```
ppg-based-rhythm-recognition-model/
│
├── app.py                     # Streamlit GUI for interactive classification
├── fetch_mimic_pleth.py       # Download PPG signals from MIMIC-III
├── qc_peaks.py                # Visual peak detection quality control
├── qc_summary.py              # Automated QC summary → keep_list.json
├── compute_ibi.py             # Full-record IBI computation
├── build_dataset.py           # Sliding-window dataset generation
├── train_cnn.py               # Model training pipeline
│
├── requirements.txt           # Python package dependencies
├── setup.sh                   # Deployment setup script
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── keep_list.json             # QC-passed record names
├── normalization_stats.npz    # Z-score stats (generated by Step 6)
├── ppg_regularity_cnn.keras   # Trained model (generated by Step 6)
├── Presentation.pdf           # Project report
│
├── ibi_data/                  # IBI .npz files (demo data committed)
├── mimic_pleth_records/       # Raw PPG .npy files (demo data committed)
│
├── dataset/                   # Generated by Step 5 (git-ignored)
├── qc_plots/                  # Generated by Step 2 (git-ignored)
└── .venv/                     # Virtual environment (git-ignored)
```

---

## Deployment — Streamlit Community Cloud (Free)

Share the app with friends via a public URL — no installation required on their end.

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[https://streamlit.io/cloud](https://streamlit.io/cloud)**
2. Click **"Sign in with GitHub"** and authorize the application.
3. Click **"New app"** and configure:

| Field        | Value                                                |
|--------------|------------------------------------------------------|
| Repository   | `geethika-2807/ppg-based-rhythm-recognition-model`   |
| Branch       | `main`                                               |
| Main file    | `app.py`                                             |

4. Click **"Deploy"**.

### Step 3: Share the URL

After deployment (~2 minutes), you'll receive a public URL:

```
https://ppg-based-rhythm-recognition-model.streamlit.app
```

Send this link to anyone — they can open it on any device with a browser and start classifying immediately.

> **How it works:** The trained model and demo data are committed directly to the repository. Streamlit Cloud installs the dependencies from `requirements.txt`, runs `app.py`, and keeps the app live while users are active. It spins down after 15 minutes of inactivity and wakes up on the next visit (cold start ~30 seconds).

---

## Dataset & Labeling Strategy

**Source:** [MIMIC-III Waveform Database Matched Subset](https://physionet.org/content/mimic3wdb-matched/1.0/) (PhysioNet)

**Labeling (unsupervised / rule-based):**

| CV of IBI | Classification |
|-----------|---------------|
| ≤ 0.08    | Regular rhythm |
| > 0.08    | Irregular rhythm |

The CV threshold (default 0.08) can be adjusted in `build_dataset.py` after inspecting the printed CV distribution.

---

## Notes

- The demo data in `ibi_data/` and `mimic_pleth_records/` lets you run the GUI and Steps 4–5 without downloading from MIMIC-III. To run the full pipeline from scratch, delete these directories and start with `fetch_mimic_pleth.py`.
- TensorFlow does **not** support Python 3.13 or 3.14. If you encounter installation errors, use Python 3.12 explicitly (see [Quick Start](#quick-start-run-the-gui-with-demo-data)).
- The `app.py` GUI reads from `ibi_data/` (IBI `.npz` files) and `mimic_pleth_records/` (raw PPG `.npy` files). Both directories must exist and contain matching files.

---

## License

This project uses data from the **MIMIC-III Waveform Database Matched Subset**. Access to MIMIC data requires:

1. Completion of [PhysioNet credentialing](https://physionet.org/settings/credentialing/)
2. Acceptance of the MIMIC-III Waveform Database Data Use Agreement

This code is provided for **research and educational purposes only**. It is not a medical device and has not been clinically validated.

---

## Citation

If you use this project in your research, please cite:

- Johnson, A.E.W., et al. *MIMIC-III, a freely accessible critical care database.* Scientific Data (2016).
- Goldberger, A.L., et al. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.* Circulation (2000).