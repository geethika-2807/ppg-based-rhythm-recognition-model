# PPG-Based Cardiac Rhythm Regularity Classification

A machine learning pipeline that classifies cardiac rhythm as **regular** or **irregular** using photoplethysmography (PPG) waveform data from the **MIMIC-III Waveform Database Matched Subset**. The system downloads PPG signals, performs quality control, extracts inter-beat intervals (IBIs), builds a labeled dataset using the coefficient of variation (CV) of IBI as a regularity proxy, trains a 1D convolutional neural network (CNN), and serves predictions through an interactive Streamlit web application.

**🌐 Live Demo:** [https://ppg-based-rhythm-recognition-model.streamlit.app](https://ppg-based-rhythm-recognition-model.streamlit.app)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Full Pipeline (End-to-End)](#full-pipeline-end-to-end)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Dataset & Labeling Strategy](#dataset--labeling-strategy)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Notes](#notes)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Quick Start

Get the app running in 3 simple steps using the pre-processed demo data.

### Prerequisites

- **Python 3.9 – 3.12** (TensorFlow 2.15 is required and doesn't support Python 3.13+)
- pip package manager
- Git (for cloning)

### 1. Clone the Repository

```bash
git clone https://github.com/geethika-2807/ppg-based-rhythm-recognition-model
cd ppg-based-rhythm-recognition-model
```

### 2. Install Dependencies

**Option A: Using default Python (if version is 3.9–3.12)**
```bash
pip install -r requirements.txt
```

**Option B: Using Python 3.12 explicitly (recommended if you have Python 3.13 or 3.14)**

**On Windows:**
```bash
C:\Users\<YOUR_USERNAME>\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

**On macOS/Linux:**
```bash
python3.12 -m pip install -r requirements.txt
```

**Option C: Create a virtual environment (recommended)**
```bash
# Create virtual environment with Python 3.11 or 3.12
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the App

```bash
# If using default Python
streamlit run app.py

# If using Python 3.12 explicitly
C:\Users\<YOUR_USERNAME>\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app.py

# If using virtual environment
streamlit run app.py
```

Open **http://localhost:8501** in your browser. Click **"🎲 Pick another random beat"** to classify different 10-beat windows.

---

## Installation

### Detailed Dependency Information

The `requirements.txt` file includes:

```
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
wfdb>=4.1
scikit-learn>=1.3
tensorflow-cpu>=2.15
streamlit>=1.30
```

**Package Descriptions:**
- **numpy** — Array operations and signal processing
- **scipy** — Signal filtering and statistical functions
- **matplotlib** — Visualization (QC plots)
- **wfdb** — PhysioNet database access for MIMIC-III data
- **scikit-learn** — Data splitting and evaluation metrics
- **tensorflow-cpu** — Deep learning framework (CPU-only version, lighter weight)
- **streamlit** — Web application framework for the GUI

### Troubleshooting Installation Issues

**Issue: "Python 3.13 or 3.14 detected"**
```bash
# Solution: Install Python 3.11 or 3.12 from python.org
# Then use the explicit path as shown in Option B above
```

**Issue: "pip not found"**
```bash
# On Windows, try:
python -m pip install --upgrade pip

# On macOS/Linux, try:
python3 -m pip install --upgrade pip
```

**Issue: "Permission denied"**
```bash
# Add --user flag or use virtual environment:
pip install --user -r requirements.txt
```

**Issue: "TensorFlow installation fails"**
```bash
# Ensure you're using Python 3.9-3.12
# Try installing tensorflow-cpu specifically:
pip install tensorflow-cpu==2.15.0
```

---

## Running the Application

### Local Development

```bash
# Activate virtual environment first (if using one)
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Run the app
streamlit run app.py
```

The app will:
1. Load the trained model (`ppg_regularity_cnn.keras`)
2. Load normalization statistics (`normalization_stats.npz`)
3. Load demo data from `ibi_data/` and `mimic_pleth_records/`
4. Display a random 10-beat window with classification results

### Using the GUI

1. **View Classification Results:**
   - The app automatically picks a random beat window
   - Shows Regular/Irregular classification with confidence
   - Displays probability of irregularity

2. **Explore More Samples:**
   - Click **"🎲 Pick another random beat"** to see different windows
   - Each click loads a new random sample from the demo data

3. **Understanding the Output:**
   - **PPG Waveform:** Raw photoplethysmography signal segment
   - **Classification:** Regular or Irregular rhythm
   - **Confidence:** Model's confidence in the prediction
   - **P(irregular):** Probability of irregular rhythm (0-1)
   - **IBI Window:** 10 consecutive inter-beat intervals in seconds
   - **CV:** Coefficient of variation (measure of rhythm regularity)
   - **Mean Rate:** Average heart rate in BPM

---

## Full Pipeline (End-to-End)

Run these steps **in order** to reproduce the entire pipeline from raw MIMIC-III data to trained model.

### Step 1: Download PPG (PLETH) Waveforms from MIMIC-III

```bash
python fetch_mimic_pleth.py
```

**What it does:**
- Searches up to 3,000 master records across the MIMIC-III Waveform Database Matched Subset
- Saves up to 40 diverse PLETH segments (max 2 per patient) to `mimic_pleth_records/`
- Each record is saved as `<name>.npy` (signal array) and `<name>.json` (metadata: sample rate, duration)
- **No credentialing required** — the MIMIC-III Waveform Database Matched Subset is openly accessible with proper citation

**Output:** `mimic_pleth_records/*.npy` and `mimic_pleth_records/*.json`

---

### Step 2: Visual Quality Control

```bash
python qc_peaks.py
```

**What it does:**
- Applies a 0.5–10 Hz bandpass filter to the first 30 seconds of each record
- Runs peak detection and generates a 2-panel plot (raw signal + filtered signal with detected peaks)
- Flags records with >5% clipping or non-physiological heart rate (<40 or >180 bpm)
- Outputs: `qc_plots/<name>_qc.png` — review these to decide which records pass QC

**Output:** `qc_plots/*.png` (visual QC plots for manual review)

**Review:** Check the generated plots and manually decide which records have good signal quality.

---

### Step 3: Automated QC Summary

```bash
python qc_summary.py
```

**What it does:**
- Repeats the QC checks programmatically (no plots)
- Prints a KEEP / REJECT tally
- Writes accepted record names to `keep_list.json`, consumed by all downstream steps

**Output:** `keep_list.json` (list of QC-passed record names)

---

### Step 4: Compute Inter-Beat Intervals (IBI)

```bash
python compute_ibi.py
```

**What it does:**
- Processes the **entire** signal in 1-hour chunks to manage memory efficiently
- Applies bandpass filtering → peak detection → computes IBI = `diff(peak_timestamps)`
- Flags physiologically impossible intervals (<0.33s or >2.0s, i.e., 30–180 bpm) via `ibi_valid_mask`
- Outputs per record to `ibi_data/<name>_ibi.npz` containing:
  - `peak_times` — beat timestamps in seconds
  - `ibi` — inter-beat intervals in seconds
  - `ibi_valid_mask` — boolean array marking physiologically valid intervals
  - `fs`, `n_samples`, `duration_sec`

**Output:** `ibi_data/*_ibi.npz` files

---

### Step 5: Build Labeled Dataset

```bash
python build_dataset.py
```

**What it does:**
- Slides a window of 10 consecutive, fully-valid IBIs across each record (step = 5, 50% overlap)
- Computes **CV = σ(IBI) / μ(IBI)** for each window
- Assigns labels:
  - **CV ≤ 0.08** → Regular (0)
  - **CV > 0.08** → Irregular (1)
- Prints the CV distribution percentile table so you can verify/adjust the threshold
- Output: `dataset/ibi_windows_dataset.npz` (X, y, record_ids)

> **Why CV?** MIMIC waveform records lack per-beat rhythm labels. The coefficient of variation of IBI is a standard, well-established proxy for rhythm irregularity widely used in atrial fibrillation screening literature.

**Output:** `dataset/ibi_windows_dataset.npz`

**Adjusting the Threshold:**
After running, check the printed CV distribution percentile table. If you want to adjust the regular/irregular threshold, edit the `CV_THRESHOLD` variable in `build_dataset.py` (default is 0.08).

---

### Step 6: Train the 1D CNN

```bash
python train_cnn.py
```

**What it does:**
- **Patient-aware split:** 70% train / 15% validation / 15% test using `GroupShuffleSplit` — prevents data leakage by ensuring windows from the same patient stay in the same split
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

**Training Output:**
- Progress bars for each epoch
- Validation accuracy and AUC metrics
- Best model saved automatically
- Training history plots (if enabled)

**Output:** `ppg_regularity_cnn.keras` and `normalization_stats.npz`

---

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

## Deployment

Deploy your app to the web so others can use it without installing anything.

### Option 1: Streamlit Community Cloud (Recommended) ✅

**Status:** Already deployed at [https://ppg-based-rhythm-recognition-model.streamlit.app](https://ppg-based-rhythm-recognition-model.streamlit.app)

**To deploy a new version:**

1. **Push changes to GitHub:**
   ```bash
   git add .
   git commit -m "Update app"
   git push origin main
   ```

2. **Streamlit Cloud will auto-deploy:**
   - Any push to the `main` branch triggers automatic redeployment
   - Takes ~2-3 minutes to build and deploy
   - Check deployment status at [https://streamlit.io/cloud](https://streamlit.io/cloud)

3. **Configuration used:**
   - **Repository:** `geethika-2807/ppg-based-rhythm-recognition-model`
   - **Branch:** `main`
   - **Main file:** `app.py`
   - **Python version:** 3.11
   - **URL:** https://ppg-based-rhythm-recognition-model.streamlit.app

**How it works:**
- The trained model and demo data are committed directly to the repository
- Streamlit Cloud installs dependencies from `requirements.txt`
- Runs `app.py` and keeps the app live while users are active
- Spins down after 15 minutes of inactivity (saves resources)
- Wakes up on next visit (cold start ~30 seconds)

---

### Option 2: Hugging Face Spaces

**Alternative deployment option using Hugging Face Spaces:**

1. **Create a new Space:**
   - Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Name: `ppg-rhythm`
   - Select "Streamlit" as SDK
   - Set Python version to 3.10

2. **Push to Hugging Face:**
   ```bash
   # Add Hugging Face remote (if not already added)
   git remote add hf https://huggingface.co/spaces/geethika-2807/ppg-rhythm
   
   # Push to Hugging Face
   git add .
   git commit -m "Deploy to HF Spaces"
   git push hf main
   ```

3. **Your app will be available at:**
   ```
   https://geethika-2807-ppg-rhythm.hf.space
   ```

---

### Option 3: Self-Hosted Deployment

**Run on your own server or VPS:**

1. **Install dependencies on server:**
   ```bash
   git clone https://github.com/geethika-2807/ppg-based-rhythm-recognition-model
   cd ppg-based-rhythm-recognition-model
   pip install -r requirements.txt
   ```

2. **Run with Streamlit:**
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

3. **Use a reverse proxy (nginx) for production:**
   ```bash
   # Install nginx
   sudo apt install nginx
   
   # Configure nginx to proxy to Streamlit
   # See Streamlit docs for production deployment
   ```

4. **Or use Docker:**
   ```bash
   # Build image
   docker build -t ppg-rhythm-classifier .
   
   # Run container
   docker run -p 8501:8501 ppg-rhythm-classifier
   ```

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

## Dataset & Labeling Strategy

**Source:** [MIMIC-III Waveform Database Matched Subset](https://physionet.org/content/mimic3wdb-matched/1.0/) (PhysioNet)

**Access:** This dataset is openly accessible with proper citation. It should not be confused with the MIMIC-III Clinical Database, which requires PhysioNet credentialing. The waveform-only Matched Subset used in this project does not require login or credentialing.

**Labeling (unsupervised / rule-based):**

| CV of IBI | Classification |
|-----------|---------------|
| ≤ 0.08    | Regular rhythm |
| > 0.08    | Irregular rhythm |

The CV threshold (default 0.08) can be adjusted in `build_dataset.py` after inspecting the printed CV distribution.

---

## Model Performance

The 1D CNN is trained with a patient-aware train/validation/test split to prevent data leakage. Performance metrics (accuracy, AUC) are printed during training. The model uses:

- **Input:** 10 consecutive z-score-normalized IBI values
- **Architecture:** Lightweight 1D CNN with batch normalization and dropout for regularization
- **Training:** Early stopping on validation AUC with best weight restoration
- **Output:** Binary classification (Regular vs Irregular) with probability score

---

## Technical Details

### Dependencies

- **numpy** ≥ 1.24 — Array operations and signal processing
- **scipy** ≥ 1.11 — Signal filtering and statistical functions
- **matplotlib** ≥ 3.7 — Visualization (QC plots)
- **wfdb** ≥ 4.1 — PhysioNet database access
- **scikit-learn** ≥ 1.3 — Data splitting and metrics
- **tensorflow-cpu** ≥ 2.15 — Deep learning framework
- **streamlit** ≥ 1.30 — Web application framework

### Python Version Compatibility

**Required:** Python 3.9 – 3.12

TensorFlow 2.15 supports Python 3.9–3.12. If you encounter installation errors on Python 3.13 or 3.14, use Python 3.11 or 3.12 explicitly as shown in the Installation section.

**Recommended:** Python 3.11 or 3.12 for best compatibility

---

## Notes

- The demo data in `ibi_data/` and `mimic_pleth_records/` lets you run the GUI and Steps 4–5 without downloading from MIMIC-III. To run the full pipeline from scratch, delete these directories and start with `fetch_mimic_pleth.py`.
- The `app.py` GUI reads from `ibi_data/` (IBI `.npz` files) and `mimic_pleth_records/` (raw PPG `.npy` files). Both directories must exist and contain matching files.
- The CV threshold of 0.08 was chosen based on the distribution of IBI variability in the dataset. Adjust if needed for your specific use case.
- The app is deployed on Streamlit Community Cloud and is freely accessible at https://ppg-based-rhythm-recognition-model.streamlit.app
- No installation required for end users — just open the URL in a browser

---

## License

This project uses the **MIMIC-III Waveform Database Matched Subset** (PhysioNet), which is openly accessible with proper citation. It should not be confused with the MIMIC-III Clinical Database, which requires PhysioNet credentialing.

- **Dataset:** [MIMIC-III Waveform Database Matched Subset](https://physionet.org/content/mimic3wdb-matched/1.0/) — Open Access with citation
- **Code:** Provided for **research and educational purposes only**. Not a medical device; not clinically validated.

---

## Citation

If you use this project in your research, please cite:

- Johnson, A.E.W., et al. *MIMIC-III, a freely accessible critical care database.* Scientific Data (2016).
- Goldberger, A.L., et al. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.* Circulation (2000).

---

## Contact

For questions or issues, please open an issue on GitHub or contact [geethika-2807](https://github.com/geethika-2807).

**Live App:** [https://ppg-based-rhythm-recognition-model.streamlit.app](https://ppg-based-rhythm-recognition-model.streamlit.app)