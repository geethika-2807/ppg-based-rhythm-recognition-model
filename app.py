"""
Streamlit GUI: pick a random beat window from the dataset, show the PPG
waveform around it, and classify regular vs irregular.

Run locally:
    pip install streamlit
    streamlit run app.py
"""

import json
import random
import numpy as np
import streamlit as st
from tensorflow import keras

IBI_DIR = 'ibi_data'
RAW_DIR = 'mimic_pleth_records'
WINDOW_BEATS = 10
MODEL_PATH = 'ppg_regularity_cnn.keras'
NORM_PATH = 'normalization_stats.npz'


@st.cache_resource
def load_model_and_stats():
    model = keras.models.load_model(MODEL_PATH)
    norm = np.load(NORM_PATH)
    return model, float(norm['mean']), float(norm['std'])


@st.cache_data
def load_keep_list():
    with open('keep_list.json') as f:
        return json.load(f)


def pick_random_window(keep_names):
    """Keep retrying until we land on a fully-valid contiguous window."""
    for _ in range(200):
        name = random.choice(keep_names)
        ibi_data = np.load(f'{IBI_DIR}/{name}_ibi.npz')
        ibi = ibi_data['ibi']
        valid = ibi_data['ibi_valid_mask']
        peak_times = ibi_data['peak_times']
        fs = int(ibi_data['fs'])

        if len(ibi) < WINDOW_BEATS:
            continue
        start = random.randint(0, len(ibi) - WINDOW_BEATS)
        if not valid[start:start + WINDOW_BEATS].all():
            continue

        window_ibi = ibi[start:start + WINDOW_BEATS]
        window_peak_times = peak_times[start:start + WINDOW_BEATS + 1]
        return name, fs, window_ibi, window_peak_times
    return None


def classify(model, mean, std, window_ibi):
    x = (window_ibi - mean) / std
    x = x.reshape(1, -1, 1).astype(np.float32)
    prob_irregular = float(model.predict(x, verbose=0)[0, 0])
    label = 'Irregular' if prob_irregular > 0.5 else 'Regular'
    confidence = prob_irregular if label == 'Irregular' else 1 - prob_irregular
    return label, confidence, prob_irregular


def main():
    st.set_page_config(page_title='PPG Rhythm Classifier', layout='centered')
    st.title('PPG Rhythm Regularity Classifier')
    st.caption('Pulls a random beat window from held-out MIMIC-III PPG records '
               'and classifies it as regular or irregular using a 1D CNN.')

    model, mean, std = load_model_and_stats()
    keep_names = load_keep_list()

    if 'current' not in st.session_state:
        st.session_state.current = pick_random_window(keep_names)

    if st.button('🎲 Pick another random beat', use_container_width=True):
        st.session_state.current = pick_random_window(keep_names)

    result = st.session_state.current
    if result is None:
        st.error('Could not find a valid window. Try again.')
        return

    name, fs, window_ibi, window_peak_times = result
    label, confidence, prob_irregular = classify(model, mean, std, window_ibi)

    # Load raw signal segment to plot
    raw = np.load(f'{RAW_DIR}/{name}.npy')
    t_start = window_peak_times[0]
    t_end = window_peak_times[-1]
    pad = 1.0  # seconds of context on each side
    s0 = max(0, int((t_start - pad) * fs))
    s1 = min(len(raw), int((t_end + pad) * fs))
    segment = raw[s0:s1]
    t_axis = (np.arange(s0, s1) / fs)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f'Record: {name}')
        st.line_chart(
            {'PPG': segment},
            x=None,
            use_container_width=True,
        )
        st.caption(f'Window spans {t_end - t_start:.2f}s, {WINDOW_BEATS} beats, '
                   f'record sampled at {fs}Hz')

    with col2:
        if label == 'Irregular':
            st.error(f'### {label}')
        else:
            st.success(f'### {label}')
        st.metric('Confidence', f'{confidence*100:.1f}%')
        st.metric('P(irregular)', f'{prob_irregular:.3f}')
        st.write('**IBI window (s):**')
        st.write(np.round(window_ibi, 3))
        cv = np.std(window_ibi) / np.mean(window_ibi)
        st.write(f'**CV:** {cv:.4f}')
        mean_bpm = 60 / np.mean(window_ibi)
        st.write(f'**Mean rate:** {mean_bpm:.0f} bpm')


if __name__ == '__main__':
    main()