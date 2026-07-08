#!/bin/bash
# setup.sh — Prepares the project for first use or deployment
# This script is used by Streamlit Cloud during deployment.

echo "=== PPG Rhythm Recognition Model — Setup ==="

# Create required directories if they don't exist
mkdir -p ibi_data mimic_pleth_records

echo "✓ Directories ready"

# All demo data is committed to git (ibi_data/ + mimic_pleth_records/)
# No download needed — the app is ready to run.

echo "✓ Setup complete — the app is ready to launch!"
echo ""
echo "To run locally:  streamlit run app.py"