#!/bin/bash
# ============================================================
# 00_setup.sh — Install all dependencies for the NLP project
# Run once from Terminal: bash ~/Desktop/MSIS/TEXT_MINING_NLP/Project/00_setup.sh
# ============================================================

echo ">>> Creating virtual environment..."
cd ~/Desktop/MSIS/TEXT_MINING_NLP/Project/code
python3 -m venv venv
source venv/bin/activate

echo ">>> Installing packages..."
pip install --upgrade pip
pip install \
    pandas \
    numpy \
    pyarrow \
    fastparquet \
    vaderSentiment \
    spacy \
    scikit-learn \
    factor_analyzer \
    statsmodels \
    matplotlib \
    seaborn \
    tqdm \
    bertopic \
    sentence-transformers \
    umap-learn \
    hdbscan \
    keybert \
    openpyxl \
    jupyterlab \
    langdetect

echo ">>> Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm

echo ""
echo "✅ Setup complete!"
echo "To activate the environment later, run:"
echo "   source ~/Desktop/MSIS/TEXT_MINING_NLP/Project/venv/bin/activate"
