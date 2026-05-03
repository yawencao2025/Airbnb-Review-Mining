# Airbnb Review NLP Analysis Repository

This repository contains an end-to-end text mining workflow for Airbnb reviews across 15 US cities, including:

- exploratory data checks on raw CSV files,
- multi-city ingestion and cleaning into SQLite/Parquet,
- full-data NLP exploratory analysis,
- sampled NLP modeling pipeline for sentiment, topics, and rating prediction.

The core workflow is notebook-first, with one script version for ingestion (`01_ingest.py`) for reproducible reruns.

## Project Goals

The analysis focuses on:

- understanding review data quality and structure before modeling,
- quantifying sentiment (overall + aspect-based),
- extracting key topics with LDA,
- predicting listing quality tiers from review text/features,
- evaluating model transferability across cities (LOCO CV in sampled pipeline).

## Repository Structure

Main files and folders:

- `00_explore.ipynb`  
  Pre-analysis data inspection (file inventory, single-city checks, all-city summary, optional SQLite checks after ingest).
- `01_ingest.ipynb`  
  Full ingest + cleaning + EDA + SQLite and Parquet writes.
- `01_ingest.py`  
  Script version of Phase 1 ingest for command-line execution.
- `02_nlp_full_data_EDA.ipynb`  
  Full-data NLP/EDA pipeline (language detection, preprocessing, TF-IDF, VADER, aspect sentiment, LDA, rating prediction baseline/comparison).
- `02_nlp_sampled_pipeline.ipynb`  
  Sampled modeling pipeline with balanced per-city sampling, negation-safe preprocessing, LDA, multiple classifiers, and LOCO evaluation.
- `00_setup.sh`  
  One-time environment setup helper (virtualenv + pip installs + spaCy model).
- `csv/`  
  Input city files (`listings_*_clean.csv`, `reviews_*_clean.csv` expected by ingest).
- `parquet/`  
  Intermediate and final parquet datasets.
- `outputs/`  
  Model/topic/prediction CSVs and generated figures.
- `airbnb.db`  
  SQLite database produced by ingestion.

## Data Flow Overview

1. **Explore raw files** (`00_explore.ipynb`)  
   Verify inventory and inspect listing/review quality before large-scale processing.
2. **Ingest + clean** (`01_ingest.ipynb` or `01_ingest.py`)  
   Combine all cities, normalize key fields, deduplicate IDs, validate join coverage, write SQLite and Parquet.
3. **Run NLP analysis**:
   - Full-data analysis in `02_nlp_full_data_EDA.ipynb`, or
   - Sampled modeling pipeline in `02_nlp_sampled_pipeline.ipynb`.

## Outputs Produced

### From Ingest Phase

- `airbnb.db` with tables:
  - `listings`
  - `reviews`
- `parquet/listings_all.parquet`
- `parquet/reviews_raw.parquet`
- `parquet/reviews_with_features.parquet` (joined review + listing features)

### From Full NLP Notebook (`02_nlp_full_data_EDA.ipynb`)

- `parquet/reviews_sentiment.parquet`
- `outputs/lda_topics.csv`
- `outputs/predictions.csv`
- plus in-notebook figures and diagnostics

### From Sampled NLP Notebook (`02_nlp_sampled_pipeline.ipynb`)

- checkpoint parquet during pipeline steps
- final sentiment/topic/model parquet (current notebook cell writes `reviews_sentiment_lda_machinelearning4.parquet`)
- `outputs/lda_topics.csv`
- `outputs/lda_topics_neg_neu.csv`
- `outputs/predictions.csv`
- figure files such as:
  - `vader_distribution.png`
  - `tfidf_ngrams.png`

## Environment Setup

### 1) Create and activate a virtual environment

From repo root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install required packages

```bash
pip install pandas numpy pyarrow fastparquet vaderSentiment spacy scikit-learn factor_analyzer statsmodels matplotlib seaborn tqdm bertopic sentence-transformers umap-learn hdbscan keybert openpyxl jupyterlab langdetect xgboost nltk
python -m spacy download en_core_web_sm
```

Notes:

- `xgboost` is required by `02_nlp_sampled_pipeline.ipynb`.
- `nltk` resources are downloaded inside notebooks (`punkt`, `punkt_tab`, `stopwords`).

## Important Path Configuration

The notebooks/scripts currently hardcode:

`~/Desktop/MSIS/TEXT_MINING_NLP/Project/code2`

If your actual folder differs (for example this repo is under `code2 copy`), update `BASE_DIR` in notebook config cells and in any scripts before running.

At minimum, confirm these variables resolve correctly:

- `BASE_DIR`
- `CSV_DIR`
- `PARQUET_DIR`
- `OUTPUT_DIR`
- `DB_PATH`

## How To Run (Recommended Order)

### Option A: Notebook-first workflow

1. Launch Jupyter:

   ```bash
   jupyter lab
   ```

2. Run notebooks in this order:
   - `00_explore.ipynb`
   - `01_ingest.ipynb`
   - one of:
     - `02_nlp_full_data_EDA.ipynb` (full data, slower), or
     - `02_nlp_sampled_pipeline.ipynb` (balanced sampled pipeline, additional model comparisons)

### Option B: Script ingest + notebook NLP

1. Build core data assets:

   ```bash
   python3 01_ingest.py
   ```

2. Open and run either NLP notebook (`02_nlp_full_data_EDA.ipynb` or `02_nlp_sampled_pipeline.ipynb`).

## Notebook-by-Notebook Documentation

### `00_explore.ipynb` (Pre-analysis Inspection)

Key sections:

- Imports and path config
- file inventory checks across cities
- single-city listing inspection
- single-city review inspection
- all-city summary report
- SQLite checks (after ingest is complete)

Purpose:

- validate data availability/consistency before expensive processing,
- quickly inspect anomalies and distributions.

### `01_ingest.ipynb` / `01_ingest.py` (Phase 1 Ingest)

Key logic:

- force ID columns to string to avoid float precision loss on large IDs,
- clean listing and review data per city,
- deduplicate IDs (city-level and cross-city),
- check review-to-listing join coverage,
- write outputs to SQLite + Parquet,
- run all-city EDA summaries.

Important implementation note:

- ID precision is explicitly protected by reading ID columns as `str` to prevent 17-digit ID rounding from float conversion.

### `02_nlp_full_data_EDA.ipynb` (Phase 2 Full Dataset)

Key sections:

- load joined parquet (`reviews_with_features.parquet`)
- language detection analysis
- English filtering + text preprocessing
- TF-IDF feature extraction
- VADER sentiment + aspect-based sentiment
- LDA topic modeling (coherence-guided setup)
- rating prediction models (TF-IDF baseline + Sentence-BERT comparison)
- save parquet and outputs

Characteristics:

- highest data coverage,
- longest runtime (language detection and full-data transforms are expensive).

### `02_nlp_sampled_pipeline.ipynb` (Sampled/Modeling Pipeline)

Key sections:

- deterministic two-stage per-city sampling
  - listing-level cap (`REVIEWS_PER_LISTING`)
  - fixed per-city target (`SAMPLE_N_PER_CITY`)
- short-review filtering (`MIN_REVIEW_WORDS`)
- language filtering post-sampling
- negation-safe preprocessing for TF-IDF/LDA
- VADER + aspect sentiment
- LDA feature pipeline
- rating prediction model family comparison
- LOCO (Leave-One-City-Out) generalization evaluation
- save outputs

Characteristics:

- more balanced training design across cities,
- additional model benchmarking and cross-city robustness checks.

## Runtime Notes

Approximate guidance (depends on hardware and dataset size):

- `01_ingest.py`: minutes
- `02_nlp_full_data_EDA.ipynb`: potentially long (language detection on full corpus can be substantial)
- `02_nlp_sampled_pipeline.ipynb`: shorter than full-data path but still model-intensive

For best stability:

- run heavy notebooks with sufficient RAM,
- avoid interrupting model/training cells midway,
- save checkpoints frequently.

## Reproducibility Recommendations

- keep `RANDOM_STATE` fixed where defined in notebooks,
- keep library versions consistent within one project run,
- do not mix outputs from different `BASE_DIR` locations,
- clear/rename old output files if running multiple experimental variants.

## Common Troubleshooting

- **File not found for CSV/Parquet/DB**  
  Usually `BASE_DIR` mismatch. Update path variables and rerun from top.
- **Missing package errors**  
  Re-activate `venv` and reinstall dependencies.
- **spaCy model not found**  
  Run `python -m spacy download en_core_web_sm`.
- **Long language detection runtime**  
  Use sampled notebook for faster experimentation.
- **Output filename mismatch in sampled notebook**  
  The notebook currently writes `reviews_sentiment_lda_machinelearning4.parquet`; rename in notebook if you prefer a stable canonical name.

## Suggested Minimal Run Checklist

1. Activate `venv`.
2. Confirm `BASE_DIR` in notebook config cell(s).
3. Run `01_ingest.py` (or `01_ingest.ipynb`).
4. Verify created files in `parquet/`.
5. Run one NLP notebook end-to-end.
6. Verify `outputs/` artifacts and final parquet outputs.

---

If you plan to share this repository publicly, consider adding:

- `requirements.txt` (pinned versions),
- environment metadata (Python version),
- a small sample dataset or download instructions,
- and a single canonical final-output naming convention in the sampled pipeline notebook.
