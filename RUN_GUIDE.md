# 1. One-time setup (creates a virtual environment + installs everything)
bash ~/Desktop/MSIS/TEXT_MINING_NLP/Project/00_setup.sh

# 2. Activate the environment (do this every new Terminal session)
source /Users/yawencao2024/Desktop/MSIS/TEXT_MINING_NLP/Project/code2/venv/bin/activate

# 3. Run the three phases
python3 01_ingest.py        # ~5 min — builds SQLite + Parquet files
python3 02_nlp.py           # ~30-60 min — VADER, aspect sentiment, LDA
python3 03_factor_analysis.py  # ~5 min — EFA + regressions + charts



# workflow 
01_ingest.py          → reviews_raw.parquet + airbnb.db
02_nlp.py             → reviews_sentiment.parquet
                                    ↓
              (when ready for factor analysis)
                                    ↓
01b_join.py           → reviews_with_features.parquet
03_factor_analysis.py → outputs/




# Note: 
1. check current nlp conduct sentence or review-level sentiment analysis 
2. Although, we'll focus on the English review analysis, we also want to see the percentage of different language reviews. 
3. analyze the length of reviews
