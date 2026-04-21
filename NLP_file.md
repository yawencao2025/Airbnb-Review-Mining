- What's the differences between 02_nlp.ipynb, 02_nlp_v4_revised.ipynb, and 02_nlp_v6_final.ipynb


- 02_nlp_v6_final changes: 

 1. Config RATING_HIGH_MIN changed from 4.85 → 4.

 2. Text Preprocessing: deal with negations more carefully for TF-IDF and LDA analysis, aoviding missing negation in negative reviews. 
Defined NEGATIONS set covering not, never, didn't, can't, hardly, barely and 15 others
Built STOP_WORDS_BASE = set(stopwords.words("english")) - NEGATIONS — all downstream stopword sets derive from this
Added assert guards to catch any future accidental negation leakage
Added encode_negations() function that fuses not clean → not_clean before LDA cleaning
preprocess_lda() now calls encode_negations() first so compound tokens survive into the LDA vocabulary
TfidfVectorizer in Cell 56 now receives stop_words=list(STOP_WORDS_TFIDF) instead of nothing — sklearn's built-in 'english' list has the same negation problem as NLTK'same
3. Three new sections appended before Save Results
Section 8: Random Forest + Sentence-BERT (8.1) and XGBoost + Sentence-BERT (8.2), with confusion matrices and classification reports for both
Section 9: Leave-One-City-Out CV with per-city macro F1 bar chart, per-class grouped bar chart, and a generalisation gap interpretation
Section 10: Full 5-model comparison heatmap and per-class F1 bar chart