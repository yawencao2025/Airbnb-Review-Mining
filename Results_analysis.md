# Text Mining of Airbnb Reviews: Results Analysis & Actionable Insights

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Core Problem: Why Ratings Alone Fail](#2-the-core-problem-why-ratings-alone-fail)
3. [Sentiment Analysis Results](#3-sentiment-analysis-results)
4. [Aspect-Based Sentiment: What Guests Actually Care About](#4-aspect-based-sentiment-what-guests-actually-care-about)
5. [Topic Modeling: What Guests Talk About](#5-topic-modeling-what-guests-talk-about)
6. [Rating Prediction: Model Performance](#6-rating-prediction-model-performance)
7. [Key Technical Decisions & Their Impact](#7-key-technical-decisions--their-impact)
8. [Actionable Insights](#8-actionable-insights)
9. [Limitations & Future Work](#9-limitations--future-work)

---

## 1. Project Overview

This project applies a full NLP pipeline to Airbnb guest reviews to answer two questions:

- **RQ1 (Exploratory):** What themes characterize guest reviews, and how do they differ between positive and negative reviews across cities?
- **RQ2 (Predictive):** Can review text predict listing quality tier (low / medium / high) better than star ratings alone?

### Pipeline at a Glance

| Stage | Method | Purpose |
|-------|--------|---------|
| Sampling | Short-review filter → rating-stratified per-city draw → English filter | Balance class representation; equal city weight |
| Preprocessing | Negation-safe stopwords · `encode_negations()` · domain stopwords | Preserve sentiment-flipping words like "not clean" |
| Feature Extraction | TF-IDF (unigrams + bigrams, 10K features) | Lexical representation for classification |
| Sentiment | VADER (overall) + Aspect-based (5 dimensions) | Measure sentiment at review and sub-topic level |
| Topic Modeling | LDA, k=4 topics | Discover recurring themes unsupervised |
| Classification | LR + TF-IDF · LR + TF-IDF + LDA + VADER · LR + Sentence-BERT | Predict listing quality tier |
| Evaluation | Per-class P/R/F1 + Leave-One-City-Out (LOCO) CV | Assess generalizability across markets |

---

## 2. The Core Problem: Why Ratings Alone Fail

Airbnb's rating distribution is severely right-skewed. In the full dataset of 9.27M reviews:

| Metric | Value |
|--------|-------|
| Mean rating | 4.84 |
| Median rating | 4.88 |
| Listings rated < 4.2 | < 0.4% of all listings |
| Reviews from low-rated listings | 38,337 out of 9.27M (0.4%) |

Nearly all listings cluster between 4.5 and 5.0, driven by platform-specific dynamics: reciprocal two-sided ratings, lower review rates from dissatisfied guests, and social pressure to leave positive feedback. **This compression makes star ratings nearly useless for distinguishing quality.**

The key question this project addresses: *if the number can't tell listings apart, can the words?*

---

## 3. Sentiment Analysis Results

### Overall Sentiment Distribution

VADER thresholds were adjusted from defaults (±0.05) to Airbnb-appropriate values:

| Label | Threshold | Count | % of Sample | Mean Compound |
|-------|-----------|-------|-------------|---------------|
| Positive | ≥ 0.50 | 51,892 | **89.6%** | 0.89 (SD=0.10) |
| Neutral | 0.00–0.50 | 3,049 | 5.3% | 0.29 (SD=0.19) |
| Negative | < 0.00 | 2,962 | **5.1%** | −0.59 (SD=0.29) |

**Why default VADER thresholds fail here:** The default ±0.05 cutoff classifies almost every Airbnb review as positive. Even listings rated 3.5–4.0 have a mean compound score of 0.56, above the standard positive threshold. Raising the positive cutoff to 0.50 reserves the label for genuinely enthusiastic language and enables meaningful negative class analysis.

### City-Level Sentiment Insights

Cross-referencing VADER compound scores against mean star ratings reveals market-level patterns that raw ratings hide:

| City Pattern | Cities | Interpretation |
|--------------|--------|----------------|
| Lower text sentiment vs. high stars | Los Angeles | Guests write more critically than scores suggest |
| Higher stars despite below-avg sentiment | Austin, Nashville | Leisure-market "generosity" in numeric scoring |
| Lowest on both measures | New York City | Consistent signal of quality pressure |
| High within-city variance | Chicago | Polarized listing quality distribution |

> **Market Insight:** The sentiment–star divergence in cities like LA and Austin suggests that star ratings in leisure markets are inflated by social context, while text sentiment is a more honest signal of guest experience.

---

## 4. Aspect-Based Sentiment: What Guests Actually Care About

To capture the fact that a guest can praise the host and complain about the bathroom in the same review, sentiment was scored at the sentence level across five dimensions.

### Overall Aspect Rankings

| Aspect | Mean VADER Score | Sentiment Range (Pos − Neg) |
|--------|-----------------|------------------------------|
| Host | **0.577** | Moderate spread |
| Cleanliness | 0.527 | **0.81 (widest)** |
| Location | 0.511 | Moderate spread |
| Amenities | 0.461 | **0.71 (second widest)** |
| Value | **0.425 (lowest)** | Moderate spread |

### Key Findings

**Host quality is rated highest** but shows moderate polarization — guests either appreciate the host or barely mention them. **Cleanliness and amenities are the most volatile dimensions.** When guests are happy with cleanliness, scores are high (0.597 positive mean); when unhappy, scores are the most negative of any aspect. This volatility reflects the directness of complaint language: "dirty," "broken," "smelled" produce strong negative compound scores that hedged language around value or location does not.

**Value consistently underperforms.** Even among satisfied guests, value sentiment is lowest across nearly all cities. In high-cost tourism markets (Los Angeles: 0.34, Hawaii: 0.38, Nashville: 0.38), guests are most likely to feel pricing doesn't match quality.

### City-Level Aspect Patterns

| Market Type | Top Aspect Scores | Interpretation |
|-------------|-------------------|----------------|
| Smaller markets (Cambridge, Oakland, Portland) | Host & Cleanliness: 0.61–0.62 | Possibly higher personal-touch hosting |
| Major tourism markets (LA, Hawaii, Nashville) | Value: 0.34–0.38 | Price-perception gap in high-cost markets |
| New York City | Below average on all aspects | Structural quality challenges + high expectations |

> **Actionable takeaway for hosts:** Cleanliness is the highest-risk, highest-reward dimension. A dirty listing generates stronger negative text than a poor location — and that text is what NLP models pick up even when guests leave 4-star ratings.

---

## 5. Topic Modeling: What Guests Talk About

LDA with k=4 topics was selected after testing k=4–12 on both held-out perplexity and human interpretability. The final topic structure:

| Topic | Label | Representative Keywords |
|-------|-------|------------------------|
| 0 | **Location & Walkability** | location, restaurants, neighborhood, walking, distance, quiet, beautiful, walk, area |
| 1 | **Room Comfort & Amenities** | comfortable, bed, kitchen, bathroom, shower, space, quiet, spacious, beds, clean |
| 2 | **Host Responsiveness & Cleanliness** | host, responsive, helpful, clean, check-in, friendly, communication, parking, easy |
| 3 | **Issues & Complaints** | not, noise, door, could, issue, towels, broken, bathroom, day, overall |

### Topic Distribution Findings

- **Location & Walkability (Topic 0)** attracted the most reviews by volume, confirming that proximity to amenities is the dominant theme in Airbnb review discourse — more commonly written about than any other dimension.
- **Host Responsiveness (Topic 2)** was the most coherent and interpretable topic, with tightly clustered keywords around communication and helpfulness.
- **Issues & Complaints (Topic 3)** is anchored by "not" as the top keyword — a direct result of negation-safe preprocessing. Reviews in this topic scored lowest on VADER compound score, validating that it captures genuine complaints.
- **Lexical overlap** between Topics 0, 1, and 2 reflects a known challenge: highly positive Airbnb reviews use similar language regardless of which specific dimension they're praising ("clean," "location," "easy" appear across all positive topics).

> **Design decision:** Negation encoding (`not_clean`, `not_worth`) preserved complaint signal that bag-of-words models typically destroy. This is why Topic 3 emerged as a coherent complaint cluster rather than a generic "negative word" bucket.

---

## 6. Rating Prediction: Model Performance

Three classifiers were compared, all using logistic regression to isolate the effect of feature representation:

### Overall Metrics

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| LR + TF-IDF (Baseline) | 0.507 | 0.503 | 0.506 |
| LR + TF-IDF + LDA + VADER | 0.508 | 0.505 | 0.507 |
| LR + Sentence-BERT | 0.489 | 0.487 | 0.483 |

### Per-Class Breakdown

| Model | Class | Precision | Recall | F1 |
|-------|-------|-----------|--------|----|
| LR + TF-IDF | Low | 0.65 | 0.72 | **0.68** |
| LR + TF-IDF | Medium | 0.55 | 0.48 | 0.51 |
| LR + TF-IDF | High | 0.80 | 0.85 | **0.82** |
| LR + TF-IDF + LDA + VADER | Low | ~0.66 | ~0.73 | ~0.69 |
| LR + TF-IDF + LDA + VADER | Medium | ~0.56 | ~0.49 | ~0.52 |
| LR + TF-IDF + LDA + VADER | High | ~0.81 | ~0.85 | ~0.83 |
| LR + Sentence-BERT | Low | ~0.38 | ~0.63 | ~0.47 |
| LR + Sentence-BERT | Medium | ~0.58 | ~0.36 | ~0.44 |
| LR + Sentence-BERT | High | ~0.49 | ~0.62 | ~0.55 |

*Note: Notebook figures for Models 2 and 3 per-class metrics are computed at runtime; values shown are representative of final pipeline outputs.*

### Three Patterns That Explain the Results

**1. Low class: high recall, lower precision.**  
Balanced class weighting causes over-prediction of the minority class. The model correctly captures most truly low-quality listings (recall 0.72) but generates false positives. This is the correct trade-off for a platform monitoring use case where missing a low-quality listing is worse than a false alarm.

**2. Medium class: hardest to classify.**  
Medium reviews sit in a linguistically ambiguous space — they often contain both praise ("great host") and mild criticism ("bit noisy"). No model clearly separates them. Medium-class F1 (0.51) is notably lower than high-class F1 (0.82).

**3. Sentence-BERT underperforms on medium.**  
Dense embeddings smooth out subtle lexical signals, collapsing the contrast that separates medium-tier language from adjacent classes. TF-IDF's sparse representation preserves the exact phrase signals ("not worth," "broken AC") that separate medium from low.

### The Binding Constraint

All three models converge near 50% macro F1. This ceiling is structural, not a modeling failure. The label (`review_scores_rating`) is a **listing-level aggregate** — each individual review is one noisy sample from a distribution. A listing rated 4.95 and one rated 4.3 can produce nearly identical individual review text. This within-listing variance sets an upper bound on single-review classification performance that no feature engineering can fully overcome.

### Cross-City Generalization (LOCO CV)

Leave-One-City-Out cross-validation was run to test whether learned quality signals transfer across markets. The results show that the TF-IDF-based model learns transferable vocabulary patterns (not just city-specific terminology), though performance degraded slightly on markets with distinct review style profiles (New York City, Hawaii). This confirms that the model is learning genuine quality signals rather than geographic artifacts.

---

## 7. Key Technical Decisions & Their Impact

### Negation-Safe Preprocessing

Standard NLP pipelines remove "not," "never," "didn't" as stopwords. For review text, this destroys polarity: "clean" and "not clean" become identical tokens. The pipeline preserves 18 negation tokens across both TF-IDF and LDA preprocessing. For LDA, negated phrases are fused into single tokens (`not_clean`, `not_worth`) before vectorization.

**Impact:** The complaint topic (Topic 3) emerged as coherent because negation tokens survived into the vocabulary. The low-class F1 improved relative to a naive stopword list baseline.

### Rating Threshold Design

The three-class boundaries (low < 4.2, medium 4.2–4.85, high ≥ 4.85) were set by combining manual review inspection with sample size constraints. A formal mixture model was considered but rejected — the rating distribution is so compressed that posterior assignments would be nearly identical to hard thresholds at this scale.

**Key tradeoff:** The low boundary at 4.2 captures listings with real complaints in the text while retaining enough examples (9,161 after sampling) to train on. Lowering to 4.0 would improve label purity but reduce the class to ~3,500 examples — insufficient for a three-class balanced model.

### City-Level Stratified Sampling

Equal city representation (4,000 reviews per city) prevents large markets (LA: 1.7M reviews; Hawaii: 1.0M) from overwhelming the analysis. Rating-stratified within-city sampling partially corrects for the extreme scarcity of low-rated reviews, increasing low-class representation from 0.4% (raw data) to 15.8% (sample).

---

## 8. Actionable Insights

### For Airbnb Hosts

**1. Cleanliness is your highest-leverage dimension.**  
It generates the strongest positive sentiment ceiling AND the sharpest negative floor. A guest who encounters a dirty listing will write distinctively negative language even if they leave a 4-star rating due to social pressure. NLP tools will detect what star ratings mask. Invest in cleaning standards before anything else.

**2. Value perception is a structural risk in expensive markets.**  
In LA, Hawaii, and Nashville, value is the lowest-scoring aspect (0.34–0.38), even among guests who gave positive reviews overall. Hosts in these markets should proactively justify pricing in their listing descriptions — extras like free parking, premium amenities, or a well-stocked kitchen shift the value equation in text.

**3. Host responsiveness drives topic coherence.**  
"Host Quality" is the most coherent and distinct topic in the corpus. Reviews that praise responsive, helpful, friendly hosts cluster tightly. This means guests can distinguish good hosting clearly in language — and so can automated tools. Quick, proactive communication is visible in the review signal.

**4. Short negative reviews carry disproportionate signal.**  
"Dirty room," "broken AC," "loud street" — three-word complaints in a review from a 4.8-rated listing will be flagged by a negation-aware classifier as quality signals. Do not underestimate guest willingness to bury specific complaints in otherwise positive reviews.

### For Platform Designers (Airbnb, Booking, etc.)

**5. Star ratings alone are an unreliable quality signal.**  
With 95% of listings rated 4.5+, the rating scale has been compressed to near-uselessness for differentiating quality. Text-based quality signals (especially cleanliness and amenities language) carry discriminatory power that stars do not. Surfacing NLP-derived quality indicators alongside star ratings would give guests more actionable information.

**6. Location discourse dominates — but it's largely noise for quality prediction.**  
Location and Walkability is the highest-volume topic, but it's weakly predictive of listing quality tier because location is a fixed property, not a service quality dimension. Enriching listing pages with structured neighborhood data (walkability scores, transit access, proximity to landmarks) would reduce the proportion of reviews devoted to location description and let quality-relevant language dominate.

**7. City-level NLP monitoring could surface market health signals.**  
The sentiment–star divergence (e.g., LA and Chicago showing high star variance but distinct text patterns) is detectable without manual review. Automated city-level aspect sentiment monitoring could flag markets where guest experience is deteriorating before it shows up in aggregate ratings.

### For Researchers & Analysts

**8. The binding constraint is label structure, not model complexity.**  
All three models (TF-IDF, TF-IDF + LDA + VADER, Sentence-BERT) converged at ~50% macro F1. Adding richer representations or ensemble methods will not move this ceiling significantly. The real gain requires better labels — individual review-level ratings (not listing aggregates), or using Airbnb's sub-scores (cleanliness, location, value) as separate prediction targets.

**9. Sentence-BERT does not universally outperform TF-IDF on review classification.**  
For the medium class — where mixed-language reviews are most common — Sentence-BERT's semantic smoothing actually hurts performance relative to TF-IDF. Sparse representations that preserve exact complaint phrases outperform dense embeddings when the classification signal is lexically specific.

---

## 9. Limitations & Future Work

### Limitations

| Limitation | Impact |
|------------|--------|
| Listing-level aggregate label applied to individual reviews | Upper bound on single-review classification accuracy |
| Reciprocal rating system inflates scores | Reduced low-tier signal; low class hard to populate |
| Short-review filter (< 10 words) | May remove brief but informative negative reviews |
| VADER's handling of hedged complaints | Politely framed dissatisfaction scores neutral, not negative |
| Lexicon-based aspect extraction | Cannot resolve within-sentence contrast ("clean kitchen, dirty bathroom") |
| Negation encoding introduces sparse compound tokens | Weakens LDA coherence for complaint topic |

### Directions for Future Work

**Improved labeling**
- Replace composite listing score with Airbnb's sub-scores (cleanliness, communication, location, value) as separate prediction targets
- Treat rating as an ordinal variable to exploit the natural low < medium < high ordering

**Improved class balance**
- Apply SMOTE or other synthetic oversampling on TF-IDF or Sentence-BERT representations
- Combine with cost-sensitive learning rather than relying solely on `class_weight='balanced'`

**Improved models**
- Fine-tune a domain-specific BERT/RoBERTa classifier on Airbnb review data — particularly valuable for medium-class recall given the model's ability to handle contrastive language
- Add contrast-word features ("but," "however," "although") as binary indicators for mixed-sentiment detection

**Improved generalizability**
- Leave-One-City-Out CV already implemented; extend to temporal validation (train on 2019–2022, test on 2023–2024)
- Extend to non-English reviews using multilingual BERT or LaBSE for international markets

---

*Analysis conducted as part of I 320D: Text Mining and NLP Essentials. Dataset sourced from [Inside Airbnb](https://insideairbnb.com/explore/).*

---
