# CareCaller Hackathon 2026 - Problem 1: Call Quality Auto-Flagger

**Team/Author**: [Your Name/Team Name]  
**Date**: March 24, 2026  
**Competition**: CareCaller AI Call Quality & Voice Agent Challenge

---

## Executive Summary

This solution addresses Problem 1: automatically detecting AI voice calls that require human review. Using advanced feature engineering and ensemble modeling, we achieved a **cross-validation F1 score of 0.9175**, positioning this solution competitively for top leaderboard placements.

### Key Results
- **Cross-Validation F1**: 0.9175 ± 0.037
- **Features Engineered**: 91
- **Models Ensembled**: 5
- **Prediction Rate**: ~9% (matches training distribution)

---

## Problem Statement

Given call metadata, transcripts, and structured Q&A responses, predict whether a call requires human review (`has_ticket = True`). The challenge involves:

- **Class Imbalance**: Only ~9% of calls have tickets
- **Multiple Failure Modes**: 6 distinct ticket categories
- **Subtle Anomalies**: Issues range from STT errors to guardrail violations
- **High Stakes**: Missing bad calls vs. overwhelming reviewers with false positives

---

## Solution Architecture

### 1. Feature Engineering (91 Features)

Our approach extracts signals from four modalities:

#### A. Metadata Features (17)
- Call duration, attempt number, turn counts
- Whisper mismatch statistics
- Temporal features (hour of day, day of week)
- Encoded categorical variables

#### B. Response Analysis (9)
- **Completeness metrics**: Answered vs. empty responses
- **Content detection**: Numeric values, medical terminology
- **Pattern matching**: Yes/no responses, medication mentions
- **Side effect indicators**: Health concern keywords

#### C. Transcript NLP (12)
- **Question analysis**: Agent vs. user question counts
- **Confusion markers**: Apologies, unclear statements
- **Sentiment scoring**: Negative word frequency
- **Domain signals**: Pricing mentions, technical issues, escalation keywords
- **Conversation quality**: Agent-to-user ratio, silence markers

#### D. Validation Features (5)
- Warning/error flags in AI validation notes
- Mismatch and verification indicators
- Note length statistics

#### E. Text Embeddings (30)
- TF-IDF vectorization (max 100 features, bigrams)
- Truncated SVD dimensionality reduction
- 20 transcript components + 10 validation components

#### F. Interaction Features (18)
- Response completeness × duration
- Whisper mismatches per turn
- Words per second (efficiency metric)
- High-risk outcome flags
- Duration anomalies
- User engagement ratios

### 2. Ensemble Modeling

**Five-Model Weighted Ensemble:**

| Model | Weight | Purpose |
|-------|--------|---------|
| Random Forest | 30% | Non-linear feature interactions |
| Gradient Boosting | 30% | Sequential error correction |
| Extra Trees | 20% | Robustness through randomization |
| Logistic Regression | 10% | Linear baseline & calibration |
| Neural Network | 10% | Deep pattern recognition |

**Configuration:**
- Class weights: Balanced (1:10 ratio for minority class)
- Feature scaling: StandardScaler applied
- Cross-validation: 5-fold stratified
- Threshold: Optimized to 0.39 (from default 0.5)

---

## Performance Analysis

### Cross-Validation Results

| Fold | F1 Score | Tickets/Total |
|------|----------|---------------|
| 1 | 0.9600 | 12/138 |
| 2 | 0.9091 | 11/138 |
| 3 | 0.9091 | 12/138 |
| 4 | 0.8571 | 12/138 |
| 5 | 0.9524 | 12/137 |
| **Mean** | **0.9175** | **59/689** |
| **Std** | **0.0369** | — |

### Model Comparison

| Model | Average F1 |
|-------|-----------|
| Random Forest | 0.8813 |
| **Gradient Boosting** | **0.9151** |
| Extra Trees | 0.8990 |
| Logistic Regression | 0.8317 |
| Neural Network | 0.7557 |
| **Ensemble** | **0.9175** |

The ensemble outperforms all individual models, demonstrating effective diversity.

### Confusion Matrix (Training)

```
                 Predicted
                 No    Yes
Actual No       630      0
Actual Yes        0     59
```

Perfect classification on training data, with CV validation confirming generalization.

---

## Technical Implementation

### Dependencies
```
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
```

### File Structure
```
call_quality_pipeline.py      # Main training & prediction script
submission.csv             # Competition predictions
README.md                  # This file
requirements.txt           # Python dependencies
```

### Usage

**Training:**
```bash
python call_quality_pipeline.py train
```

**Prediction:**
```bash
python call_quality_pipeline.py predict
```

**Output:**
- Trained models saved to `solution_models.pkl`
- Predictions saved to `submission.csv`

---

## Key Design Decisions

### 1. Why 91 Features?
Comprehensive coverage of all ticket categories:
- STT mishearing → Whisper mismatch features
- Skipped questions → Response completeness metrics
- Outcome errors → Metadata patterns
- Guardrail violations → Transcript sentiment analysis
- Data capture errors → Cross-validation between fields

### 2. Why Ensemble?
- **Robustness**: No single point of failure
- **Complementary strengths**: Trees + linear + neural
- **Variance reduction**: Weighted averaging smooths predictions
- **Proven approach**: Ensembles dominate ML competitions

### 3. Why Threshold = 0.39?
- Default 0.5 optimizes accuracy
- 0.39 optimizes F1 score specifically
- Grid search over out-of-fold predictions
- Recall matters: catching all bad calls is critical

### 4. Why Class Weights?
- 9% positive rate creates severe imbalance
- Weight ratio 1:10 prevents majority-class bias
- All models trained with balanced weights

---

## Feature Importance

Top 10 features by Random Forest importance:

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | user_word_count | 0.0903 | Metadata |
| 2 | validation_embed_6 | 0.0551 | Text Embedding |
| 3 | trans_negative_sentiment | 0.0536 | NLP |
| 4 | agent_word_count | 0.0481 | Metadata |
| 5 | transcript_embed_5 | 0.0436 | Text Embedding |
| 6 | validation_embed_0 | 0.0371 | Text Embedding |
| 7 | answered_count | 0.0301 | Response |
| 8 | avg_user_turn_words | 0.0300 | Metadata |
| 9 | transcript_embed_1 | 0.0263 | Text Embedding |
| 10 | resp_incomplete_ratio | 0.0231 | Interaction |

**Insights:**
- Engagement metrics (word counts) are critical
- Validation embeddings capture semantic issues
- Negative sentiment correlates with problems
- Response completeness is a strong signal

---

## Potential Improvements

If pursuing higher scores:

1. **Advanced Models**: XGBoost, LightGBM, CatBoost (+2-3% F1)
2. **Deep Learning**: BERT embeddings for transcripts (+1-2% F1)
3. **Stacking**: Meta-learner on OOF predictions (+0.5-1% F1)
4. **Feature Engineering**: 
   - Named entity recognition for medical terms
   - Semantic similarity between Q&A pairs
   - Time-series patterns in conversation flow
5. **Pseudo-Labeling**: High-confidence test predictions for augmentation

---

## Validation & Testing

### Robustness Checks
- ✅ 5-fold stratified cross-validation
- ✅ Out-of-fold threshold optimization
- ✅ Consistent performance across folds (low std: 0.0369)
- ✅ No data leakage (proper train/validation splits)
- ✅ Handles unseen categorical values

### Edge Cases
- ✅ Missing values (filled with 0)
- ✅ Empty transcripts (TF-IDF handles gracefully)
- ✅ Malformed JSON responses (try-except protection)
- ✅ Unseen categories in test (encoded as -1)

---

## Submission Format

**File**: `submission.csv`

**Format**:
```csv
call_id,predicted_ticket
uuid-1,False
uuid-2,True
...
```

**Specifications**:
- Column 1: `call_id` (UUID string from test file)
- Column 2: `predicted_ticket` (Boolean: True/False)
- Rows: Exact match to test file call_ids
- No index column

**Statistics**:
- Total predictions: 159
- Predicted positives: ~14 (8.8%)
- Prediction distribution matches training

---

## Acknowledgments

- CareCaller team for providing high-quality synthetic data
- Scikit-learn developers for excellent ML tools
- Competition organizers for a well-designed challenge

---

## Contact

For questions or collaboration:
- **Email**: [bagmaraashish@gmail.com]
- **GitHub**: [github.com/]
- **LinkedIn**: [www.linkedin.com/in/aashishbagmar]

---

## 📄 License

This solution is provided for the CareCaller Hackathon 2026. All rights reserved.

---

**Last Updated**: March 24, 2026  
**Version**: 1.0  
**Status**: Ready for Submission 
