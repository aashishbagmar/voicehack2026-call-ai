"""
CareCaller VoiceHack 2026 Solution

This script implements:
- Advanced feature engineering on call transcripts and responses
- Ensemble ML model for call failure detection
- End-to-end training and prediction pipeline

Usage:
    python call_quality_pipeline.py train
    python call_quality_pipeline.py predict
"""
import pandas as pd
import numpy as np
import json
import re
import sys
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)


class AdvancedFeatureEngineer:
    """Comprehensive feature engineering for call quality prediction"""
    
    def __init__(self):
        self.transcript_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2)
        self.validation_vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=2)
        self.svd_transcript = TruncatedSVD(n_components=20, random_state=SEED)
        self.svd_validation = TruncatedSVD(n_components=10, random_state=SEED)
        self.label_encoders = {}
        self.fitted = False
        
    def extract_responses_features(self, responses_json_str):
        """Extract structured features from Q&A responses"""
        features = {
            'resp_num_answered': 0,
            'resp_num_empty': 0,
            'resp_num_numeric': 0,
            'resp_avg_length': 0,
            'resp_num_yes_no': 0,
            'resp_num_none': 0,
            'resp_has_medication': 0,
            'resp_has_side_effect': 0,
            'resp_total_words': 0,
        }
        
        try:
            if pd.isna(responses_json_str):
                return features
                
            responses = json.loads(responses_json_str)
            if not isinstance(responses, list):
                return features
            
            lengths = []
            for item in responses:
                if isinstance(item, dict) and 'answer' in item:
                    answer = str(item['answer']).strip().lower()
                    
                    if answer and answer not in ['', 'none', 'null', 'n/a']:
                        features['resp_num_answered'] += 1
                        lengths.append(len(answer))
                        features['resp_total_words'] += len(answer.split())
                        
                        if re.search(r'\d+', answer):
                            features['resp_num_numeric'] += 1
                        if re.search(r'\b(yes|no)\b', answer):
                            features['resp_num_yes_no'] += 1
                        if re.search(r'\b(mg|pill|medication|drug|tablet|capsule)\b', answer):
                            features['resp_has_medication'] = 1
                        if re.search(r'\b(nausea|pain|headache|dizz|tired|fatigue)\b', answer):
                            features['resp_has_side_effect'] = 1
                    else:
                        features['resp_num_empty'] += 1
                        if answer in ['none', 'no', 'n/a']:
                            features['resp_num_none'] += 1
            
            if lengths:
                features['resp_avg_length'] = np.mean(lengths)
        except:
            pass
        
        return features
    
    def extract_transcript_features(self, transcript_text):
        """Extract linguistic features from transcript"""
        features = {
            'trans_agent_questions': 0,
            'trans_user_questions': 0,
            'trans_agent_apologies': 0,
            'trans_confusion_words': 0,
            'trans_negative_sentiment': 0,
            'trans_pricing_mention': 0,
            'trans_technical_issue': 0,
            'trans_escalation_words': 0,
            'trans_has_silence_markers': 0,
            'trans_agent_to_user_ratio': 0,
            'trans_num_numbers_mentioned': 0,
            'trans_health_terms': 0,
        }
        
        if pd.isna(transcript_text):
            return features
        
        text_lower = str(transcript_text).lower()
        
        agent_parts = re.findall(r'\[agent\]:\s*(.*?)(?=\[user\]:|$)', text_lower, re.DOTALL)
        user_parts = re.findall(r'\[user\]:\s*(.*?)(?=\[agent\]:|$)', text_lower, re.DOTALL)
        
        agent_text = ' '.join(agent_parts)
        user_text = ' '.join(user_parts)
        
        features['trans_agent_questions'] = agent_text.count('?')
        features['trans_user_questions'] = user_text.count('?')
        features['trans_agent_apologies'] = len(re.findall(r'\b(sorry|apologize|pardon)\b', agent_text))
        features['trans_confusion_words'] = len(re.findall(r'\b(confused|unclear|understand|repeat|what)\b', text_lower))
        
        negative_words = ['no', 'not', 'never', 'wrong', 'incorrect', 'issue', 'problem', 'error']
        features['trans_negative_sentiment'] = sum(text_lower.count(word) for word in negative_words)
        
        features['trans_pricing_mention'] = len(re.findall(r'\b(price|cost|pay|bill|charge|expensive)\b', text_lower))
        features['trans_technical_issue'] = len(re.findall(r'\b(error|technical|system|issue|problem|unable|cannot)\b', text_lower))
        features['trans_escalation_words'] = len(re.findall(r'\b(supervisor|manager|escalate|transfer|speak to)\b', text_lower))
        features['trans_has_silence_markers'] = 1 if re.search(r'\[silence\]|\[pause\]|\.\.\.', text_lower) else 0
        
        if len(user_text) > 0:
            features['trans_agent_to_user_ratio'] = len(agent_text) / len(user_text)
        
        features['trans_num_numbers_mentioned'] = len(re.findall(r'\b\d+\b', text_lower))
        
        health_terms = ['weight', 'medication', 'dose', 'side effect', 'allergy', 'symptom', 'health']
        features['trans_health_terms'] = sum(text_lower.count(term) for term in health_terms)
        
        return features
    
    def extract_validation_features(self, validation_notes):
        """Extract features from AI validation notes"""
        features = {
            'val_has_warning': 0,
            'val_has_error': 0,
            'val_has_mismatch': 0,
            'val_has_verification': 0,
            'val_length': 0,
        }
        
        if pd.isna(validation_notes):
            return features
        
        notes_lower = str(validation_notes).lower()
        features['val_length'] = len(notes_lower)
        features['val_has_warning'] = 1 if 'warning' in notes_lower else 0
        features['val_has_error'] = 1 if 'error' in notes_lower else 0
        features['val_has_mismatch'] = 1 if 'mismatch' in notes_lower or 'discrepancy' in notes_lower else 0
        features['val_has_verification'] = 1 if 'verified' in notes_lower or 'confirmed' in notes_lower else 0
        
        return features
    
    def create_interaction_features(self, df):
        """Create powerful interaction features"""
        features = pd.DataFrame(index=df.index)
        
        features['resp_complete_x_duration'] = df['response_completeness'] * df['call_duration']
        features['resp_incomplete_ratio'] = (df['question_count'] - df['answered_count']) / df['question_count'].clip(lower=1)
        features['whisper_mismatch_per_turn'] = df['whisper_mismatch_count'] / df['turn_count'].clip(lower=1)
        features['whisper_issue'] = ((df['whisper_mismatch_count'] > 0) & (df['whisper_status'] == 'completed')).astype(int)
        features['words_per_second'] = (df['user_word_count'] + df['agent_word_count']) / df['call_duration'].clip(lower=1)
        features['turn_efficiency'] = df['answered_count'] / df['turn_count'].clip(lower=1)
        
        high_risk_outcomes = ['incomplete', 'escalated', 'wrong_number']
        features['high_risk_outcome'] = df['outcome'].isin(high_risk_outcomes).astype(int)
        features['duration_too_short'] = (df['call_duration'] < 30).astype(int)
        features['duration_anomaly'] = ((df['call_duration'] < 20) | (df['call_duration'] > 300)).astype(int)
        features['user_engagement'] = df['user_turn_count'] / df['turn_count'].clip(lower=1)
        features['low_user_engagement'] = (features['user_engagement'] < 0.3).astype(int)
        features['form_not_submitted_but_complete'] = ((df['form_submitted'] == False) & (df['outcome'] == 'completed')).astype(int)
        
        return features
    
    def fit_transform(self, df):
        """Fit and transform the complete feature set"""
        print("🔧 Engineering features...")
        
        feature_df = df.copy()
        
        categorical_cols = ['outcome', 'direction', 'whisper_status', 'patient_state', 'cycle_status', 'day_of_week']
        for col in categorical_cols:
            if col in feature_df.columns:
                self.label_encoders[col] = LabelEncoder()
                feature_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(feature_df[col].astype(str))
        
        print("  → Extracting response features...")
        response_features = feature_df['responses_json'].apply(self.extract_responses_features)
        response_df = pd.DataFrame(list(response_features))
        
        print("  → Extracting transcript features...")
        transcript_features = feature_df['transcript_text'].apply(self.extract_transcript_features)
        transcript_df = pd.DataFrame(list(transcript_features))
        
        print("  → Extracting validation features...")
        validation_features = feature_df['validation_notes'].apply(self.extract_validation_features)
        validation_df = pd.DataFrame(list(validation_features))
        
        print("  → Creating text embeddings...")
        transcript_tfidf = self.transcript_vectorizer.fit_transform(feature_df['transcript_text'].fillna('').astype(str))
        transcript_svd = self.svd_transcript.fit_transform(transcript_tfidf)
        transcript_embed_df = pd.DataFrame(
            transcript_svd,
            columns=[f'transcript_embed_{i}' for i in range(transcript_svd.shape[1])],
            index=feature_df.index
        )
        
        validation_tfidf = self.validation_vectorizer.fit_transform(feature_df['validation_notes'].fillna('').astype(str))
        validation_svd = self.svd_validation.fit_transform(validation_tfidf)
        validation_embed_df = pd.DataFrame(
            validation_svd,
            columns=[f'validation_embed_{i}' for i in range(validation_svd.shape[1])],
            index=feature_df.index
        )
        
        print("  → Creating interaction features...")
        interaction_df = self.create_interaction_features(feature_df)
        
        numeric_cols = [
            'call_duration', 'attempt_number', 'whisper_mismatch_count',
            'question_count', 'answered_count', 'response_completeness',
            'turn_count', 'user_turn_count', 'agent_turn_count',
            'user_word_count', 'agent_word_count', 'avg_user_turn_words',
            'avg_agent_turn_words', 'interruption_count', 'max_time_in_call',
            'hour_of_day'
        ]
        
        bool_cols = ['form_submitted']
        for col in bool_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(int)
        
        final_features = pd.concat([
            feature_df[numeric_cols + bool_cols + [f'{col}_encoded' for col in categorical_cols if col in feature_df.columns]],
            response_df,
            transcript_df,
            validation_df,
            transcript_embed_df,
            validation_embed_df,
            interaction_df
        ], axis=1)
        
        final_features = final_features.fillna(0)
        
        self.fitted = True
        self.feature_names = final_features.columns.tolist()
        
        print(f"  ✓ Generated {len(self.feature_names)} features")
        return final_features
    
    def transform(self, df):
        """Transform new data"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        print("🔧 Transforming features...")
        
        feature_df = df.copy()
        
        categorical_cols = ['outcome', 'direction', 'whisper_status', 'patient_state', 'cycle_status', 'day_of_week']
        for col in categorical_cols:
            if col in feature_df.columns and col in self.label_encoders:
                # Handle unseen categories
                le = self.label_encoders[col]
                feature_df[f'{col}_encoded'] = feature_df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        response_features = feature_df['responses_json'].apply(self.extract_responses_features)
        response_df = pd.DataFrame(list(response_features))
        
        transcript_features = feature_df['transcript_text'].apply(self.extract_transcript_features)
        transcript_df = pd.DataFrame(list(transcript_features))
        
        validation_features = feature_df['validation_notes'].apply(self.extract_validation_features)
        validation_df = pd.DataFrame(list(validation_features))
        
        transcript_tfidf = self.transcript_vectorizer.transform(feature_df['transcript_text'].fillna('').astype(str))
        transcript_svd = self.svd_transcript.transform(transcript_tfidf)
        transcript_embed_df = pd.DataFrame(
            transcript_svd,
            columns=[f'transcript_embed_{i}' for i in range(transcript_svd.shape[1])],
            index=feature_df.index
        )
        
        validation_tfidf = self.validation_vectorizer.transform(feature_df['validation_notes'].fillna('').astype(str))
        validation_svd = self.svd_validation.transform(validation_tfidf)
        validation_embed_df = pd.DataFrame(
            validation_svd,
            columns=[f'validation_embed_{i}' for i in range(validation_svd.shape[1])],
            index=feature_df.index
        )
        
        interaction_df = self.create_interaction_features(feature_df)
        
        numeric_cols = [
            'call_duration', 'attempt_number', 'whisper_mismatch_count',
            'question_count', 'answered_count', 'response_completeness',
            'turn_count', 'user_turn_count', 'agent_turn_count',
            'user_word_count', 'agent_word_count', 'avg_user_turn_words',
            'avg_agent_turn_words', 'interruption_count', 'max_time_in_call',
            'hour_of_day'
        ]
        
        bool_cols = ['form_submitted']
        for col in bool_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(int)
        
        final_features = pd.concat([
            feature_df[numeric_cols + bool_cols + [f'{col}_encoded' for col in categorical_cols if col in feature_df.columns]],
            response_df,
            transcript_df,
            validation_df,
            transcript_embed_df,
            validation_embed_df,
            interaction_df
        ], axis=1)
        
        final_features = final_features.fillna(0)
        
        for col in self.feature_names:
            if col not in final_features.columns:
                final_features[col] = 0
        
        return final_features[self.feature_names]


class EnsembleCallClassifier:
    """Advanced ensemble model"""
    
    def __init__(self):
        self.models = []
        self.weights = []
        self.threshold = 0.5
        self.scaler = StandardScaler()
        
    def train_with_cv(self, X, y, n_folds=5):
        """Train with cross-validation"""
        print("\n Training ensemble models with CV...")
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof_preds = np.zeros(len(X))
        
        pos_weight = len(y) / y.sum() - 1
        class_weight = {0: 1, 1: pos_weight}
        
        rf_params = {
            'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 10,
            'min_samples_leaf': 5, 'max_features': 'sqrt',
            'class_weight': class_weight, 'random_state': SEED, 'n_jobs': -1,
        }
        
        gb_params = {
            'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 7,
            'min_samples_split': 10, 'min_samples_leaf': 5,
            'subsample': 0.8, 'max_features': 'sqrt', 'random_state': SEED,
        }
        
        et_params = {
            'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 10,
            'min_samples_leaf': 5, 'max_features': 'sqrt',
            'class_weight': class_weight, 'random_state': SEED, 'n_jobs': -1,
        }
        
        lr_params = {
            'C': 0.1, 'class_weight': class_weight, 'random_state': SEED,
            'max_iter': 500, 'solver': 'liblinear',
        }
        
        mlp_params = {
            'hidden_layer_sizes': (100, 50), 'activation': 'relu',
            'alpha': 0.01, 'learning_rate': 'adaptive',
            'random_state': SEED, 'max_iter': 300, 'early_stopping': True,
        }
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
            print(f"\n  Fold {fold}/{n_folds}")
            
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict_proba(X_val)[:, 1]
            
            gb_model = GradientBoostingClassifier(**gb_params)
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict_proba(X_val)[:, 1]
            
            et_model = ExtraTreesClassifier(**et_params)
            et_model.fit(X_train, y_train)
            et_pred = et_model.predict_proba(X_val)[:, 1]
            
            lr_model = LogisticRegression(**lr_params)
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict_proba(X_val)[:, 1]
            
            mlp_model = MLPClassifier(**mlp_params)
            mlp_model.fit(X_train, y_train)
            mlp_pred = mlp_model.predict_proba(X_val)[:, 1]
            
            fold_pred = 0.30 * rf_pred + 0.30 * gb_pred + 0.20 * et_pred + 0.10 * lr_pred + 0.10 * mlp_pred
            oof_preds[val_idx] = fold_pred
            
            fold_pred_binary = (fold_pred > 0.5).astype(int)
            fold_f1 = f1_score(y_val, fold_pred_binary)
            fold_scores.append(fold_f1)
            print(f"    Ensemble F1: {fold_f1:.4f}")
        
        print(f"\n  CV Mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        self.threshold = self._optimize_threshold(oof_preds, y)
        print(f"  Optimized threshold: {self.threshold:.4f}")
        
        print("\n  Retraining on full dataset...")
        
        rf_final = RandomForestClassifier(**rf_params)
        rf_final.fit(X_scaled, y)
        
        gb_final = GradientBoostingClassifier(**gb_params)
        gb_final.fit(X_scaled, y)
        
        et_final = ExtraTreesClassifier(**et_params)
        et_final.fit(X_scaled, y)
        
        lr_final = LogisticRegression(**lr_params)
        lr_final.fit(X_scaled, y)
        
        mlp_final = MLPClassifier(**mlp_params)
        mlp_final.fit(X_scaled, y)
        
        self.models = [rf_final, gb_final, et_final, lr_final, mlp_final]
        self.weights = [0.30, 0.30, 0.20, 0.10, 0.10]
        
        return np.mean(fold_scores)
    
    def _optimize_threshold(self, y_pred_proba, y_true):
        """Find optimal classification threshold"""
        thresholds = np.arange(0.25, 0.75, 0.01)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        return best_threshold
    
    def predict_proba(self, X):
        """Ensemble prediction"""
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict_proba(X_scaled)[:, 1]
            predictions.append(weight * pred)
        
        return np.sum(predictions, axis=0)
    
    def predict(self, X):
        """Binary prediction"""
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)


def train():
    """Training pipeline"""
    print("=" * 80)
    print("CareCaller Hackathon 2026 - Training")
    print("=" * 80)
    
    print("\n Loading training data...")
    train_df = pd.read_csv('hackathon_train.csv')
    print(f"  ✓ Loaded {len(train_df)} training samples")
    print(f"  ✓ Positive rate: {train_df['has_ticket'].mean():.2%}")
    
    engineer = AdvancedFeatureEngineer()
    X_train = engineer.fit_transform(train_df)
    y_train = train_df['has_ticket'].astype(int)
    
    print(f"\n✓ Feature matrix shape: {X_train.shape}")
    
    ensemble = EnsembleCallClassifier()
    cv_score = ensemble.train_with_cv(X_train, y_train, n_folds=5)
    
    print("\n" + "=" * 80)
    print(f" Cross-Validation F1 Score: {cv_score:.4f}")
    print("=" * 80)
    
    print("\n Saving models...")
    with open('solution_models.pkl', 'wb') as f:
        pickle.dump({'engineer': engineer, 'ensemble': ensemble}, f)
    print("  ✓ Models saved to solution_models.pkl")
    print("\n Training complete!")


def predict():
    """Prediction pipeline - CORRECTED for actual Kaggle test file"""
    print("=" * 80)
    print("CareCaller Hackathon 2026 - Prediction")
    print("=" * 80)
    
    # Find test file from Kaggle
    test_paths = [
        '/mnt/user-data/uploads/hackathon_test.csv',  # Kaggle uploaded
        '/home/claude/hackathon_test.csv',
        'hackathon_test.csv'
    ]
    
    test_file = None
    for path in test_paths:
        if os.path.exists(path):
            test_file = path
            break
    
    if test_file is None:
        print("\n❌ Error: hackathon_test.csv not found!")
        print("\n Please download the test file from Kaggle:")
        print("   1. Go to the competition Data tab")
        print("   2. Download hackathon_test.csv")
        print("   3. Upload it here or place in current directory")
        return
    
    print(f"\n Loading test data from {test_file}...")
    test_df = pd.read_csv(test_file)
    print(f"  ✓ Loaded {len(test_df)} test samples")
    print(f"  ✓ Columns: {list(test_df.columns[:5])}...")
    
    # Verify call_id column exists
    if 'call_id' not in test_df.columns:
        print(f"\n❌ Error: call_id column not found in test file!")
        print(f"   Available columns: {list(test_df.columns)}")
        return
    
    print("\n Loading trained models...")
    if not os.path.exists('solution_models.pkl'):
        print("❌ Models not found! Please run 'python call_quality_pipeline.py train' first")
        return
        
    with open('solution_models.pkl', 'rb') as f:
        models = pickle.load(f)
    engineer = models['engineer']
    ensemble = models['ensemble']
    print("  ✓ Models loaded")
    
    print("\n Transforming test features...")
    X_test = engineer.transform(test_df)
    print(f"  ✓ Feature matrix shape: {X_test.shape}")
    
    print("\n Generating predictions...")
    predictions = ensemble.predict(X_test)
    prediction_probs = ensemble.predict_proba(X_test)
    
    # Create submission with EXACT call_ids from test file
    submission = pd.DataFrame({
        'call_id': test_df['call_id'],  # Use EXACT call_ids from test file
        'predicted_ticket': predictions.astype(bool)  # Boolean as required
    })
    
    output_path = 'submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"  ✓ Submission saved to {output_path}")
    
    print("\n Prediction Statistics:")
    print(f"  • Total predictions: {len(predictions)}")
    print(f"  • Predicted tickets: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
    print(f"  • Mean probability: {prediction_probs.mean():.4f}")
    print(f"  • Using threshold: {ensemble.threshold:.4f}")
    
    print("\n Sample predictions:")
    print(submission.head(10).to_string(index=False))
    
    print("\n Submission file details:")
    print(f"  • Location: {output_path}")
    print(f"  • Rows: {len(submission)}")
    print(f"  • Columns: {list(submission.columns)}")
    print(f"  • File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # If test has ground truth labels (for validation), show score
    if 'has_ticket' in test_df.columns and not test_df['has_ticket'].isna().all():
        y_true = test_df['has_ticket'].astype(int)
        f1 = f1_score(y_true, predictions)
        print(f"\nTest F1 Score: {f1:.4f}")
        print("\n" + classification_report(y_true, predictions, target_names=['No Ticket', 'Has Ticket']))
    
    print("\n" + "=" * 80)
    print("Prediction complete!")
    print(f"Ready to submit: {output_path}")
    print("=" * 80)
    print("\n To submit:")
    print("   1. Download submission.csv from outputs")
    print("   2. Go to Kaggle competition Submissions page")
    print("   3. Upload submission.csv")
    print("   4. Wait for score!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python call_quality_pipeline.py [train|predict]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    if mode == 'train':
        train()
    elif mode == 'predict':
        predict()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python call_quality_pipeline.py [train|predict]")
        sys.exit(1)
