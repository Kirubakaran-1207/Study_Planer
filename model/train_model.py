"""
model/train_model.py
----------------------
Trains a sentence-importance classifier using TF-IDF features
and a scikit-learn pipeline.

Workflow:
  1. Load dataset/training_dataset.csv
  2. Preprocess sentences with NLTK
  3. Vectorize with TF-IDF
  4. Train a voting ensemble (Logistic Regression + Naive Bayes +
     Random Forest) for robustness
  5. Evaluate on a held-out test set
  6. Save the trained pipeline to saved_model/question_model.pkl

The saved model pipeline contains BOTH the TF-IDF vectorizer and
the classifier, so only a raw string is needed at inference time.
"""

import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# Allow imports from project root when running as standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nlp_processing.preprocess_text import preprocess_sentence

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
DATASET_PATH   = os.path.join(os.path.dirname(__file__), "..", "dataset", "training_dataset.csv")
MODEL_DIR      = os.path.join(os.path.dirname(__file__), "..", "saved_model")
MODEL_PATH     = os.path.join(MODEL_DIR, "question_model.pkl")


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Read CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path)
    # Drop any rows where 'context' is missing
    df.dropna(subset=["context"], inplace=True)
    df["label"] = df["label"].astype(int)
    print(f"[Model] Loaded {len(df)} rows | label distribution:\n{df['label'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────
# Text preparation
# ─────────────────────────────────────────────────────────────────

def prepare_text(sentence: str) -> str:
    """
    Convert a raw sentence to a cleaned, lemmatised string
    suitable for TF-IDF vectorization.
    """
    tokens = preprocess_sentence(sentence)
    return " ".join(tokens) if tokens else sentence.lower()


# ─────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline:
      TF-IDF → Soft-Voting Ensemble
                 (Logistic Regression + Naive Bayes + Random Forest)
    """
    lr  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    nb  = MultinomialNB(alpha=0.5)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)

    # Soft voting averages class probabilities — better than hard majority vote
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("nb", nb), ("rf", rf)],
        voting="soft"
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams
            max_features=5000,
            sublinear_tf=True        # log-scaled TF
        )),
        ("clf", ensemble),
    ])
    return pipeline


# ─────────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH,
          model_path:   str = MODEL_PATH,
          test_size:    float = 0.2) -> dict:
    """
    Full training run.
    Returns a dict with accuracy, report, and model path.
    """
    # 1. Load data
    df = load_dataset(dataset_path)

    # 2. Preprocess text
    print("[Model] Preprocessing sentences …")
    X = df["context"].apply(prepare_text).tolist()
    y = df["label"].tolist()

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[Model] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 4. Build and train pipeline
    pipeline = build_pipeline()
    print("[Model] Training ensemble pipeline …")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Not Q-worthy", "Q-worthy"])
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n[Model] ── Evaluation Results ──────────────────────")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}\n")

    # 6. Cross-validation score (on full dataset)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

    # 7. Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[Model] Model saved → {model_path}")

    return {
        "accuracy": round(acc, 4),
        "report":   report,
        "cv_mean":  round(cv_scores.mean(), 4),
        "cv_std":   round(cv_scores.std(), 4),
        "model_path": model_path,
    }


# ─────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH) -> object:
    """Load and return the saved sklearn Pipeline."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_importance(sentences: list[str],
                       model_path: str = MODEL_PATH) -> list[tuple[str, int, float]]:
    """
    Given a list of sentences and a saved model path, return a list of
    (sentence, predicted_label, confidence) tuples.
    """
    pipeline = load_model(model_path)
    cleaned  = [prepare_text(s) for s in sentences]
    labels   = pipeline.predict(cleaned)
    proba    = pipeline.predict_proba(cleaned)[:, 1]   # P(question-worthy)
    return list(zip(sentences, labels.tolist(), proba.tolist()))


# ─────────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate dataset first if it doesn't exist
    if not os.path.exists(DATASET_PATH):
        print("[Model] Dataset not found – generating …")
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from dataset.generate_dataset import create_dataset
        create_dataset(DATASET_PATH)

    results = train()
    print(f"\nFinal accuracy: {results['accuracy'] * 100:.2f}%")
