"""
End-to-end training pipeline for the Review Sentiment Classifier.

Loads the dataset, preprocesses text, splits into train/test sets,
extracts TF-IDF features, trains three classifiers, compares their
performance, and saves the best model artifacts.

Usage:
    python train.py [--data data/reviews.csv] [--output models/]
"""

import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import joblib

# Add the scripts directory to the path so we can import preprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_batch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the review dataset from a CSV file."""
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Run `python generate_dataset.py` first to create a sample dataset.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} reviews from {data_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}")
    print()

    return df


def prepare_data(df: pd.DataFrame):
    """
    Preprocess text, encode labels, and split into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    # Preprocess reviews
    print("Preprocessing reviews...")
    df["processed"] = preprocess_batch(df["review"].tolist())

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["sentiment"])
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed"].values,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print()

    return X_train, X_test, y_train, y_test, label_encoder


def train_tfidf(X_train: np.ndarray) -> TfidfVectorizer:
    """Fit and return the TF-IDF vectorizer on training data."""
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    print()

    return vectorizer, X_train_tfidf


def build_classifiers():
    """Return a dict of classifier name -> classifier instance."""
    return {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Linear SVM": SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-4,
            random_state=RANDOM_STATE,
            max_iter=1000,
            tol=1e-3,
        ),
    }


def evaluate_model(
    model, name: str, X_test_tfidf, y_test: np.ndarray, label_encoder
) -> dict:
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test_tfidf)

    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return metrics, y_pred


def print_comparison_table(results: list, label_encoder) -> None:
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 (macro)':>10} {'F1 (wtd)':>10}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["f1_macro"], reverse=True):
        print(
            f"{r['name']:<20} "
            f"{r['accuracy']:>10.4f} "
            f"{r['precision_macro']:>10.4f} "
            f"{r['recall_macro']:>10.4f} "
            f"{r['f1_macro']:>10.4f} "
            f"{r['f1_weighted']:>10.4f}"
        )

    print("=" * 80)
    best = max(results, key=lambda x: x["f1_macro"])
    print(f"\nBest model: {best['name']} (F1 macro: {best['f1_macro']:.4f})")
    print()


def save_best_model(
    best_name: str,
    best_model,
    vectorizer,
    label_encoder,
    output_dir: str,
    all_results: list,
) -> None:
    """Save the best model, vectorizer, and label encoder to disk."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pkl")
    vectorizer_path = os.path.join(output_dir, "vectorizer.pkl")
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)

    # Save metadata
    metadata = {
        "best_model": best_name,
        "all_results": all_results,
        "tfidf_params": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved best model ({best_name}) to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved label encoder to: {encoder_path}")
    print(f"Saved metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument(
        "--data",
        default="data/reviews.csv",
        help="Path to the CSV dataset (default: data/reviews.csv)",
    )
    parser.add_argument(
        "--output",
        default="models",
        help="Directory to save model artifacts (default: models/)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("REVIEW SENTIMENT CLASSIFIER — TRAINING PIPELINE")
    print("=" * 80)
    print()

    # Step 1: Load data
    df = load_dataset(args.data)

    # Step 2: Preprocess and split
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(df)

    # Step 3: TF-IDF vectorization
    vectorizer, X_train_tfidf = train_tfidf(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 4: Train and evaluate classifiers
    classifiers = build_classifiers()
    results = []

    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train_tfidf, y_train)
        metrics, y_pred = evaluate_model(clf, name, X_test_tfidf, y_test, label_encoder)
        results.append(metrics)
        print(f"  {name} — Accuracy: {metrics['accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")

    # Step 5: Compare models
    print_comparison_table(results, label_encoder)

    # Step 6: Save the best model
    best_result = max(results, key=lambda x: x["f1_macro"])
    best_name = best_result["name"]
    best_model = classifiers[best_name]

    save_best_model(best_name, best_model, vectorizer, label_encoder, args.output, results)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
