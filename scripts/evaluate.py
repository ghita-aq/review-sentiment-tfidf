"""
Standalone evaluation script for the Review Sentiment Classifier.

Loads saved model artifacts (best_model.pkl, vectorizer.pkl, label_encoder.pkl),
evaluates on the test set, and produces detailed metrics including a confusion
matrix visualization.

Usage:
    python evaluate.py [--data data/reviews.csv] [--models models/]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns

# Add the scripts directory to the path so we can import preprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_batch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_model_artifacts(models_dir: str):
    """Load the saved model, vectorizer, and label encoder."""
    model_path = os.path.join(models_dir, "best_model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")

    for path, name in [(model_path, "model"), (vectorizer_path, "vectorizer"), (encoder_path, "label encoder")]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            print("Run `python train.py` first to train and save the model.")
            sys.exit(1)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)

    print(f"Loaded model from: {model_path}")
    print(f"Loaded vectorizer from: {vectorizer_path}")
    print(f"Loaded label encoder from: {encoder_path}")
    print()

    return model, vectorizer, label_encoder


def load_and_prepare_data(data_path: str):
    """Load the dataset and prepare labels using a fresh LabelEncoder."""
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} reviews from {data_path}")

    # Preprocess
    print("Preprocessing reviews...")
    df["processed"] = preprocess_batch(df["review"].tolist())

    # Encode labels with a fresh encoder (classes will match the saved one)
    le = LabelEncoder()
    y = le.fit_transform(df["sentiment"])

    return df, y


def prepare_test_set(df: pd.DataFrame, y, label_encoder):
    """Split data into train/test (same split as training) and return test set."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed"].values,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_test, y_test


def evaluate(model, vectorizer, label_encoder, X_test, y_test):
    """Run predictions and compute all metrics."""
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    classes = label_encoder.classes_

    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classes": classes,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def print_metrics(results: dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro):    {results['recall_macro']:.4f}")
    print(f"  F1-Score (macro):  {results['f1_macro']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 50)
    for i, cls in enumerate(results["classes"]):
        print(
            f"{cls:<15} "
            f"{results['precision_per_class'][i]:>10.4f} "
            f"{results['recall_per_class'][i]:>10.4f} "
            f"{results['f1_per_class'][i]:>10.4f}"
        )

    print("\n" + "-" * 80)
    print("Classification Report:")
    print("-" * 80)
    report = classification_report(
        results["y_test"],
        results["y_pred"],
        target_names=results["classes"],
        zero_division=0,
    )
    print(report)

    print("=" * 80)


def plot_confusion_matrix(results: dict, output_path: str) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = results["confusion_matrix"]
    classes = results["classes"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Confusion Matrix — Sentiment Classifier", fontsize=14)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier")
    parser.add_argument(
        "--data",
        default="data/reviews.csv",
        help="Path to the CSV dataset (default: data/reviews.csv)",
    )
    parser.add_argument(
        "--models",
        default="models",
        help="Directory with saved model artifacts (default: models/)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("REVIEW SENTIMENT CLASSIFIER — EVALUATION")
    print("=" * 80)
    print()

    # Load model artifacts
    model, vectorizer, label_encoder = load_model_artifacts(args.models)

    # Load and prepare data
    df, y = load_and_prepare_data(args.data)
    X_test, y_test = prepare_test_set(df, y, label_encoder)
    print(f"Test set size: {len(X_test)}")
    print()

    # Evaluate
    print("Running predictions...")
    results = evaluate(model, vectorizer, label_encoder, X_test, y_test)

    # Print metrics
    print_metrics(results)

    # Plot confusion matrix
    cm_path = os.path.join(args.models, "confusion_matrix.png")
    plot_confusion_matrix(results, cm_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
