import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    return text.lower().strip()


def preprocess_batch(X):
    return [preprocess(text) for text in X]


# Mengatur input-input wajib dari terminal saat menjalankan script.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train fake news detection model "
        "using TF-IDF and Logistic Regression."
    )

    parser.add_argument(
        "--data", type=str, required=True, help="Path to training dataset CSV file."
    )

    parser.add_argument(
        "--model-out",
        type=str,
        required=True,
        help="Output path for trained model, for example: model/pipeline.pkl",
    )

    parser.add_argument(
        "--metrics-out",
        type=str,
        required=True,
        help="Output path for metrics JSON, for example: reports/metrics.json",
    )

    parser.add_argument(
        "--text-col", type=str, required=True, help="Column name containing news text."
    )

    parser.add_argument(
        "--label-col",
        type=str,
        required=True,
        help="Column name containing target label.",
    )

    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set proportion. Default: 0.2"
    )

    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed. Default: 42"
    )

    return parser.parse_args()


# Membuat folder baru otomatis jika folder tujuan penyimpanan belum ada.
def prepare_output_dirs(*paths):
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


# Menyaring data yang rusak/kosong agar tidak merusak proses latihan.
def validate_dataset(df, text_col, label_col):
    if text_col not in df.columns:
        raise ValueError(
            f"Text column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].copy()

    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()

    df = df.dropna()
    df = df[df[text_col] != ""]
    df = df[df[label_col] != ""]

    if len(df) < 10:
        raise ValueError(
            "Dataset is too small after cleaning. Minimum 10 rows required."
        )

    if df[label_col].nunique() < 2:
        raise ValueError("Dataset must contain at least two label classes.")

    return df


def build_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", FunctionTransformer(preprocess_batch)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def get_stratify_target(y):
    class_counts = y.value_counts()

    if len(class_counts) < 2:
        return None

    if class_counts.min() < 2:
        return None

    return y


def main():
    args = parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    prepare_output_dirs(args.model_out, args.metrics_out)

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    print("Validating dataset...")
    df = validate_dataset(df, args.text_col, args.label_col)

    X = df[args.text_col]
    y = df[args.label_col]

    stratify_target = get_stratify_target(y)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_target,
    )

    print("Building pipeline...")
    model = build_pipeline()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    model_version = os.getenv("GITHUB_SHA", "local")[:12]

    metrics = {
        "model_name": "fake-news-detection-tfidf-logistic-regression",
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data_path),
        "text_column": args.text_col,
        "label_column": args.label_col,
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "labels": sorted(y.unique().tolist()),
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "classification_report": report,
    }

    mlflow.set_experiment("fake-news-detection")

    with mlflow.start_run(run_name=model_version):
        mlflow.log_params(
            {
                "text_col": args.text_col,
                "label_col": args.label_col,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "tfidf_max_features": TFIDF_MAX_FEATURES,
                "tfidf_ngram_range": str(TFIDF_NGRAM_RANGE),
                "classifier": "LogisticRegression",
                "total_rows": int(len(df)),
                "train_rows": int(len(X_train)),
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": round(float(accuracy), 4),
                "precision_weighted": round(float(precision), 4),
                "recall_weighted": round(float(recall), 4),
                "f1_weighted": round(float(f1), 4),
            }
        )

        mlflow.set_tags(
            {
                "model_version": model_version,
                "dataset_path": str(data_path),
            }
        )

        mlflow.sklearn.log_model(model, "model")

        print("Saving model...")
        joblib.dump(model, args.model_out)

        print("Saving metrics...")
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        mlflow.log_artifact(args.metrics_out)

    print("Training completed successfully.")
    print(f"Model saved to: {args.model_out}")
    print(f"Metrics saved to: {args.metrics_out}")
    print(f"Weighted F1 score: {metrics['f1_weighted']}")


if __name__ == "__main__":
    main()
