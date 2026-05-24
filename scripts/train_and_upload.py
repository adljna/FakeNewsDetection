import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional

import joblib
import pandas as pd
from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)

    if required and not value:
        raise ValueError(f"Environment variable '{name}' wajib diisi.")

    return value


def parse_gcs_uri(gcs_uri: str):
    """
    Mengubah:
    gs://bucket-name/path/file.csv

    Menjadi:
    bucket-name, path/file.csv
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"GCS URI harus diawali dengan gs://, diterima: {gcs_uri}")

    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    if not bucket_name or not blob_name:
        raise ValueError(f"Format GCS URI tidak valid: {gcs_uri}")

    return bucket_name, blob_name


def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"Downloading from {gcs_uri} to {local_path}")
    blob.download_to_filename(local_path)

    return local_path


def upload_to_gcs(local_path: str, gcs_uri: str):
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"Uploading {local_path} to {gcs_uri}")
    blob.upload_from_filename(local_path)

    print(f"Upload success: {gcs_uri}")


def resolve_dataset_path(data_uri: str) -> str:
    """
    DATA_URI bisa berupa:
    1. Local path:
       data/train.csv

    2. GCS path:
       gs://fake-news-mlops-models/data/train.csv
    """
    if data_uri.startswith("gs://"):
        local_path = str(Path(tempfile.gettempdir()) / "train_data.csv")
        return download_from_gcs(data_uri, local_path)

    if not Path(data_uri).exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_uri}")

    return data_uri


def pick_column(
    df: pd.DataFrame,
    preferred_column: Optional[str],
    candidate_columns: list,
    column_description: str,
) -> str:
    """
    Memilih nama kolom berdasarkan:
    1. Env variable dari GitHub Actions
    2. Candidate default
    """
    if preferred_column and preferred_column in df.columns:
        return preferred_column

    for column in candidate_columns:
        if column in df.columns:
            return column

    raise ValueError(
        f"Kolom untuk {column_description} tidak ditemukan.\n"
        f"Kolom yang tersedia: {list(df.columns)}\n"
        f"Solusi: set TEXT_COL dan LABEL_COL di GitHub Variables sesuai nama kolom dataset."
    )


def normalize_label(label):
    """
    Menormalkan label supaya lebih konsisten.
    Tidak memaksa format tertentu, tapi membantu untuk kasus umum.
    """
    label_str = str(label).strip().lower()

    if label_str in ["fake", "false", "1", "hoax", "palsu"]:
        return "fake"

    if label_str in ["real", "true", "0", "valid", "asli"]:
        return "real"

    return label_str


def main():
    print("Starting training pipeline...")

    data_uri = get_env("DATA_URI", required=True)
    model_bucket = get_env("MODEL_BUCKET", required=True)
    model_prefix = get_env("MODEL_PREFIX", "models/fakenews")
    text_col_env = get_env("TEXT_COL")
    label_col_env = get_env("LABEL_COL")
    f1_threshold = float(get_env("F1_THRESHOLD", "0.80"))

    commit_sha = get_env("COMMIT_SHA", "local")
    short_sha = commit_sha[:7] if commit_sha else "local"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_version = get_env("MODEL_VERSION", f"{timestamp}-{short_sha}")

    print(f"DATA_URI: {data_uri}")
    print(f"MODEL_BUCKET: {model_bucket}")
    print(f"MODEL_PREFIX: {model_prefix}")
    print(f"MODEL_VERSION: {model_version}")
    print(f"F1_THRESHOLD: {f1_threshold}")

    dataset_path = resolve_dataset_path(data_uri)

    print(f"Reading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")

    text_col = pick_column(
        df=df,
        preferred_column=text_col_env,
        candidate_columns=[
            "text",
            "news",
            "content",
            "article",
            "statement",
            "title_text",
            "title",
        ],
        column_description="teks berita",
    )

    label_col = pick_column(
        df=df,
        preferred_column=label_col_env,
        candidate_columns=[
            "label",
            "class",
            "target",
            "truth",
            "fake",
            "category",
        ],
        column_description="label",
    )

    print(f"Using text column: {text_col}")
    print(f"Using label column: {label_col}")

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].apply(normalize_label)

    df = df[df[text_col].str.strip() != ""]
    df = df[df[label_col].str.strip() != ""]

    print(f"Cleaned dataset shape: {df.shape}")
    print("Label distribution:")
    print(df[label_col].value_counts())

    if len(df) < 10:
        raise ValueError("Dataset terlalu sedikit untuk training. Minimal butuh lebih banyak data.")

    if df[label_col].nunique() < 2:
        raise ValueError("Label hanya punya 1 kelas. Model klasifikasi butuh minimal 2 kelas.")

    X = df[text_col]
    y = df[label_col]

    min_class_count = y.value_counts().min()
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=5000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "model_version": model_version,
        "data_uri": data_uri,
        "text_column": text_col,
        "label_column": label_col,
        "row_count": int(len(df)),
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "f1_threshold": float(f1_threshold),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
        ),
    }

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    local_model_path = output_dir / "final_model.sav"
    local_metrics_path = output_dir / "metrics.json"

    print(f"Saving model to: {local_model_path}")
    joblib.dump(model, local_model_path)

    print(f"Saving metrics to: {local_metrics_path}")
    with open(local_metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    if f1 < f1_threshold:
        print("Model failed quality gate.")
        print(f"F1 score: {f1:.4f}")
        print(f"Threshold: {f1_threshold:.4f}")
        sys.exit(1)

    print("Model passed quality gate.")

    model_gcs_uri = f"gs://{model_bucket}/{model_prefix}/{model_version}/final_model.sav"
    metrics_gcs_uri = f"gs://{model_bucket}/{model_prefix}/{model_version}/metrics.json"

    upload_to_gcs(str(local_model_path), model_gcs_uri)
    upload_to_gcs(str(local_metrics_path), metrics_gcs_uri)

    Path("model_uri.txt").write_text(model_gcs_uri, encoding="utf-8")
    Path("model_version.txt").write_text(model_version, encoding="utf-8")

    print("Training pipeline completed successfully.")
    print(f"MODEL_GCS_URI={model_gcs_uri}")
    print(f"MODEL_VERSION={model_version}")


if __name__ == "__main__":
    main()