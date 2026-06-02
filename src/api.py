import os
import re
import pickle
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template


BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

_model = None
_model_source = None


def reset_model_cache():
    global _model
    global _model_source

    _model = None
    _model_source = None


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    return text.lower().strip()


def parse_gcs_uri(gcs_uri):
    parsed = urlparse(gcs_uri)

    if parsed.scheme != "gs":
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    return bucket_name, blob_name


def download_model_from_gcs(gcs_uri, destination_path):
    from google.cloud import storage

    bucket_name, blob_name = parse_gcs_uri(gcs_uri)

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"Model artifact not found in GCS: {gcs_uri}")

    blob.download_to_filename(destination_path)

    return destination_path


def load_pickle_or_joblib(model_path):
    try:
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as file:
            return pickle.load(file)


def load_model():
    model_gcs_uri = os.getenv("MODEL_GCS_URI")
    model_local_path = os.getenv(
        "MODEL_LOCAL_PATH",
        str(BASE_DIR / "model" / "pipeline_v1.pkl")
    )

    if model_gcs_uri:
        local_cache_path = "/tmp/model/pipeline.pkl"
        print(f"Downloading model from Cloud Storage: {model_gcs_uri}")
        model_path = download_model_from_gcs(model_gcs_uri, local_cache_path)
        source = model_gcs_uri
    else:
        model_path = Path(model_local_path)
        source = str(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model not found: {model_path}. "
                "Set MODEL_GCS_URI for Cloud Run or MODEL_LOCAL_PATH for local testing."
            )

    print(f"Loading model from: {model_path}")
    model = load_pickle_or_joblib(model_path)

    return model, source


def get_model():
    global _model
    global _model_source

    if _model is None:
        _model, _model_source = load_model()

    return _model


def get_model_source():
    return _model_source


def is_fake_prediction(prediction):
    fake_values = {"0", "fake", "fake news", "false", "palsu"}
    return str(prediction).strip().lower() in fake_values


def get_class_probability(model, probabilities, target_values):
    if not hasattr(model, "classes_"):
        return None

    target_values = {str(value).strip().lower() for value in target_values}

    for index, class_value in enumerate(model.classes_):
        if str(class_value).strip().lower() in target_values:
            return float(probabilities[index]) * 100

    return None


def predict_news(news_text):
    model = get_model()
    clean_text = preprocess(news_text)

    prediction = model.predict([clean_text])[0]

    confidence = None
    prob_fake = None
    prob_real = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([clean_text])[0]
        confidence = float(np.max(probabilities)) * 100

        prob_fake = get_class_probability(
            model,
            probabilities,
            target_values={"0", "fake", "fake news", "false", "palsu"}
        )
        prob_real = get_class_probability(
            model,
            probabilities,
            target_values={"1", "real", "real news", "true", "asli"}
        )

        if prob_fake is None and len(probabilities) >= 1:
            prob_fake = float(probabilities[0]) * 100
        if prob_real is None and len(probabilities) >= 2:
            prob_real = float(probabilities[1]) * 100

    if is_fake_prediction(prediction):
        label = "FAKE NEWS (Berita Palsu)"
        is_fake = True
        confidence_value = prob_fake if prob_fake is not None else confidence
    else:
        label = "REAL NEWS (Berita Asli)"
        is_fake = False
        confidence_value = prob_real if prob_real is not None else confidence

    confidence_display = None
    if confidence_value is not None:
        confidence_display = f"{confidence_value:.2f}"

    return {
        "raw_prediction": str(prediction),
        "label": label,
        "confidence": confidence_display,
        "is_fake": is_fake,
        "model_version": os.getenv("MODEL_VERSION", "unknown"),
        "model_source": get_model_source(),
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    try:
        get_model()

        return jsonify(
            {
                "status": "healthy",
                "model_loaded": True,
                "model_source": get_model_source(),
                "model_version": os.getenv("MODEL_VERSION", "unknown"),
            }
        ), 200

    except Exception as error:
        return jsonify(
            {
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(error),
            }
        ), 503


@app.route("/predict", methods=["POST"])
def predict_web():
    message = request.form.get("news", "").strip()

    if not message:
        return render_template(
            "index.html",
            error="Masukkan teks berita terlebih dahulu.",
            news_text="",
        )

    result = predict_news(message)

    return render_template(
        "index.html",
        news_text=message,
        pred_label=result["label"],
        pred_confidence=result["confidence"],
        pred_is_fake=result["is_fake"],
    )


@app.route("/api/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify(
            {
                "error": "Invalid request. JSON body is required."
            }
        ), 400

    text = payload.get("text", "")

    if not str(text).strip():
        return jsonify(
            {
                "error": "Field 'text' is required and cannot be empty."
            }
        ), 400

    try:
        result = predict_news(text)

        return jsonify(
            {
                "prediction": result["raw_prediction"],
                "label": result["label"],
                "confidence": result["confidence"],
                "is_fake": result["is_fake"],
                "model_version": result["model_version"],
                "model_source": result["model_source"],
            }
        ), 200

    except Exception as error:
        return jsonify(
            {
                "error": "Prediction failed.",
                "detail": str(error),
            }
        ), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
