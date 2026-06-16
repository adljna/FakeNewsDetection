import pickle
import pytest
import numpy as np
import joblib

import src.api as api
from src.api import app

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@pytest.fixture()
def client(tmp_path, monkeypatch):
    model_path = tmp_path / "pipeline.pkl"

    dummy_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(lowercase=True, max_features=100),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    X_train = [
        "government confirms official policy update",
        "scientists publish verified research findings",
        "shocking secret celebrity claim without proof",
        "fake miracle cure spreads online",
    ]

    y_train = ["REAL", "REAL", "FAKE", "FAKE"]

    dummy_model.fit(X_train, y_train)
    joblib.dump(dummy_model, model_path)

    monkeypatch.setenv("MODEL_LOCAL_PATH", str(model_path))
    monkeypatch.delenv("MODEL_GCS_URI", raising=False)
    monkeypatch.setenv("MODEL_VERSION", "test-version")

    from src import api

    api.reset_model_cache()

    with api.app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.get_json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-version"


def test_api_predict_endpoint_success(client):
    response = client.post(
        "/api/predict", json={"text": "scientists publish verified research findings"}
    )

    assert response.status_code == 200

    body = response.get_json()

    assert "prediction" in body
    assert body["prediction"] in ["REAL", "FAKE"]
    assert "confidence" in body
    assert body["model_version"] == "test-version"


def test_api_predict_endpoint_empty_text(client):
    response = client.post("/api/predict", json={"text": ""})

    assert response.status_code == 400

    body = response.get_json()

    assert "error" in body


def test_api_predict_endpoint_invalid_json(client):
    response = client.post("/api/predict", data="not json", content_type="text/plain")

    assert response.status_code == 400

    body = response.get_json()

    assert "error" in body


def test_preprocess_removes_symbols_and_lowercase():
    result = api.preprocess("BREAKING NEWS!!! 123 Fake-News.")

    assert result == "breaking news  fakenews"


def test_validate_news_text_too_short():
    error = api.validate_news_text("short")

    assert error == "Teks berita terlalu pendek. Minimal 20 karakter."


def test_validate_news_text_less_than_five_words():
    error = api.validate_news_text("abcdefghij klmnopqrst uvwxyzabcd efghijklmn")

    assert error == "Teks berita harus mengandung minimal 5 kata."


def test_validate_news_text_without_letters():
    error = api.validate_news_text("12345 67890 11111 22222 33333")

    assert error == "Teks berita harus mengandung huruf."


def test_validate_news_text_repeated_characters():
    error = api.validate_news_text("aaaa aaaa aaaa aaaa aaaa")

    assert error == "Input tidak valid. Masukkan teks berita yang bermakna."


def test_validate_news_text_too_long():
    long_text = "word " * 2500

    error = api.validate_news_text(long_text)

    assert error == "Teks berita terlalu panjang. Maksimal 10000 karakter."


def test_validate_news_text_valid_input():
    error = api.validate_news_text(
        "This is a meaningful news text with enough words for testing."
    )

    assert error is None


def test_parse_gcs_uri_valid():
    bucket_name, blob_name = api.parse_gcs_uri(
        "gs://fake-news-mlops-models/models/production/latest/pipeline.pkl"
    )

    assert bucket_name == "fake-news-mlops-models"
    assert blob_name == "models/production/latest/pipeline.pkl"


def test_parse_gcs_uri_invalid_scheme():
    with pytest.raises(ValueError):
        api.parse_gcs_uri("http://fake-news-mlops-models/model.pkl")


def test_parse_gcs_uri_missing_blob():
    with pytest.raises(ValueError):
        api.parse_gcs_uri("gs://fake-news-mlops-models")


def test_is_fake_prediction_true_values():
    assert api.is_fake_prediction("fake") is True
    assert api.is_fake_prediction("FAKE NEWS") is True
    assert api.is_fake_prediction("palsu") is True


def test_is_fake_prediction_false_values():
    assert api.is_fake_prediction("real") is False
    assert api.is_fake_prediction("true") is False
    assert api.is_fake_prediction("asli") is False


def test_get_class_probability_found():
    class FakeModel:
        classes_ = ["fake", "real"]

    probabilities = np.array([0.8, 0.2])

    result = api.get_class_probability(
        FakeModel(),
        probabilities,
        target_values={"fake"},
    )

    assert result == 80.0


def test_get_class_probability_without_classes():
    class FakeModel:
        pass

    probabilities = np.array([0.8, 0.2])

    result = api.get_class_probability(
        FakeModel(),
        probabilities,
        target_values={"fake"},
    )

    assert result is None


def test_load_pickle_or_joblib_fallback(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pkl"
    expected_model = {"name": "fallback-model"}

    with open(model_path, "wb") as file:
        pickle.dump(expected_model, file)

    def fake_joblib_load(path):
        raise Exception("joblib failed")

    monkeypatch.setattr(api.joblib, "load", fake_joblib_load)

    result = api.load_pickle_or_joblib(model_path)

    assert result == expected_model


def test_predict_api_model_error(client, monkeypatch):
    class ErrorModel:
        def predict(self, texts):
            raise RuntimeError("model prediction error")

    monkeypatch.setattr(api, "_model", ErrorModel())
    monkeypatch.setattr(api, "_model_source", "test-model")

    response = client.post(
        "/api/predict",
        json={"text": "This is a valid news text for testing model error handling."},
    )

    assert response.status_code == 500

    data = response.get_json()
    assert data["error"] == "Prediction failed."
    assert "model prediction error" in data["detail"]


def test_health_unhealthy_when_model_missing(client, monkeypatch):
    api.reset_model_cache()

    monkeypatch.delenv("MODEL_GCS_URI", raising=False)
    monkeypatch.setenv("MODEL_LOCAL_PATH", "model/not-found.pkl")

    response = client.get("/health")

    assert response.status_code == 503

    data = response.get_json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False

    api.reset_model_cache()
