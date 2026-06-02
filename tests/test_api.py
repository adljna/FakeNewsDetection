import joblib
import pytest

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
