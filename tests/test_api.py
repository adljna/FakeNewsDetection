from src.api import app

client = app.test_client()


def test_health():

    response = client.get('/health')

    assert response.status_code == 200


def test_predict():

    response = client.post(
        '/api/predict',
        json={
            "text": "Aliens built the pyramids and secretly control the world"
        }
    )

    assert response.status_code == 200


def test_empty_input():

    response = client.post(
        '/api/predict',
        json={}
    )

    assert response.status_code == 400