import pandas as pd
import joblib
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)

model = joblib.load("model/random_forest.joblib")


def test_model_prediction():
    features = pd.DataFrame([{
        "temps_reponse": 2.3,
        "nb_erreurs": 0,
        "score": 100
    }])

    prediction = model.predict(features)

    assert prediction[0] in [0, 1]


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={
            "temps_reponse": 2.3,
            "nb_erreurs": 0,
            "score": 100
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "satisfait" in data
    assert "confidence" in data

    assert data["satisfait"] in [0, 1]
    assert 0 <= data["confidence"] <= 1


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_requests" in data
    assert "successful_predictions" in data
    assert "errors" in data
    assert "average_response_time_ms" in data
def test_invalid_score():
    response = client.post(
        "/predict",
        json={
            "temps_reponse": 2.3,
            "nb_erreurs": 0,
            "score": 150
        }
    )

    assert response.status_code == 422


def test_negative_errors():
    response = client.post(
        "/predict",
        json={
            "temps_reponse": 2.3,
            "nb_erreurs": -1,
            "score": 80
        }
    )

    assert response.status_code == 422


def test_negative_response_time():
    response = client.post(
        "/predict",
        json={
            "temps_reponse": -2,
            "nb_erreurs": 0,
            "score": 80
        }
    )

    assert response.status_code == 422