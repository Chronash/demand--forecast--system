from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "store_id": 1,
    "region": "Москва",
    "store_type": "Супермаркет",
    "month": 7,
    "weekday": 5,
    "day_of_year": 200,
    "is_weekend": True,
    "is_holiday": False,
    "promo_flag": False,
    "temperature": 28.5,
    "price": 58.0,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_200():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_required_fields():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    assert "predicted_demand" in data
    assert "predicted_revenue" in data
    assert "model_name" in data
    assert "model_version" in data


def test_predict_values_are_valid():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    assert isinstance(data["predicted_demand"], int)
    assert data["predicted_demand"] >= 0
    assert isinstance(data["predicted_revenue"], (int, float))
    assert data["predicted_revenue"] >= 0


def test_predict_invalid_region():
    invalid_payload = {**VALID_PAYLOAD, "region": "НесуществующийРегион"}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422