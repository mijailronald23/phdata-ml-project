import pytest


def test_health_endpoint(test_client):
    """Test the /health endpoint returns correct status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


def test_ready_endpoint(test_client):
    """Test the /ready endpoint reflects initialized inference resources."""
    response = test_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_predict_endpoint_valid_input(test_client, sample_home_features):
    """Test the /predict endpoint with valid input."""
    response = test_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)
    assert response_data["predicted_price"] > 0


def test_predict_endpoint_one_missing_numeric_value(
    test_client, one_missing_home_features
):
    response = test_client.post("/predict", json=one_missing_home_features)

    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_predict_endpoint_omitted_numeric_value(test_client, sample_home_features):
    payload = dict(sample_home_features)
    payload.pop("bathrooms")

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_predict_endpoint_multiple_missing_numeric_values(
    test_client, multiple_missing_home_features
):
    response = test_client.post("/predict", json=multiple_missing_home_features)

    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_predict_endpoint_all_numeric_values_missing(
    test_client, all_missing_home_features
):
    response = test_client.post("/predict", json=all_missing_home_features)

    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_predict_endpoint_rejects_missing_zipcode(test_client, sample_home_features):
    payload = dict(sample_home_features)
    payload.pop("zipcode")

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_zipcode_format(
    test_client, sample_home_features
):
    payload = dict(sample_home_features)
    payload["zipcode"] = "98A42"

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_unsupported_zipcode(
    test_client, unsupported_home_features
):
    response = test_client.post("/predict", json=unsupported_home_features)

    assert response.status_code == 422
    assert "Unsupported zipcode" in response.json()["detail"]


def test_predict_endpoint_rejects_invalid_numeric_domain(
    test_client, sample_home_features
):
    payload = dict(sample_home_features)
    payload["sqft_living"] = -1

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 422
