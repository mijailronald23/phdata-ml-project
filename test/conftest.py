import pytest
from fastapi.testclient import TestClient
import httpx
import os


@pytest.fixture
def test_client():
    """Fixture for unit tests using FastAPI TestClient."""
    from main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_base_url():
    """Fixture providing API base URL for integration tests."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture
def http_client(api_base_url):
    """Fixture for integration tests using httpx."""
    return httpx.Client(base_url=api_base_url, timeout=10.0)


@pytest.fixture
def sample_home_features():
    """Fixture providing sample input data for tests."""
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500.0,
        "sqft_lot": 5000.0,
        "floors": 1.0,
        "sqft_above": 1200.0,
        "sqft_basement": 300.0,
        "zipcode": "98042"
    }


@pytest.fixture
def unsupported_home_features(sample_home_features):
    payload = dict(sample_home_features)
    payload["zipcode"] = "99999"
    return payload


@pytest.fixture
def one_missing_home_features(sample_home_features):
    payload = dict(sample_home_features)
    payload["bathrooms"] = None
    return payload


@pytest.fixture
def multiple_missing_home_features(sample_home_features):
    payload = dict(sample_home_features)
    payload["bathrooms"] = None
    payload["sqft_lot"] = None
    payload["sqft_basement"] = None
    return payload


@pytest.fixture
def all_missing_home_features(sample_home_features):
    payload = {"zipcode": sample_home_features["zipcode"]}
    for key in sample_home_features:
        if key != "zipcode":
            payload[key] = None
    return payload
