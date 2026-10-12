import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import create_app
from services.inference import InferenceService
from services.inference import REQUEST_PROPERTY_FEATURES
from utils.loader import ArtifactPaths
from utils.loader import ResourceInitializationError


def phase2_predict(home_features):
    artifact_paths = ArtifactPaths()

    with artifact_paths.model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    with artifact_paths.features_path.open(encoding="utf-8") as features_file:
        model_features = json.load(features_file)

    input_data = pd.DataFrame([home_features])
    demographics = pd.read_csv(artifact_paths.demographics_path, dtype={"zipcode": str})
    demographic_info = demographics[
        demographics["zipcode"] == home_features["zipcode"]
    ].drop(columns="zipcode").reset_index(drop=True)
    input_data = pd.concat([input_data, demographic_info], axis=1)
    input_data = input_data[model_features]
    prediction = model.predict(input_data)
    return float(prediction[0])


def test_inference_service_initializes_expected_contract(sample_home_features):
    service = InferenceService.from_artifacts()

    assert len(service.model_features) == 33
    assert service.model_features[: len(REQUEST_PROPERTY_FEATURES)] == list(
        REQUEST_PROPERTY_FEATURES
    )
    assert sample_home_features["zipcode"] in service.supported_zipcodes


def test_inference_service_prepare_features_matches_model_order(sample_home_features):
    service = InferenceService.from_artifacts()

    prepared = service.prepare_features(sample_home_features)

    assert list(prepared.model_input.columns) == service.model_features
    assert prepared.model_input.shape == (1, 33)
    assert prepared.demographics
    assert prepared.used_imputation is False


def test_inference_service_loads_imputer_artifact_metadata():
    service = InferenceService.from_artifacts()

    assert service.imputer_metadata["n_neighbors"] == 5
    assert service.imputer_metadata["weights"] == "distance"
    assert service.imputer_metadata["scaled_distances"] is True
    assert service.imputer_metadata["reference_row_count"] > 0


def test_inference_service_preserves_phase2_prediction(sample_home_features):
    service = InferenceService.from_artifacts()

    optimized_prediction = service.predict(sample_home_features)
    legacy_prediction = phase2_predict(sample_home_features)

    assert optimized_prediction == pytest.approx(legacy_prediction)


def test_inference_service_imputes_one_missing_value(one_missing_home_features):
    service = InferenceService.from_artifacts()

    prepared = service.prepare_features(one_missing_home_features)

    assert prepared.used_imputation is True
    assert prepared.missing_property_features == ("bathrooms",)
    assert not prepared.model_input.isnull().any().any()


def test_inference_service_imputes_multiple_missing_values(
    multiple_missing_home_features,
):
    service = InferenceService.from_artifacts()

    prepared = service.prepare_features(multiple_missing_home_features)

    assert prepared.used_imputation is True
    assert set(prepared.missing_property_features) == {
        "bathrooms",
        "sqft_lot",
        "sqft_basement",
    }
    assert not prepared.model_input.isnull().any().any()


def test_inference_service_imputes_all_missing_property_features(
    all_missing_home_features,
):
    service = InferenceService.from_artifacts()

    prepared = service.prepare_features(all_missing_home_features)

    assert prepared.used_imputation is True
    assert set(prepared.missing_property_features) == set(REQUEST_PROPERTY_FEATURES)
    assert not prepared.model_input.isnull().any().any()
    assert prepared.model_input.shape == (1, 33)


def test_inference_service_repeated_inference_is_stable(one_missing_home_features):
    service = InferenceService.from_artifacts()

    first_prediction = service.predict(one_missing_home_features)
    second_prediction = service.predict(one_missing_home_features)

    assert first_prediction == pytest.approx(second_prediction)


def test_application_fails_fast_when_artifacts_are_missing(tmp_path):
    missing_paths = ArtifactPaths(
        model_path=tmp_path / "missing-model.pkl",
        features_path=Path("model/model_features.json"),
        demographics_path=Path("data/zipcode_demographics.csv"),
    )

    with pytest.raises(ResourceInitializationError):
        with TestClient(create_app(artifact_paths=missing_paths)):
            pass
