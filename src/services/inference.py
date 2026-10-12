from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from preprocessing.imputation import ImputationArtifactError
from preprocessing.imputation import transform_with_imputer_artifact
from preprocessing.imputation import validate_imputer_artifact
from utils.loader import ArtifactPaths
from utils.loader import ResourceInitializationError
from utils.loader import load_demographics
from utils.loader import load_feature_names
from utils.loader import load_imputer_artifact
from utils.loader import load_model


REQUEST_PROPERTY_FEATURES: Sequence[str] = (
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
)
EXPECTED_MODEL_FEATURE_COUNT = 33


class InferenceClientError(ValueError):
    """Raised when a client request cannot be served as submitted."""


class UnsupportedZipcodeError(InferenceClientError):
    """Raised when a zipcode is syntactically valid but unsupported."""


class InferenceServiceError(RuntimeError):
    """Raised when serving fails after resources were initialized."""


@dataclass(frozen=True)
class PreparedFeatures:
    """Prepared model input plus the resolved zipcode demographics."""

    model_input: pd.DataFrame
    demographics: Mapping[str, float]
    used_imputation: bool
    missing_property_features: Sequence[str]


class InferenceService:
    """Loads inference resources once and serves predictions from memory."""

    def __init__(
        self,
        model: Any,
        model_features: Sequence[str],
        imputer_artifact: Mapping[str, Any],
        demographics_by_zipcode: Mapping[str, Mapping[str, float]],
    ) -> None:
        self._model = model
        self._model_features = list(model_features)
        self._imputer_artifact = dict(imputer_artifact)
        self._demographics_by_zipcode = {
            zipcode: dict(values) for zipcode, values in demographics_by_zipcode.items()
        }
        self._supported_zipcodes = frozenset(self._demographics_by_zipcode.keys())
        self._demographic_features = [
            feature
            for feature in self._model_features
            if feature not in REQUEST_PROPERTY_FEATURES
        ]

        self._validate_contract()
        self._validate_model_compatibility()

    @classmethod
    def from_artifacts(cls, artifact_paths: ArtifactPaths | None = None) -> "InferenceService":
        paths = artifact_paths or ArtifactPaths()
        model = load_model(paths.model_path)
        model_features = load_feature_names(paths.features_path)
        imputer_artifact = load_imputer_artifact(paths.imputer_path)
        demographics_by_zipcode = load_demographics(paths.demographics_path)
        return cls(
            model=model,
            model_features=model_features,
            imputer_artifact=imputer_artifact,
            demographics_by_zipcode=demographics_by_zipcode,
        )

    @property
    def model_features(self) -> List[str]:
        return list(self._model_features)

    @property
    def supported_zipcodes(self) -> Sequence[str]:
        return tuple(sorted(self._supported_zipcodes))

    @property
    def imputer_metadata(self) -> Mapping[str, Any]:
        return {
            "n_neighbors": self._imputer_artifact["n_neighbors"],
            "weights": self._imputer_artifact["weights"],
            "scaled_distances": self._imputer_artifact["scaled_distances"],
            "reference_row_count": self._imputer_artifact["reference_row_count"],
        }

    def prepare_features(self, home_features: Mapping[str, Any]) -> PreparedFeatures:
        zipcode = str(home_features["zipcode"])
        demographics = self._demographics_by_zipcode.get(zipcode)
        if demographics is None:
            raise UnsupportedZipcodeError(
                f"Unsupported zipcode '{zipcode}'. Supported zipcodes are limited "
                "to those present in the demographics dataset."
            )

        combined_features: Dict[str, Any] = {
            feature: home_features.get(feature) for feature in REQUEST_PROPERTY_FEATURES
        }
        combined_features.update(demographics)
        missing_property_features = [
            feature
            for feature in REQUEST_PROPERTY_FEATURES
            if combined_features.get(feature) is None
        ]

        missing_features = [
            feature for feature in self._model_features if feature not in combined_features
        ]
        if missing_features:
            raise InferenceServiceError(
                "Prepared model input is missing required features: "
                + ", ".join(missing_features)
            )

        ordered_features = {
            feature: combined_features[feature] for feature in self._model_features
        }
        raw_model_input = pd.DataFrame([ordered_features], columns=self._model_features)
        model_input = raw_model_input
        used_imputation = False

        if missing_property_features:
            try:
                model_input = transform_with_imputer_artifact(
                    self._imputer_artifact,
                    raw_model_input,
                )
            except ImputationArtifactError as exc:
                raise InferenceServiceError("Failed to apply KNN imputation.") from exc
            used_imputation = True

        if model_input.isnull().any().any():
            raise InferenceServiceError(
                "Prepared model input contains null values after preprocessing."
            )

        return PreparedFeatures(
            model_input=model_input,
            demographics=demographics,
            used_imputation=used_imputation,
            missing_property_features=tuple(missing_property_features),
        )

    def predict(self, home_features: Mapping[str, Any]) -> float:
        prepared = self.prepare_features(home_features)
        try:
            prediction = self._model.predict(prepared.model_input)
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise InferenceServiceError("Model prediction failed.") from exc

        return float(prediction[0])

    def _validate_contract(self) -> None:
        if not hasattr(self._model, "predict"):
            raise ResourceInitializationError(
                "Loaded model artifact does not expose a predict() method."
            )

        if len(self._model_features) != EXPECTED_MODEL_FEATURE_COUNT:
            raise ResourceInitializationError(
                f"Expected exactly {EXPECTED_MODEL_FEATURE_COUNT} model features, "
                f"found {len(self._model_features)}."
            )

        property_features = [
            feature
            for feature in self._model_features
            if feature in REQUEST_PROPERTY_FEATURES
        ]
        if list(property_features) != list(REQUEST_PROPERTY_FEATURES):
            raise ResourceInitializationError(
                "Model feature contract is incompatible with the public request schema."
            )

        if not self._supported_zipcodes:
            raise ResourceInitializationError("Demographics dataset is empty.")

        if len(self._demographic_features) != (
            EXPECTED_MODEL_FEATURE_COUNT - len(REQUEST_PROPERTY_FEATURES)
        ):
            raise ResourceInitializationError(
                "Demographic feature count is incompatible with the trained model."
            )

        try:
            validate_imputer_artifact(
                self._imputer_artifact,
                feature_names=self._model_features,
                property_features=REQUEST_PROPERTY_FEATURES,
            )
        except ImputationArtifactError as exc:
            raise ResourceInitializationError(
                "Imputation artifact is incompatible with the serving contract."
            ) from exc

        missing_zipcode_rows = [
            zipcode
            for zipcode, values in self._demographics_by_zipcode.items()
            if any(feature not in values for feature in self._demographic_features)
        ]
        if missing_zipcode_rows:
            raise ResourceInitializationError(
                "Demographics dataset is missing required model features for one or "
                "more zipcodes."
            )

    def _validate_model_compatibility(self) -> None:
        if hasattr(self._model, "feature_names_in_"):
            feature_names_in = list(self._model.feature_names_in_)
            if feature_names_in != self._model_features:
                raise ResourceInitializationError(
                    "Serialized model feature_names_in_ does not match "
                    "model_features.json."
                )

        sample_zipcode = next(iter(self._supported_zipcodes))
        sample_home = {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 1500.0,
            "sqft_lot": 5000.0,
            "floors": 1.0,
            "sqft_above": 1200.0,
            "sqft_basement": 300.0,
            "zipcode": sample_zipcode,
        }
        prepared = self.prepare_features(sample_home)
        try:
            self._model.predict(prepared.model_input)
        except Exception as exc:
            raise ResourceInitializationError(
                "Model artifact and feature schema are incompatible."
            ) from exc

        missing_sample_home = {
            "bedrooms": None,
            "bathrooms": None,
            "sqft_living": None,
            "sqft_lot": None,
            "floors": None,
            "sqft_above": None,
            "sqft_basement": None,
            "zipcode": sample_zipcode,
        }
        missing_prepared = self.prepare_features(missing_sample_home)
        if missing_prepared.model_input.isnull().any().any():
            raise ResourceInitializationError(
                "Imputation artifact did not resolve missing property features."
            )
