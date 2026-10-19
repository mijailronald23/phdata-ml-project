from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from preprocessing.imputation import ImputationArtifactError
from preprocessing.imputation import transform_array_with_imputer_artifact
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
    """Prepared model input and serving metadata for one request."""

    model_input: pd.DataFrame
    demographics: Mapping[str, float]
    used_imputation: bool
    missing_property_features: Sequence[str]


class InferenceService:
    """Load inference artifacts once and serve predictions from memory."""

    def __init__(
        self,
        model: Any,
        model_features: Sequence[str],
        imputer_artifact: Mapping[str, Any],
        demographics_by_zipcode: Mapping[str, Mapping[str, float]],
    ) -> None:
        """Initialize the in-memory inference service.

        Args:
            model: Deserialized prediction model with a ``predict`` method.
            model_features: Ordered internal feature names required by the model.
            imputer_artifact: Offline-fitted KNN imputation artifact.
            demographics_by_zipcode: Zipcode-indexed demographic lookup table.
        """
        self._model = model
        self._model_features = list(model_features)
        self._imputer_artifact = dict(imputer_artifact)
        self._demographics_by_zipcode = {
            zipcode: dict(values) for zipcode, values in demographics_by_zipcode.items()
        }
        self._supported_zipcodes = frozenset(self._demographics_by_zipcode.keys())
        self._feature_positions = {
            feature: idx for idx, feature in enumerate(self._model_features)
        }
        self._property_feature_positions = [
            self._feature_positions[feature] for feature in REQUEST_PROPERTY_FEATURES
        ]
        self._demographic_features = [
            feature
            for feature in self._model_features
            if feature not in REQUEST_PROPERTY_FEATURES
        ]
        self._demographic_arrays_by_zipcode = self._build_demographic_arrays()

        self._validate_contract()
        self._validate_model_compatibility()

    @classmethod
    def from_artifacts(cls, artifact_paths: ArtifactPaths | None = None) -> "InferenceService":
        """Build an inference service from serialized project artifacts.

        Args:
            artifact_paths: Optional paths overriding default artifact locations.

        Returns:
            Initialized inference service ready to serve predictions.

        Raises:
            ResourceInitializationError: If any required artifact is missing or invalid.
        """
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
        """Return the ordered internal model feature contract."""
        return list(self._model_features)

    @property
    def supported_zipcodes(self) -> Sequence[str]:
        """Return supported zipcodes available in the demographics dataset."""
        return tuple(sorted(self._supported_zipcodes))

    @property
    def imputer_metadata(self) -> Mapping[str, Any]:
        """Return non-sensitive metadata describing the imputation artifact."""
        return {
            "n_neighbors": self._imputer_artifact["n_neighbors"],
            "weights": self._imputer_artifact["weights"],
            "scaled_distances": self._imputer_artifact["scaled_distances"],
            "reference_row_count": self._imputer_artifact["reference_row_count"],
        }

    def prepare_features(self, home_features: Mapping[str, Any]) -> PreparedFeatures:
        """Prepare the ordered model input from validated property data.

        Enriches the request with zipcode demographics and applies missing-value
        imputation only when at least one property field is missing.

        Args:
            home_features: Validated property features and mandatory zipcode.

        Returns:
            PreparedFeatures containing the ordered model input.

        Raises:
            UnsupportedZipcodeError: If demographic data is unavailable.
            InferenceServiceError: If the feature contract cannot be satisfied.
        """
        zipcode = str(home_features["zipcode"])
        zipcode_demographics = self._demographics_by_zipcode.get(zipcode)
        zipcode_demographic_array = self._demographic_arrays_by_zipcode.get(zipcode)
        if zipcode_demographics is None:
            raise UnsupportedZipcodeError(
                f"Unsupported zipcode '{zipcode}'. Supported zipcodes are limited "
                "to those present in the demographics dataset."
            )

        missing_property_features = [
            feature
            for feature in REQUEST_PROPERTY_FEATURES
            if home_features.get(feature) is None
        ]

        if zipcode_demographic_array is None:
            raise InferenceServiceError("Demographic lookup array is not initialized.")

        # Demographics are resolved before imputation so both the KNN artifact and
        # the model see the same 33-feature representation at serving time.
        ordered_feature_array = zipcode_demographic_array.copy()
        for idx, feature in zip(
            self._property_feature_positions,
            REQUEST_PROPERTY_FEATURES,
        ):
            value = home_features.get(feature)
            if value is not None:
                ordered_feature_array[idx] = float(value)

        ordered_feature_array = ordered_feature_array.reshape(1, -1)
        used_imputation = False

        if missing_property_features:
            try:
                ordered_feature_array = transform_array_with_imputer_artifact(
                    self._imputer_artifact,
                    ordered_feature_array,
                )
            except ImputationArtifactError as exc:
                raise InferenceServiceError("Failed to apply KNN imputation.") from exc
            used_imputation = True
        # Complete requests bypass imputation so the original no-missing prediction
        # path stays behaviorally identical to the earlier optimized service.

        if np.isnan(ordered_feature_array).any():
            raise InferenceServiceError(
                "Prepared model input contains null values after preprocessing."
            )

        model_input = pd.DataFrame(ordered_feature_array, columns=self._model_features)

        return PreparedFeatures(
            model_input=model_input,
            demographics=zipcode_demographics,
            used_imputation=used_imputation,
            missing_property_features=tuple(missing_property_features),
        )

    def predict(self, home_features: Mapping[str, Any]) -> float:
        """Predict a home price from validated property features.

        Args:
            home_features: Validated property features and mandatory zipcode.

        Returns:
            Predicted price as a native float.

        Raises:
            UnsupportedZipcodeError: If the zipcode is unsupported.
            InferenceServiceError: If preprocessing or model prediction fails.
        """
        prepared = self.prepare_features(home_features)
        try:
            prediction_result = self._model.predict(prepared.model_input)
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise InferenceServiceError("Model prediction failed.") from exc

        return float(prediction_result[0])

    def _validate_contract(self) -> None:
        """Validate artifact compatibility with the serving contract."""
        if not hasattr(self._model, "predict"):
            raise ResourceInitializationError(
                "Loaded model artifact does not expose a predict() method."
            )

        # The trained model was serialized against this exact 33-column feature
        # order, so startup validation rejects any contract drift immediately.
        if len(self._model_features) != EXPECTED_MODEL_FEATURE_COUNT:
            raise ResourceInitializationError(
                f"Expected exactly {EXPECTED_MODEL_FEATURE_COUNT} model features, "
                f"found {len(self._model_features)}."
            )

        model_property_features = [
            feature
            for feature in self._model_features
            if feature in REQUEST_PROPERTY_FEATURES
        ]
        if list(model_property_features) != list(REQUEST_PROPERTY_FEATURES):
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

        zipcodes_with_missing_features = [
            zipcode
            for zipcode, values in self._demographics_by_zipcode.items()
            if any(feature not in values for feature in self._demographic_features)
        ]
        if zipcodes_with_missing_features:
            raise ResourceInitializationError(
                "Demographics dataset is missing required model features for one or "
                "more zipcodes."
            )

    def _validate_model_compatibility(self) -> None:
        """Run a lightweight inference check against the loaded artifacts."""
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

    def _build_demographic_arrays(self) -> Dict[str, np.ndarray]:
        """Precompute zipcode demographic arrays aligned to model feature order."""
        demographic_arrays: Dict[str, np.ndarray] = {}
        for zipcode, zipcode_demographics in self._demographics_by_zipcode.items():
            demographic_array = np.full(len(self._model_features), np.nan, dtype=float)
            for feature in self._demographic_features:
                demographic_array[self._feature_positions[feature]] = float(
                    zipcode_demographics[feature]
                )
            demographic_arrays[zipcode] = demographic_array
        return demographic_arrays
