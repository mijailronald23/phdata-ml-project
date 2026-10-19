from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


KNN_IMPUTER_NEIGHBORS = 5
KNN_IMPUTER_WEIGHTS = "distance"


class ImputationArtifactError(RuntimeError):
    """Raised when the imputation artifact is malformed or incompatible."""


def fit_knn_imputer_artifact(
    reference_frame: pd.DataFrame,
    feature_names: Sequence[str],
    property_features: Sequence[str],
) -> dict[str, Any]:
    """Fit and package the offline KNN imputation artifact.

    Args:
        reference_frame: Training-reference feature matrix used to fit preprocessing.
        feature_names: Ordered internal feature names expected at serving time.
        property_features: Public request fields that may be missing.

    Returns:
        Serializable artifact containing the scaler, imputer, and metadata.
    """
    _validate_feature_frame(reference_frame, feature_names)

    # Robust scaling keeps large-magnitude demographic and area features from
    # dominating KNN distances during imputation.
    scaler = RobustScaler()
    scaled_reference = scaler.fit_transform(reference_frame.loc[:, feature_names])

    imputer = KNNImputer(
        n_neighbors=KNN_IMPUTER_NEIGHBORS,
        weights=KNN_IMPUTER_WEIGHTS,
    )
    imputer.fit(scaled_reference)

    return {
        "feature_names": list(feature_names),
        "property_features": list(property_features),
        "scaler": scaler,
        "imputer": imputer,
        "n_neighbors": KNN_IMPUTER_NEIGHBORS,
        "weights": KNN_IMPUTER_WEIGHTS,
        "scaled_distances": True,
        "reference_row_count": int(reference_frame.shape[0]),
    }


def validate_imputer_artifact(
    artifact: Mapping[str, Any],
    feature_names: Sequence[str],
    property_features: Sequence[str],
) -> None:
    """Validate that an imputation artifact matches the serving contract.

    Args:
        artifact: Deserialized imputation artifact.
        feature_names: Ordered internal feature names required by the model.
        property_features: Public request fields that may be missing.

    Raises:
        ImputationArtifactError: If the artifact is missing required metadata or
            is incompatible with the serving contract.
    """
    required_keys = {
        "feature_names",
        "property_features",
        "scaler",
        "imputer",
        "n_neighbors",
        "weights",
        "scaled_distances",
        "reference_row_count",
    }
    missing_keys = sorted(required_keys.difference(artifact.keys()))
    if missing_keys:
        raise ImputationArtifactError(
            "Imputation artifact is missing required keys: "
            + ", ".join(missing_keys)
        )

    if list(artifact["feature_names"]) != list(feature_names):
        raise ImputationArtifactError(
            "Imputation artifact feature_names do not match model_features.json."
        )

    if list(artifact["property_features"]) != list(property_features):
        raise ImputationArtifactError(
            "Imputation artifact property_features do not match the public request schema."
        )

    if artifact["n_neighbors"] != KNN_IMPUTER_NEIGHBORS:
        raise ImputationArtifactError("Imputation artifact n_neighbors is incompatible.")

    if artifact["weights"] != KNN_IMPUTER_WEIGHTS:
        raise ImputationArtifactError("Imputation artifact weights are incompatible.")

    if not artifact["scaled_distances"]:
        raise ImputationArtifactError(
            "Imputation artifact must use scaled distances for KNN consistency."
        )

    scaler = artifact["scaler"]
    imputer = artifact["imputer"]
    if not hasattr(scaler, "transform") or not hasattr(scaler, "inverse_transform"):
        raise ImputationArtifactError("Imputation artifact scaler is invalid.")
    if not hasattr(imputer, "transform"):
        raise ImputationArtifactError("Imputation artifact imputer is invalid.")


def transform_with_imputer_artifact(
    artifact: Mapping[str, Any],
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Impute missing values in a feature frame using a fitted artifact.

    Args:
        artifact: Validated imputation artifact.
        features: Feature frame aligned to the model feature contract.

    Returns:
        Feature frame with missing values imputed and observed values preserved.
    """
    feature_names = list(artifact["feature_names"])
    _validate_feature_frame(features, feature_names)

    feature_frame = features.loc[:, feature_names].copy()
    missing_mask = feature_frame.isna()
    if not missing_mask.any().any():
        return feature_frame

    scaler = artifact["scaler"]
    imputer = artifact["imputer"]

    scaled_feature_array = scaler.transform(feature_frame)
    imputed_scaled_array = imputer.transform(scaled_feature_array)
    imputed_feature_frame = pd.DataFrame(
        scaler.inverse_transform(imputed_scaled_array),
        columns=feature_names,
        index=feature_frame.index,
    )
    return imputed_feature_frame.where(missing_mask, feature_frame)


def transform_array_with_imputer_artifact(
    artifact: Mapping[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    """Impute missing values in a feature array using a fitted artifact.

    Args:
        artifact: Validated imputation artifact.
        features: Two-dimensional feature array aligned to the model contract.

    Returns:
        Array with missing values imputed and observed values preserved.

    Raises:
        ImputationArtifactError: If the feature array shape is incompatible.
    """
    feature_names = list(artifact["feature_names"])
    feature_count = len(feature_names)
    if features.ndim != 2 or features.shape[1] != feature_count:
        raise ImputationArtifactError(
            "Feature array must be 2-dimensional and match the artifact feature count."
        )

    if not np.isnan(features).any():
        return features

    scaler = artifact["scaler"]
    imputer = artifact["imputer"]

    # The imputer is fit offline on reference training features only; serving
    # requests must reuse that artifact rather than refitting on live traffic.
    feature_frame = pd.DataFrame(features, columns=feature_names)
    scaled_array = scaler.transform(feature_frame)
    imputed_scaled = imputer.transform(scaled_array)
    imputed_array = scaler.inverse_transform(imputed_scaled)
    return np.where(np.isnan(features), imputed_array, features)


def _validate_feature_frame(
    features: pd.DataFrame,
    feature_names: Sequence[str],
) -> None:
    """Ensure a feature frame contains every required model column."""
    missing_columns = [feature for feature in feature_names if feature not in features.columns]
    if missing_columns:
        raise ImputationArtifactError(
            "Feature frame is missing required columns: " + ", ".join(missing_columns)
        )
