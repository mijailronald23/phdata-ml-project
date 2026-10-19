from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd


class ResourceInitializationError(RuntimeError):
    """Raised when required serving artifacts are missing or incompatible."""


def _resolve_default_path(*candidates: str) -> Path:
    """Return the first existing artifact path from the provided candidates."""
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    return Path(candidates[0])


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved filesystem locations for serving artifacts."""

    model_path: Path = field(
        default_factory=lambda: _resolve_default_path(
            "model/model.pkl",
            "src/model/model.pkl",
        )
    )
    features_path: Path = field(
        default_factory=lambda: _resolve_default_path(
            "model/model_features.json",
            "src/model/model_features.json",
        )
    )
    imputer_path: Path = field(
        default_factory=lambda: _resolve_default_path(
            "model/knn_imputer.pkl",
            "src/model/knn_imputer.pkl",
        )
    )
    demographics_path: Path = field(
        default_factory=lambda: _resolve_default_path(
            "data/zipcode_demographics.csv",
            "src/data/zipcode_demographics.csv",
        )
    )


def _ensure_file_exists(path: Path, label: str) -> Path:
    """Validate that an artifact path exists and points to a file."""
    if not path.exists():
        raise ResourceInitializationError(f"Required {label} artifact not found: {path}")
    if not path.is_file():
        raise ResourceInitializationError(f"Expected {label} artifact to be a file: {path}")
    return path


def load_model(model_path: str | Path) -> Any:
    """Load the serialized prediction model from disk."""
    artifact_path = _ensure_file_exists(Path(model_path), "model")
    with artifact_path.open("rb") as model_file:
        return pickle.load(model_file)


def load_feature_names(features_path: str | Path) -> List[str]:
    """Load the ordered model feature list from JSON metadata."""
    artifact_path = _ensure_file_exists(Path(features_path), "feature metadata")
    with artifact_path.open("r", encoding="utf-8") as features_file:
        feature_names = json.load(features_file)

    if not isinstance(feature_names, list) or not feature_names or not all(
        isinstance(feature_name, str) for feature_name in feature_names
    ):
        raise ResourceInitializationError(
            "Feature metadata must be a non-empty JSON array of strings."
        )

    return list(feature_names)


def load_imputer_artifact(imputer_path: str | Path) -> Mapping[str, Any]:
    """Load the serialized KNN imputation artifact from disk."""
    artifact_path = _ensure_file_exists(Path(imputer_path), "imputation artifact")
    with artifact_path.open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)

    if not isinstance(artifact, dict):
        raise ResourceInitializationError(
            "Imputation artifact must deserialize to a dictionary."
        )

    return artifact


def load_demographics(
    demographics_path: str | Path,
) -> Dict[str, Mapping[str, float]]:
    """Load zipcode demographics as a dictionary keyed by zipcode.

    Args:
        demographics_path: CSV path containing zipcode demographic features.

    Returns:
        Dictionary mapping zipcode strings to demographic feature mappings.

    Raises:
        ResourceInitializationError: If the CSV is missing required structure.
    """
    artifact_path = _ensure_file_exists(Path(demographics_path), "demographics data")
    demographic_frame = pd.read_csv(artifact_path, dtype={"zipcode": str})

    if "zipcode" not in demographic_frame.columns:
        raise ResourceInitializationError("Demographics dataset must include a zipcode column.")

    if demographic_frame["zipcode"].duplicated().any():
        raise ResourceInitializationError(
            "Demographics dataset must contain unique zipcode rows."
        )

    demographic_frame = demographic_frame.set_index("zipcode")
    return {
        zipcode: row.dropna().to_dict()
        for zipcode, row in demographic_frame.iterrows()
    }