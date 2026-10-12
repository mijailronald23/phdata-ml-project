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
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    return Path(candidates[0])


@dataclass(frozen=True)
class ArtifactPaths:
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
    demographics_path: Path = field(
        default_factory=lambda: _resolve_default_path(
            "data/zipcode_demographics.csv",
            "src/data/zipcode_demographics.csv",
        )
    )


def _ensure_file_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise ResourceInitializationError(f"Required {label} artifact not found: {path}")
    if not path.is_file():
        raise ResourceInitializationError(f"Expected {label} artifact to be a file: {path}")
    return path


def load_model(model_path: str | Path) -> Any:
    artifact_path = _ensure_file_exists(Path(model_path), "model")
    with artifact_path.open("rb") as model_file:
        return pickle.load(model_file)


def load_feature_names(features_path: str | Path) -> List[str]:
    artifact_path = _ensure_file_exists(Path(features_path), "feature metadata")
    with artifact_path.open("r", encoding="utf-8") as features_file:
        features = json.load(features_file)

    if not isinstance(features, list) or not features or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ResourceInitializationError(
            "Feature metadata must be a non-empty JSON array of strings."
        )

    return list(features)


def load_demographics(
    demographics_path: str | Path,
) -> Dict[str, Mapping[str, float]]:
    artifact_path = _ensure_file_exists(Path(demographics_path), "demographics data")
    demographics = pd.read_csv(artifact_path, dtype={"zipcode": str})

    if "zipcode" not in demographics.columns:
        raise ResourceInitializationError("Demographics dataset must include a zipcode column.")

    if demographics["zipcode"].duplicated().any():
        raise ResourceInitializationError(
            "Demographics dataset must contain unique zipcode rows."
        )

    demographics = demographics.set_index("zipcode")
    return {
        zipcode: row.dropna().to_dict()
        for zipcode, row in demographics.iterrows()
    }