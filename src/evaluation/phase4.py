from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection

from create_model import DEMOGRAPHICS_PATH
from create_model import SALES_COLUMN_SELECTION
from create_model import SALES_PATH
from preprocessing.imputation import fit_knn_imputer_artifact
from preprocessing.imputation import transform_with_imputer_artifact
from services.inference import REQUEST_PROPERTY_FEATURES
from utils.loader import ArtifactPaths
from utils.loader import load_feature_names
from utils.loader import load_model


STRUCTURED_MISSING_GROUPS: Sequence[tuple[str, ...]] = (
    ("bedrooms", "bathrooms"),
    ("sqft_living", "sqft_above", "sqft_basement"),
    ("sqft_lot", "floors"),
)


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    description: str
    masked_properties: pd.DataFrame
    missing_mask: pd.DataFrame
    is_simulation: bool


@dataclass(frozen=True)
class EvaluationContext:
    model: Any
    model_features: list[str]
    raw_train_properties: pd.DataFrame
    raw_test_properties: pd.DataFrame
    train_model_frame: pd.DataFrame
    test_model_frame: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    demographics: pd.DataFrame
    artifact_paths: ArtifactPaths = field(default_factory=ArtifactPaths)


def load_evaluation_context(random_state: int = 42) -> EvaluationContext:
    artifact_paths = ArtifactPaths()
    model = load_model(artifact_paths.model_path)
    model_features = load_feature_names(artifact_paths.features_path)

    sales = pd.read_csv(
        SALES_PATH,
        usecols=SALES_COLUMN_SELECTION,
        dtype={"zipcode": str},
    )
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})

    y = sales["price"].copy()
    raw_properties = sales.drop(columns=["price"]).reset_index(drop=True)
    full_model_frame = enrich_with_demographics(
        raw_properties=raw_properties,
        demographics=demographics,
        model_features=model_features,
    )

    (
        raw_train_properties,
        raw_test_properties,
        train_model_frame,
        test_model_frame,
        y_train,
        y_test,
    ) = model_selection.train_test_split(
        raw_properties,
        full_model_frame,
        y,
        random_state=random_state,
    )

    return EvaluationContext(
        model=model,
        model_features=model_features,
        raw_train_properties=raw_train_properties.reset_index(drop=True),
        raw_test_properties=raw_test_properties.reset_index(drop=True),
        train_model_frame=train_model_frame.reset_index(drop=True),
        test_model_frame=test_model_frame.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        demographics=demographics.reset_index(drop=True),
        artifact_paths=artifact_paths,
    )


def build_scenarios(
    raw_test_properties: pd.DataFrame,
    seed: int = 42,
) -> list[EvaluationScenario]:
    property_frame = raw_test_properties.loc[:, REQUEST_PROPERTY_FEATURES].copy()
    scenarios: list[EvaluationScenario] = [
        EvaluationScenario(
            name="complete_input",
            description="No missing property values. Baseline evaluation.",
            masked_properties=property_frame.copy(),
            missing_mask=pd.DataFrame(
                False,
                index=property_frame.index,
                columns=property_frame.columns,
            ),
            is_simulation=False,
        )
    ]

    for rate in (0.05, 0.15, 0.30):
        mask = create_random_missing_mask(property_frame, missing_rate=rate, seed=seed)
        scenarios.append(
            EvaluationScenario(
                name=f"random_missing_{int(rate * 100)}pct",
                description=(
                    f"Synthetic MCAR simulation over API-allowed property features at "
                    f"{int(rate * 100)}% missingness."
                ),
                masked_properties=apply_missing_mask(property_frame, mask),
                missing_mask=mask,
                is_simulation=True,
            )
        )

    structured_mask = create_structured_missing_mask(property_frame, seed=seed)
    scenarios.append(
        EvaluationScenario(
            name="structured_missing_groups",
            description=(
                "Synthetic grouped missingness where correlated property fields are "
                "removed together."
            ),
            masked_properties=apply_missing_mask(property_frame, structured_mask),
            missing_mask=structured_mask,
            is_simulation=True,
        )
    )
    return scenarios


def enrich_with_demographics(
    raw_properties: pd.DataFrame,
    demographics: pd.DataFrame,
    model_features: Sequence[str],
) -> pd.DataFrame:
    demographics_indexed = demographics.set_index("zipcode")
    demographic_features = [
        feature for feature in model_features if feature not in REQUEST_PROPERTY_FEATURES
    ]
    enriched = raw_properties.copy()
    demographic_rows = demographics_indexed.loc[
        enriched["zipcode"], demographic_features
    ].reset_index(drop=True)
    model_frame = pd.concat(
        [
            enriched.loc[:, REQUEST_PROPERTY_FEATURES].reset_index(drop=True),
            demographic_rows,
        ],
        axis=1,
    )
    return model_frame.loc[:, model_features]


def fit_strategy_artifacts(context: EvaluationContext) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer

    mean_imputer = SimpleImputer(strategy="mean").fit(context.train_model_frame)
    median_imputer = SimpleImputer(strategy="median").fit(context.train_model_frame)
    knn_artifact = fit_knn_imputer_artifact(
        reference_frame=context.train_model_frame,
        feature_names=context.model_features,
        property_features=REQUEST_PROPERTY_FEATURES,
    )
    return {
        "mean": mean_imputer,
        "median": median_imputer,
        "knn": knn_artifact,
    }


def apply_strategy(
    strategy_name: str,
    strategy: Any,
    model_frame: pd.DataFrame,
    missing_mask: pd.DataFrame,
) -> pd.DataFrame:
    if strategy_name == "mean":
        return pd.DataFrame(
            strategy.transform(model_frame),
            columns=model_frame.columns,
            index=model_frame.index,
        )
    if strategy_name == "median":
        return pd.DataFrame(
            strategy.transform(model_frame),
            columns=model_frame.columns,
            index=model_frame.index,
        )
    if strategy_name in {"knn", "knn_scaled_distance", "production"}:
        if not missing_mask.any().any():
            return model_frame.copy()
        return transform_with_imputer_artifact(strategy, model_frame)
    raise ValueError(f"Unsupported strategy: {strategy_name}")


def compute_prediction_metrics(
    y_true: pd.Series,
    predicted: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict[str, float]:
    mae = metrics.mean_absolute_error(y_true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, predicted))
    r2 = metrics.r2_score(y_true, predicted)
    baseline_delta_mae = metrics.mean_absolute_error(baseline_predictions, predicted)
    baseline_delta_rmse = np.sqrt(
        metrics.mean_squared_error(baseline_predictions, predicted)
    )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "baseline_delta_mae": float(baseline_delta_mae),
        "baseline_delta_rmse": float(baseline_delta_rmse),
    }


def compute_imputation_error(
    imputed_frame: pd.DataFrame,
    original_frame: pd.DataFrame,
    missing_mask: pd.DataFrame,
) -> dict[str, float | None]:
    missing_values = missing_mask.to_numpy()
    if not missing_values.any():
        return {"mae": None, "rmse": None}

    original = original_frame.loc[:, REQUEST_PROPERTY_FEATURES].to_numpy()[missing_values]
    imputed = imputed_frame.loc[:, REQUEST_PROPERTY_FEATURES].to_numpy()[missing_values]
    mae = metrics.mean_absolute_error(original, imputed)
    rmse = np.sqrt(metrics.mean_squared_error(original, imputed))
    return {"mae": float(mae), "rmse": float(rmse)}


def create_random_missing_mask(
    property_frame: pd.DataFrame,
    missing_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mask = rng.random(property_frame.shape) < missing_rate
    return pd.DataFrame(mask, columns=property_frame.columns, index=property_frame.index)


def create_structured_missing_mask(
    property_frame: pd.DataFrame,
    seed: int,
    row_missing_rate: float = 0.20,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(
        False,
        columns=property_frame.columns,
        index=property_frame.index,
    )
    selected_rows = rng.random(property_frame.shape[0]) < row_missing_rate
    selected_indices = np.where(selected_rows)[0]
    if selected_indices.size == 0:
        return mask

    group_choices = rng.integers(0, len(STRUCTURED_MISSING_GROUPS), size=selected_indices.size)
    for row_index, group_index in zip(selected_indices, group_choices):
        for column in STRUCTURED_MISSING_GROUPS[group_index]:
            mask.iat[row_index, property_frame.columns.get_loc(column)] = True
    return mask


def apply_missing_mask(
    property_frame: pd.DataFrame,
    missing_mask: pd.DataFrame,
) -> pd.DataFrame:
    masked = property_frame.copy()
    masked[missing_mask] = np.nan
    return masked
