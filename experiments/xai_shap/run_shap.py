from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn

from evaluation.phase4 import load_evaluation_context
from services.inference import InferenceService
from services.inference import REQUEST_PROPERTY_FEATURES


SEED = 42
BACKGROUND_SAMPLE_SIZE = 20
EXPLANATION_SAMPLE_SIZE = 5
MISSING_FIELDS = ("bathrooms", "sqft_lot")
GLOBAL_TOP_FEATURES = 10
LOCAL_TOP_FEATURES = 10
RESULTS_DIR = Path(__file__).resolve().parent / "results"
GLOBAL_PLOT_PATH = RESULTS_DIR / "shap_global_importance.png"
WATERFALL_PLOT_PATH = RESULTS_DIR / "shap_local_waterfall.png"
SUMMARY_PATH = RESULTS_DIR / "shap_summary.json"


def main() -> None:
    """Run a small, reproducible SHAP experiment for the existing model."""
    run_started_at = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    service = InferenceService.from_artifacts()
    evaluation_context = load_evaluation_context(random_state=SEED)
    complete_candidates = build_complete_candidate_frame(
        evaluation_context.raw_test_properties,
        service,
    )
    selected_indices = select_sample_indices(len(complete_candidates))

    background_payloads = complete_candidates.iloc[selected_indices["background"]]
    explanation_payloads = complete_candidates.iloc[selected_indices["explanations"]]

    background_frame = prepare_model_inputs(service, background_payloads)
    explanation_frame = prepare_model_inputs(service, explanation_payloads)

    feature_count = explanation_frame.shape[1]
    max_evals = 2 * feature_count + 1
    model_predictor = build_model_predictor(service)
    explainer = shap.Explainer(
        model_predictor,
        background_frame,
        algorithm="permutation",
        seed=SEED,
    )

    explanation_started_at = time.perf_counter()
    explanation_values = explainer(explanation_frame, max_evals=max_evals)
    explanation_runtime_seconds = time.perf_counter() - explanation_started_at

    local_payload = row_to_payload(explanation_payloads.iloc[0])
    local_complete_prepared = service.prepare_features(local_payload)
    local_complete_prediction = service.predict(local_payload)

    local_missing_payload = dict(local_payload)
    for field_name in MISSING_FIELDS:
        local_missing_payload[field_name] = None

    local_missing_prepared = service.prepare_features(local_missing_payload)
    local_missing_prediction = service.predict(local_missing_payload)
    local_missing_explanation = explainer(
        local_missing_prepared.model_input,
        max_evals=max_evals,
    )

    global_importance = compute_global_importance(
        explanation_frame.columns,
        explanation_values.values,
        top_k=GLOBAL_TOP_FEATURES,
    )
    save_global_importance_plot(global_importance, GLOBAL_PLOT_PATH)
    save_local_waterfall_plot(explanation_values[0], WATERFALL_PLOT_PATH)

    local_complete_baseline = float(explanation_values.base_values[0])
    local_complete_shap_sum = float(explanation_values.values[0].sum())
    local_complete_approximation = local_complete_baseline + local_complete_shap_sum
    local_complete_additivity_error = abs(
        local_complete_prediction - local_complete_approximation
    )

    local_missing_baseline = float(local_missing_explanation.base_values[0])
    local_missing_shap_sum = float(local_missing_explanation.values[0].sum())
    local_missing_approximation = local_missing_baseline + local_missing_shap_sum
    local_missing_additivity_error = abs(
        local_missing_prediction - local_missing_approximation
    )

    summary = {
        "metadata": {
            "seed": SEED,
            "model_type": type(service._model).__module__
            + "."
            + type(service._model).__name__,
            "final_estimator_type": type(service._model.steps[-1][1]).__module__
            + "."
            + type(service._model.steps[-1][1]).__name__,
            "model_feature_count": feature_count,
            "model_features": service.model_features,
            "model_sha256": sha256_for_file(Path("src/model/model.pkl")),
            "knn_imputer_sha256": sha256_for_file(Path("src/model/knn_imputer.pkl")),
            "scikit_learn_version": sklearn.__version__,
            "shap_version": shap.__version__,
            "background_sample_size": BACKGROUND_SAMPLE_SIZE,
            "explanation_sample_size": EXPLANATION_SAMPLE_SIZE,
            "explainer_algorithm": "permutation",
            "explainer_class": type(explainer).__name__,
            "max_evals": max_evals,
            "execution_time_seconds": round(time.perf_counter() - run_started_at, 6),
            "explanation_runtime_seconds": round(explanation_runtime_seconds, 6),
        },
        "global_feature_importance": {
            "label": "Exploratory SHAP feature importance - small sample",
            "top_features": global_importance,
        },
        "local_explanation": {
            "selected_observation": {
                "source_row_index": int(explanation_payloads.iloc[0]["source_row_index"]),
                "raw_complete_input": local_payload,
            },
            "prediction": float(local_complete_prediction),
            "baseline_value": local_complete_baseline,
            "approximation_value": local_complete_approximation,
            "additivity_error": local_complete_additivity_error,
            "feature_contributions": summarize_feature_contributions(
                explanation_frame.columns,
                explanation_values.values[0],
                local_complete_prepared.model_input.iloc[0],
                top_k=LOCAL_TOP_FEATURES,
            ),
            "explainer_configuration": {
                "background_sample_size": BACKGROUND_SAMPLE_SIZE,
                "max_evals": max_evals,
                "algorithm": "permutation",
                "feature_count": feature_count,
            },
        },
        "missing_data_comparison": {
            "raw_missing_input": local_missing_payload,
            "prediction_complete": float(local_complete_prediction),
            "prediction_missing": float(local_missing_prediction),
            "absolute_prediction_difference": float(
                abs(local_complete_prediction - local_missing_prediction)
            ),
            "relative_prediction_difference_pct": float(
                abs(local_complete_prediction - local_missing_prediction)
                / local_complete_prediction
                * 100
            ),
            "imputed_features": list(local_missing_prepared.missing_property_features),
            "imputed_feature_values": {
                feature_name: float(local_missing_prepared.model_input.iloc[0][feature_name])
                for feature_name in local_missing_prepared.missing_property_features
            },
            "changed_model_input_features": compare_model_inputs(
                local_complete_prepared.model_input.iloc[0],
                local_missing_prepared.model_input.iloc[0],
            ),
            "complete_explanation": {
                "baseline_value": local_complete_baseline,
                "approximation_value": local_complete_approximation,
                "additivity_error": local_complete_additivity_error,
                "top_feature_contributions": summarize_feature_contributions(
                    explanation_frame.columns,
                    explanation_values.values[0],
                    local_complete_prepared.model_input.iloc[0],
                    top_k=5,
                ),
            },
            "missing_explanation": {
                "baseline_value": local_missing_baseline,
                "approximation_value": local_missing_approximation,
                "additivity_error": local_missing_additivity_error,
                "top_feature_contributions": summarize_feature_contributions(
                    local_missing_prepared.model_input.columns,
                    local_missing_explanation.values[0],
                    local_missing_prepared.model_input.iloc[0],
                    top_k=5,
                ),
            },
            "interpretation_note": (
                "SHAP explains the downstream KNeighborsRegressor on the final 33-feature "
                "model input. It does not isolate a causal effect of missingness or "
                "separately explain the KNN imputation mechanism."
            ),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(make_json_serializable(summary), indent=2),
        encoding="utf-8",
    )
    print(f"Saved SHAP experiment summary to {SUMMARY_PATH}")
    print(f"Saved global importance plot to {GLOBAL_PLOT_PATH}")
    print(f"Saved local waterfall plot to {WATERFALL_PLOT_PATH}")


def build_complete_candidate_frame(
    raw_test_properties: pd.DataFrame,
    service: InferenceService,
) -> pd.DataFrame:
    """Return supported, complete request payload candidates for the experiment."""
    complete_candidates = (
        raw_test_properties.dropna(subset=list(REQUEST_PROPERTY_FEATURES))
        .reset_index()
        .rename(columns={"index": "source_row_index"})
    )
    supported_zipcode_mask = complete_candidates["zipcode"].astype(str).isin(
        service.supported_zipcodes
    )
    return complete_candidates.loc[supported_zipcode_mask].reset_index(drop=True)


def select_sample_indices(candidate_count: int) -> dict[str, np.ndarray]:
    """Select deterministic background and explanation sample indices."""
    required_count = BACKGROUND_SAMPLE_SIZE + EXPLANATION_SAMPLE_SIZE
    if candidate_count < required_count:
        raise ValueError(
            f"Need at least {required_count} complete candidates, found {candidate_count}."
        )

    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(candidate_count)
    return {
        "background": shuffled_indices[:BACKGROUND_SAMPLE_SIZE],
        "explanations": shuffled_indices[
            BACKGROUND_SAMPLE_SIZE : BACKGROUND_SAMPLE_SIZE + EXPLANATION_SAMPLE_SIZE
        ],
    }


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert a raw property row into the public request payload format."""
    payload = {
        feature_name: to_python_scalar(row[feature_name])
        for feature_name in REQUEST_PROPERTY_FEATURES
    }
    payload["zipcode"] = str(row["zipcode"])
    return payload


def prepare_model_inputs(
    service: InferenceService,
    payload_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare final model inputs using the production inference service."""
    prepared_frames = []
    for _, row in payload_frame.iterrows():
        prepared_frames.append(service.prepare_features(row_to_payload(row)).model_input)
    return pd.concat(prepared_frames, ignore_index=True)


def build_model_predictor(service: InferenceService):
    """Return a predictor for finalized 33-feature model inputs.

    SHAP explains the downstream prediction model after the production
    ``InferenceService`` has already produced model-compliant inputs.
    """

    def predict_model_inputs(model_input_frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(model_input_frame, pd.DataFrame):
            model_input_frame = pd.DataFrame(
                model_input_frame,
                columns=service.model_features,
            )
        return service._model.predict(model_input_frame)

    return predict_model_inputs


def compute_global_importance(
    feature_names: pd.Index,
    shap_values: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, float | str]]:
    """Compute global importance from mean absolute SHAP values."""
    mean_absolute_shap = np.abs(shap_values).mean(axis=0)
    importance_frame = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": mean_absolute_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    top_features = importance_frame.head(top_k)
    return [
        {
            "feature": row["feature"],
            "mean_abs_shap": float(row["mean_abs_shap"]),
        }
        for _, row in top_features.iterrows()
    ]


def summarize_feature_contributions(
    feature_names: pd.Index,
    shap_values: np.ndarray,
    feature_values: pd.Series,
    *,
    top_k: int,
) -> list[dict[str, float | str]]:
    """Return the top absolute SHAP contributions for one explanation."""
    contribution_frame = pd.DataFrame(
        {
            "feature": list(feature_names),
            "feature_value": [float(feature_values[name]) for name in feature_names],
            "shap_value": shap_values,
        }
    )
    contribution_frame["abs_shap_value"] = contribution_frame["shap_value"].abs()
    top_contributions = contribution_frame.sort_values(
        "abs_shap_value",
        ascending=False,
    ).head(top_k)
    return [
        {
            "feature": row["feature"],
            "feature_value": float(row["feature_value"]),
            "shap_value": float(row["shap_value"]),
        }
        for _, row in top_contributions.iterrows()
    ]


def compare_model_inputs(
    complete_features: pd.Series,
    missing_features: pd.Series,
) -> list[dict[str, float | str]]:
    """Return model-input features whose values changed after imputation."""
    changed_features = []
    for feature_name in complete_features.index:
        complete_value = float(complete_features[feature_name])
        missing_value = float(missing_features[feature_name])
        if not np.isclose(complete_value, missing_value):
            changed_features.append(
                {
                    "feature": feature_name,
                    "complete_value": complete_value,
                    "missing_value": missing_value,
                }
            )
    return changed_features


def save_global_importance_plot(
    global_importance: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    """Save the top-10 global SHAP importance chart."""
    plot_frame = pd.DataFrame(global_importance).sort_values("mean_abs_shap", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(plot_frame["feature"], plot_frame["mean_abs_shap"], color="#4C78A8")
    axis.set_xlabel("Mean |SHAP value|")
    axis.set_ylabel("Feature")
    axis.set_title("Exploratory SHAP feature importance - small sample")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_local_waterfall_plot(
    explanation: shap.Explanation,
    output_path: Path,
) -> None:
    """Save the local SHAP waterfall plot for one prediction."""
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(explanation, max_display=LOCAL_TOP_FEATURES, show=False)
    plt.gcf().suptitle("Exploratory SHAP local explanation - complete input", y=1.02)
    plt.gcf().savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(plt.gcf())


def sha256_for_file(path: Path) -> str:
    """Return the SHA256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_python_scalar(value: Any) -> Any:
    """Convert NumPy and pandas scalar values into native Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_json_serializable(value: Any) -> Any:
    """Recursively convert nested experiment results into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]
    return to_python_scalar(value)


if __name__ == "__main__":
    main()
