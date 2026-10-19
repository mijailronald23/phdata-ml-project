from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from evaluation.phase4 import EvaluationContext
from evaluation.phase4 import EvaluationScenario
from evaluation.phase4 import apply_strategy
from evaluation.phase4 import build_scenarios
from evaluation.phase4 import compute_imputation_error
from evaluation.phase4 import compute_prediction_metrics
from evaluation.phase4 import enrich_with_demographics
from evaluation.phase4 import fit_strategy_artifacts
from evaluation.phase4 import load_evaluation_context
from services.inference import REQUEST_PROPERTY_FEATURES


OUTPUT_PATH = Path("artifacts/phase4/evaluation_results.json")


def main() -> None:
    """Run the Phase 4 offline evaluation and persist the results JSON."""
    context = load_evaluation_context()
    scenarios = build_scenarios(context.raw_test_properties)
    fitted_strategies = fit_strategy_artifacts(context)

    with context.artifact_paths.imputer_path.open("rb") as artifact_file:
        production_imputer_artifact = pickle.load(artifact_file)

    strategies = {
        "mean": fitted_strategies["mean"],
        "median": fitted_strategies["median"],
        "knn_scaled_distance": fitted_strategies["knn"],
        "current_production": production_imputer_artifact,
    }

    baseline_predictions = context.model.predict(context.test_model_frame)
    results = {
        "metadata": build_metadata(context),
        "strategies": {},
    }

    for strategy_name, strategy in strategies.items():
        strategy_results = {}
        for scenario in scenarios:
            scenario_result = evaluate_scenario(
                context=context,
                scenario=scenario,
                strategy_name=strategy_name,
                strategy=strategy,
                baseline_predictions=baseline_predictions,
            )
            strategy_results[scenario.name] = scenario_result
        results["strategies"][strategy_name] = strategy_results

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved Phase 4 evaluation results to {OUTPUT_PATH}")


def evaluate_scenario(
    *,
    context: EvaluationContext,
    scenario: EvaluationScenario,
    strategy_name: str,
    strategy: object,
    baseline_predictions,
) -> dict[str, object]:
    """Evaluate one imputation strategy against one missingness scenario.

    Args:
        context: Shared evaluation inputs and fitted model.
        scenario: Missingness scenario to evaluate.
        strategy_name: Human-readable strategy identifier.
        strategy: Fitted imputation strategy or production artifact.
        baseline_predictions: Predictions produced from complete model inputs.

    Returns:
        Scenario-level metrics and runtime information.
    """
    property_payload = scenario.masked_properties.copy()
    property_payload["zipcode"] = context.raw_test_properties["zipcode"].values
    masked_model_frame = enrich_with_demographics(
        raw_properties=property_payload,
        demographics=context.demographics,
        model_features=context.model_features,
    )

    start_time = time.perf_counter()
    imputed_frame = apply_strategy(
        strategy_name=strategy_name if strategy_name != "current_production" else "production",
        strategy=strategy,
        model_frame=masked_model_frame,
        missing_mask=scenario.missing_mask,
    )
    predictions = context.model.predict(imputed_frame)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    prediction_metrics = compute_prediction_metrics(
        y_true=context.y_test,
        predicted=predictions,
        baseline_predictions=baseline_predictions,
    )
    imputation_metrics = compute_imputation_error(
        imputed_frame=imputed_frame,
        original_frame=context.raw_test_properties,
        missing_mask=scenario.missing_mask,
    )

    return {
        "description": scenario.description,
        "is_simulation": scenario.is_simulation,
        "prediction_metrics": prediction_metrics,
        "imputation_metrics": imputation_metrics,
        "success_rate_pct": 100.0,
        "record_count": int(len(context.y_test)),
        "missing_value_count": int(scenario.missing_mask.to_numpy().sum()),
        "runtime_ms_total": float(elapsed_ms),
        "runtime_ms_per_record": float(elapsed_ms / len(context.y_test)),
    }


def build_metadata(context: EvaluationContext) -> dict[str, object]:
    """Build top-level metadata for the saved evaluation artifact."""
    return {
        "evaluation_type": (
            "Reconstructed validation split from training code. This is not a "
            "cryptographically verified independent holdout for the serialized model."
        ),
        "train_row_count": int(context.train_model_frame.shape[0]),
        "test_row_count": int(context.test_model_frame.shape[0]),
        "model_feature_count": len(context.model_features),
        "property_feature_count": len(REQUEST_PROPERTY_FEATURES),
        "model_features": context.model_features,
    }


if __name__ == "__main__":
    main()
