# SHAP Experiment for Interview Support

This directory contains a lightweight, non-production SHAP experiment for the
existing Seattle home-price model. It is intentionally separate from the
FastAPI service and does not change production inference behavior.

## Goal
- Estimate exploratory global feature importance on a very small sample
- Produce one local prediction explanation
- Compare a complete-input prediction with the same property after missing-value
  imputation through the real `InferenceService`

## Production contract reused
- Model artifact: `src/model/model.pkl`
- Feature schema: `src/model/model_features.json`
- Imputer artifact: `src/model/knn_imputer.pkl`
- Inference path: existing `InferenceService`

The experiment does **not** retrain the model, refit the imputer, or reimplement
the serving pipeline.

## Environment
Use an isolated environment and install:

```bash
/home/mijailvargas/ENTREVISTAS/phd_data_test/phdata_env/bin/pip install -r requirements.txt -r requirements-xai.txt
```

Pinned XAI dependencies:
- `shap==0.54.1`
- `matplotlib==3.11.2`

## Method selection
The serialized model is a scikit-learn `Pipeline` whose final estimator is
`KNeighborsRegressor`. Because this is not a tree-based model, the experiment
uses SHAP's model-agnostic `PermutationExplainer` rather than `TreeExplainer`.

Configuration:
- random seed: `42`
- background sample size: `20`
- explanation sample size: `5`
- feature count: `33`
- max evaluations per explanation: `67` (`2 * 33 + 1`)

This keeps the computational budget small and makes the experiment easy to
defend in an interview.

## Run

```bash
PYTHONPATH=src:. /home/mijailvargas/ENTREVISTAS/phd_data_test/phdata_env/bin/python experiments/xai_shap/run_shap.py
```

## Outputs
- `results/shap_global_importance.png`
- `results/shap_local_waterfall.png`
- `results/shap_summary.json`

## Actual measured results
From `results/shap_summary.json`:

- explainer: `PermutationExplainer`
- total execution time: `1.172299 s`
- explanation-only runtime: `0.167346 s`
- top 5 exploratory global features:
  1. `hous_val_amt`
  2. `bathrooms`
  3. `sqft_living`
  4. `sqft_above`
  5. `bedrooms`

### Local explanation
- selected observation source index: `870`
- zipcode: `98168`
- complete-input prediction: `337477.6`
- SHAP baseline value: `470669.13`
- additivity error: `5.82e-11`

Top local contributions for the complete record:
- `sqft_lot`: `+73998.055`
- `bathrooms`: `-53344.355`
- `sqft_basement`: `+53223.265`
- `edctn_high_schl_qty`: `-32535.93`
- `per_less_than_9`: `+30873.68`

### Missing-data comparison
Same property with `bathrooms` and `sqft_lot` set to missing:

- missing-input prediction: `271110.0`
- absolute prediction difference: `66367.6`
- relative prediction difference: `19.6658%`
- imputed features:
  - `bathrooms -> 1.397872624403325`
  - `sqft_lot -> 15638.217274826096`

## Interpretation guardrails
- This is an exploratory SHAP analysis on a tiny sample, not a market-wide
  importance study.
- SHAP values explain model behavior, not causal effects.
- For the missing-data comparison, SHAP explains the downstream model on the
  final 33-feature input. It does not isolate the causal effect of the KNN
  imputation step.
- Rerunning the script should reproduce the same selected rows and substantive
  explanation values. The JSON file hash may still change because execution-time
  metadata is intentionally recorded.
