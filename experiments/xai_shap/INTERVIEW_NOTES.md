# Interview Notes: SHAP Experiment

## Why SHAP was evaluated
SHAP was evaluated as a lightweight, interview-friendly way to explain how the
existing home-price model behaves without changing the production inference
service. The goal was explanatory support, not a production feature.

## How it explains the existing KNN regressor
The serialized model is a scikit-learn pipeline ending in
`KNeighborsRegressor`. Because the estimator is not tree-based, the experiment
used SHAP `PermutationExplainer`, which is model-agnostic and works with a
prediction function over finalized model inputs.

The explainer was applied to the same 33-feature model inputs produced by the
real `InferenceService`, so the experiment stayed aligned with production
behavior.

## Small-sample feature importance
On the small explanation sample, the top global features by mean absolute SHAP
value were:
1. `hous_val_amt`
2. `bathrooms`
3. `sqft_living`
4. `sqft_above`
5. `bedrooms`

This ranking is exploratory only. It should not be interpreted as a stable
ranking for the entire Seattle housing market.

## Missing-data effect on the chosen example
For the selected property in zipcode `98168`:
- complete-input prediction: `337477.6`
- missing-input prediction: `271110.0`
- absolute difference: `66367.6`
- relative difference: `19.6658%`

The production imputer filled:
- `bathrooms -> 1.397872624403325`
- `sqft_lot -> 15638.217274826096`

This illustrates that the missing-data path can materially change the final
prediction even when the public API contract stays the same.

## Why SHAP is not a substitute for Phase 4 evaluation
SHAP explains how the model distributed credit across features for individual
predictions. It does not measure predictive quality. Phase 4 metrics such as
MAE, RMSE, and R² are still the correct tools for evaluating predictive
performance and comparing imputation strategies.

## SHAP limitations for correlated features
Many of the 33 inputs are correlated, especially property size features and
zipcode demographics. SHAP may redistribute credit across correlated variables,
so the explanations are useful for understanding model behavior but should not
be overinterpreted as unique or causal feature effects.

## Why this experiment stayed outside production
- It adds optional dependencies (`shap`, `matplotlib`)
- It introduces extra compute not needed for serving
- It explains model behavior on a small sample only
- It is useful for interview discussion, but not required for the phData
  challenge deliverable

## 60-90 second English explanation
"I ran a small SHAP experiment to explain the existing Seattle home-price model
without changing the production service. The serialized model is a scikit-learn
pipeline ending in KNeighborsRegressor, so I used SHAP's model-agnostic
PermutationExplainer rather than a tree-specific explainer. To stay aligned
with production, I generated explanations from the exact 33-feature model inputs
produced by the real InferenceService. On a very small sample, the most
influential features included zipcode-level house value, bathrooms, and living
area. For one representative property, the complete-input prediction was
337,477.6, while removing bathrooms and lot size and letting the production KNN
imputer fill them reduced the prediction to 271,110. The key point is that SHAP
helped me explain model behavior, but it does not replace Phase 4 evaluation
metrics like MAE, RMSE, and R², and it does not give a causal explanation of
missingness." 
