# Technical Demo Script

## Goal
Deliver a 10 to 15 minute live walkthrough of a simple, production-oriented ML inference service that handles missing property data without changing the trained model.

## 1. Open the repository and explain the structure
Highlight these files first:
- `src/main.py`: FastAPI app and startup lifecycle
- `src/api/endpoints.py`: request validation and routes
- `src/services/inference.py`: inference orchestration
- `src/preprocessing/imputation.py`: KNN imputation logic
- `create_imputer.py`: offline artifact preparation
- `test/unit` and `test/integration`: automated validation
- `artifacts/phase4`: evaluation and benchmark evidence

Suggested explanation:
"The service keeps the API surface small, keeps inference artifacts in memory after startup, and separates validation, preprocessing, and prediction."

## 2. Start the FastAPI application

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
PYTHONPATH=src:. uvicorn main:app --host 127.0.0.1 --port 8000
```

## 3. Show `/health` and `/ready`

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Talking point:
- `/health` proves the API process is alive.
- `/ready` proves model, feature metadata, imputer artifact, and demographics loaded successfully at startup.

## 4. Submit a complete prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1500,
    "sqft_lot": 5000,
    "floors": 1.0,
    "sqft_above": 1200,
    "sqft_basement": 300,
    "zipcode": "98042"
  }'
```

Talking point:
"A complete request goes through validation, zipcode enrichment, and directly into the model without imputation."

## 5. Submit a prediction request with missing numeric values

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": null,
    "sqft_living": 1500,
    "sqft_lot": null,
    "floors": 1.0,
    "sqft_above": 1200,
    "sqft_basement": 300,
    "zipcode": "98042"
  }'
```

Talking point:
"Missing numeric values are allowed. The service enriches the request to the model's 33-feature space, applies the offline-fitted KNN imputer, then calls the existing model."

## 6. Demonstrate input validation

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sqft_living": -1, "zipcode": "98A42"}'
```

Talking point:
- `zipcode` is mandatory and must match a 5-digit pattern.
- Numeric fields have domain validation.
- Unsupported but well-formed zipcodes return `422`.

## 7. Show where KNN imputation happens in code
Open:
- `src/api/endpoints.py`
- `src/services/inference.py`
- `src/preprocessing/imputation.py`

Key code path to explain:
1. `HomeFeatures` validates the public request contract.
2. `InferenceService.prepare_features()` assembles the 33-feature internal row.
3. `transform_array_with_imputer_artifact()` applies `RobustScaler -> KNNImputer -> inverse_transform` only when values are missing.
4. The trained model predicts from the ordered 33-feature frame.

## 8. Explain startup initialization and performance improvements
Open `src/main.py` and explain FastAPI lifespan.

Talking points:
- The old bottleneck was repeated file loading and deserialization on every request.
- The current implementation loads `model.pkl`, `model_features.json`, `knn_imputer.pkl`, and `zipcode_demographics.csv` once per worker.
- This improved the complete-input mean latency from the stored historical baseline of `2.785 ms` to `1.709 ms` in the Phase 4 benchmark.

## 9. Show automated tests

```bash
PYTHONPATH=src:. pytest test/unit -v
PYTHONPATH=src:. uvicorn main:app --host 127.0.0.1 --port 8001
API_BASE_URL=http://127.0.0.1:8001 PYTHONPATH=src:. pytest test/integration -v
```

Talking point:
"Tests cover startup initialization, complete and missing requests, invalid input, unsupported zipcodes, feature-order compatibility, and repeated inference stability."

## 10. Explain the key limitations and trade-offs
Close with these points:
- The prediction model itself was not retrained with imputed inputs.
- The validation split used in Phase 4 is reconstructed from training code, not independently verified from the original training run.
- Missing-value requests are valid but slower because the KNN path is CPU-heavy.
- Each worker keeps its own model and imputation artifacts in memory.

## Recommended flow for Q&A
- Why KNN over mean or median: show Phase 4 evidence that KNN preserved prediction quality better under missingness.
- Why 33 features instead of only 7: the model contract already requires 33, and demographics still provide signal when all property fields are missing.
- Why synchronous endpoint: inference is CPU-bound, so running it in FastAPI's threadpool is simpler and safer than blocking the event loop with async code.
