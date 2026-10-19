# FastAPI Housing Price API

This repository implements the phData Machine Learning Engineer Project Challenge as a FastAPI service that serves a pre-trained house price model. Version `v0.3.0` adds offline-fitted KNN missing-data imputation while preserving the existing trained model, the `/predict` endpoint, and the response shape:

```json
{
  "predicted_price": 450000.0
}
```

## What is deployed

The deployed inference path is:

`FastAPI -> Input Validation -> InferenceService -> Zipcode Enrichment -> KNN Imputation (when needed) -> Existing ML Model -> Prediction`

Important invariants:

- `src/model/model.pkl` is preserved and not retrained by the API.
- The model still consumes exactly 33 ordered internal features.
- Public API input remains property fields plus mandatory `zipcode`.
- Complete inputs bypass imputation so predictions remain compatible with Phase 2.

## Project structure

```text
phdata-ml-project
├── src/
│   ├── api/
│   │   └── endpoints.py
│   ├── preprocessing/
│   │   └── imputation.py
│   ├── services/
│   │   └── inference.py
│   ├── utils/
│   │   └── loader.py
│   ├── model/
│   │   ├── model.pkl
│   │   ├── model_features.json
│   │   └── knn_imputer.pkl
│   └── data/
│       ├── kc_house_data.csv
│       ├── zipcode_demographics.csv
│       └── future_unseen_examples.csv
├── create_model.py
├── create_imputer.py
├── Dockerfile
├── Dockerfile.test
├── docker-compose.test.yml
└── test/
```

## Environment compatibility

The existing model artifact was serialized with `scikit-learn 1.7.2`. To avoid runtime incompatibility, `requirements.txt` now pins:

```text
scikit-learn==1.7.2
```

This is required for both local execution and container builds. Do not modify `model.pkl`.

## Input contract

`/predict` accepts:

- `zipcode` (required, 5-digit string)
- numeric property features:
  - `bedrooms`
  - `bathrooms`
  - `sqft_living`
  - `sqft_lot`
  - `floors`
  - `sqft_above`
  - `sqft_basement`

Phase 3 behavior:

- Numeric property fields may be omitted or set to `null`
- `zipcode` remains mandatory
- unsupported zipcodes return `422`
- malformed inputs still return validation errors

## Missing-data handling

Phase 3 uses a fitted offline KNN imputation artifact:

- estimator: `KNNImputer`
- `n_neighbors=5`
- `weights="distance"`
- distance space: robustly scaled numeric features

### Why scale before KNN?

KNN distance is sensitive to feature magnitude. In this dataset, raw dimensions such as `sqft_lot`, population counts, income and property value can dominate the distance calculation if left unscaled. The imputation artifact therefore fits:

1. `RobustScaler` on the reference feature matrix
2. `KNNImputer` on the scaled matrix

At inference time the request is enriched, scaled, imputed, and inverse-transformed back to raw feature space before being passed to the existing model pipeline.

### Why use the enriched 33-feature representation instead of only 7 property fields?

The model already expects the 33-feature enriched representation, and using it for KNN imputation has two advantages:

1. It matches the serving feature space used by the deployed model.
2. It allows imputation even when all 7 property features are missing, because the 26 demographic features remain available from `zipcode`.

### Training-serving consistency choice

The notebook research was exploratory and operated on synthetic missingness over the raw housing dataset. The deployed model was not trained with an imputer in its pipeline. To minimize serving drift:

- complete records bypass imputation entirely
- the imputer is fit offline on the model's reference training features only
- the target column (`price`) is never used during imputer fitting

This keeps the imputer aligned with the model input schema while avoiding target leakage. It does **not** eliminate the broader limitation that the model itself was not retrained on imputed inputs.

## Offline imputer preparation

`create_imputer.py` fits `src/model/knn_imputer.pkl` offline using the same feature construction and train split used by `create_model.py`:

- loads the historical sales data
- joins zipcode demographics
- removes `price`
- recreates the original `train_test_split(random_state=42)`
- fits the scaler + KNN imputer on `x_train` only

### Local artifact generation

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src:. python create_imputer.py
```

### Docker artifact generation

Build the artifact image:

```bash
docker build -f src/model/Dockerfile -t create-model .
```

Generate the original model artifacts if needed:

```bash
docker run --rm -v "$(pwd)/src/model:/app/model" create-model
```

Generate the imputer artifact:

```bash
docker run --rm -v "$(pwd)/src/model:/app/model" create-model python create_imputer.py
```

## Startup artifact loading

FastAPI loads the following resources once per worker during lifespan startup:

- `model.pkl`
- `model_features.json`
- `knn_imputer.pkl`
- `zipcode_demographics.csv`

Startup fails fast if any mandatory artifact is missing or incompatible. `/health` remains a liveness endpoint, and `/ready` verifies inference readiness.

## Phase 4 evaluation and benchmarking

Phase 4 adds reproducible offline evaluation and performance benchmarking utilities:

- `evaluate_phase4.py`
  - reconstructs the original train/validation split from the training code
  - evaluates complete input plus synthetic missingness scenarios
  - compares `mean`, `median`, offline KNN, and the current production configuration
  - reports prediction error and imputation error
- `benchmark_phase4.py`
  - profiles the inference path step-by-step
  - compares current service latency against the Phase 3 baseline
  - runs reproducible HTTP load tests for complete, missing and mixed traffic

Outputs are written to:

```text
artifacts/phase4/evaluation_results.json
artifacts/phase4/benchmark_results.json
```

### Important evaluation limitation

The evaluation split is reconstructed from the source training code (`train_test_split(random_state=42)`) and the current datasets. This is appropriate for internal comparison of strategies, but it is not a cryptographically verified independent holdout for the serialized `model.pkl`. Treat the resulting metrics as reconstructed validation measurements, not as guaranteed unbiased generalization estimates.

## Running the API

### Local

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src:. uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t phdata-ml-api .
docker run -d -p 8000:8000 --name housing-api phdata-ml-api
```

Docs will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Testing

### Docker-based test workflow

```bash
make test-unit
make test-integration
make test-all
```

### Local pytest workflow

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
PYTHONPATH=src:. pytest test/unit -v
PYTHONPATH=src:. uvicorn main:app --host 127.0.0.1 --port 8001
API_BASE_URL=http://127.0.0.1:8001 PYTHONPATH=src:. pytest test/integration -v
```

### Phase 4 offline evaluation

```bash
PYTHONPATH=src:. python evaluate_phase4.py
```

### Phase 4 profiling and HTTP benchmarking

Start the API in one shell:

```bash
PYTHONPATH=src:. uvicorn main:app --host 127.0.0.1 --port 8001
```

Run the benchmark in another shell:

```bash
PYTHONPATH=src:. python benchmark_phase4.py \
  --base-url http://127.0.0.1:8001 \
  --iterations 200 \
  --requests 120 \
  --concurrency 12
```

## Architecture and trade-offs

### Decisions

- Inference resources are loaded once at startup for performance.
- The endpoint stays synchronous so CPU-bound preprocessing and model inference run in FastAPI's threadpool instead of blocking the event loop.
- Imputation is separated from API validation and from artifact loading.

### Trade-offs

- Each worker keeps its own model and imputer copy in memory.
- Missing-value inference is slower than complete-record inference because KNN preprocessing is substantially more expensive than the no-missing fast path.
- For missing-value requests, the end-to-end path performs two neighbor-based operations: KNN imputation and KNN regression.
- The model pipeline itself still was not retrained with imputed inputs.

## Known limitations

- The model was not retrained to explicitly account for imputed requests.
- KNN imputation uses historical training features as its neighborhood reference set; this is appropriate for serving, but still assumes production missingness resembles the observed training distribution.
- Integration test coverage reported by `pytest --cov` against an external running server is not representative of in-process code coverage.
- CORS remains permissive and observability is still minimal for a production-grade deployment.