# FastAPI Housing Price API

## 1. Project Overview
Sound Realty uses this API to estimate Seattle-area home prices from property data. The original service could not handle missing numeric inputs and did repeated artifact loading on every request. This implementation keeps the existing trained model intact, adds KNN-based missing-data handling, and improves inference performance.

```json
{"predicted_price": 450000.0}
```

## 2. Architecture
`FastAPI -> Pydantic validation -> InferenceService -> zipcode demographic enrichment -> KNN imputation when needed -> existing model -> JSON response`

Important invariants:
- `src/model/model.pkl` is preserved and not retrained.
- The model still receives exactly 33 ordered internal features.
- The public API remains 8 fields: 7 numeric property fields plus required `zipcode`.
- Complete inputs bypass imputation and preserve the pre-imputation prediction path.

Core files:
- `src/main.py`: FastAPI app and lifespan startup
- `src/api/endpoints.py`: `/health`, `/ready`, `/predict`
- `src/services/inference.py`: in-memory inference path
- `src/preprocessing/imputation.py`: `RobustScaler + KNNImputer`
- `create_imputer.py`: offline artifact generation

## 3. Quick Start with Docker
```bash
docker build -t phdata-ml-api .
docker run --rm -p 8000:8000 --name phdata-ml-api phdata-ml-api
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

```bash
docker build -f Dockerfile.test -t ml-api-test .
make test-unit
make test-integration
make test-all
```

## 4. API Usage
- Required: `zipcode` as a 5-digit string
- Optional or nullable: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `sqft_above`, `sqft_basement`

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

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sqft_living": -1, "zipcode": "98A42"}'
```

## 5. KNN Missing Data Imputation
The deployed model expects 33 features: 7 property features and 26 demographic features joined by `zipcode`. When any numeric property value is missing, the request is first enriched with zipcode demographics and then imputed in the same 33-feature space used by the model.

Serving uses an offline-fitted artifact with:
- `RobustScaler`
- `KNNImputer(n_neighbors=5, weights="distance")`
- training reference data only

The imputer is fit offline by `create_imputer.py` on the reconstructed training split from `create_model.py`. It never uses `price` and never fits during API requests.

## 6. Performance Improvements
Artifacts are loaded once per worker during FastAPI lifespan startup:
- `model.pkl`
- `model_features.json`
- `knn_imputer.pkl`
- `zipcode_demographics.csv`

Stored historical baseline used by the benchmark comparison:
- complete input mean `2.785 ms`, median `2.743 ms`, P95 `3.824 ms`

Phase 4 service benchmark:
- complete input mean `1.709 ms`, median `1.227 ms`, P95 `2.798 ms`
- missing input mean `99.984 ms`, median `99.147 ms`, P95 `120.260 ms`
- complete-path mean improved `38.64%` versus the stored baseline
- missing-value path adds `98.275 ms` mean overhead due to scaling, KNN imputation, and downstream model prediction

Phase 4 HTTP benchmark:
- complete traffic: `205.431 req/s`, `p95 101.668 ms`
- missing traffic: `61.519 req/s`, `p95 280.395 ms`
- mixed traffic: `95.079 req/s`, `p95 271.814 ms`

## 7. Testing
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
```

Run tests locally:

```bash
PYTHONPATH=src:. pytest test/unit -v
PYTHONPATH=src:. uvicorn main:app --host 127.0.0.1 --port 8001
API_BASE_URL=http://127.0.0.1:8001 PYTHONPATH=src:. pytest test/integration -v
```

Phase 4 utilities:

```bash
PYTHONPATH=src:. python evaluate_phase4.py
PYTHONPATH=src:. python benchmark_phase4.py --base-url http://127.0.0.1:8001 --iterations 200 --requests 120 --concurrency 12
```

## 8. Limitations
- The evaluation split is reconstructed from training code and current data; it is not a cryptographically verified independent holdout for `model.pkl`.
- The model itself was not retrained with imputed inputs.
- Missing-value requests are much slower than complete requests because the KNN path is CPU-heavy.
- Each worker keeps its own in-memory copy of the model and imputation resources.
- Docker commands are provided here, but container execution must be verified in an environment where Docker is available.