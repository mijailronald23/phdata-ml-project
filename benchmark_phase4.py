from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import sklearn

from services.inference import InferenceService
from services.inference import REQUEST_PROPERTY_FEATURES


OUTPUT_PATH = Path("artifacts/phase4/benchmark_results.json")
PHASE3_BASELINE = {
    "complete_mean_ms": 2.785,
    "complete_median_ms": 2.743,
    "complete_p95_ms": 3.824,
    "missing_mean_ms": 92.477,
    "missing_median_ms": 94.618,
    "missing_p95_ms": 114.606,
    "knn_only_mean_ms": 33.858,
    "knn_only_median_ms": 29.368,
    "knn_only_p95_ms": 60.846,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    service = InferenceService.from_artifacts()
    complete_payload = build_complete_payload()
    missing_payload = build_missing_payload(complete_payload)

    service_benchmarks = run_service_benchmarks(
        service=service,
        complete_payload=complete_payload,
        missing_payload=missing_payload,
        iterations=args.iterations,
    )
    component_profiles = run_component_profiles(
        service=service,
        complete_payload=complete_payload,
        missing_payload=missing_payload,
        iterations=args.iterations,
    )
    http_benchmarks = asyncio.run(
        run_http_benchmarks(
            base_url=args.base_url,
            total_requests=args.requests,
            concurrency=args.concurrency,
        )
    )

    results = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "iterations": args.iterations,
            "http_requests": args.requests,
            "http_concurrency": args.concurrency,
        },
        "phase3_baseline": PHASE3_BASELINE,
        "service_benchmarks": service_benchmarks,
        "component_profiles": component_profiles,
        "http_benchmarks": http_benchmarks,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved Phase 4 benchmark results to {OUTPUT_PATH}")


def build_complete_payload() -> dict:
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500.0,
        "sqft_lot": 5000.0,
        "floors": 1.0,
        "sqft_above": 1200.0,
        "sqft_basement": 300.0,
        "zipcode": "98042",
    }


def build_missing_payload(complete_payload: dict) -> dict:
    payload = dict(complete_payload)
    payload["bathrooms"] = None
    payload["sqft_lot"] = None
    payload["sqft_basement"] = None
    return payload


def run_service_benchmarks(
    *,
    service: InferenceService,
    complete_payload: dict,
    missing_payload: dict,
    iterations: int,
) -> dict:
    benchmarks = {
        "complete": lambda: service.predict(complete_payload),
        "missing": lambda: service.predict(missing_payload),
    }
    results = {
        name: measure_latency(fn, iterations)
        for name, fn in benchmarks.items()
    }
    results["derived"] = {
        "knn_overhead_mean_ms": round(
            results["missing"]["avg_ms"] - results["complete"]["avg_ms"], 3
        ),
        "complete_vs_phase3_mean_pct": round(
            ((results["complete"]["avg_ms"] - PHASE3_BASELINE["complete_mean_ms"])
             / PHASE3_BASELINE["complete_mean_ms"]) * 100,
            2,
        ),
        "missing_vs_phase3_mean_pct": round(
            ((results["missing"]["avg_ms"] - PHASE3_BASELINE["missing_mean_ms"])
             / PHASE3_BASELINE["missing_mean_ms"]) * 100,
            2,
        ),
    }
    return results


def run_component_profiles(
    *,
    service: InferenceService,
    complete_payload: dict,
    missing_payload: dict,
    iterations: int,
) -> dict:
    profiles = {
        "complete": profile_request_path,
        "missing": profile_request_path,
    }
    payloads = {
        "complete": complete_payload,
        "missing": missing_payload,
    }
    results = {}
    for name, profiler in profiles.items():
        per_step = {}
        for _ in range(iterations):
            metrics = profiler(service, payloads[name])
            for key, value in metrics.items():
                per_step.setdefault(key, []).append(value)
        results[name] = {
            step: summarize_timings(values) for step, values in per_step.items()
        }
    return results


def profile_request_path(service: InferenceService, payload: dict) -> dict[str, float]:
    timings_ms: dict[str, float] = {}

    started = time.perf_counter_ns()
    zipcode = str(payload["zipcode"])
    missing_property_features = [
        feature for feature in REQUEST_PROPERTY_FEATURES if payload.get(feature) is None
    ]
    timings_ms["input_preparation"] = elapsed_ms(started)

    started = time.perf_counter_ns()
    demographics = service._demographics_by_zipcode[zipcode]
    demographic_array = service._demographic_arrays_by_zipcode[zipcode]
    timings_ms["demographic_lookup"] = elapsed_ms(started)

    started = time.perf_counter_ns()
    raw_row = demographic_array.copy()
    for idx, feature in zip(service._property_feature_positions, REQUEST_PROPERTY_FEATURES):
        value = payload.get(feature)
        if value is not None:
            raw_row[idx] = float(value)
    raw_array = raw_row.reshape(1, -1)
    timings_ms["array_creation"] = elapsed_ms(started)

    scaler = service._imputer_artifact["scaler"]
    imputer = service._imputer_artifact["imputer"]
    model_input_array = raw_array

    if missing_property_features:
        started = time.perf_counter_ns()
        feature_frame = pd.DataFrame(raw_array, columns=service.model_features)
        scaled_array = scaler.transform(feature_frame)
        timings_ms["scaling"] = elapsed_ms(started)

        started = time.perf_counter_ns()
        imputed_scaled = imputer.transform(scaled_array)
        timings_ms["knn_imputation"] = elapsed_ms(started)

        started = time.perf_counter_ns()
        inverse_scaled = scaler.inverse_transform(imputed_scaled)
        model_input_array = np.where(np.isnan(raw_array), inverse_scaled, raw_array)
        timings_ms["inverse_transformation"] = elapsed_ms(started)
    else:
        timings_ms["scaling"] = 0.0
        timings_ms["knn_imputation"] = 0.0
        timings_ms["inverse_transformation"] = 0.0

    started = time.perf_counter_ns()
    model_input = pd.DataFrame(model_input_array, columns=service.model_features)
    timings_ms["dataframe_creation"] = elapsed_ms(started)

    started = time.perf_counter_ns()
    prediction = service._model.predict(model_input)
    timings_ms["model_prediction"] = elapsed_ms(started)

    started = time.perf_counter_ns()
    float(prediction[0])
    timings_ms["response_serialization"] = elapsed_ms(started)

    return timings_ms


async def run_http_benchmarks(
    *,
    base_url: str,
    total_requests: int,
    concurrency: int,
) -> dict:
    payloads = load_http_payloads()
    complete_payloads = payloads["complete"]
    missing_payloads = payloads["missing"]
    mixed_payloads = build_mixed_payloads(complete_payloads, missing_payloads)

    return {
        "complete_traffic": await benchmark_http_scenario(
            base_url, complete_payloads, total_requests, concurrency
        ),
        "missing_traffic": await benchmark_http_scenario(
            base_url, missing_payloads, total_requests, concurrency
        ),
        "mixed_traffic": await benchmark_http_scenario(
            base_url, mixed_payloads, total_requests, concurrency
        ),
    }


def load_http_payloads() -> dict[str, list[dict]]:
    df = pd.read_csv("src/data/future_unseen_examples.csv", dtype={"zipcode": str})
    complete_payloads = (
        df.loc[:, [*REQUEST_PROPERTY_FEATURES, "zipcode"]]
        .head(60)
        .to_dict(orient="records")
    )

    missing_payloads = []
    for idx, payload in enumerate(complete_payloads):
        candidate = dict(payload)
        if idx % 3 == 0:
            candidate["bathrooms"] = None
        elif idx % 3 == 1:
            candidate["sqft_living"] = None
            candidate["sqft_above"] = None
            candidate["sqft_basement"] = None
        else:
            for feature in REQUEST_PROPERTY_FEATURES:
                candidate[feature] = None
        missing_payloads.append(candidate)

    return {"complete": complete_payloads, "missing": missing_payloads}


def build_mixed_payloads(
    complete_payloads: list[dict],
    missing_payloads: list[dict],
) -> list[dict]:
    mixed = []
    for complete, missing in zip(complete_payloads, missing_payloads):
        mixed.append(complete)
        mixed.append(missing)
    return mixed


async def benchmark_http_scenario(
    base_url: str,
    payloads: list[dict],
    total_requests: int,
    concurrency: int,
) -> dict:
    latencies: list[float] = []
    error_count = 0
    semaphore = asyncio.Semaphore(concurrency)
    started_at = time.perf_counter()

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        async def run_one(request_index: int) -> None:
            nonlocal error_count
            payload = payloads[request_index % len(payloads)]
            async with semaphore:
                request_started_at = time.perf_counter()
                response = await client.post("/predict", json=payload)
                latency_ms = (time.perf_counter() - request_started_at) * 1000
                latencies.append(latency_ms)
                if response.status_code != 200:
                    error_count += 1

        await asyncio.gather(*(run_one(i) for i in range(total_requests)))

    total_elapsed_s = time.perf_counter() - started_at
    success_count = total_requests - error_count
    return {
        "request_count": total_requests,
        "concurrency": concurrency,
        "error_rate_pct": round((error_count / total_requests) * 100, 3),
        "requests_per_second": round(success_count / total_elapsed_s, 3),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
    }


def measure_latency(fn, iterations: int) -> dict[str, float]:
    latencies = []
    for _ in range(iterations):
        started_at = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - started_at) * 1000)
    return summarize_timings(latencies)


def summarize_timings(latencies: list[float]) -> dict[str, float]:
    return {
        "avg_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": percentile(latencies, 95),
    }


def percentile(latencies: list[float], q: float) -> float:
    return round(float(np.percentile(latencies, q)), 3)


def elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000


if __name__ == "__main__":
    main()
