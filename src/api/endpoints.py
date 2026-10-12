import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from services.inference import InferenceClientError
from services.inference import InferenceService
from services.inference import InferenceServiceError
from services.inference import UnsupportedZipcodeError

router = APIRouter()
logger = logging.getLogger(__name__)


class HomeFeatures(BaseModel):
    bedrooms: int | None = Field(default=None, ge=0, le=20)
    bathrooms: float | None = Field(default=None, ge=0, le=20)
    sqft_living: float | None = Field(default=None, gt=0, le=50000)
    sqft_lot: float | None = Field(default=None, gt=0, le=10_000_000)
    floors: float | None = Field(default=None, gt=0, le=10)
    sqft_above: float | None = Field(default=None, ge=0, le=50000)
    sqft_basement: float | None = Field(default=None, ge=0, le=50000)
    zipcode: str = Field(..., pattern=r"^\d{5}$")

    @field_validator("sqft_above", "sqft_basement")
    @classmethod
    def validate_component_area(
        cls, value: float | None, info: ValidationInfo
    ) -> float | None:
        if value is None:
            return value
        sqft_living = info.data.get("sqft_living") if info.data else None
        if sqft_living is not None and value > sqft_living:
            raise ValueError(f"{info.field_name} cannot exceed sqft_living.")
        return value


class PredictionResponse(BaseModel):
    predicted_price: float


def get_inference_service(request: Request) -> InferenceService:
    inference_service = getattr(request.app.state, "inference_service", None)
    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service is not initialized.",
        )
    return inference_service


@router.get("/health")
def health_check():
    """
    Health check endpoint for container orchestration.
    Returns 200 if the API process is alive.
    """
    return {"status": "healthy"}


@router.get("/ready")
def readiness_check(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Inference resources are not ready.")
    get_inference_service(request)
    return {"status": "ready"}


@router.post("/predict")
def predict(home_features: HomeFeatures, request: Request) -> PredictionResponse:
    inference_service = get_inference_service(request)
    payload = home_features.model_dump()

    try:
        prediction = inference_service.predict(payload)
    except (UnsupportedZipcodeError, InferenceClientError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceServiceError as exc:
        logger.exception("Prediction failed for zipcode %s", payload["zipcode"])
        raise HTTPException(
            status_code=500,
            detail="Prediction service is temporarily unavailable.",
        ) from exc

    return PredictionResponse(predicted_price=prediction)
