from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as api_router
from services.inference import InferenceService
from utils.loader import ArtifactPaths


def create_app(
    *,
    inference_service: InferenceService | None = None,
    artifact_paths: ArtifactPaths | None = None,
) -> FastAPI:
    """Create the FastAPI application and initialize inference resources.

    Args:
        inference_service: Optional prebuilt inference service, primarily for tests.
        artifact_paths: Optional artifact locations used when loading resources.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load inference artifacts once per worker lifespan."""
        app.state.ready = False
        # Load artifacts once at startup so requests avoid repeated I/O and fail fast
        # if the model contract or required resources are invalid.
        app.state.inference_service = inference_service or InferenceService.from_artifacts(
            artifact_paths
        )
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)