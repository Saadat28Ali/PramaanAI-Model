"""FastAPI REST + WebSocket API for DocuNet."""

import cv2
import numpy as np
import io
import os
import uuid
import base64
import time
import asyncio
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from starlette.middleware.base import BaseHTTPMiddleware

from src.config import DocuNetConfig
from src.pipeline import DocuNetPipeline


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject X-Request-ID into every request/response for log correlation."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        with logger.contextualize(request_id=request_id):
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response


app = FastAPI(
    title="DocuNet API",
    description="ID Card Tamper Detection & Robust OCR Pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Environment-driven CORS
_allowed_origins = os.environ.get(
    "DOCUNET_CORS_ORIGINS", "http://localhost:3000,http://localhost:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(RequestIDMiddleware)

# Rate limiting (requires slowapi)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _HAS_LIMITER = True
except ImportError:  # pragma: no cover
    _HAS_LIMITER = False
    logger.warning("slowapi not installed — API rate limiting disabled")

# Lazy-load the pipeline
_pipeline: Optional[DocuNetPipeline] = None


def get_pipeline() -> DocuNetPipeline:
    """Get or initialize the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        config = DocuNetConfig.default()
        _pipeline = DocuNetPipeline(config)
    return _pipeline


class VerifyResponse(BaseModel):
    success: bool
    stage_reached: str
    error_message: Optional[str] = None
    total_time_ms: float
    quality: Optional[dict] = None
    rectification: Optional[dict] = None
    glare: Optional[dict] = None
    tamper_detection: Optional[dict] = None
    ocr: Optional[dict] = None
    document: Optional[dict] = None
    timings: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    pipeline_loaded: bool




def decode_upload(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded file bytes to OpenCV image with size validation."""
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
        )
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Supported formats: JPEG, PNG, BMP.",
        )
    return image


def encode_image_base64(image: np.ndarray, format: str = ".jpg") -> str:
    """Encode OpenCV image to base64 string."""
    _, buffer = cv2.imencode(format, image)
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        pipeline_loaded=_pipeline is not None,
    )


@app.post("/api/v1/verify", response_model=VerifyResponse)
async def verify_document(
    file: UploadFile = File(...),
    skip_quality_gate: bool = False,
):
    """Verify a single document image through the full pipeline."""
    logger.info(f"Received verification request: {file.filename}")

    # Read and decode
    file_bytes = await file.read()
    image = decode_upload(file_bytes)

    # Process
    pipeline = get_pipeline()
    result = pipeline.process(image, skip_quality_gate=skip_quality_gate)

    return VerifyResponse(**result.to_dict())


@app.post("/api/v1/ela-only")
async def ela_analysis(file: UploadFile = File(...)):
    """Run only ELA tamper detection (no OCR) and return a base64 heatmap."""
    file_bytes = await file.read()
    image = decode_upload(file_bytes)

    pipeline = get_pipeline()
    result = pipeline.process(image, skip_ocr=True, skip_quality_gate=True)

    response = {
        "tamper_detection": result.ela_result.to_dict() if result.ela_result else None,
    }

    if result.ela_result and result.ela_result.heatmap is not None:
        response["heatmap_base64"] = encode_image_base64(result.ela_result.heatmap)

    if "ela_overlay" in result.images:
        response["overlay_base64"] = encode_image_base64(result.images["ela_overlay"])

    return JSONResponse(content=response)


@app.post("/api/v1/batch")
async def batch_verify(files: List[UploadFile] = File(...)):
    """Batch verification of multiple document images."""
    logger.info(f"Received batch request: {len(files)} images")

    pipeline = get_pipeline()
    results = []

    for file in files:
        try:
            file_bytes = await file.read()
            image = decode_upload(file_bytes)
            result = pipeline.process(image)
            results.append({
                "filename": file.filename,
                "result": result.to_dict(),
            })
        except (ValueError, RuntimeError, cv2.error) as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })

    return JSONResponse(content={"results": results, "total": len(results)})


@app.websocket("/ws/live-capture")
async def live_capture(websocket: WebSocket):
    """WebSocket endpoint for real-time camera quality feedback and smart capture."""
    await websocket.accept()

    pipeline = get_pipeline()
    quality_gate = pipeline.quality_gate

    logger.info("WebSocket live-capture session started")

    try:
        while True:
            data = await websocket.receive_text()

            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Could not decode frame",
                    })
                    continue

                # Quick quality check
                report = quality_gate.evaluate(frame)

                if report.passed:
                    await websocket.send_json({
                        "type": "ready",
                        "message": "Quality OK — capturing...",
                        "quality": report.to_dict(),
                    })

                    result = pipeline.process(frame)

                    await websocket.send_json({
                        "type": "result",
                        "data": result.to_dict(),
                    })
                else:
                    await websocket.send_json({
                        "type": "guidance",
                        "quality": report.to_dict(),
                        "issues": report.issues,
                    })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket live-capture session ended")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
