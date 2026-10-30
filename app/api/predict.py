import base64
import cv2
import numpy as np
from loguru import logger

from fastapi import APIRouter, HTTPException, status

from app.schemas.request import DocumentPredictRequest, PredictRequest
from app.schemas.response import (
    ErrorDetail,
    ErrorResponse,
    PredictResponse,
    SuccessResponse,
)

router = APIRouter()


@router.post("/v1/predict/document")
async def predict_document(request: DocumentPredictRequest):
    from app.main import loader, inference_executor

    try:
        model = loader.get_model(request.model)
    except ValueError as exc:
        return ErrorResponse(
            error=ErrorDetail(code="MODEL_NOT_FOUND", message=str(exc))
        )
    if not callable(getattr(model, "predict_document", None)):
        return ErrorResponse(
            error=ErrorDetail(
                code="UNSUPPORTED_TASK",
                message="Model does not support document inference",
            )
        )
    images = []
    try:
        for encoded in request.images:
            image = cv2.imdecode(
                np.frombuffer(
                    base64.b64decode(encoded.split(",", 1)[-1], validate=True),
                    np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError("Failed to decode document page")
            images.append(image)
    except (ValueError, cv2.error) as exc:
        return ErrorResponse(
            error=ErrorDetail(code="INVALID_IMAGE", message=str(exc))
        )
    try:
        result = await inference_executor.execute(
            request.model, images, request.params, document=True
        )
        return SuccessResponse(data=result)
    except RuntimeError as exc:
        if "queue is full" in str(exc).lower():
            raise HTTPException(status_code=503, detail=str(exc))
        return ErrorResponse(
            error=ErrorDetail(code="INFERENCE_ERROR", message=str(exc))
        )
    except Exception as exc:
        logger.error(f"Document inference error: {exc}")
        return ErrorResponse(
            error=ErrorDetail(code="INFERENCE_ERROR", message=str(exc))
        )


@router.post("/v1/predict")
async def predict(request: PredictRequest):
    """Execute prediction on an image.

    Args:
        request: Prediction request with model, image, and parameters.

    Returns:
        Success response with prediction results or error response.
    """
    from app.main import loader, inference_executor

    try:
        _ = loader.get_model(request.model)
    except ValueError as e:
        return ErrorResponse(
            error=ErrorDetail(code="MODEL_NOT_FOUND", message=str(e))
        )

    try:
        image_data = (
            request.image.split(",")[1]
            if "," in request.image
            else request.image
        )
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_IMAGE", message="Failed to decode image"
                )
            )
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code="INVALID_IMAGE",
                message=f"Failed to decode image: {str(e)}",
            )
        )

    try:
        result = await inference_executor.execute(
            request.model, image, request.params
        )
        return SuccessResponse(data=PredictResponse(**result))
    except RuntimeError as e:
        if "queue is full" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task queue is full, please try again later",
            )
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return ErrorResponse(
            error=ErrorDetail(code="INFERENCE_ERROR", message=str(e))
        )
