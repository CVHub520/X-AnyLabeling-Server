from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from .shape import Shape


class ErrorDetail(BaseModel):
    """Error detail schema."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Success response schema."""

    success: bool = True
    data: Any


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = False
    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    models_loaded: int
    timestamp: str


class Relation(BaseModel):
    subject_id: str
    object_id: str
    label: str
    score: Optional[float] = Field(None, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Prediction response data schema."""

    shapes: List[Shape] = []
    description: str = ""
    replace: Optional[bool] = None
    relations: Optional[List[Relation]] = None

    class Config:
        extra = "allow"
        exclude_none = True
