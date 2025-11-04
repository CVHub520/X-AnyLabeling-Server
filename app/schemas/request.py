from pydantic import BaseModel
from typing import Dict, Any


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    model: str
    image: str
    params: Dict[str, Any] = {}
