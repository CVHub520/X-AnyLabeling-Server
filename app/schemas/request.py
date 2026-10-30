from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    model: str
    image: str
    params: Dict[str, Any] = {}


class DocumentPredictRequest(BaseModel):
    model: str
    images: List[str] = Field(min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)


class VideoInitRequest(BaseModel):
    """Request schema for video session initialization."""

    model: str
    frames: List[str]
    start_frame_index: int = 0


class VideoPromptRequest(BaseModel):
    """Request schema for video frame prompt."""

    session_id: str
    model: str
    text_prompt: Optional[str] = None
    frame_index: int = 0
    points: Optional[List[List[float]]] = None
    point_labels: Optional[List[int]] = None
    obj_id: Optional[int] = None
    params: Dict[str, Any] = {}


class VideoPropagateRequest(BaseModel):
    """Request schema for video propagation."""

    session_id: str
    model: str
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
