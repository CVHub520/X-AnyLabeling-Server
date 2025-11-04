import math
from pydantic import BaseModel, Field, field_validator
from typing import Any, ClassVar, Dict, List, Literal, Optional


class Shape(BaseModel):
    """Shape data schema for annotations."""

    SUPPORTED_SHAPES: ClassVar[List[str]] = [
        "polygon",
        "rectangle",
        "rotation",
        "point",
        "line",
        "circle",
        "linestrip",
    ]

    label: str
    shape_type: Literal[
        "polygon",
        "rectangle",
        "rotation",
        "point",
        "line",
        "circle",
        "linestrip",
    ]
    points: List[List[float]]
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    difficult: bool = False
    direction: float = Field(0, ge=0.0, le=2 * math.pi)
    flags: Optional[Dict[str, Any]] = None
    group_id: Optional[int] = Field(None, gt=0)
    kie_linking: List[Any] = Field(default_factory=list)

    @field_validator("group_id")
    @classmethod
    def validate_group_id(cls, v: Optional[int]) -> Optional[int]:
        """Validate that group_id is a positive integer when provided."""
        if v is not None and v <= 0:
            raise ValueError("group_id must be a positive integer")
        return v

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: Optional[float]) -> Optional[float]:
        """Validate that score is between 0 and 1 when provided."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("score must be between 0.0 and 1.0")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: float) -> float:
        """Validate that direction is between 0 and 2π radians."""
        if v < 0.0 or v > 2 * math.pi:
            raise ValueError(
                f"direction must be between 0 and 2π radians (0 to {2 * math.pi})"
            )
        return v

    class Config:
        extra = "allow"
