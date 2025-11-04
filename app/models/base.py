import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model.

        Args:
            config: Complete configuration dictionary from auto_labeling/{model_id}.yaml.
        """
        self.config = config
        self.model_id = config["model_id"]
        self.display_name = config["display_name"]
        self.params = config.get("params", {})

    @abstractmethod
    def load(self):
        """Load model into memory.

        Called by framework during service startup.
        Should raise exception if loading fails.
        """
        pass

    @abstractmethod
    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute inference.

        Args:
            image: Input image array
            params: Inference parameters from client request.

        Returns:
            Dictionary with inference results:
                {
                    "shapes": List[Dict[str, Any]],  # List of detected shapes (optional)
                    "description": str                # Text description (optional)
                }

            Shape dictionary fields:
                - label (str): Label name
                - shape_type (str): Shape type (rectangle, polygon, etc.)
                - points (List[List[float]]): Coordinate points
                - score (float): Confidence score
                - attributes (Dict[str, Any]): Additional attributes
                - description (str | None): Description
                - difficult (bool): Difficult flag
                - direction (int): Direction
                - flags (Dict | None): Flags
                - group_id (int | None): Group ID
                - kie_linking (List): KIE linking
        """
        pass

    @abstractmethod
    def unload(self):
        """Unload model and free resources.

        Called by framework during service shutdown.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata for /v1/models endpoint.

        Returns:
            Dictionary containing model information.
        """
        return {
            "display_name": self.display_name,
            "widgets": self.config.get("widgets", []),
            "params": self.params,
        }
