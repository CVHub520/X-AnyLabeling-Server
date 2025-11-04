import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import Dict, Any

from app.core.registry import ModelRegistry


class InferenceExecutor:
    """Inference task executor with concurrency control."""

    def __init__(
        self, loader: ModelRegistry, max_workers: int, max_queue_size: int
    ):
        """Initialize inference executor.

        Args:
            loader: Model loader instance.
            max_workers: Maximum number of concurrent inference tasks.
            max_queue_size: Maximum task queue size.
        """
        self.loader = loader
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_queue_size = max_queue_size
        self._queue_size = 0

    async def execute(
        self, model_id: str, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute inference asynchronously.

        Args:
            model_id: Model identifier.
            image: Input image.
            params: Inference parameters.

        Returns:
            Inference results.

        Raises:
            RuntimeError: If queue is full.
        """
        if self._queue_size >= self.max_queue_size:
            raise RuntimeError("Task queue is full")

        self._queue_size += 1
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._run_inference, model_id, image, params
            )
            return result
        finally:
            self._queue_size -= 1

    def _run_inference(
        self, model_id: str, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run inference in thread pool.

        Args:
            model_id: Model identifier.
            image: Input image.
            params: Inference parameters.

        Returns:
            Inference results with shapes and description.
        """
        model = self.loader.get_model(model_id)
        return model.predict(image, params)

    def shutdown(self):
        """Shutdown executor and wait for pending tasks."""
        logger.info("Shutting down inference executor...")
        self.executor.shutdown(wait=True)
