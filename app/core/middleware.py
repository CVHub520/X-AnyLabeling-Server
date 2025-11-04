import time
from loguru import logger
from typing import Callable

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    def __init__(self, app, enabled: bool, api_key: str, header_name: str):
        """Initialize API key middleware.

        Args:
            app: FastAPI application.
            enabled: Whether authentication is enabled.
            api_key: Expected API key.
            header_name: Header name for API key.
        """
        super().__init__(app)
        self.enabled = enabled
        self.api_key = api_key
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with API key validation.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from next handler or 401 error.
        """
        if request.url.path == "/health":
            return await call_next(request)

        if self.enabled:
            provided_key = request.headers.get(self.header_name)
            if not provided_key or provided_key != self.api_key:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "success": False,
                        "error": {
                            "code": "UNAUTHORIZED",
                            "message": "Invalid or missing API Key",
                        },
                    },
                )

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware."""

    async def dispatch(self, request: Request, call_next: Callable):
        """Log request and response.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from handler.
        """
        start_time = time.time()

        logger.info(f"Request: {request.method} {request.url.path}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Duration: {process_time:.3f}s"
        )

        return response


def setup_cors(app, origins: list):
    """Setup CORS middleware.

    Args:
        app: FastAPI application.
        origins: Allowed origins.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
