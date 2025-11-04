import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console_level: str = "INFO"
    file_enabled: bool = True
    file_path: Optional[str] = None
    rotation: str = "500 MB"
    retention: str = "30 days"
    format: str = "json"


class SecurityConfig(BaseModel):
    api_key_enabled: bool = False
    api_key: str = ""
    api_key_header: str = "Token"
    cors_origins: List[str] = ["*"]


class PerformanceConfig(BaseModel):
    request_timeout: int = 300
    max_image_size: int = 0
    rate_limit_enabled: bool = False
    rate_limit: str = "100/minute"


class ConcurrencyConfig(BaseModel):
    max_workers: int = 4
    max_queue_size: int = 50


class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)

    @classmethod
    def load_from_yaml(cls, config_path: Path) -> "Settings":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Settings instance with loaded configuration.
        """
        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        if data.get("security", {}).get("api_key") == "":
            env_key = os.getenv("XANYLABELING_API_KEY", "")
            if env_key:
                data["security"]["api_key"] = env_key

        return cls(**data)


def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings instance.
    """
    config_path = (
        Path(__file__).parent.parent.parent / "configs" / "server.yaml"
    )
    return Settings.load_from_yaml(config_path)
