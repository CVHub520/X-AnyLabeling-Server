import sys
from datetime import datetime
from loguru import logger
from pathlib import Path

from .config import LoggingConfig


def setup_logging(config: LoggingConfig):
    """Configure loguru logger based on settings.

    Args:
        config: Logging configuration.
    """
    logger.remove()

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=console_format,
        level=config.console_level,
        colorize=True,
    )

    if config.file_enabled:
        if config.file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"logs/server_{timestamp}.log"
        else:
            file_path = config.file_path

        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if config.format == "json":
            logger.add(
                file_path,
                format="{time} {level} {name} {function} {message}",
                level=config.level,
                rotation=config.rotation,
                retention=config.retention,
                serialize=True,
            )
        else:
            logger.add(
                file_path,
                format=console_format,
                level=config.level,
                rotation=config.rotation,
                retention=config.retention,
            )
