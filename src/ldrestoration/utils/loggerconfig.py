import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
)

__all__ = ["logger"]
