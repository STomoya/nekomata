"""Logger."""

import logging
from typing import Literal

# Type alias for acceptable logging levels
type LogLevel = int | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

# Add a NullHandler to the 'nekomata' root logger to avoid warnings when the library
# is used without logging configuration.
logging.getLogger('nekomata').addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Internal modules should use this to ensure logs propagate to the root 'nekomata' logger.

    Args:
        name (str): The name of the logger (typically `__name__`).

    Returns:
        logging.Logger: The logger instance.

    """
    return logging.getLogger(name)


def setup_logger(
    name: str = 'nekomata',
    level: LogLevel = logging.INFO,
    log_format: str | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """Configure and return a logging.Logger instance for the package.

    This is an opt-in utility for users who want a "batteries-included" logging setup.
    It supports console logging with a default detailed format.

    Args:
        name: The name of the logger. Defaults to 'nekomata'.
        level: The logging level to set. Defaults to logging.INFO.
        log_format: A custom logging format string. If None, a detailed default format is used.
        propagate: Whether to propagate logs to parent loggers. Defaults to False when manually configured.

    Returns:
        logging.Logger: The fully configured logger instance.

    """
    logger = logging.getLogger(name)

    # Set the base logging level
    if isinstance(level, str):
        level = logging.getLevelNamesMapping()[level.upper()]
    logger.setLevel(level)

    # Control propagation
    logger.propagate = propagate

    # Idempotency: Clear existing handlers to prevent duplicate log lines
    if logger.hasHandlers():
        # Only clear handlers if we are re-configuring (to avoid clearing NullHandler alone if it's the only one)
        # But usually in setup_logger, we want a fresh start for this specific name.
        logger.handlers.clear()

    # Define the log format
    if log_format is None:
        log_format = '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    formatter = logging.Formatter(log_format)

    # Configure Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
