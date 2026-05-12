"""Utilities."""

import datetime

from .logger import get_logger, setup_logger
from .uuid import create_uuid


def get_utc_timestamp() -> float:
    """Get UTC UNIX timestamp."""
    return datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
