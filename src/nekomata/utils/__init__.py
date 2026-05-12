"""Utilities."""

import datetime

from .logger import get_logger, setup_logger
from .misc import get_env
from .uuid import create_uuid


def get_utc_timestamp() -> float:
    """Get UTC UNIX timestamp."""
    return datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
