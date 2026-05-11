"""Tests for logger utility."""

import logging

from nekomata.utils import get_logger, setup_logger


def test_get_logger():
    """Test get_logger returns a logging.Logger with the correct name."""
    name = 'test_logger'
    logger = get_logger(name)
    assert isinstance(logger, logging.Logger)
    assert logger.name == name
    # Internal loggers should propagate by default
    assert logger.propagate is True


def test_root_logger_has_null_handler():
    """Test that the 'nekomata' root logger has a NullHandler."""
    root_logger = logging.getLogger('nekomata')
    handlers = root_logger.handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_setup_logger_configuration():
    """Test setup_logger correctly configures handlers and level."""
    logger_name = 'test_setup_logger'
    logger = setup_logger(name=logger_name, level='DEBUG', propagate=True)

    assert logger.level == logging.DEBUG
    assert logger.propagate is True

    # Check if StreamHandler was added
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    # Check idempotency (calling again shouldn't duplicate handlers)
    original_handler_count = len(logger.handlers)
    setup_logger(name=logger_name, level='INFO')
    assert len(logger.handlers) == original_handler_count
    assert logger.level == logging.INFO
