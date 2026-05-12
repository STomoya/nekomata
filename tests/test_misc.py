"""Tests for the misc utility module."""

import os
from unittest.mock import patch

from nekomata.utils.misc import get_env


class TestMiscUtils:
    """Test suite for misc utilities."""

    def test_get_env_success_str(self) -> None:
        """Test getting an existing environment variable as string."""
        with patch.dict(os.environ, {'TEST_VAR': 'hello'}):
            value = get_env('TEST_VAR', type=str, default='default')
            assert value == 'hello'
            assert isinstance(value, str)

    def test_get_env_success_int(self) -> None:
        """Test getting an existing environment variable as integer."""
        with patch.dict(os.environ, {'TEST_VAR': '123'}):
            value = get_env('TEST_VAR', type=int, default=0)
            assert value == 123  # noqa: PLR2004
            assert isinstance(value, int)

    def test_get_env_success_float(self) -> None:
        """Test getting an existing environment variable as float."""
        with patch.dict(os.environ, {'TEST_VAR': '1.23'}):
            value = get_env('TEST_VAR', type=float, default=0.0)
            assert value == 1.23  # noqa: PLR2004
            assert isinstance(value, float)

    def test_get_env_missing_with_default(self) -> None:
        """Test fallback to default when variable is missing."""
        if 'MISSING_VAR' in os.environ:
            del os.environ['MISSING_VAR']
        value = get_env('MISSING_VAR', type=int, default=42)
        assert value == 42  # noqa: PLR2004

    def test_get_env_missing_no_default(self) -> None:
        """Test returning None when variable is missing and no default is provided."""
        if 'MISSING_VAR' in os.environ:
            del os.environ['MISSING_VAR']
        value = get_env('MISSING_VAR', type=str, default=None)
        assert value is None
