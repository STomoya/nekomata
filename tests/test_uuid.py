"""Tests for the UUID utility module."""

import pytest

from nekomata.utils.uuid import create_uuid, fastuuid4, fastuuid7

UUID_STRING_LENGTH = 36


class TestUUIDUtils:
    """Test suite for UUID utilities."""

    def test_fastuuid4(self) -> None:
        """Test fastuuid4 returns a valid UUID4 string."""
        u = fastuuid4()
        assert isinstance(u, str)
        # Simple check for UUID length
        assert len(u) == UUID_STRING_LENGTH

    def test_fastuuid7(self) -> None:
        """Test fastuuid7 returns a valid UUID7 string."""
        u = fastuuid7()
        assert isinstance(u, str)
        # Simple check for UUID length
        assert len(u) == UUID_STRING_LENGTH

    def test_create_uuid_v4(self) -> None:
        """Test create_uuid with version 4."""
        u = create_uuid(version=4)
        assert isinstance(u, str)
        assert len(u) == UUID_STRING_LENGTH

    def test_create_uuid_v7(self) -> None:
        """Test create_uuid with version 7."""
        u = create_uuid(version=7)
        assert isinstance(u, str)
        assert len(u) == UUID_STRING_LENGTH

    def test_create_uuid_str_version(self) -> None:
        """Test create_uuid with version as string."""
        u = create_uuid(version='4')
        assert isinstance(u, str)
        assert len(u) == UUID_STRING_LENGTH

    def test_create_uuid_unsupported(self) -> None:
        """Test create_uuid with unsupported version."""
        with pytest.raises(ValueError, match='Unkown or unsupported UUID version'):
            create_uuid(version=1)
