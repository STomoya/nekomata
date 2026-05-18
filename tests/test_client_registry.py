"""Tests for client registry."""

import pytest

from nekomata.clients.base import ClientABC
from nekomata.clients.registry import (
    CLIENT_REGISTRY,
    PACKAGE_DEFINED,
    get_client_entrypoint,
    list_client_keys,
    register_client,
)


class DummyClient(ClientABC):
    """Dummy client."""


class TestClientRegistry:
    """Tests for the client registry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry."""
        yield
        CLIENT_REGISTRY._registry.clear()

    def test_register_client_success(self):
        """Test register_client decorator."""
        register_client(name='test')(DummyClient)

    def test_register_client_predefined(self):
        """Test decorator rejects known names."""
        with pytest.raises(ValueError):
            register_client('openai')(DummyClient)

    def test_register_client_dup_name(self):
        """Test decorator rejects duplicate names."""
        register_client(name='test')(DummyClient)
        with pytest.raises(ValueError):
            register_client(name='test')(DummyClient)

    def test_register_client_not_client_subclass(self):
        """Test decorator rejects non-ClientABC subclass registration."""

        class Dummy:
            pass

        with pytest.raises(TypeError):
            register_client(name='test')(Dummy)  # ty: ignore[invalid-argument-type]

    def test_get_client_entrypoint(self):
        """Test get_client_entrypoint."""
        register_client('test')(DummyClient)
        cls = get_client_entrypoint(name='test')
        assert cls == DummyClient

    def test_get_client_entrypoint_unkown_name(self):
        """Test get_client_entrypoint raises Error on unregitered name input."""
        with pytest.raises(KeyError):
            get_client_entrypoint(name='unkown')

    def test_list_client_keys(self):
        """Test list_client_keys."""
        keys = list_client_keys()
        # list_client_keys returns a sorted list.
        # We only want to check if the list is complete, so we convert to a set.
        assert set(keys) == PACKAGE_DEFINED

        register_client('test')(DummyClient)

        keys = list_client_keys()
        assert set(keys) == {'test', *PACKAGE_DEFINED}
