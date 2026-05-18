"""Client registry."""

from typing import Callable, Hashable

from nekomata.clients.base import ClientABC
from nekomata.const import SUPPORTED_PROVIDERS
from nekomata.utils.registry import Registry

PACKAGE_DEFINED: set[str] = SUPPORTED_PROVIDERS
CLIENT_REGISTRY = Registry[type[ClientABC]]()


def register_client(name: str) -> Callable[[type[ClientABC]], type[ClientABC]]:
    """Register a user defined client implementation to be used in the package."""
    if name in PACKAGE_DEFINED:
        raise ValueError(f'"{name}" is a predefined client name. Use an alternative name.')

    def register_and_pass_through(client_cls: type[ClientABC]) -> type[ClientABC]:
        if not issubclass(client_cls, ClientABC):
            raise TypeError('The registered client class must be a subclass of ClientABC.')
        CLIENT_REGISTRY.register(name, client_cls)
        return client_cls

    return register_and_pass_through


def get_client_entrypoint(name: str) -> type[ClientABC]:
    """Get client class by name."""
    return CLIENT_REGISTRY.get(name)


def list_client_keys() -> list[Hashable]:
    """List known client keys."""
    package_keys: list[str] = list(PACKAGE_DEFINED)
    registry_keys = CLIENT_REGISTRY.list_keys()
    return sorted(package_keys + registry_keys)
