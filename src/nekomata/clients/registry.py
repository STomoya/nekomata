"""Client registry."""

from typing import Callable, Hashable

from nekomata.clients.base import ClientABC
from nekomata.const import SUPPORTED_PROVIDERS
from nekomata.utils.registry import Registry

PACKAGE_DEFINED: set[str] = SUPPORTED_PROVIDERS
CLIENT_REGISTRY = Registry[type[ClientABC]]()


def register_client(name: str) -> Callable[[type[ClientABC]], type[ClientABC]]:
    """Register a user defined client implementation to be used in the package.

    Args:
        name (str): The name of the Client class to register. Must be identical. Names reserved by the package are
            "openai", "anthropic", and "google".

    Examples::

        from nekomata.clients import create_client
        from nekomata.clients.registry import register_client
        from nekomata.clients.base import ClientABC

        @register_client(name="cats-ai")
        class CatsAIClient(ClientABC):
            # Implement required functions.
            ...

        my_client: CatsAIClient = create_client("cats-ai", ...)

        # Registering reserved or duplicate names will raise an error:
        try:
            @register_client(name="openai")  # package reserved name
            class MyOpenAIClient(ClientABC):
                ...
            @register_client(name="cats-ai")  # duplicate name
            class CatsAIClient2(ClientABC):
                ...
        except Exception as e:
            print(e)  # ValueError

        # Client class must be a ClientABC subclass
        try:
            @register_client(name="not-subclass")
            class NotSubclass:
                pass
        except Exception as e:
            print(e)  # TypeError

        # The module with the client definition must be imported before creation.
        import my_client_module  # This will run the registration.
        client = create_client("my-client-name", ...)

    """
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
