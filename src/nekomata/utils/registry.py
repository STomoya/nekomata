"""Thread-safe registry inplementation."""

import threading
from typing import Hashable, TypeVar

RegistryItemT = TypeVar('RegistryItemT')


class Registry[RegistryItemT]:
    """Thread-safe registry."""

    def __init__(self) -> None:
        """Construct registry."""
        self._registry: dict[Hashable, RegistryItemT] = {}
        self._lock: threading.Lock = threading.Lock()

    def register(self, key: Hashable, value: RegistryItemT) -> None:
        """Register an item to the registry.

        Args:
            key (Hashable): The unique identifier for the item.. Must be a hashable object.
            value (RegistryItemT): The item to register.

        """
        with self._lock:
            if key in self._registry:
                raise ValueError(f"Item with key '{key}' is already registered.")
            self._registry[key] = value

    def get(self, key: Hashable) -> RegistryItemT:
        """Retrieve the item from the registry using the corresponding key.

        Args:
            key (Hashable): The unique identifier for the item.. Must be a hashable object.

        """
        with self._lock:
            if key not in self._registry:
                raise KeyError(f"Item with key '{key}' not found in the registry.")
            return self._registry[key]

    def list_keys(self) -> list[Hashable]:
        """Return a list of all currently registered keys.

        Returns:
            list[Hashable]: A list of keys.

        """
        with self._lock:
            # We cast to list to evaluate the keys immediately while under the lock,
            # preventing "dictionary changed size during iteration" errors.
            return list(self._registry.keys())
