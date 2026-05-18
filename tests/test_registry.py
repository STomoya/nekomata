"""Tests for registry."""

import concurrent
import concurrent.futures

import pytest
from pytest_mock import MockerFixture

from nekomata.utils.registry import Registry


@pytest.fixture
def registry() -> Registry[str]:
    """Fixture to provide a clean registry instance for each test.

    Returns:
        Registry[str]: A new instance of the Registry class typed for strings.

    """
    return Registry[str]()


class TestRegistry:
    """Test suite for the Registry class."""

    def test_register_and_get_success(self, registry: Registry[str]) -> None:
        """Test successful registration and retrieval of an item."""
        registry.register('service_a', 'Service A Instance')
        result = registry.get('service_a')
        assert result == 'Service A Instance'

    def test_register_duplicate_key_raises_error(self, registry: Registry[str]) -> None:
        """Test that registering a duplicate key raises a ValueError."""
        registry.register('service_a', 'Instance 1')

        with pytest.raises(ValueError, match='is already registered'):
            registry.register('service_a', 'Instance 2')

    def test_get_missing_key_raises_error(self, registry: Registry[str]) -> None:
        """Test that retrieving a non-existent key raises a KeyError."""
        with pytest.raises(KeyError, match='not found in the registry'):
            registry.get('missing_service')

    def test_list_keys(self, registry: Registry[str]) -> None:
        """Test that list_keys accurately returns all registered keys."""
        registry.register('service_a', 'Instance 1')
        registry.register('service_b', 'Instance 2')

        keys = registry.list_keys()

        assert len(keys) == 2  # noqa: PLR2004
        assert 'service_a' in keys
        assert 'service_b' in keys

    # --- Threading & Concurrency Tests ---

    def test_concurrent_registration_distinct_keys(self, registry: Registry[str]) -> None:
        """Test registering multiple distinct keys concurrently.

        Ensures that concurrent write operations do not corrupt internal state.
        """
        num_threads = 100

        def register_task(index: int) -> None:
            registry.register(f'key_{index}', f'value_{index}')

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_task, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        assert len(registry.list_keys()) == num_threads
        assert registry.get('key_50') == 'value_50'

    def test_concurrent_registration_same_key(self, registry: Registry[str]) -> None:
        """Test race condition mitigation when multiple threads register the same key.

        If 100 threads try to register "duplicate_key" at the exact same time,
        exactly 1 should succeed, and 99 should raise a ValueError.
        """
        num_threads = 100
        success_count = 0
        value_error_count = 0

        def register_task() -> None:
            registry.register('duplicate_key', 'some_value')

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_task) for i in range(num_threads)]

            # Tally up exceptions directly from the futures
            for future in concurrent.futures.as_completed(futures):
                exc = future.exception()
                if exc is None:
                    success_count += 1
                elif isinstance(exc, ValueError):
                    value_error_count += 1
                else:
                    pytest.fail(f'Unexpected exception raised: {exc}')

        assert success_count == 1, 'Exactly one thread should have succeeded.'
        assert value_error_count == num_threads - 1, 'All other threads must fail with ValueError.'
        assert registry.get('duplicate_key') == 'some_value'

    def test_concurrent_read_and_write(self, registry: Registry[str]) -> None:
        """Test reading keys while other threads are actively registering items.

        This explicitly checks for `RuntimeError: dictionary changed size during iteration`
        which would occur if `list_keys` didn't properly use locks and evaluate immediately.
        """
        num_writers = 50
        num_readers = 50

        def write_task(index: int) -> None:
            registry.register(f'concurrent_key_{index}', 'value')

        def read_task() -> int:
            keys = registry.list_keys()
            return len(keys)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_writers + num_readers) as executor:
            # Submit both write and read tasks simultaneously
            futures = []
            for i in range(num_writers):
                futures.append(executor.submit(write_task, i))
                futures.append(executor.submit(read_task))

            # If a RuntimeError occurred in a read thread, getting its result will raise it
            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(registry.list_keys()) == num_writers

    def test_mocking_lock_interaction(self, registry: Registry[str], mocker: MockerFixture) -> None:
        """Demonstrate pytest-mock usage by replacing the lock to verify it gets acquired.

        Note: We replace `self._lock` with a MagicMock instead of spying on the existing
        `threading.Lock` because C-extension objects cannot always be spied upon directly.
        """
        mock_lock = mocker.MagicMock()
        registry._lock = mock_lock  # Inject the mock

        registry.register('mock_key', 'mock_val')

        # Verify the mock lock was used as a context manager (__enter__/__exit__)
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
