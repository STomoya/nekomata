"""Tests for the SyncLLMDispatcher."""

import concurrent.futures

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from nekomata.dispatcher.sync import SyncLLMDispatcher


class TestSyncLLMDispatcher:
    """Test suite for SyncLLMDispatcher."""

    def test_start_stop(self, mocker) -> None:
        """Test starting and stopping the dispatcher."""
        # Mock anyio portal functions
        mock_start_portal = mocker.patch('nekomata.dispatcher.sync.start_blocking_portal')
        mock_cm = mock_start_portal.return_value
        mock_portal = mock_cm.__enter__.return_value

        # Mock async dispatcher
        _mock_async_dispatcher_class = mocker.patch('nekomata.dispatcher.sync.AsyncLLMDispatcher')

        dispatcher = SyncLLMDispatcher()
        dispatcher.start()

        assert dispatcher._leases == 1
        mock_start_portal.assert_called_once()

        dispatcher.stop()
        assert dispatcher._leases == 0
        # The stop() method calls self._portal.call(self._async_dispatcher.close)
        mock_portal.call.assert_called_once_with(dispatcher._async_dispatcher.close)
        mock_cm.__exit__.assert_called_once()

    def test_submit(self, mocker) -> None:
        """Test submitting a request."""
        mocker.patch('nekomata.dispatcher.sync.start_blocking_portal')
        mock_portal = mocker.MagicMock()

        mock_future = concurrent.futures.Future()
        mock_portal.start_task_soon.return_value = mock_future

        dispatcher = SyncLLMDispatcher()
        # Manually set portal for testing
        dispatcher._portal = mock_portal

        class MyResponse(BaseModel):
            answer: str

        future = dispatcher.submit(
            endpoint_name='test',
            model='gpt-4',
            prompt='Say hello',
            response_format=MyResponse,
            system_prompt='system',
            max_output_tokens=100,
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            seed=42,
            reasoning_effort='high',
            extra_body={'chat_template_kwargs': {'enable_thinking': True}},
            custom_id='id-000',
            max_model_retry=2,
        )

        assert future == mock_future
        mock_portal.start_task_soon.assert_called_once_with(
            dispatcher._async_dispatcher.submit,
            'test',
            'gpt-4',
            'Say hello',
            MyResponse,
            'system',
            100,
            0.7,
            0.95,
            64,
            0.1,
            0.1,
            42,
            'high',
            {'chat_template_kwargs': {'enable_thinking': True}},
            'id-000',
            2,
        )

    def test_submit_not_running(self) -> None:
        """Test that submit raises RuntimeError if dispatcher is not running."""
        dispatcher = SyncLLMDispatcher()
        with pytest.raises(RuntimeError, match='Dispatcher is not running'):
            dispatcher.submit(endpoint_name='test', model='m', prompt='p')

    def test_stop_not_running(self) -> None:
        """Test that stop raises RuntimeError if dispatcher is not running."""
        dispatcher = SyncLLMDispatcher()
        with pytest.raises(RuntimeError, match='Dispatcher is not running'):
            dispatcher.stop()

    def test_context_manager(self, mocker) -> None:
        """Test context manager usage."""
        mock_start_portal = mocker.patch('nekomata.dispatcher.sync.start_blocking_portal')
        mock_cm = mock_start_portal.return_value

        with SyncLLMDispatcher() as dispatcher:
            assert dispatcher._leases == 1

        assert dispatcher._leases == 0
        mock_cm.__exit__.assert_called_once()

    def test_register_endpoint(self, mocker: MockerFixture) -> None:
        """Test registering an endpoint through the sync dispatcher."""
        _mock_async_dispatcher_class = mocker.patch('nekomata.dispatcher.sync.AsyncLLMDispatcher')
        dispatcher = SyncLLMDispatcher()

        dispatcher.register_endpoint(name='test', provider='openai')

        dispatcher._async_dispatcher.register_endpoint.assert_called_once_with(name='test', provider='openai')  # ty: ignore[unresolved-attribute]

    def test_stop_close_failure(self, mocker: MockerFixture) -> None:
        """Test that failure during close() is logged but doesn't prevent portal shutdown."""
        mock_start_portal = mocker.patch('nekomata.dispatcher.sync.start_blocking_portal')
        mock_cm = mock_start_portal.return_value
        mock_portal = mock_cm.__enter__.return_value

        # Mock portal.call to raise an exception
        mock_portal.call.side_effect = Exception('Close error')

        # Mock logger to verify exception is logged
        mock_logger = mocker.patch('nekomata.dispatcher.sync.logger')

        dispatcher = SyncLLMDispatcher()
        dispatcher.start()

        # Should not raise exception
        dispatcher.stop()

        assert dispatcher._leases == 0
        mock_logger.exception.assert_called_once_with('Error closing async dispatcher during shutdown.')
        # Portal cleanup should still happen
        mock_cm.__exit__.assert_called_once()
        assert dispatcher._portal is None
        assert dispatcher._portal_cm is None
