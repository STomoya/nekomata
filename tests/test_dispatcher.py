"""Tests for the AsyncLLMDispatcher."""

import time

import anyio
import pytest
from openai import OpenAIError
from pydantic import BaseModel
from pytest_mock import MockerFixture

from nekomata.dispatcher.dispatcher import AsyncLLMDispatcher
from nekomata.types.dispatcher import EndpointConfig
from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus


class TestAsyncLLMDispatcher:
    """Test suite for AsyncLLMDispatcher."""

    @pytest.fixture
    def mock_create_client(self, mocker: MockerFixture):
        """Mock create_client factory."""
        mock = mocker.patch('nekomata.dispatcher.dispatcher.create_client')
        return mock

    @pytest.mark.anyio
    async def test_registor_endpoint(self):
        """Test registoration."""
        dispatcher = AsyncLLMDispatcher()
        dispatcher.register_endpoint(
            name='test',
            provider='openai',
            base_url='http://localhost:8000/v1',
            api_key='local',
            max_concurrent=10,
            max_connections=10,
            max_keepalive=10,
            keepalive_expiry=10,
            timeout=10,
            ssl_verify=True,
        )

        assert 'test' in dispatcher._configs
        config = dispatcher._configs['test']
        assert isinstance(config, EndpointConfig)
        assert config.name == 'test'
        assert config.provider == 'openai'
        assert config.base_url == 'http://localhost:8000/v1'
        assert config.api_key == 'local'
        assert config.max_concurrent == 10  # noqa: PLR2004
        assert config.max_connections == 10  # noqa: PLR2004
        assert config.max_keepalive == 10  # noqa: PLR2004
        assert config.keepalive_expiry == 10  # noqa: PLR2004
        assert config.timeout == 10  # noqa: PLR2004
        assert config.ssl_verify is True

    @pytest.mark.anyio
    async def test_lazy_client_creation(self, mocker: MockerFixture, mock_create_client):
        """Test lazy client creation."""
        mock_client = mocker.AsyncMock()
        mock_create_client.return_value = mock_client

        dispatcher = AsyncLLMDispatcher()
        dispatcher.register_endpoint(name='test', provider='openai', api_key='local')

        # Check that the client is not instanciated at this point.
        mock_create_client.assert_not_called()
        assert dispatcher._clients == {}

        async with dispatcher:
            await dispatcher.submit(endpoint_name='test', model='model', prompt='prompt')

            # First call to submit should trigger the client creation.
            mock_create_client.assert_called_once()
            assert 'test' in dispatcher._clients
            client = dispatcher._clients['test']
            assert client == mock_client

            await dispatcher.submit(endpoint_name='test', model='model', prompt='prompt')

            # The second call should reuse the client.
            assert mock_create_client.call_count == 1

    @pytest.mark.anyio
    async def test_submit_success(self, mocker: MockerFixture, mock_create_client) -> None:
        """Test registering an endpoint and submitting a request."""
        mock_client = mocker.AsyncMock()
        mock_create_client.return_value = mock_client

        # Mock acompletion return value
        mock_response = ChatCompletionResponse(
            created_at=int(time.time()),
            elapsed=10,
            content='Hello',
            finish_reason='stop',
        )
        mock_client.acompletion.return_value = mock_response

        class MyResponse(BaseModel):
            answer: str

        async with AsyncLLMDispatcher() as dispatcher:
            dispatcher.register_endpoint(name='test', provider='openai', api_key='sk-test')

            response = await dispatcher.submit(
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
            )

            assert response == mock_response
            assert response.status == ChatCompletionStatus.SUCCESS
            mock_create_client.assert_called_once()
            mock_client.acompletion.assert_called_once_with(
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
            )

    @pytest.mark.anyio
    async def test_submit_failed(self, mocker: MockerFixture, mock_create_client) -> None:
        """Test registering an endpoint and submitting a request."""
        mock_client = mocker.AsyncMock()
        mock_create_client.return_value = mock_client

        mock_logger = mocker.patch('nekomata.dispatcher.dispatcher.logger')

        mock_response = ChatCompletionResponse(
            status=ChatCompletionStatus.FAILED,
            created_at=time.time(),
            elapsed=10,
            content=None,
            finish_reason=None,
        )
        # Mock acompletion return value
        mock_client.acompletion.return_value = mock_response

        async with AsyncLLMDispatcher() as dispatcher:
            dispatcher.register_endpoint(name='test', provider='openai', api_key='sk-test')

            response = await dispatcher.submit(
                endpoint_name='test',
                model='gpt-4',
                prompt='Say hello',
            )

            assert isinstance(response, ChatCompletionResponse)
            assert response.status == ChatCompletionStatus.FAILED
            mock_logger.debug.assert_called_with("Request to 'test' failed. Check error logs.")

    @pytest.mark.anyio
    async def test_submit_unregistered_endpoint(self) -> None:
        """Test submitting to an unregistered endpoint raises ValueError."""
        async with AsyncLLMDispatcher() as dispatcher:
            with pytest.raises(ValueError, match="Endpoint 'invalid' not registered"):
                await dispatcher.submit(endpoint_name='invalid', model='m', prompt='p')

    @pytest.mark.anyio
    async def test_close_clients(self, mocker: MockerFixture, mock_create_client) -> None:
        """Test that close() calls aclose() on all registered clients."""
        mock_client1 = mocker.AsyncMock()
        mock_client2 = mocker.AsyncMock()
        mock_clients = [mock_client1, mock_client2]
        mock_create_client.side_effect = mock_clients

        dispatcher = AsyncLLMDispatcher()
        dispatcher.register_endpoint(name='ep1', provider='openai')
        dispatcher.register_endpoint(name='ep2', provider='openai')

        # Trigger client creation via private method for testing
        dispatcher._get_or_create_client('ep1')
        dispatcher._get_or_create_client('ep2')

        await dispatcher.close()
        # Should call close on all client instances.
        for mock_client in mock_clients:
            mock_client.aclose.assert_called_once()
        assert len(dispatcher._clients) == 0

    @pytest.mark.anyio
    async def test_context_manager(self, mocker: MockerFixture, mock_create_client) -> None:
        """Test that the context manager calls close()."""
        mock_client = mocker.AsyncMock()
        mock_create_client.return_value = mock_client

        async with AsyncLLMDispatcher() as dispatcher:
            dispatcher.register_endpoint(name='ep1', provider='openai')
            dispatcher._get_or_create_client('ep1')

        mock_client.aclose.assert_called_once()

    @pytest.mark.anyio
    async def test_submit_package_error(self, mocker: MockerFixture, mock_create_client) -> None:
        """Test that client package errors are logged and re-raised."""
        mock_client = mocker.AsyncMock()
        mock_create_client.return_value = mock_client

        mock_client.acompletion.side_effect = OpenAIError('OpenAI Error')

        async with AsyncLLMDispatcher() as dispatcher:
            dispatcher.register_endpoint(name='ep1', provider='openai')
            with pytest.raises(OpenAIError):
                await dispatcher.submit(endpoint_name='ep1', model='m', prompt='p')

    @pytest.mark.anyio
    async def test_submit_unexpected_error(self, mocker: MockerFixture, mock_create_client) -> None:
        """Test that unexpected errors are logged and re-raised."""
        mock_client = mocker.AsyncMock()
        mock_create_client.return_value = mock_client

        mock_client.acompletion.side_effect = RuntimeError('Unexpected')

        async with AsyncLLMDispatcher() as dispatcher:
            dispatcher.register_endpoint(name='ep1', provider='openai')
            with pytest.raises(RuntimeError):
                await dispatcher.submit(endpoint_name='ep1', model='m', prompt='p')

    @pytest.mark.anyio
    async def test_submit_cancellation(self, mocker: MockerFixture, mock_create_client) -> None:
        """Test that the dispatcher handles cancellation correctly."""
        mock_client = mocker.AsyncMock()
        mock_create_client.return_value = mock_client

        cancel_exc = anyio.get_cancelled_exc_class()
        mock_client.acompletion.side_effect = cancel_exc()

        async with AsyncLLMDispatcher() as dispatcher:
            dispatcher.register_endpoint(name='ep1', provider='openai')
            with pytest.raises(cancel_exc):
                await dispatcher.submit(endpoint_name='ep1', model='m', prompt='p')
