"""Tests for the Anthropic client."""

import pytest
from anthropic import Omit
from anthropic.types import Message, ParsedMessage, TextBlock, ThinkingBlock
from pydantic import BaseModel
from pytest_mock import MockerFixture

from nekomata.clients.providers.anthropic import AnthropicClient
from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus


class TestAnthropicClient:
    """Test suite for AnthropicClient."""

    @pytest.fixture
    def messages_response_factory(self, mocker: MockerFixture):
        """Create factory function of messages API response object."""

        def factory(response_cls=Message):
            mock_response = mocker.MagicMock(spec=response_cls)

            mock_response.stop_reason = 'end_turn'

            mock_thinking_block = mocker.MagicMock(spec=ThinkingBlock)
            mock_thinking_block.type = 'thinking'
            mock_thinking_block.thinking = 'thinking'

            mock_text_block = mocker.MagicMock(spec=TextBlock)
            mock_text_block.type = 'text'
            mock_text_block.text = 'text'

            mock_response.content = [mock_thinking_block, mock_text_block]

            mock_usage = mocker.MagicMock()
            mock_usage.input_tokens = 20
            mock_usage.output_tokens = 10
            mock_usage.cache_read_input_tokens = 5

            mock_response.usage = mock_usage
            return mock_response

        return factory

    @pytest.mark.anyio
    async def test_acompletion_success(self, mocker: MockerFixture) -> None:
        """Test successful acompletion call."""
        # Mock anthropic client.
        mock_lib_client_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_lib_client = mock_lib_client_class.return_value

        # Mock messages create.
        mock_lib_response = mocker.MagicMock(spec=Message)
        mock_lib_client.messages.create = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.anthropic.AnthropicClient.convert_output',
            return_value=mock_convert_result,
        )

        client = AnthropicClient(api_key='test')
        response = await client.acompletion(model='model', prompt='prompt')

        mock_lib_client.messages.create.assert_called_once()
        mock_convert_output.assert_called_once_with(mock_lib_response)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output(self, mocker) -> None:
        """Test successful acompletion call with structured output."""
        # Mock anthropic client.
        mock_lib_client_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_lib_client = mock_lib_client_class.return_value

        # Mock messages create.
        mock_lib_response = mocker.MagicMock(spec=ParsedMessage)
        mock_lib_client.messages.parse = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.anthropic.AnthropicClient.convert_output',
            return_value=mock_convert_result,
        )

        class MyResponse(BaseModel):
            answer: str

        client = AnthropicClient(api_key='test')
        response = await client.acompletion(model='model', prompt='prompt', response_format=MyResponse)

        mock_lib_client.messages.parse.assert_called_once()
        mock_convert_output.assert_called_once_with(mock_lib_response)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_failure(self, mocker) -> None:
        """Test failure during acompletion call."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value

        mock_instance.messages.create.side_effect = Exception('Anthropic Error')

        client = AnthropicClient(api_key='test-key')
        response = await client.acompletion(model='claude-3-5-sonnet', prompt='hi')

        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason
        assert 'Anthropic Error' in response.fail_reason

    @pytest.mark.anyio
    async def test_aclose(self, mocker) -> None:
        """Test aclose() method."""
        mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')

        client = AnthropicClient(api_key='test-key')
        mock_aclose = mocker.patch.object(client._http_client, 'aclose', new_callable=mocker.AsyncMock)

        await client.aclose()

        mock_aclose.assert_called_once()

    @pytest.mark.anyio
    async def test_convert_output_dispatch(self, mocker: MockerFixture) -> None:
        """Test that convert_output dispatches convert method based on input response type."""
        mock_create_response = mocker.MagicMock(spec=Message)
        mock_parse_response = mocker.MagicMock(spec=ParsedMessage)

        mock_convert_create = mocker.patch(
            'nekomata.clients.providers.anthropic.AnthropicClient._convert_create_response', return_value='create'
        )
        mock_convert_parse = mocker.patch(
            'nekomata.clients.providers.anthropic.AnthropicClient._convert_parse_response', return_value='parse'
        )

        mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        client = AnthropicClient(api_key='test-key')

        # .messages.create response conversion.
        create_result = client.convert_output(mock_create_response)

        mock_convert_create.assert_called_once_with(mock_create_response)
        assert create_result == 'create'

        # .messages.parse response conversion.
        parse_result = client.convert_output(mock_parse_response)

        mock_convert_parse.assert_called_once_with(mock_parse_response)
        assert parse_result == 'parse'

    @pytest.mark.anyio
    async def test_extract_common_attrs_success(self, mocker: MockerFixture, messages_response_factory) -> None:
        """Test successful _extract_common_attrs call."""
        mock_response = messages_response_factory()

        mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        client = AnthropicClient(api_key='test')

        common_attrs = client._extract_common_attrs(mock_response)

        assert common_attrs.content == 'text'
        assert common_attrs.reason == 'thinking'
        assert common_attrs.finish_reason == 'end_turn'
        assert common_attrs.total_tokens == 30  # noqa: PLR2004
        assert common_attrs.input_tokens == 20  # noqa: PLR2004
        assert common_attrs.output_tokens == 10  # noqa: PLR2004
        assert common_attrs.cache_tokens == 5  # noqa: PLR2004
        assert common_attrs.reason_tokens is None

    @pytest.mark.anyio
    async def test_extract_common_attrs_empty_content(self, mocker) -> None:
        """Test that _extract_common_attrs raises ValueError for empty content."""
        mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        client = AnthropicClient(api_key='test-key')

        mock_response = mocker.MagicMock()
        mock_response.content = []

        with pytest.raises(ValueError, match='Response object has an empty `content` field'):
            client._extract_common_attrs(mock_response)

    @pytest.mark.anyio
    async def test_convert_create_response_success(self, mocker: MockerFixture, messages_response_factory) -> None:
        """Test successful _convert_create_response call."""
        mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        client = AnthropicClient(api_key='test-key')

        mock_response = messages_response_factory()

        converted = client._convert_create_response(mock_response)

        assert converted.original == mock_response
        assert converted.parsed is None
        # Assert propagation of common attrs extracted via _extract_common_attrs
        assert converted.content == 'text'
        assert converted.reason == 'thinking'
        assert converted.finish_reason == 'end_turn'
        assert converted.total_tokens == 30  # noqa: PLR2004
        assert converted.input_tokens == 20  # noqa: PLR2004
        assert converted.output_tokens == 10  # noqa: PLR2004
        assert converted.cache_tokens == 5  # noqa: PLR2004
        assert converted.reason_tokens is None

    @pytest.mark.anyio
    async def test_convert_parse_response_success(self, mocker: MockerFixture, messages_response_factory) -> None:
        """Test successful _convert_create_response call."""
        mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        client = AnthropicClient(api_key='test-key')

        class MyResponse(BaseModel):
            answer: str

        mock_response = messages_response_factory(ParsedMessage)
        mock_response.parsed_output = MyResponse(answer='hello')

        converted = client._convert_parse_response(mock_response)

        assert converted.original == mock_response
        assert converted.parsed == MyResponse(answer='hello')
        # Assert propagation of common attrs extracted via _extract_common_attrs
        assert converted.content == 'text'
        assert converted.reason == 'thinking'
        assert converted.finish_reason == 'end_turn'
        assert converted.total_tokens == 30  # noqa: PLR2004
        assert converted.input_tokens == 20  # noqa: PLR2004
        assert converted.output_tokens == 10  # noqa: PLR2004
        assert converted.cache_tokens == 5  # noqa: PLR2004
        assert converted.reason_tokens is None

    @pytest.mark.anyio
    async def test_acompletion_reasoning_effort(self, mocker) -> None:
        """Test that reasoning_effort correctly configures thinking and output_config."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value

        # Mock response to avoid conversion errors
        mock_response = mocker.MagicMock()
        mock_response.content = [mocker.MagicMock(text='hi', type='text')]
        mocker.patch('nekomata.clients.providers.anthropic.isinstance', return_value=True)
        mock_instance.messages.create = mocker.AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key='test-key')

        await client.acompletion(model='c', prompt='p', reasoning_effort='high')

        _args, kwargs = mock_instance.messages.create.call_args
        assert kwargs['thinking']['type'] == 'adaptive'
        assert kwargs['output_config']['effort'] == 'high'

        # Remove reasoning on unsupported 'minimal' effort.
        await client.acompletion(model='c', prompt='p', reasoning_effort='minimal')

        _args, kwargs = mock_instance.messages.create.call_args
        assert isinstance(kwargs['thinking'], Omit)
        assert isinstance(kwargs['output_config'], Omit)
