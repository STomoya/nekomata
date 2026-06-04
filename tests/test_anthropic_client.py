"""Tests for the Anthropic client."""

import time
from unittest.mock import ANY

import pytest
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
            'nekomata.clients.providers.anthropic.AnthropicClient._convert_output',
            return_value=mock_convert_result,
        )

        client = AnthropicClient(api_key='test')
        response = await client.acompletion(model='model', prompt='prompt')

        mock_lib_client.messages.create.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
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
            'nekomata.clients.providers.anthropic.AnthropicClient._convert_output',
            return_value=mock_convert_result,
        )

        class MyResponse(BaseModel):
            answer: str

        client = AnthropicClient(api_key='test')
        response = await client.acompletion(model='model', prompt='prompt', response_format=MyResponse)

        mock_lib_client.messages.parse.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
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

        created_at = time.time()

        # .messages.create response conversion.
        create_result = client._convert_output(mock_create_response, created_at)

        mock_convert_create.assert_called_once_with(
            response=mock_create_response, created_at=created_at, custom_id=None
        )
        assert create_result == 'create'

        # .messages.parse response conversion.
        parse_result = client._convert_output(mock_parse_response, created_at)

        mock_convert_parse.assert_called_once_with(response=mock_parse_response, created_at=created_at, custom_id=None)
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
        assert common_attrs.total_tokens == 30
        assert common_attrs.input_tokens == 20
        assert common_attrs.output_tokens == 10
        assert common_attrs.cache_tokens == 5
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

        created_at = time.time()

        converted = client._convert_create_response(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.parsed is None
        # Assert propagation of common attrs extracted via _extract_common_attrs
        assert converted.content == 'text'
        assert converted.reason == 'thinking'
        assert converted.finish_reason == 'end_turn'
        assert converted.total_tokens == 30
        assert converted.input_tokens == 20
        assert converted.output_tokens == 10
        assert converted.cache_tokens == 5
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

        created_at = time.time()

        converted = client._convert_parse_response(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.parsed == MyResponse(answer='hello')
        # Assert propagation of common attrs extracted via _extract_common_attrs
        assert converted.content == 'text'
        assert converted.reason == 'thinking'
        assert converted.finish_reason == 'end_turn'
        assert converted.total_tokens == 30
        assert converted.input_tokens == 20
        assert converted.output_tokens == 10
        assert converted.cache_tokens == 5
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

        # Check if a reasoning effort unsupported by other providers is valid for anthropic client.
        await client.acompletion(model='c', prompt='p', reasoning_effort='max')

        _args, kwargs = mock_instance.messages.create.call_args
        assert kwargs['thinking']['type'] == 'adaptive'
        assert kwargs['output_config']['effort'] == 'max'


class TestAnthropicBatchAPI:
    """Test suite for Anthropic Batch API operations."""

    @pytest.mark.anyio
    async def test_acreate_batch(self, mocker: MockerFixture) -> None:
        """Test acreate_batch call with generated custom_id."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value
        mock_create = mock_instance.beta.messages.batches.create = mocker.AsyncMock(return_value='mock-batch')

        client = AnthropicClient(api_key='test-key')

        class DummyResponse(BaseModel):
            answer: str

        res = await client.acreate_batch(
            model='claude-3-5-sonnet',
            prompt=['hello', 'world'],
            system_prompt='sys prompt',
            max_output_tokens=100,
            reasoning_effort='high',
            response_format=DummyResponse,
        )

        mock_create.assert_called_once()
        called_args = mock_create.call_args[1]
        requests = called_args['requests']
        assert len(requests) == 2
        assert requests[0]['custom_id'].startswith('req-')
        assert requests[1]['custom_id'].startswith('req-')
        assert requests[0]['params']['messages'] == [{'role': 'user', 'content': 'hello'}]
        assert requests[1]['params']['messages'] == [{'role': 'user', 'content': 'world'}]
        assert requests[0]['params']['max_tokens'] == 100
        assert requests[0]['params']['system'] == 'sys prompt'
        assert requests[0]['params']['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
        assert requests[0]['params']['output_config'] == {'effort': 'high'}
        assert requests[0]['params']['output_format'] == DummyResponse
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_with_custom_id_string(self, mocker: MockerFixture) -> None:
        """Test acreate_batch call with single custom_id string."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value
        mock_create = mock_instance.beta.messages.batches.create = mocker.AsyncMock(return_value='mock-batch')

        client = AnthropicClient(api_key='test-key')

        res = await client.acreate_batch(
            model='claude-3-5-sonnet',
            prompt=['hello', 'world'],
            custom_id='custom-id-prefix',
        )

        mock_create.assert_called_once()
        called_args = mock_create.call_args[1]
        requests = called_args['requests']
        assert len(requests) == 2
        assert requests[0]['custom_id'] == 'custom-id-prefix-0'
        assert requests[1]['custom_id'] == 'custom-id-prefix-1'
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_with_custom_id_list(self, mocker: MockerFixture) -> None:
        """Test acreate_batch call with custom_id list."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value
        mock_create = mock_instance.beta.messages.batches.create = mocker.AsyncMock(return_value='mock-batch')

        client = AnthropicClient(api_key='test-key')

        res = await client.acreate_batch(
            model='claude-3-5-sonnet',
            prompt=['hello', 'world'],
            custom_id=['id-1', 'id-2'],
        )

        mock_create.assert_called_once()
        called_args = mock_create.call_args[1]
        requests = called_args['requests']
        assert len(requests) == 2
        assert requests[0]['custom_id'] == 'id-1'
        assert requests[1]['custom_id'] == 'id-2'
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_length_mismatch(self, mocker: MockerFixture) -> None:
        """Test acreate_batch call with list length mismatch."""
        mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        client = AnthropicClient(api_key='test-key')
        with pytest.raises(ValueError, match='Lengths of list arguments do not match'):
            await client.acreate_batch(
                model='claude-3-5-sonnet',
                prompt=['hello', 'world'],
                custom_id=['id-1', 'id-2', 'id-3'],
            )

    @pytest.mark.anyio
    async def test_acreate_batch_validation_error(self, mocker: MockerFixture) -> None:
        """Test acreate_batch validation error when no list is provided."""
        mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        client = AnthropicClient(api_key='test-key')
        with pytest.raises(ValueError, match=r'At least one of prompt,.* must be a list'):
            await client.acreate_batch(model='claude-3-5-sonnet', prompt='hello')

    @pytest.mark.anyio
    async def test_aretrieve_batch(self, mocker: MockerFixture) -> None:
        """Test aretrieve_batch call."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value
        mock_retrieve = mock_instance.beta.messages.batches.retrieve = mocker.AsyncMock(return_value='mock-batch')

        client = AnthropicClient(api_key='test-key')

        res = await client.aretrieve_batch('batch-id', timeout=10.0)

        mock_retrieve.assert_called_once_with('batch-id', timeout=10.0)
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acancel_batch(self, mocker: MockerFixture) -> None:
        """Test acancel_batch call."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value
        mock_cancel = mock_instance.beta.messages.batches.cancel = mocker.AsyncMock(return_value='mock-batch')

        client = AnthropicClient(api_key='test-key')

        res = await client.acancel_batch('batch-id', timeout=10.0)

        mock_cancel.assert_called_once_with('batch-id', timeout=10.0)
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_alist_batches(self, mocker: MockerFixture) -> None:
        """Test alist_batches call."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value
        mock_list = mock_instance.beta.messages.batches.list = mocker.AsyncMock(return_value='mock-batches')

        client = AnthropicClient(api_key='test-key')

        res = await client.alist_batches(after_id='after-id', limit=5)

        mock_list.assert_called_once_with(after_id='after-id', limit=5)
        assert res == 'mock-batches'

    @pytest.mark.anyio
    async def test_adelete_batch(self, mocker: MockerFixture) -> None:
        """Test adelete_batch call."""
        mock_anthropic_class = mocker.patch('nekomata.clients.providers.anthropic.AsyncAnthropic')
        mock_instance = mock_anthropic_class.return_value
        mock_delete = mock_instance.beta.messages.batches.delete = mocker.AsyncMock(return_value='mock-deleted')

        client = AnthropicClient(api_key='test-key')

        res = await client.adelete_batch('batch-id', timeout=10.0)

        mock_delete.assert_called_once_with('batch-id', timeout=10.0)
        assert res == 'mock-deleted'
