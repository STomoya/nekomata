"""Tests for the OpenAI client."""

import time
from unittest.mock import ANY

import pytest
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel
from pytest_mock import MockerFixture

from nekomata.clients.providers.openai import OpenAIClient
from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus


class TestOpenAIClient:
    """Test suite for OpenAIClient."""

    @pytest.fixture
    def chat_completion_factory(self, mocker: MockerFixture):
        """Create factory function of chat completion response object."""

        def factory(response_cls=ChatCompletion):
            mock_response = mocker.MagicMock(spec=response_cls)

            mock_choice = mocker.MagicMock()
            mock_choice.message.content = 'Test content'
            mock_choice.message.reasoning_content = 'Test reasoning'
            mock_choice.finish_reason = 'stop'
            mock_response.choices = [mock_choice]

            mock_usage = mocker.MagicMock()
            mock_usage.total_tokens = 100
            mock_usage.prompt_tokens = 40
            mock_usage.completion_tokens = 60

            mock_prompt_details = mocker.MagicMock()
            mock_prompt_details.cached_tokens = 10
            mock_usage.prompt_tokens_details = mock_prompt_details

            mock_completion_details = mocker.MagicMock()
            mock_completion_details.reasoning_tokens = 5
            mock_usage.completion_tokens_details = mock_completion_details

            mock_response.usage = mock_usage
            return mock_response

        return factory

    @pytest.mark.anyio
    async def test_acompletion_success(self, mocker: MockerFixture) -> None:
        """Test successful acompletion call."""
        # Mock OpenAI client.
        mock_lib_client_class = mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        mock_lib_client = mock_lib_client_class.return_value

        # Mock completion create.
        mock_lib_response = mocker.MagicMock(spec=ChatCompletion)
        mock_lib_client.chat.completions.create = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient.convert_output',
            return_value=mock_convert_result,
        )

        client = OpenAIClient(api_key='test')
        response = await client.acompletion(model='model', prompt='prompt')

        mock_lib_client.chat.completions.create.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output(self, mocker: MockerFixture) -> None:
        """Test successful acompletion call with structured output."""
        # Mock OpenAI client.
        mock_lib_client_class = mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        mock_lib_client = mock_lib_client_class.return_value

        # Mock completion parse.
        mock_lib_response = mocker.MagicMock(spec=ParsedChatCompletion)
        mock_lib_client.chat.completions.parse = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient.convert_output',
            return_value=mock_convert_result,
        )

        class MyResponse(BaseModel):
            answer: str

        client = OpenAIClient(api_key='test')
        response = await client.acompletion(model='model', prompt='prompt', response_format=MyResponse)

        mock_lib_client.chat.completions.parse.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_failure(self, mocker: MockerFixture) -> None:
        """Test failure during acompletion call."""
        mock_openai_class = mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        mock_instance = mock_openai_class.return_value

        mock_instance.chat.completions.create.side_effect = Exception('API Error')

        client = OpenAIClient(api_key='test-key')
        response = await client.acompletion(model='gpt-4o', prompt='hello')

        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason
        assert 'API Error' in response.fail_reason

    @pytest.mark.anyio
    async def test_aclose(self, mocker: MockerFixture) -> None:
        """Test aclose() method."""
        mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')

        client = OpenAIClient(api_key='test-key')
        mock_aclose = mocker.patch.object(client._http_client, 'aclose', new_callable=mocker.AsyncMock)

        await client.aclose()

        mock_aclose.assert_called_once()

    def test_initialized_property(self, mocker: MockerFixture) -> None:
        """Test the initialized property."""
        mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        client = OpenAIClient(api_key='test-key')
        assert client.initialized is True

    @pytest.mark.anyio
    async def test_convert_output_dispatch(self, mocker: MockerFixture) -> None:
        """Test that convert_output dispatches convert method based on input response type."""
        mock_create_response = mocker.MagicMock(spec=ChatCompletion)
        mock_parse_response = mocker.MagicMock(spec=ParsedChatCompletion)

        mock_convert_create = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_create_output', return_value='create'
        )
        mock_convert_parse = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_parse_output', return_value='parse'
        )

        mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        client = OpenAIClient(api_key='test-key')

        created_at = time.time()

        # .chat.completions.create response conversion.
        create_result = client.convert_output(mock_create_response, created_at)

        mock_convert_create.assert_called_once_with(
            response=mock_create_response, created_at=created_at, custom_id=None
        )
        assert create_result == 'create'

        # .chat.completions.parse response conversion.
        parse_result = client.convert_output(mock_parse_response, created_at)

        mock_convert_parse.assert_called_once_with(response=mock_parse_response, created_at=created_at, custom_id=None)
        assert parse_result == 'parse'

    @pytest.mark.anyio
    async def test_extract_common_attrs_success(self, mocker: MockerFixture, chat_completion_factory) -> None:
        """Test successful _extract_common_attrs call."""
        mock_response = chat_completion_factory()

        mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        client = OpenAIClient(api_key='test')

        common_attrs = client._extract_common_attrs(mock_response)

        assert common_attrs.content == 'Test content'
        assert common_attrs.reason == 'Test reasoning'
        assert common_attrs.finish_reason == 'stop'
        assert common_attrs.total_tokens == 100  # noqa: PLR2004
        assert common_attrs.input_tokens == 40  # noqa: PLR2004
        assert common_attrs.output_tokens == 60  # noqa: PLR2004
        assert common_attrs.cache_tokens == 10  # noqa: PLR2004
        assert common_attrs.reason_tokens == 5  # noqa: PLR2004

    @pytest.mark.anyio
    async def test_extract_common_attrs_empty_choices(self, mocker: MockerFixture) -> None:
        """Test that _extract_common_attrs raises ValueError for empty choices."""
        mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        client = OpenAIClient(api_key='test-key')

        mock_response = mocker.MagicMock()
        mock_response.choices = []

        with pytest.raises(ValueError, match='Response object has an empty `choices` field'):
            client._extract_common_attrs(mock_response)

    @pytest.mark.anyio
    async def test_convert_create_output_success(self, mocker: MockerFixture, chat_completion_factory) -> None:
        """Test successful _convert_create_output call."""
        mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        client = OpenAIClient(api_key='test-key')

        mock_response = chat_completion_factory()

        created_at = time.time()

        converted = client._convert_create_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.parsed is None
        # Assert propagation of common attrs
        assert converted.content == 'Test content'
        assert converted.reason == 'Test reasoning'
        assert converted.finish_reason == 'stop'
        assert converted.total_tokens == 100  # noqa: PLR2004
        assert converted.cache_tokens == 10  # noqa: PLR2004
        assert converted.reason_tokens == 5  # noqa: PLR2004

    @pytest.mark.anyio
    async def test_convert_parse_output_success(self, mocker: MockerFixture, chat_completion_factory) -> None:
        """Test successful _convert_parse_output call."""
        mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        client = OpenAIClient(api_key='test-key')

        class MyResponse(BaseModel):
            answer: str

        mock_response = chat_completion_factory(ParsedChatCompletion)
        mock_response.choices[0].message.parsed = MyResponse(answer='hello')

        created_at = time.time()

        converted = client._convert_parse_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.parsed == MyResponse(answer='hello')
        # Assert propagation of common attrs
        assert converted.content == 'Test content'
        assert converted.reason == 'Test reasoning'
        assert converted.finish_reason == 'stop'
        assert converted.total_tokens == 100  # noqa: PLR2004

    @pytest.mark.anyio
    async def test_acompletion_system_prompt_and_top_k(self, mocker: MockerFixture) -> None:
        """Test that system_prompt and top_k are correctly handled."""
        mock_openai_class = mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')
        mock_instance = mock_openai_class.return_value

        # Mock response to avoid conversion error
        mock_lib_response = mocker.MagicMock(spec=ChatCompletion)
        mock_instance.chat.completions.create = mocker.AsyncMock(return_value=mock_lib_response)

        mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient.convert_output',
            return_value=mocker.MagicMock(spec=ChatCompletionResponse),
        )

        client = OpenAIClient(api_key='test-key')
        await client.acompletion(model='gpt-4o', prompt='hello', system_prompt='be helpful', top_k=50)

        _args, kwargs = mock_instance.chat.completions.create.call_args

        # Verify messages
        messages = kwargs['messages']
        assert len(messages) == 2  # noqa: PLR2004
        assert messages[0] == {'role': 'system', 'content': 'be helpful'}
        assert messages[1] == {'role': 'user', 'content': 'hello'}

        # Verify top_k in extra_body
        assert kwargs['extra_body'] == {'top_k': 50}
