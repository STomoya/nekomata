"""Tests for the OpenAI client."""

import time
from collections.abc import Callable
from unittest.mock import ANY, MagicMock

import pytest
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from openai.types.responses import ParsedResponse, Response
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from nekomata.clients.providers.openai import OpenAIClient
from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus
from nekomata.types.openai import OpenAIArgs

# --- Shared Test Artifacts ---


class _MockStructuredResponse(BaseModel):
    """Shared Pydantic model for testing structured output."""

    answer: str


def _get_validation_error() -> ValidationError:
    """Generate a realistic ValidationError for retry tests."""
    try:
        _MockStructuredResponse(answer=1)  # ty: ignore[invalid-argument-type]
    except ValidationError as e:
        return e
    raise RuntimeError('Expected ValidationError was not raised.')


# --- Shared Fixtures ---


@pytest.fixture
def mock_async_openai_class(mocker: MockerFixture) -> MagicMock:
    """Mock the AsyncOpenAI class globally to prevent HTTP client initialization."""
    return mocker.patch('nekomata.clients.providers.openai.AsyncOpenAI')


@pytest.fixture
def mock_async_openai(mock_async_openai_class: MagicMock) -> MagicMock:
    """Provide the mocked instance of AsyncOpenAI."""
    return mock_async_openai_class.return_value


@pytest.fixture
def client(mock_async_openai_class: MagicMock) -> OpenAIClient:
    """Provide a standard, pre-initialized OpenAIClient."""
    return OpenAIClient(api_key='test-key')


class TestOpenAIClient:
    """Test suite for OpenAIClient."""

    @pytest.mark.anyio
    async def test_acompletion_dispatch(self, mocker: MockerFixture, client: OpenAIClient) -> None:
        """Test acompletion dispatch by API configuration."""
        mock_chat_completion_response = mocker.MagicMock()
        mock_chat_completion = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._chat_completion',
            return_value=mock_chat_completion_response,
        )
        mock_responses_response = mocker.MagicMock()
        mock_responses = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._responses', return_value=mock_responses_response
        )

        completion_args = OpenAIArgs(api='chat_completions')
        response = await client.acompletion(model='model', prompt='prompt', args=completion_args)

        mock_chat_completion.assert_called_once()
        # The responses should not be called.
        mock_responses.assert_not_called()
        assert response == mock_chat_completion_response

        responses_args = OpenAIArgs(api='responses')
        response = await client.acompletion(model='model', prompt='prompt', args=responses_args)

        mock_responses.assert_called_once()
        # chat completion should not be called (==1)
        mock_chat_completion.assert_called_once()
        assert response == mock_responses_response

    @pytest.mark.anyio
    async def test_aclose(self, mocker: MockerFixture, client: OpenAIClient) -> None:
        """Test aclose() method."""
        mock_aclose = mocker.patch.object(client._http_client, 'aclose', new_callable=mocker.AsyncMock)

        await client.aclose()

        mock_aclose.assert_called_once()

    def test_initialized_property(self, client: OpenAIClient) -> None:
        """Test the initialized property."""
        assert client.initialized is True


class TestOpenAIChatCompletion:
    """Test suit for the _chat_completion API call of OpenAI SDK."""

    @pytest.fixture
    def chat_completion_factory(self, mocker: MockerFixture) -> Callable[..., MagicMock]:
        """Create factory function of chat completion response object."""

        def factory(response_cls: type = ChatCompletion) -> MagicMock:
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
    async def test_acompletion_success(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test successful acompletion call."""
        # Mock completion create.
        mock_lib_response = mocker.MagicMock(spec=ChatCompletion)
        mock_async_openai.chat.completions.create = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_chat_completion_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(model='model', prompt='prompt')

        mock_async_openai.chat.completions.create.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test successful acompletion call with structured output."""
        # Mock completion parse.
        mock_lib_response = mocker.MagicMock(spec=ParsedChatCompletion)
        mock_async_openai.chat.completions.parse = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_chat_completion_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(model='model', prompt='prompt', response_format=_MockStructuredResponse)

        mock_async_openai.chat.completions.parse.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output_retry(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test successful acompletion call with structured output."""
        # Mock completion parse.
        mock_lib_response = mocker.MagicMock(spec=ParsedChatCompletion)
        mock_async_openai.chat.completions.parse = mocker.AsyncMock(
            side_effect=[_get_validation_error(), mock_lib_response]
        )

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_chat_completion_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(
            model='model', prompt='prompt', response_format=_MockStructuredResponse, max_model_retry=2
        )

        assert mock_async_openai.chat.completions.parse.call_count == 2
        mock_convert_output.assert_called_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_failure(self, mock_async_openai: MagicMock, client: OpenAIClient) -> None:
        """Test failure during acompletion call."""
        mock_async_openai.chat.completions.create.side_effect = Exception('API Error')

        response = await client.acompletion(model='gpt-4o', prompt='hello')

        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason
        assert 'API Error' in response.fail_reason

    @pytest.mark.anyio
    async def test_convert_output_dispatch(self, mocker: MockerFixture, client: OpenAIClient) -> None:
        """Test that convert_output dispatches convert method based on input response type."""
        mock_create_response = mocker.MagicMock(spec=ChatCompletion)
        mock_parse_response = mocker.MagicMock(spec=ParsedChatCompletion)

        mock_convert_create = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_chat_completion_create_output',
            return_value='create',
        )
        mock_convert_parse = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_chat_completion_parse_output',
            return_value='parse',
        )

        created_at = time.time()

        # .chat.completions.create response conversion.
        create_result = client._convert_chat_completion_output(mock_create_response, created_at)

        mock_convert_create.assert_called_once_with(
            response=mock_create_response, created_at=created_at, custom_id=None
        )
        assert create_result == 'create'

        # .chat.completions.parse response conversion.
        parse_result = client._convert_chat_completion_output(mock_parse_response, created_at)

        mock_convert_parse.assert_called_once_with(response=mock_parse_response, created_at=created_at, custom_id=None)
        assert parse_result == 'parse'

    @pytest.mark.anyio
    async def test_extract_common_attrs_success(
        self, chat_completion_factory: Callable[..., MagicMock], client: OpenAIClient
    ) -> None:
        """Test successful _extract_common_attrs call."""
        mock_response = chat_completion_factory()
        common_attrs = client._extract_chat_completion_common_attrs(mock_response)

        assert common_attrs.content == 'Test content'
        assert common_attrs.reason == 'Test reasoning'
        assert common_attrs.finish_reason == 'stop'
        assert common_attrs.total_tokens == 100
        assert common_attrs.input_tokens == 40
        assert common_attrs.output_tokens == 60
        assert common_attrs.cache_tokens == 10
        assert common_attrs.reason_tokens == 5

    @pytest.mark.anyio
    async def test_extract_common_attrs_empty_choices(self, mocker: MockerFixture, client: OpenAIClient) -> None:
        """Test that _extract_common_attrs raises ValueError for empty choices."""
        mock_response = mocker.MagicMock()
        mock_response.choices = []

        with pytest.raises(ValueError, match='Response object has an empty `choices` field'):
            client._extract_chat_completion_common_attrs(mock_response)

    @pytest.mark.anyio
    async def test_convert_create_output_success(
        self, chat_completion_factory: Callable[..., MagicMock], client: OpenAIClient
    ) -> None:
        """Test successful _convert_create_output call."""
        mock_response = chat_completion_factory()
        created_at = time.time()

        converted = client._convert_chat_completion_create_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.parsed is None
        # Assert propagation of common attrs
        assert converted.content == 'Test content'
        assert converted.reason == 'Test reasoning'
        assert converted.finish_reason == 'stop'
        assert converted.total_tokens == 100
        assert converted.cache_tokens == 10
        assert converted.reason_tokens == 5

    @pytest.mark.anyio
    async def test_convert_parse_output_success(
        self, chat_completion_factory: Callable[..., MagicMock], client: OpenAIClient
    ) -> None:
        """Test successful _convert_parse_output call."""
        mock_response = chat_completion_factory(ParsedChatCompletion)
        mock_response.choices[0].message.parsed = _MockStructuredResponse(answer='hello')

        created_at = time.time()
        converted = client._convert_chat_completion_parse_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.parsed == _MockStructuredResponse(answer='hello')
        # Assert propagation of common attrs
        assert converted.content == 'Test content'
        assert converted.reason == 'Test reasoning'
        assert converted.finish_reason == 'stop'
        assert converted.total_tokens == 100

    @pytest.mark.anyio
    async def test_acompletion_system_prompt_and_top_k(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test that system_prompt and top_k are correctly handled."""
        # Mock response to avoid conversion error
        mock_lib_response = mocker.MagicMock(spec=ChatCompletion)
        mock_async_openai.chat.completions.create = mocker.AsyncMock(return_value=mock_lib_response)

        mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_chat_completion_output',
            return_value=mocker.MagicMock(spec=ChatCompletionResponse),
        )

        await client.acompletion(model='gpt-4o', prompt='hello', system_prompt='be helpful', top_k=50)

        _args, kwargs = mock_async_openai.chat.completions.create.call_args

        # Verify messages
        messages = kwargs['messages']
        assert len(messages) == 2
        assert messages[0] == {'role': 'system', 'content': 'be helpful'}
        assert messages[1] == {'role': 'user', 'content': 'hello'}

        # Verify top_k in extra_body
        assert kwargs['extra_body'] == {'top_k': 50}


class TestOpenAIResponses:
    """Test responses API call of the OpenAI SDK."""

    @pytest.fixture
    def responses_factory(self, mocker: MockerFixture) -> Callable[..., MagicMock]:
        """Create factory function of reponses response object."""

        def factory(response_cls: type = Response, reason_has_content: bool = True) -> MagicMock:
            mock_response = mocker.MagicMock(spec=response_cls)
            mock_response.status = 'completed'
            mock_response.output_text = 'content'

            mock_response.output = [
                mocker.MagicMock(
                    type='reasoning',
                    summary=[mocker.MagicMock(text='summary...')],
                    content=[mocker.MagicMock(text='thinking...')] if reason_has_content else None,
                ),
            ]

            mock_response.usage = mocker.MagicMock()
            mock_response.usage.total_tokens = 20
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 10
            mock_response.usage.input_tokens_details.cached_tokens = 5
            mock_response.usage.output_tokens_details.reasoning_tokens = 5

            return mock_response

        return factory

    @pytest.fixture
    def responses_args(self) -> OpenAIArgs:
        """Args for responses."""
        return OpenAIArgs(api='responses')

    @pytest.mark.anyio
    async def test_acompletion_success(
        self,
        mocker: MockerFixture,
        mock_async_openai: MagicMock,
        client: OpenAIClient,
        responses_args: OpenAIArgs,
    ) -> None:
        """Test successful acompletion call."""
        # Mock completion create.
        mock_lib_response = mocker.MagicMock(spec=Response)
        mock_async_openai.responses.create = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_responses_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(model='model', prompt='prompt', args=responses_args)

        mock_async_openai.responses.create.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output(
        self,
        mocker: MockerFixture,
        mock_async_openai: MagicMock,
        client: OpenAIClient,
        responses_args: OpenAIArgs,
    ) -> None:
        """Test successful acompletion call with structured output."""
        # Mock completion create.
        mock_lib_response = mocker.MagicMock(spec=Response)
        mock_async_openai.responses.parse = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_responses_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(
            model='model', prompt='prompt', response_format=_MockStructuredResponse, args=responses_args
        )

        mock_async_openai.responses.parse.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output_retry(
        self,
        mocker: MockerFixture,
        mock_async_openai: MagicMock,
        client: OpenAIClient,
        responses_args: OpenAIArgs,
    ) -> None:
        """Test successful acompletion call with structured output."""
        # Mock completion create.
        mock_lib_response = mocker.MagicMock(spec=Response)
        mock_async_openai.responses.parse = mocker.AsyncMock(side_effect=[_get_validation_error(), mock_lib_response])

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_responses_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(
            model='model',
            prompt='prompt',
            response_format=_MockStructuredResponse,
            max_model_retry=2,
            args=responses_args,
        )

        assert mock_async_openai.responses.parse.call_count == 2
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_failure(
        self,
        mock_async_openai: MagicMock,
        client: OpenAIClient,
        responses_args: OpenAIArgs,
    ) -> None:
        """Test failure during acompletion call."""
        mock_async_openai.responses.create.side_effect = Exception('API Error')

        response = await client.acompletion(model='gpt-4o', prompt='hello', args=responses_args)

        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason
        assert 'API Error' in response.fail_reason

    @pytest.mark.anyio
    async def test_convert_output_dispatch(
        self,
        mocker: MockerFixture,
        client: OpenAIClient,
    ) -> None:
        """Test that convert_output dispatches convert method based on input response type."""
        mock_create_response = mocker.MagicMock(spec=Response)
        mock_parse_response = mocker.MagicMock(spec=ParsedResponse)

        mock_convert_create = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_responses_create_output',
            return_value='create',
        )
        mock_convert_parse = mocker.patch(
            'nekomata.clients.providers.openai.OpenAIClient._convert_responses_parse_output',
            return_value='parse',
        )

        created_at = time.time()

        # .chat.completions.create response conversion.
        create_result = client._convert_responses_output(mock_create_response, created_at)

        mock_convert_create.assert_called_once_with(
            response=mock_create_response, created_at=created_at, custom_id=None
        )
        assert create_result == 'create'

        # .chat.completions.parse response conversion.
        parse_result = client._convert_responses_output(mock_parse_response, created_at)

        mock_convert_parse.assert_called_once_with(response=mock_parse_response, created_at=created_at, custom_id=None)
        assert parse_result == 'parse'

    @pytest.mark.anyio
    async def test_extract_common_attrs_success(
        self, responses_factory: Callable[..., MagicMock], client: OpenAIClient
    ) -> None:
        """Test successful _extract_common_attrs call."""
        mock_response = responses_factory()
        common_attrs = client._extract_responses_common_attrs(mock_response)

        assert common_attrs.content == 'content'
        assert common_attrs.reason == 'thinking...'
        assert common_attrs.finish_reason == 'completed'
        assert common_attrs.total_tokens == 20
        assert common_attrs.input_tokens == 10
        assert common_attrs.output_tokens == 10
        assert common_attrs.cache_tokens == 5
        assert common_attrs.reason_tokens == 5

    @pytest.mark.anyio
    async def test_extract_common_attrs_reason_summary(
        self, responses_factory: Callable[..., MagicMock], client: OpenAIClient
    ) -> None:
        """Test successful _extract_common_attrs call with only reason summaries."""
        mock_response = responses_factory()
        mock_response.output[0].content = None
        common_attrs = client._extract_responses_common_attrs(mock_response)

        assert common_attrs.reason == 'summary...'

    @pytest.mark.anyio
    async def test_extract_common_attrs_no_reasoning(
        self, responses_factory: Callable[..., MagicMock], client: OpenAIClient
    ) -> None:
        """Test successful _extract_common_attrs call."""
        mock_response = responses_factory()
        mock_response.output = []

        common_attrs = client._extract_responses_common_attrs(mock_response)

        assert common_attrs.reason is None

    @pytest.mark.anyio
    async def test_convert_create_output_success(
        self, responses_factory: Callable[..., MagicMock], client: OpenAIClient
    ) -> None:
        """Test successful _convert_create_output call."""
        mock_response = responses_factory()
        created_at = time.time()

        converted = client._convert_responses_create_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.parsed is None
        # Assert propagation of common attrs
        assert converted.content == 'content'
        assert converted.reason == 'thinking...'
        assert converted.finish_reason == 'completed'
        assert converted.total_tokens == 20
        assert converted.input_tokens == 10
        assert converted.output_tokens == 10

    @pytest.mark.anyio
    async def test_convert_parse_output_success(
        self, responses_factory: Callable[..., MagicMock], client: OpenAIClient
    ) -> None:
        """Test successful _convert_parse_output call."""
        mock_response = responses_factory(ParsedResponse)
        mock_response.output_parsed = _MockStructuredResponse(answer='hello')

        created_at = time.time()
        converted = client._convert_responses_parse_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.parsed == _MockStructuredResponse(answer='hello')
        # Assert propagation of common attrs
        assert converted.content == 'content'
        assert converted.reason == 'thinking...'
        assert converted.finish_reason == 'completed'
        assert converted.total_tokens == 20


class TestOpenAIBatchAPI:
    """Test suite for OpenAI Batch API operations."""

    @pytest.mark.anyio
    async def test_acreate_batch(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test acreate_batch call with single custom_id."""
        mock_upload = mock_async_openai.files.create = mocker.AsyncMock()
        mock_upload.return_value.id = 'file-id'
        mock_create = mock_async_openai.batches.create = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acreate_batch(
            model='gpt-4o',
            prompt=['hello', 'world'],
            system_prompt='sys prompt',
            max_output_tokens=100,
            reasoning_effort='medium',
            response_format=_MockStructuredResponse,
            custom_id='my-custom-id',
        )

        mock_upload.assert_called_once()
        mock_create.assert_called_once_with(
            completion_window='24h',
            endpoint='/v1/chat/completions',
            input_file_id='file-id',
        )
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_with_custom_id_list(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test acreate_batch call with a list of custom_ids."""
        mock_upload = mock_async_openai.files.create = mocker.AsyncMock()
        mock_upload.return_value.id = 'file-id'
        mock_create = mock_async_openai.batches.create = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acreate_batch(
            model='gpt-4o',
            prompt=['hello', 'world'],
            custom_id=['id-1', 'id-2'],
        )

        mock_upload.assert_called_once()
        mock_create.assert_called_once_with(
            completion_window='24h',
            endpoint='/v1/chat/completions',
            input_file_id='file-id',
        )
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_warns_file_mode(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test acreate_batch call with custom_id list."""
        mock_upload = mock_async_openai.files.create = mocker.AsyncMock()
        mock_upload.return_value.id = 'file-id'
        mock_create = mock_async_openai.batches.create = mocker.AsyncMock(return_value='mock-batch')

        mock_logger = mocker.patch('nekomata.clients.plugins.openai.logger')
        mock_logger.warning = mocker.MagicMock()

        client = OpenAIClient(api_key='test-key')

        res = await client.acreate_batch(
            model='gpt-4o',
            prompt=['hello', 'world'],
            custom_id=['id-1', 'id-2'],
            mode='inline',
        )

        # Create should be called successfully reguardless of "mode" validity.
        mock_upload.assert_called_once()
        mock_create.assert_called_once_with(
            completion_window='24h',
            endpoint='/v1/chat/completions',
            input_file_id='file-id',
        )
        assert res == 'mock-batch'

        # The users should be warned.
        mock_logger.warning.assert_called_once_with(
            'OpenAI only supports Files API-based batch requests. Proceeding with "file".'
        )

    @pytest.mark.anyio
    async def test_acreate_batch_validation_error(self, client: OpenAIClient) -> None:
        """Test acreate_batch validation error when no list is provided."""
        with pytest.raises(ValueError, match=r'At least one of prompt,.* must be a list'):
            await client.acreate_batch(model='gpt-4o', prompt='hello')

    @pytest.mark.anyio
    async def test_acreate_batch_length_mismatch(self, client: OpenAIClient) -> None:
        """Test acreate_batch validation error when list lengths mismatch."""
        with pytest.raises(ValueError, match='Lengths of list arguments do not match'):
            await client.acreate_batch(
                model='gpt-4o', prompt=['hello', 'world'], system_prompt=['sys1', 'sys2', 'sys3']
            )

    @pytest.mark.anyio
    async def test_aretrieve_batch(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test aretrieve_batch call."""
        mock_retrieve = mock_async_openai.batches.retrieve = mocker.AsyncMock(return_value='mock-batch')

        res = await client.aretrieve_batch('batch-id', timeout=10.0)

        mock_retrieve.assert_called_once_with('batch-id', timeout=10.0)
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acancel_batch(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test acancel_batch call."""
        mock_cancel = mock_async_openai.batches.cancel = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acancel_batch('batch-id', timeout=10.0)

        mock_cancel.assert_called_once_with('batch-id', timeout=10.0)
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_alist_batches(
        self, mocker: MockerFixture, mock_async_openai: MagicMock, client: OpenAIClient
    ) -> None:
        """Test alist_batches call."""
        mock_list = mock_async_openai.batches.list = mocker.AsyncMock(return_value='mock-batches')

        res = await client.alist_batches(after='after-id', limit=5)

        mock_list.assert_called_once_with(after='after-id', limit=5)
        assert res == 'mock-batches'

    @pytest.mark.anyio
    async def test_adelete_batch(self, client: OpenAIClient) -> None:
        """Test adelete_batch raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r'OpenAI does not support deleting batches\.'):
            await client.adelete_batch('batch-id')
