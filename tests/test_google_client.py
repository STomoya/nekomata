"""Tests for the Google client."""

import json
import time
from collections.abc import Callable
from unittest.mock import ANY, MagicMock

import pytest
from google.genai import types
from google.genai.interactions import Interaction, TextContent, ThoughtStep
from google.genai.types import GenerateContentResponse
from pydantic import BaseModel
from pytest_mock import MockerFixture

from nekomata.clients.providers.google import GoogleClient
from nekomata.types.google import GoogleArgs, InteractionsArgs
from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus

# --- Shared Test Artifacts ---


class _MockStructuredResponse(BaseModel):
    """Shared Pydantic model for testing structured output."""

    answer: str


# --- Shared Fixtures ---


@pytest.fixture
def mock_google_class(mocker: MockerFixture) -> MagicMock:
    """Mock the Google GenAI Client class globally."""
    return mocker.patch('nekomata.clients.providers.google.Client')


@pytest.fixture
def mock_google_client(mock_google_class: MagicMock) -> MagicMock:
    """Provide the mocked instance of the Google GenAI Client."""
    return mock_google_class.return_value


@pytest.fixture
def client(mock_google_class: MagicMock) -> GoogleClient:
    """Provide a standard, pre-initialized GoogleClient."""
    return GoogleClient(api_key='test-key')


class TestGoogleClient:
    """Test suite for base GoogleClient functionalities."""

    @pytest.mark.anyio
    async def test_acompletion_dispatch(self, mocker: MockerFixture, client: GoogleClient) -> None:
        """Test acompletion dispatch by API configuration."""
        mock_generate_content_response = mocker.MagicMock()
        mock_generate_content = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._generate_content',
            return_value=mock_generate_content_response,
        )

        mock_interactions_response = mocker.MagicMock()
        mock_interactions = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._interactions', return_value=mock_interactions_response
        )

        # Test dispatch to generateContent (default/explicit)
        generate_args = GoogleArgs(api='generateContent')
        response = await client.acompletion(model='model', prompt='prompt', args=generate_args)

        mock_generate_content.assert_called_once()
        mock_interactions.assert_not_called()
        assert response == mock_generate_content_response

        # Reset mocks
        mock_generate_content.reset_mock()
        mock_interactions.reset_mock()

        # Test dispatch to interactions
        interactions_args = GoogleArgs(api='interactions', interactions_args=InteractionsArgs())
        response = await client.acompletion(model='model', prompt='prompt', args=interactions_args)

        mock_interactions.assert_called_once()
        mock_generate_content.assert_not_called()
        assert response == mock_interactions_response

    @pytest.mark.anyio
    async def test_aclose(self, mocker: MockerFixture, client: GoogleClient) -> None:
        """Test aclose() method."""
        mock_aclose = mocker.patch.object(client._http_client, 'aclose', new_callable=mocker.AsyncMock)

        await client.aclose()

        mock_aclose.assert_called_once()

    def test_initialized_property(self, client: GoogleClient) -> None:
        """Test the initialized property."""
        assert client.initialized is True


class TestGoogleGenerateContent:
    """Test suite for the _generate_content API call of Google SDK."""

    @pytest.fixture
    def generate_content_response_factory(self, mocker: MockerFixture) -> Callable[..., MagicMock]:
        """Create factory function of generate content response object."""

        def factory() -> MagicMock:
            mock_response = mocker.MagicMock(spec=GenerateContentResponse)

            # Mock candidate
            mock_candidate = mocker.MagicMock()
            mock_candidate.finish_reason.value = 'STOP'

            # Mock parts
            mock_thought_part = mocker.MagicMock()
            mock_thought_part.text = 'thought'
            mock_thought_part.thought = True

            mock_text_part = mocker.MagicMock()
            mock_text_part.text = 'text'
            mock_text_part.thought = False

            mock_candidate.content.parts = [mock_thought_part, mock_text_part]
            mock_response.candidates = [mock_candidate]

            # Mock usage metadata
            mock_usage = mocker.MagicMock()
            mock_usage.total_token_count = 30
            mock_usage.prompt_token_count = 20
            mock_usage.candidates_token_count = 10
            mock_usage.cached_content_token_count = 5
            mock_usage.thoughts_token_count = 2
            mock_response.usage_metadata = mock_usage

            mock_response.parsed = None
            return mock_response

        return factory

    @pytest.mark.anyio
    async def test_acompletion_success(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test successful acompletion call."""
        # Mock generate content.
        mock_lib_response = mocker.MagicMock(spec=GenerateContentResponse)
        mock_google_client.aio.models.generate_content = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_generate_content_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(model='gemini-1.5-pro', prompt='hi')

        mock_google_client.aio.models.generate_content.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_with_thoughts(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test successful acompletion call with thinking blocks."""
        # Mock response to avoid conversion error
        mock_lib_response = mocker.MagicMock(spec=GenerateContentResponse)
        mock_google_client.aio.models.generate_content = mocker.AsyncMock(return_value=mock_lib_response)

        mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_generate_content_output',
            return_value=mocker.MagicMock(spec=ChatCompletionResponse),
        )

        await client.acompletion(model='gemini-2.0-flash-thinking', prompt='hi', reasoning_effort='medium')

        # Verify thinking config was passed
        _args, kwargs = mock_google_client.aio.models.generate_content.call_args
        config = kwargs['config']
        assert config.thinking_config.include_thoughts is True
        # The SDK converts string to Enum, we check that it equals the uppercase version or just the value
        assert (
            str(config.thinking_config.thinking_level).upper() == 'MEDIUM'
            or config.thinking_config.thinking_level == 'MEDIUM'
        )

    @pytest.mark.anyio
    async def test_acompletion_structured_output(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test successful acompletion call with structured output."""
        # Mock response to avoid conversion error
        mock_lib_response = mocker.MagicMock(spec=GenerateContentResponse)
        mock_google_client.aio.models.generate_content = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_result.parsed = _MockStructuredResponse(answer='a')
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_generate_content_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(
            model='gemini-1.5-pro', prompt='hi', response_format=_MockStructuredResponse
        )

        mock_google_client.aio.models.generate_content.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output_fail(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test acompletion raises error on structured output.

        `google.genai` package silently ignores the ValidationError. We force the _acompletion function to raise the
        ValidationError when we detect this pattern and propage the error to the acompletion function, which should
        handle the errors and notice the user.
        """
        # Mock response to avoid conversion error
        mock_lib_response = mocker.MagicMock(spec=GenerateContentResponse)
        mock_google_client.aio.models.generate_content = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_result.parsed = None  # Empty parsed field
        mock_convert_result.content = '{"answer": 10}'  # With an invalid content field.
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_generate_content_output',
            return_value=mock_convert_result,
        )

        response = await client.acompletion(
            model='gemini-1.5-pro', prompt='hi', response_format=_MockStructuredResponse
        )

        mock_google_client.aio.models.generate_content.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason is not None
        assert 'validation' in response.fail_reason

    @pytest.mark.anyio
    async def test_acompletion_failure(self, mock_google_client: MagicMock, client: GoogleClient) -> None:
        """Test failure during acompletion call."""
        mock_google_client.aio.models.generate_content.side_effect = Exception('Google API Error')

        response = await client.acompletion(model='gemini-1.5-pro', prompt='hi')

        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason
        assert 'Google API Error' in response.fail_reason

    def test_convert_output_success(
        self, generate_content_response_factory: Callable[..., MagicMock], client: GoogleClient
    ) -> None:
        """Test successful convert_output call."""
        mock_response = generate_content_response_factory()
        created_at = time.time()

        converted = client._convert_generate_content_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.content == 'text'
        assert converted.reason == 'thought'
        assert converted.finish_reason == 'STOP'
        assert converted.total_tokens == 30
        assert converted.input_tokens == 20
        assert converted.output_tokens == 10
        assert converted.cache_tokens == 5
        assert converted.reason_tokens == 2

    def test_convert_output_no_candidates(self, mocker: MockerFixture, client: GoogleClient) -> None:
        """Test that convert_output raises ValueError when candidates list is empty."""
        mock_response = mocker.MagicMock()
        mock_response.candidates = []
        created_at = time.time()

        with pytest.raises(ValueError, match='Response object has an empty `candidates` field'):
            client._convert_generate_content_output(mock_response, created_at)

    def test_convert_output_empty_content(self, mocker: MockerFixture, client: GoogleClient) -> None:
        """Test that convert_output raises ValueError when candidate content is empty."""
        mock_response = mocker.MagicMock()
        mock_candidate = mocker.MagicMock()
        mock_candidate.content = None
        mock_response.candidates = [mock_candidate]
        created_at = time.time()

        with pytest.raises(ValueError, match='Response object has an candidate with empty content'):
            client._convert_generate_content_output(mock_response, created_at)

    def test_convert_output_skips_multimodal(self, mocker: MockerFixture, client: GoogleClient) -> None:
        """Test that convert_output skips parts with no text (multimodal)."""
        mock_response = mocker.MagicMock()
        mock_candidate = mocker.MagicMock()
        mock_candidate.finish_reason = None

        # One multimodal part (text=None) and one text part
        mock_multimodal_part = mocker.MagicMock()
        mock_multimodal_part.text = None

        mock_text_part = mocker.MagicMock()
        mock_text_part.text = 'Hello'
        mock_text_part.thought = False

        mock_candidate.content.parts = [mock_multimodal_part, mock_text_part]
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None
        mock_response.parsed = None

        created_at = time.time()

        response = client._convert_generate_content_output(mock_response, created_at)
        assert response.content == 'Hello'


class TestGoogleInteractions:
    """Test suite for the _interactions API call of Google SDK."""

    @pytest.fixture
    def interactions_response_factory(self, mocker: MockerFixture) -> Callable[..., MagicMock]:
        """Create factory function of interaction response object."""

        def factory() -> MagicMock:
            mock_response = mocker.MagicMock(spec=Interaction)
            mock_response.status = 'COMPLETED'
            mock_response.output_text = 'hello text'

            # Mock steps for reasoning
            mock_thought_step = mocker.MagicMock(spec=ThoughtStep)
            mock_thought_step.type = 'thought'

            mock_text_content = mocker.MagicMock(spec=TextContent)
            mock_text_content.text = 'thinking text'
            mock_thought_step.summary = [mock_text_content]

            mock_response.steps = [mock_thought_step]

            # Mock usage metadata
            mock_usage = mocker.MagicMock()
            mock_usage.total_tokens = 50
            mock_usage.total_input_tokens = 25
            mock_usage.total_output_tokens = 25
            mock_usage.total_cached_tokens = 10
            mock_usage.total_thought_tokens = 5
            mock_response.usage = mock_usage

            return mock_response

        return factory

    @pytest.mark.anyio
    async def test_acompletion_success(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test successful acompletion call routing to _interactions."""
        mock_lib_response = mocker.MagicMock(spec=Interaction)
        mock_google_client.aio.interactions.create = mocker.AsyncMock(return_value=mock_lib_response)

        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_interactions_output',
            return_value=mock_convert_result,
        )

        interactions_args = GoogleArgs(api='interactions')
        response = await client.acompletion(model='gemini-2.5-pro', prompt='hi', args=interactions_args)

        mock_google_client.aio.interactions.create.assert_called_once()
        mock_convert_output.assert_called_once_with(
            response=mock_lib_response, created_at=ANY, custom_id=None, response_format=None
        )
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test _interactions call correctly passes response_format and parses JSON schemas."""
        mock_lib_response = mocker.MagicMock(spec=Interaction)
        mock_google_client.aio.interactions.create = mocker.AsyncMock(return_value=mock_lib_response)

        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_interactions_output',
            return_value=mock_convert_result,
        )

        interactions_args = GoogleArgs(api='interactions')

        response = await client.acompletion(
            model='gemini-1.5-pro', prompt='hi', response_format=_MockStructuredResponse, args=interactions_args
        )

        _args, kwargs = mock_google_client.aio.interactions.create.call_args

        # Verify store argument logic passed down.
        assert kwargs['store'] is True

        # Verify schema mapping format passed to the API
        assert kwargs['response_format'] == [
            {
                'type': 'text',
                'mime_type': 'application/json',
                'schema': _MockStructuredResponse.model_json_schema(),
            }
        ]

        mock_convert_output.assert_called_once_with(
            response=mock_lib_response, created_at=ANY, custom_id=None, response_format=_MockStructuredResponse
        )
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_failure(self, mock_google_client: MagicMock, client: GoogleClient) -> None:
        """Test failure during _interactions call."""
        mock_google_client.aio.interactions.create.side_effect = Exception('Interactions API Error')

        interactions_args = GoogleArgs(api='interactions')
        response = await client.acompletion(model='gemini-1.5-pro', prompt='hi', args=interactions_args)

        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason
        assert 'Interactions API Error' in response.fail_reason

    def test_convert_interactions_output_success(
        self, interactions_response_factory: Callable[..., MagicMock], client: GoogleClient
    ) -> None:
        """Test successful _convert_interactions_output extraction logic."""
        mock_response = interactions_response_factory()
        created_at = time.time()

        converted = client._convert_interactions_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.content == 'hello text'
        assert converted.reason == 'thinking text'
        assert converted.finish_reason == 'COMPLETED'
        assert converted.total_tokens == 50
        assert converted.input_tokens == 25
        assert converted.output_tokens == 25
        assert converted.cache_tokens == 10
        assert converted.reason_tokens == 5

    def test_convert_interactions_output_structured(
        self, interactions_response_factory: Callable[..., MagicMock], client: GoogleClient
    ) -> None:
        """Test _convert_interactions_output parses JSON if response_format is provided."""
        mock_response = interactions_response_factory()
        # Ensure output is valid json for the model
        mock_response.output_text = '{"answer": "parsed_value"}'
        created_at = time.time()

        converted = client._convert_interactions_output(
            response=mock_response, created_at=created_at, response_format=_MockStructuredResponse
        )

        assert converted.parsed == _MockStructuredResponse(answer='parsed_value')
        assert converted.content == '{"answer": "parsed_value"}'

    def test_convert_interactions_empty_thoughts(
        self, interactions_response_factory: Callable[..., MagicMock], client: GoogleClient
    ) -> None:
        """Test _convert_interactions_output sets reason to None if thought steps are empty or absent."""
        mock_response = interactions_response_factory()

        # Test case where step.summary is missing/None
        mock_thought_step = MagicMock(spec=ThoughtStep)
        mock_thought_step.type = 'thought'
        mock_thought_step.summary = None
        mock_response.steps = [mock_thought_step]

        created_at = time.time()
        converted = client._convert_interactions_output(mock_response, created_at)

        # The empty thought parsing logic sets it to None
        assert converted.reason is None


class TestGoogleBatchAPI:
    """Test suite for Google GenAI Batch API operations."""

    @pytest.mark.anyio
    async def test_acreate_batch_file_mode(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test acreate_batch in file mode (default)."""
        mock_upload = mock_google_client.aio.files.upload = mocker.AsyncMock()
        mock_upload.return_value.name = 'files/123'
        mock_create = mock_google_client.aio.batches.create = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acreate_batch(
            model='gemini-1.5-pro',
            prompt=['hello', 'world'],
            system_prompt='sys prompt',
            max_output_tokens=100,
            reasoning_effort='medium',
            response_format=_MockStructuredResponse,
            mode='file',
        )

        mock_upload.assert_called_once()
        kwargs = mock_upload.call_args[1]
        uploaded_bytes = kwargs['file'].getvalue()
        uploaded_text = uploaded_bytes.decode('utf-8')
        lines = uploaded_text.strip().split('\n')
        assert len(lines) == 2

        req0 = json.loads(lines[0])
        assert req0['key'].startswith('req-')
        assert req0['request']['contents'] == 'hello'
        assert req0['request']['config']['systemInstruction'] == 'sys prompt'
        assert req0['request']['config']['maxOutputTokens'] == 100
        assert req0['request']['config']['responseSchema'] == _MockStructuredResponse.model_json_schema()
        assert req0['request']['config']['thinkingConfig'] == {'includeThoughts': True, 'thinkingLevel': 'MEDIUM'}

        mock_create.assert_called_once_with(
            model='gemini-1.5-pro',
            src=types.BatchJobSource(file_name='files/123'),
        )
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_with_custom_id_string_file_mode(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test acreate_batch with single custom_id string in file mode."""
        mock_upload = mock_google_client.aio.files.upload = mocker.AsyncMock()
        mock_upload.return_value.name = 'files/123'
        mock_create = mock_google_client.aio.batches.create = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acreate_batch(
            model='gemini-1.5-pro',
            prompt=['hello', 'world'],
            custom_id='custom-id-prefix',
            mode='file',
        )

        mock_upload.assert_called_once()
        mock_create.assert_called_once()
        kwargs = mock_upload.call_args[1]
        uploaded_bytes = kwargs['file'].getvalue()
        uploaded_text = uploaded_bytes.decode('utf-8')
        lines = uploaded_text.strip().split('\n')
        assert len(lines) == 2

        req0 = json.loads(lines[0])
        req1 = json.loads(lines[1])
        assert req0['key'] == 'custom-id-prefix-0'
        assert req1['key'] == 'custom-id-prefix-1'
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_with_custom_id_list_file_mode(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test acreate_batch with custom_id list in file mode."""
        mock_upload = mock_google_client.aio.files.upload = mocker.AsyncMock()
        mock_upload.return_value.name = 'files/123'
        mock_create = mock_google_client.aio.batches.create = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acreate_batch(
            model='gemini-1.5-pro',
            prompt=['hello', 'world'],
            custom_id=['id-1', 'id-2'],
            mode='file',
        )

        mock_upload.assert_called_once()
        mock_create.assert_called_once()
        kwargs = mock_upload.call_args[1]
        uploaded_bytes = kwargs['file'].getvalue()
        uploaded_text = uploaded_bytes.decode('utf-8')
        lines = uploaded_text.strip().split('\n')
        assert len(lines) == 2

        req0 = json.loads(lines[0])
        req1 = json.loads(lines[1])
        assert req0['key'] == 'id-1'
        assert req1['key'] == 'id-2'
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_inline_mode(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test acreate_batch in inline mode."""
        mock_create = mock_google_client.aio.batches.create = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acreate_batch(
            model='gemini-1.5-pro',
            prompt=['hello', 'world'],
            system_prompt='sys prompt',
            max_output_tokens=100,
            reasoning_effort='medium',
            response_format=_MockStructuredResponse,
            mode='inline',
        )

        mock_create.assert_called_once()
        called_args = mock_create.call_args[1]
        assert called_args['model'] == 'gemini-1.5-pro'
        src = called_args['src']
        assert len(src) == 2
        assert src[0].contents == 'hello'
        assert src[0].metadata['custom_id'].startswith('req-')
        assert src[1].metadata['custom_id'].startswith('req-')
        assert src[0].config.system_instruction == 'sys prompt'
        assert src[0].config.max_output_tokens == 100
        assert src[0].config.response_schema == _MockStructuredResponse
        assert src[0].config.thinking_config.thinking_level == 'MEDIUM'
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_with_custom_id_string_inline_mode(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test acreate_batch with single custom_id string in inline mode."""
        mock_create = mock_google_client.aio.batches.create = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acreate_batch(
            model='gemini-1.5-pro',
            prompt=['hello', 'world'],
            custom_id='custom-id-prefix',
            mode='inline',
        )

        mock_create.assert_called_once()
        called_args = mock_create.call_args[1]
        src = called_args['src']
        assert len(src) == 2
        assert src[0].metadata['custom_id'] == 'custom-id-prefix-0'
        assert src[1].metadata['custom_id'] == 'custom-id-prefix-1'
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_with_custom_id_list_inline_mode(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test acreate_batch with custom_id list in inline mode."""
        mock_create = mock_google_client.aio.batches.create = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acreate_batch(
            model='gemini-1.5-pro',
            prompt=['hello', 'world'],
            custom_id=['id-1', 'id-2'],
            mode='inline',
        )

        mock_create.assert_called_once()
        called_args = mock_create.call_args[1]
        src = called_args['src']
        assert len(src) == 2
        assert src[0].metadata['custom_id'] == 'id-1'
        assert src[1].metadata['custom_id'] == 'id-2'
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acreate_batch_length_mismatch(self, mock_google_client: MagicMock, client: GoogleClient) -> None:
        """Test acreate_batch length mismatch validation."""
        with pytest.raises(ValueError, match='Lengths of list arguments do not match'):
            await client.acreate_batch(
                model='gemini-1.5-pro',
                prompt=['hello', 'world'],
                custom_id=['id-1', 'id-2', 'id-3'],
            )

    @pytest.mark.anyio
    async def test_acreate_batch_validation_error(self, mock_google_client: MagicMock, client: GoogleClient) -> None:
        """Test acreate_batch validation error when no list is provided."""
        with pytest.raises(ValueError, match=r'At least one of prompt,.* must be a list'):
            await client.acreate_batch(model='gemini', prompt='hello')

    @pytest.mark.anyio
    async def test_aretrieve_batch(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test aretrieve_batch call."""
        mock_get = mock_google_client.aio.batches.get = mocker.AsyncMock(return_value='mock-batch')

        res = await client.aretrieve_batch('batch-id', config={'val': 1})

        mock_get.assert_called_once_with(name='batch-id', config={'val': 1})
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_acancel_batch(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test acancel_batch call."""
        mock_cancel = mock_google_client.aio.batches.cancel = mocker.AsyncMock(return_value='mock-batch')

        res = await client.acancel_batch('batch-id', config={'val': 2})

        mock_cancel.assert_called_once_with(name='batch-id', config={'val': 2})
        assert res == 'mock-batch'

    @pytest.mark.anyio
    async def test_alist_batches(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test alist_batches call."""
        mock_list = mock_google_client.aio.batches.list = mocker.AsyncMock(return_value='mock-batches')

        res = await client.alist_batches(config={'val': 3})

        mock_list.assert_called_once_with(config={'val': 3})
        assert res == 'mock-batches'

    @pytest.mark.anyio
    async def test_adelete_batch(
        self, mocker: MockerFixture, mock_google_client: MagicMock, client: GoogleClient
    ) -> None:
        """Test adelete_batch call."""
        mock_delete = mock_google_client.aio.batches.delete = mocker.AsyncMock(return_value='mock-deleted')

        res = await client.adelete_batch('batch-id', config={'val': 4})

        mock_delete.assert_called_once_with(name='batch-id', config={'val': 4})
        assert res == 'mock-deleted'
