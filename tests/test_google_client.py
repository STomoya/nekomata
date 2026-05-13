"""Tests for the Google client."""

import time
from unittest.mock import ANY

import pytest
from google.genai.types import GenerateContentResponse
from pydantic import BaseModel
from pytest_mock import MockerFixture

from nekomata.clients.providers.google import GoogleClient
from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus


class TestGoogleClient:
    """Test suite for GoogleClient."""

    @pytest.fixture
    def generate_content_response_factory(self, mocker: MockerFixture):
        """Create factory function of generate content response object."""

        def factory():
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
    async def test_acompletion_success(self, mocker: MockerFixture) -> None:
        """Test successful acompletion call."""
        # Mock Google GenAI Client
        mock_google_class = mocker.patch('nekomata.clients.providers.google.Client')
        mock_instance = mock_google_class.return_value

        # Mock generate content.
        mock_lib_response = mocker.MagicMock(spec=GenerateContentResponse)
        mock_instance.aio.models.generate_content = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_output',
            return_value=mock_convert_result,
        )

        client = GoogleClient(api_key='test-key')
        response = await client.acompletion(model='gemini-1.5-pro', prompt='hi')

        mock_instance.aio.models.generate_content.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_with_thoughts(self, mocker: MockerFixture) -> None:
        """Test successful acompletion call with thinking blocks."""
        mock_google_class = mocker.patch('nekomata.clients.providers.google.Client')
        mock_instance = mock_google_class.return_value

        # Mock response to avoid conversion error
        mock_lib_response = mocker.MagicMock(spec=GenerateContentResponse)
        mock_instance.aio.models.generate_content = mocker.AsyncMock(return_value=mock_lib_response)

        mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_output',
            return_value=mocker.MagicMock(spec=ChatCompletionResponse),
        )

        client = GoogleClient(api_key='test-key')
        await client.acompletion(model='gemini-2.0-flash-thinking', prompt='hi', reasoning_effort='medium')

        # Verify thinking config was passed
        _args, kwargs = mock_instance.aio.models.generate_content.call_args
        config = kwargs['config']
        assert config.thinking_config.include_thoughts is True
        # The SDK converts string to Enum, we check that it equals the uppercase version or just the value
        assert (
            str(config.thinking_config.thinking_level).upper() == 'MEDIUM'
            or config.thinking_config.thinking_level == 'MEDIUM'
        )

    @pytest.mark.anyio
    async def test_acompletion_structured_output(self, mocker: MockerFixture) -> None:
        """Test successful acompletion call with structured output."""
        mock_google_class = mocker.patch('nekomata.clients.providers.google.Client')
        mock_instance = mock_google_class.return_value

        # Mock response to avoid conversion error
        mock_lib_response = mocker.MagicMock(spec=GenerateContentResponse)
        mock_instance.aio.models.generate_content = mocker.AsyncMock(return_value=mock_lib_response)

        class MyResponse(BaseModel):
            answer: str

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_result.parsed = MyResponse(answer='a')
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_output',
            return_value=mock_convert_result,
        )

        client = GoogleClient(api_key='test-key')
        response = await client.acompletion(model='gemini-1.5-pro', prompt='hi', response_format=MyResponse)

        mock_instance.aio.models.generate_content.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response == mock_convert_result

    @pytest.mark.anyio
    async def test_acompletion_structured_output_fail(self, mocker: MockerFixture) -> None:
        """Test acompletion raises error on structured output.

        `google.genai` package silently ignores the ValidationError. We force the _acompletion function to raise the
        ValidationError when we detect this pattern and propage the error to the acompletion function, which should
        handle the errors and notice the user.
        """
        mock_google_class = mocker.patch('nekomata.clients.providers.google.Client')
        mock_instance = mock_google_class.return_value

        # Mock response to avoid conversion error
        mock_lib_response = mocker.MagicMock(spec=GenerateContentResponse)
        mock_instance.aio.models.generate_content = mocker.AsyncMock(return_value=mock_lib_response)

        # Mock response conversion.
        mock_convert_result = mocker.MagicMock(spec=ChatCompletionResponse)
        mock_convert_result.parsed = None  # Empty parsed field
        mock_convert_result.content = '{"answer": 10}'  # With an invalid content field.
        mock_convert_output = mocker.patch(
            'nekomata.clients.providers.google.GoogleClient._convert_output',
            return_value=mock_convert_result,
        )

        class MyResponse(BaseModel):
            answer: str

        client = GoogleClient(api_key='test-key')

        response = await client.acompletion(model='gemini-1.5-pro', prompt='hi', response_format=MyResponse)

        mock_instance.aio.models.generate_content.assert_called_once()
        mock_convert_output.assert_called_once_with(response=mock_lib_response, created_at=ANY, custom_id=None)
        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason is not None
        assert 'validation' in response.fail_reason

    @pytest.mark.anyio
    async def test_acompletion_failure(self, mocker: MockerFixture) -> None:
        """Test failure during acompletion call."""
        mock_google_class = mocker.patch('nekomata.clients.providers.google.Client')
        mock_instance = mock_google_class.return_value

        mock_instance.aio.models.generate_content.side_effect = Exception('Google API Error')

        client = GoogleClient(api_key='test-key')
        response = await client.acompletion(model='gemini-1.5-pro', prompt='hi')

        assert response.status == ChatCompletionStatus.FAILED
        assert response.fail_reason
        assert 'Google API Error' in response.fail_reason

    @pytest.mark.anyio
    async def test_aclose(self, mocker: MockerFixture) -> None:
        """Test aclose() method."""
        mocker.patch('nekomata.clients.providers.google.Client')

        client = GoogleClient(api_key='test-key')
        mock_aclose = mocker.patch.object(client._http_client, 'aclose', new_callable=mocker.AsyncMock)

        await client.aclose()

        mock_aclose.assert_called_once()

    def test_convert_output_success(self, mocker: MockerFixture, generate_content_response_factory) -> None:
        """Test successful convert_output call."""
        mock_response = generate_content_response_factory()

        mocker.patch('nekomata.clients.providers.google.Client')
        client = GoogleClient(api_key='test-key')

        created_at = time.time()

        converted = client._convert_output(mock_response, created_at)

        assert converted.original == mock_response
        assert converted.content == 'text'
        assert converted.reason == 'thought'
        assert converted.finish_reason == 'STOP'
        assert converted.total_tokens == 30  # noqa: PLR2004
        assert converted.input_tokens == 20  # noqa: PLR2004
        assert converted.output_tokens == 10  # noqa: PLR2004
        assert converted.cache_tokens == 5  # noqa: PLR2004
        assert converted.reason_tokens == 2  # noqa: PLR2004

    def test_convert_output_no_candidates(self, mocker: MockerFixture) -> None:
        """Test that convert_output raises ValueError when candidates list is empty."""
        mocker.patch('nekomata.clients.providers.google.Client')
        client = GoogleClient(api_key='test-key')

        mock_response = mocker.MagicMock()
        mock_response.candidates = []

        created_at = time.time()

        with pytest.raises(ValueError, match='Response object has an empty `candidates` field'):
            client._convert_output(mock_response, created_at)

    def test_convert_output_empty_content(self, mocker: MockerFixture) -> None:
        """Test that convert_output raises ValueError when candidate content is empty."""
        mocker.patch('nekomata.clients.providers.google.Client')
        client = GoogleClient(api_key='test-key')

        mock_response = mocker.MagicMock()
        mock_candidate = mocker.MagicMock()
        mock_candidate.content = None
        mock_response.candidates = [mock_candidate]

        created_at = time.time()

        with pytest.raises(ValueError, match='Response object has an candidate with empty content'):
            client._convert_output(mock_response, created_at)

    def test_convert_output_skips_multimodal(self, mocker: MockerFixture) -> None:
        """Test that convert_output skips parts with no text (multimodal)."""
        mocker.patch('nekomata.clients.providers.google.Client')
        client = GoogleClient(api_key='test-key')

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

        response = client._convert_output(mock_response, created_at)
        assert response.content == 'Hello'
