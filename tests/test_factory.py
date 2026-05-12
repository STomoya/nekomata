"""Tests for the client factory module."""

import pytest

from nekomata.clients.factory import SUPPORTED_PROVIDERS, create_client


class TestClientFactory:
    """Test suite for the client factory."""

    def test_create_client_openai(self, mocker) -> None:
        """Test that the factory correctly creates an OpenAI client."""
        mock_openai = mocker.patch('nekomata.clients.factory.OpenAIClient')
        client = create_client(provider='openai', api_key='test-key')

        mock_openai.assert_called_once_with(
            api_key='test-key',
            base_url=None,
            max_concurrent=None,
            max_connections=100,
            max_keepalive=10,
            keepalive_expiry=None,
            timeout=60.0,
            ssl_verify=True,
        )
        assert client == mock_openai.return_value

    def test_create_client_google(self, mocker) -> None:
        """Test that the factory correctly creates a Google client."""
        mock_google = mocker.patch('nekomata.clients.factory.GoogleClient')
        client = create_client(provider='google', api_key='test-key')

        mock_google.assert_called_once_with(
            api_key='test-key',
            max_concurrent=None,
            max_connections=100,
            max_keepalive=10,
            keepalive_expiry=None,
            timeout=60.0,
        )
        assert client == mock_google.return_value

    def test_create_client_anthropic(self, mocker) -> None:
        """Test that the factory correctly creates an Anthropic client."""
        mock_anthropic = mocker.patch('nekomata.clients.factory.AnthropicClient')
        client = create_client(provider='anthropic', api_key='test-key')

        mock_anthropic.assert_called_once_with(
            api_key='test-key',
            max_concurrent=None,
            max_connections=100,
            max_keepalive=10,
            keepalive_expiry=None,
            timeout=60.0,
        )
        assert client == mock_anthropic.return_value

    def test_create_client_unsupported(self) -> None:
        """Test that the factory raises ValueError for unsupported providers."""
        with pytest.raises(ValueError, match='Unsupport provider'):
            # Casting to ignore type warning for test case
            create_client(provider='unsupported')  # type: ignore

    def test_supported_providers_constant(self) -> None:
        """Test that SUPPORTED_PROVIDERS contains the expected values."""
        expected = {'openai', 'google', 'anthropic'}
        assert expected == SUPPORTED_PROVIDERS
