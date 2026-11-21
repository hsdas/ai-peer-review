import pytest
from unittest.mock import patch, MagicMock

from ai_peer_review.llm_clients.local_client import LocalLLMClient
from ai_peer_review.llm_clients.base_client import BaseLLMClient


class TestLocalLLMClient:
    def test_local_client_inheritance(self):
        # LocalLLMClient must implement the BaseLLMClient contract
        assert issubclass(LocalLLMClient, BaseLLMClient)

    @patch('ai_peer_review.llm_clients.local_client.get_local_server_config')
    @patch('ai_peer_review.llm_clients.local_client.OpenAI')
    def test_init_with_explicit_args(self, mock_openai, mock_server_cfg):
        # No config / env lookup should happen when everything is passed
        mock_server_cfg.return_value = {}
        client = LocalLLMClient(
            model="qwen3:32b",
            base_url="http://localhost:11434/v1",
            api_key="local",
            temperature=0.2,
            max_tokens=4096,
        )

        # The OpenAI SDK is pointed at the local base_url (not api.openai.com)
        mock_openai.assert_called_once()
        _, kwargs = mock_openai.call_args
        assert kwargs["base_url"] == "http://localhost:11434/v1"
        assert kwargs["api_key"] == "local"

        assert client.model == "qwen3:32b"
        assert client.base_url == "http://localhost:11434/v1"
        assert client.api_key == "local"
        assert client.temperature == 0.2
        assert client.max_tokens == 4096

    @patch('ai_peer_review.llm_clients.local_client.get_local_server_config')
    @patch('ai_peer_review.llm_clients.local_client.OpenAI')
    def test_init_inherits_server_defaults(self, mock_openai, mock_server_cfg):
        # When base_url/api_key are omitted, the client inherits them from the
        # local_server config section.
        mock_server_cfg.return_value = {
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        }
        client = LocalLLMClient(model="llama3.3:70b")

        assert client.base_url == "http://localhost:1234/v1"
        assert client.api_key == "lm-studio"
        _, kwargs = mock_openai.call_args
        assert kwargs["base_url"] == "http://localhost:1234/v1"
        assert kwargs["api_key"] == "lm-studio"

    @patch.dict('os.environ', {
        "AI_PEER_REVIEW_BASE_URL": "http://env-host:8080/v1",
        "AI_PEER_REVIEW_API_KEY": "env-key",
    })
    @patch('ai_peer_review.llm_clients.local_client.get_local_server_config')
    @patch('ai_peer_review.llm_clients.local_client.OpenAI')
    def test_init_reads_env_vars(self, mock_openai, mock_server_cfg):
        # Env vars take precedence over config defaults
        mock_server_cfg.return_value = {
            "base_url": "http://from-config:11434/v1",
            "api_key": "from-config",
        }
        client = LocalLLMClient(model="gemma3:27b")

        assert client.base_url == "http://env-host:8080/v1"
        assert client.api_key == "env-key"

    @patch('ai_peer_review.llm_clients.local_client.get_local_server_config')
    @patch('ai_peer_review.llm_clients.local_client.OpenAI')
    def test_generate_sends_chat_completion(self, mock_openai, mock_server_cfg):
        mock_server_cfg.return_value = {}
        # Build a fake OpenAI client whose .chat.completions.create returns a
        # response with one choice.
        mock_message = MagicMock()
        mock_message.content = "  Generated review  "
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_create = MagicMock(return_value=mock_response)
        mock_instance = MagicMock()
        mock_instance.chat.completions.create = mock_create
        mock_openai.return_value = mock_instance

        with patch('ai_peer_review.llm_clients.local_client.get_prompt',
                   return_value="You are a neuroscientist and expert in brain imaging."):
            client = LocalLLMClient(
                model="qwen3:32b",
                base_url="http://localhost:11434/v1",
                api_key="local",
            )
            response = client.generate("Review this paper.")

        # The request hits the local model with system+user messages
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["model"] == "qwen3:32b"
        assert kwargs["messages"] == [
            {"role": "system", "content": "You are a neuroscientist and expert in brain imaging."},
            {"role": "user", "content": "Review this paper."},
        ]
        assert kwargs["temperature"] == 0.1

        # The response is stripped of surrounding whitespace
        assert response == "Generated review"
