import os
from typing import Optional

from openai import OpenAI

from .base_client import BaseLLMClient
from ..utils.config import get_local_server_config, get_prompt


class LocalLLMClient(BaseLLMClient):
    """Client for a locally-served LLM that exposes an OpenAI-compatible API.

    This client replaces the original cloud-API clients (OpenAI, Anthropic,
    Google, Together.ai). It never contacts a third-party LLM provider: the
    ``base_url`` points at an inference server that *you* run yourself, so the
    manuscript text never leaves a machine you control. This is the intended
    mode for peer review of unpublished work.

    It works with any local server that implements the OpenAI
    ``/v1/chat/completions`` endpoint, including:

    - **Ollama**        (default ``http://localhost:11434/v1``)
    - **LM Studio**     (``http://localhost:1234/v1``)
    - **llama.cpp** ``llama-server`` (``http://localhost:8080/v1``)
    - **vLLM**          (``http://localhost:8000/v1``)
    - **LocalAI**, **text-generation-webui** (OpenAI extension), **TGI** (with
      ``--chat-template``), and any other OpenAI-compatible server.

    No real API key is required: local servers ignore the ``Authorization``
    header, but the OpenAI Python SDK requires a non-empty string, so the
    placeholder ``"local"`` is sent by default.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 8000,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the local LLM client.

        Args:
            model: Model tag as recognized by the local server
                (e.g. ``"qwen3:32b"``, ``"deepseek-r1:70b"``).
            base_url: Base URL of the OpenAI-compatible local server. If
                ``None``, falls back to the ``AI_PEER_REVIEW_BASE_URL`` env
                var, then to ``local_server.base_url`` in the config, then to
                the Ollama default.
            api_key: Placeholder API key. If ``None``, falls back to the
                ``AI_PEER_REVIEW_API_KEY`` env var, then to
                ``local_server.api_key`` in the config, then to ``"local"``.
            temperature: Sampling temperature (low for reproducible reviews).
            max_tokens: Maximum tokens to generate. Default 8000 (raised from
                the original 4000) because local long-context models can and
                should produce full-length peer reviews.
            timeout: Per-request timeout in seconds. ``None`` uses the OpenAI
                SDK default. Local inference on large models can be slow, so
                you may want to raise this (e.g. 600) for very large models.
        """
        self.model = model

        server_cfg = get_local_server_config()
        self.base_url = (
            base_url
            or os.environ.get("AI_PEER_REVIEW_BASE_URL")
            or server_cfg.get("base_url")
            or "http://localhost:11434/v1"
        )
        self.api_key = (
            api_key
            or os.environ.get("AI_PEER_REVIEW_API_KEY")
            or server_cfg.get("api_key")
            or "local"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the local LLM.

        Args:
            prompt: Input prompt (the user message).

        Returns:
            Generated response text.
        """
        # The system prompt is read from the config so the scientific-validity
        # profile (configs/scientific-only.json) overrides the default persona.
        system_prompt = get_prompt("system")
        if not system_prompt:
            system_prompt = "You are a neuroscientist and expert in brain imaging."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return (response.choices[0].message.content or "").strip()
