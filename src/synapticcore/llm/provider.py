"""
LLM Provider Abstraction

Configurable interface for LLM backends. Supports DeepSeek, Anthropic,
and auto-configuration from environment variables.
"""

import os
import json
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM text generation."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt

        Returns:
            Generated text
        """
        ...


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider."""

    def __init__(self, api_key: str = None, model: str = "deepseek-chat",
                 api_endpoint: str = "https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.api_endpoint = api_endpoint
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY or pass api_key.")

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        response = requests.post(self.api_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key.")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text


class OpenAICompatibleProvider(LLMProvider):
    """Provider for any OpenAI-compatible API (OpenAI, Together, Groq, etc.)."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini",
                 base_url: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key.")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


# Provider registry
PROVIDERS = {
    "deepseek": DeepSeekProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAICompatibleProvider,
}


def create_provider(provider_name: str = None, **kwargs) -> LLMProvider:
    """
    Create an LLM provider from name or auto-detect from environment.

    Args:
        provider_name: One of "deepseek", "anthropic", "openai". If None, auto-detects.
        **kwargs: Passed to the provider constructor.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If no provider could be configured.
    """
    # Explicit provider
    if provider_name:
        name = provider_name.lower()
        if name not in PROVIDERS:
            raise ValueError(f"Unknown provider '{name}'. Options: {list(PROVIDERS.keys())}")
        return PROVIDERS[name](**kwargs)

    # Check env var for explicit choice
    env_provider = os.environ.get("SYNAPTICCORE_LLM_PROVIDER", "").lower()
    if env_provider and env_provider in PROVIDERS:
        return PROVIDERS[env_provider](**kwargs)

    # Auto-detect from available API keys
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(**kwargs)
    if os.environ.get("DEEPSEEK_API_KEY"):
        return DeepSeekProvider(**kwargs)
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleProvider(**kwargs)

    raise ValueError(
        "No LLM provider configured. Set one of: "
        "ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, "
        "or set SYNAPTICCORE_LLM_PROVIDER explicitly."
    )
