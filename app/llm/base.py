"""LLM interface: every provider returns a JSON object that follows the given schema."""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """`model` overrides the configured model name for the provider (used by the evaluation)."""
    from app.config import get_settings

    s = get_settings()
    provider = provider or s.llm_provider
    if provider == "ollama":
        from app.llm.ollama import OllamaLLM

        return OllamaLLM(s.ollama_url, model or s.ollama_model, s.ollama_num_ctx, s.ollama_keep_alive)
    if provider == "gemini":
        from app.llm.gemini import GeminiLLM

        if not s.gemini_api_key:
            raise LLMError("LLM_PROVIDER=gemini needs GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)")
        return GeminiLLM(s.gemini_api_key, model or s.gemini_model)
    raise LLMError(f"unknown LLM provider {provider!r}")
