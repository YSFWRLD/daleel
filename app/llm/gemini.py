"""Google Gemini (free API tier) as the switchable cloud model. Structured output via responseSchema."""

from __future__ import annotations

import json

import httpx

from app.llm.base import LLMError

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiLLM:
    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model
        self.name = f"gemini:{model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        try:
            r = httpx.post(
                API.format(model=self.model),
                json=body,
                headers={"x-goog-api-key": self.api_key},  # header, not URL, so the key never lands in logs
                timeout=120,
            )
            if r.status_code == 429:
                raise LLMError("Gemini free-tier rate limit reached; wait a minute or switch LLM_PROVIDER=ollama")
            r.raise_for_status()
            return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"Gemini call failed: {e}") from e
