"""Local model through Ollama (default: Qwen 2.5 7B, 4-bit, on the GPU). Structured output via `format`."""

from __future__ import annotations

import json

import httpx

from app.llm.base import LLMError


class OllamaLLM:
    def __init__(self, url: str, model: str, num_ctx: int, keep_alive: str):
        self.url, self.model, self.num_ctx, self.keep_alive = url, model, num_ctx, keep_alive
        self.name = f"ollama:{model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 1536},  # hard stop for runaway output
        }
        try:
            r = httpx.post(f"{self.url}/api/chat", json=body, timeout=300)
            r.raise_for_status()
            return json.loads(r.json()["message"]["content"])
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"Ollama call failed: {e}") from e

    def warm_up(self) -> None:
        httpx.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": "",
                "keep_alive": self.keep_alive,
                "options": {"num_ctx": self.num_ctx},
            },
            timeout=300,
        )
