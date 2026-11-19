from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/compliance"

    ollama_url: str = "http://localhost:11434"
    llm_provider: Literal["ollama", "gemini"] = "ollama"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    embed_model: str = "bge-m3"
    ollama_num_ctx: int = 8192

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    chunking: Literal["article", "fixed"] = "article"
    reranker: Literal["bge", "none"] = "bge"
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
