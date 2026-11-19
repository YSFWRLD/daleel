from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@127.0.0.1:5432/compliance"

    ollama_url: str = "http://127.0.0.1:11434"
    llm_provider: Literal["ollama", "gemini"] = "ollama"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    embed_model: str = "bge-m3"
    ollama_num_ctx: int = 8192
    ollama_keep_alive: str = "30m"  # keep models in VRAM between questions (avoids ~5-10 s reloads)

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    chunking: Literal["article", "fixed"] = "article"
    reranker: Literal["bge", "none"] = "bge"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    hf_home: str | None = None  # Hugging Face cache dir for the reranker model (e.g. D:/HFCache)
    reranker_max_length: int = 256  # tokens of query + passage; 256 keeps the same top-1 as 512, ~3x faster
    reranker_int8: bool = True  # dynamic int8 on CPU: ~1.6x faster, same top-1 on the benchmark
    top_k: int = 5  # chunks passed to the LLM
    candidates: int = 30  # per retriever, before fusion
    rerank_top: int = 10  # fused candidates sent to the reranker (CPU cost grows linearly)


@lru_cache
def get_settings() -> Settings:
    return Settings()
