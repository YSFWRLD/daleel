"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3, multilingual). Runs on CPU to leave the GPU to Ollama.

Speed knobs (measured on a 6-core CPU, see README): `reranker_max_length` (tokens per query+passage)
and `reranker_int8` (dynamic int8 quantization of the linear layers).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from app.config import get_settings


@lru_cache
def _model():
    s = get_settings()
    if s.hf_home:
        os.environ.setdefault("HF_HOME", s.hf_home)  # keep the ~2.3 GB model off a full C: drive
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if (cache / ("models--" + s.reranker_model.replace("/", "--"))).exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # already downloaded: skip slow/failed Hub checks
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(s.reranker_model, device="cpu", max_length=s.reranker_max_length)
    if s.reranker_int8:
        for mod in model.modules():
            if type(mod).__name__.endswith("ForSequenceClassification"):
                torch.quantization.quantize_dynamic(mod, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return model


def rerank(query: str, passages: list[str]) -> list[float]:
    """Relevance scores in [0, 1] (sigmoid of the cross-encoder logit), same order as `passages`."""
    if not passages:
        return []
    scores = _model().predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)
    return [float(x) for x in scores]


def warm_up() -> None:
    """Load the model at startup so the first question doesn't pay for it."""
    rerank("warm up", ["warm up"])
