from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import SecretStr

from .settings import Settings, VectorStoreProvider


@dataclass(kw_only=True, frozen=True)
class Config:
    ragserver_base_url: str = Settings.RAGSERVER_BASE_URL
    vector_store_provider: VectorStoreProvider = Settings.VECTOR_STORE_PROVIDER
    llm_openai_model: str = Settings.LLM_OPENAI_MODEL
    openai_api_key: Optional[SecretStr] = Settings.OPENAI_API_KEY
    device: Literal["cpu", "cuda", "mps"] = Settings.DEVICE
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        Settings.LOG_LEVEL
    )
