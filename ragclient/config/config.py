from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import SecretStr

from ragclient.config.settings import LLMProvider, Settings, VectorStoreProvider


@dataclass(kw_only=True, frozen=True)
class Config:
    ragserver_base_url: str = Settings.RAGSERVER_BASE_URL
    vector_store_provider: VectorStoreProvider = Settings.VECTOR_STORE_PROVIDER
    llm_provider: LLMProvider = Settings.LLM_PROVIDER
    llm_local_base_url: str = Settings.LLM_LOCAL_BASE_URL
    llm_local_model: str = Settings.LLM_LOCAL_MODEL
    llm_openai_model: str = Settings.LLM_OPENAI_MODEL
    openai_api_key: Optional[SecretStr] = Settings.OPENAI_API_KEY
    device: Literal["cpu", "cuda", "mps"] = Settings.DEVICE
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        Settings.LOG_LEVEL
    )
