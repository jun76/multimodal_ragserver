from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import SecretStr

from .settings import Settings


@dataclass(kw_only=True, frozen=True)
class Config:
    ragserver_base_url: str = Settings.RAGSERVER_BASE_URL
    openai_llm_model: str = Settings.OPENAI_LLM_MODEL
    openai_api_key: Optional[SecretStr] = Settings.OPENAI_API_KEY
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        Settings.LOG_LEVEL
    )
