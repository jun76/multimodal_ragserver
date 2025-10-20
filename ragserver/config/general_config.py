from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ragserver.config.settings import (
    EmbedProvider,
    RerankProvider,
    Settings,
    VectorStoreProvider,
)


@dataclass(kw_only=True, frozen=True)
class GeneralConfig:
    project_name: str = Settings.PROJECT_NAME
    version: str = Settings.VERSION
    knowledgebase_name: str = Settings.KNOWLEDGEBASE_NAME
    vector_store: VectorStoreProvider = Settings.VECTOR_STORE
    text_embed_provider: EmbedProvider = Settings.TEXT_EMBED_PROVIDER
    image_embed_provider: Optional[EmbedProvider] = Settings.IMAGE_EMBED_PROVIDER
    audio_embed_provider: Optional[EmbedProvider] = Settings.AUDIO_EMBED_PROVIDER
    rerank_provider: RerankProvider = Settings.RERANK_PROVIDER
    device: Literal["cpu", "cuda", "mps"] = Settings.DEVICE
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        Settings.LOG_LEVEL
    )
