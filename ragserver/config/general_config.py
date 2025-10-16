from dataclasses import dataclass, field

from ragserver.config.settings import (
    EmbedProvider,
    RerankProvider,
    Settings,
    VectorStoreProvider,
)


@dataclass(kw_only=True)
class GeneralConfig:
    project_name: str = field(default=Settings.PROJECT_NAME)
    knowledgebase_name: str = field(default=Settings.KNOWLEDGEBASE_NAME)
    vector_store: VectorStoreProvider = field(default=Settings.VECTOR_STORE)
    text_embed_provider: EmbedProvider = field(default=Settings.TEXT_EMBED_PROVIDER)
    image_embed_provider: EmbedProvider = field(default=Settings.IMAGE_EMBED_PROVIDER)
    rerank_provider: RerankProvider = field(default=Settings.RERANK_PROVIDER)
