from dataclasses import dataclass, field
from typing import Optional

from ragserver.config.settings import Settings


@dataclass(kw_only=True)
class EmbeddingConfig:
    text_embed_provider: str = field(default=Settings.TEXT_EMBED_PROVIDER)
    image_embed_provider: str = field(default=Settings.IMAGE_EMBED_PROVIDER)
    openai_embed_model_text: str = field(default=Settings.OPENAI_EMBED_MODEL_TEXT)
    openai_api_key: str = field(default=Settings.OPENAI_API_KEY)
    openai_base_url: Optional[str] = field(default=Settings.OPENAI_BASE_URL)
    cohere_embed_model_text: str = field(default=Settings.COHERE_EMBED_MODEL_TEXT)
    cohere_embed_model_image: str = field(default=Settings.COHERE_EMBED_MODEL_IMAGE)
    cohere_api_key: Optional[str] = field(default=Settings.COHERE_API_KEY)
    clip_embed_model_text: str = field(default=Settings.CLIP_EMBED_MODEL_TEXT)
    clip_embed_model_image: str = field(default=Settings.CLIP_EMBED_MODEL_IMAGE)
