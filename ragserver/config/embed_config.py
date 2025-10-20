from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .settings import Settings


@dataclass(kw_only=True, frozen=True)
class EmbedConfig:
    """埋め込み関連の設定用データクラス"""

    # Text
    openai_embed_model_text: str = Settings.OPENAI_EMBED_MODEL_TEXT
    cohere_embed_model_text: str = Settings.COHERE_EMBED_MODEL_TEXT
    clip_embed_model_text: str = Settings.CLIP_EMBED_MODEL_TEXT
    huggingface_embed_model_text: str = Settings.HUGGINGFACE_EMBED_MODEL_TEXT

    # Image
    cohere_embed_model_image: str = Settings.COHERE_EMBED_MODEL_IMAGE
    clip_embed_model_image: str = Settings.CLIP_EMBED_MODEL_IMAGE
    huggingface_embed_model_image: str = Settings.HUGGINGFACE_EMBED_MODEL_IMAGE

    # Audio
    clap_embed_model_audio: Literal[
        "effect_short", "effect_varlen", "music", "speech", "general"
    ] = Settings.CLAP_EMBED_MODEL_AUDIO
