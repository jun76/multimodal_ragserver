from __future__ import annotations

import os
from enum import StrEnum
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


class VectorStoreProvider(StrEnum):
    CHROMA = "chroma"
    PGVECTOR = "pgvector"


class EmbedProvider(StrEnum):
    CLIP = "clip"
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CLAP = "clap"


class RerankProvider(StrEnum):
    FLAGEMBEDDING = "flagembedding"
    COHERE = "cohere"


class Settings:
    """各種設定値の指定用クラス

    以下の値を設定ファイルのように書き換えて使用する想定。
    API キーやパスワード等は予め .env ファイルに記述しておく。

    """

    # general
    PROJECT_NAME: str = "ragserver"
    VERSION: str = "1.0"
    KNOWLEDGEBASE_NAME: str = "default"
    VECTOR_STORE: VectorStoreProvider = VectorStoreProvider.CHROMA
    TEXT_EMBED_PROVIDER: EmbedProvider = EmbedProvider.HUGGINGFACE
    IMAGE_EMBED_PROVIDER: Optional[EmbedProvider] = EmbedProvider.CLIP
    AUDIO_EMBED_PROVIDER: Optional[EmbedProvider] = EmbedProvider.CLAP
    RERANK_PROVIDER: RerankProvider = RerankProvider.FLAGEMBEDDING
    DEVICE: Literal["cpu", "cuda", "mps"] = "cuda"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    # vector store
    LOAD_LIMIT: int = 10000
    CHECK_UPDATE: bool = False
    CHROMA_PERSIST_DIR: str = f"{PROJECT_NAME}_db"
    CHROMA_HOST: Optional[str] = None
    CHROMA_PORT: Optional[int] = None
    CHROMA_API_KEY: Optional[SecretStr] = (
        SecretStr(os.getenv("CHROMA_API_KEY", "")) or None
    )
    CHROMA_TENANT: Optional[str] = None
    CHROMA_DATABASE: Optional[str] = None
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DATABASE: str = PROJECT_NAME
    PGVECTOR_USER: str = PROJECT_NAME
    PGVECTOR_PASSWORD: Optional[SecretStr] = (
        SecretStr(os.getenv("PGVECTOR_PASSWORD", "")) or None
    )

    # embedding
    OPENAI_EMBED_MODEL_TEXT: str = "text-embedding-3-small"
    OPENAI_API_KEY: Optional[SecretStr] = (
        SecretStr(os.getenv("OPENAI_API_KEY", "")) or None
    )
    OPENAI_BASE_URL: Optional[str] = None
    COHERE_EMBED_MODEL_TEXT: str = "embed-v4.0"
    COHERE_EMBED_MODEL_IMAGE: str = "embed-v4.0"
    COHERE_API_KEY: Optional[SecretStr] = (
        SecretStr(os.getenv("COHERE_API_KEY", "")) or None
    )
    CLIP_EMBED_MODEL_TEXT: str = "ViT-B/32"
    CLIP_EMBED_MODEL_IMAGE: str = "ViT-B/32"
    HUGGINGFACE_EMBED_MODEL_TEXT: str = "intfloat/multilingual-e5-base"
    HUGGINGFACE_EMBED_MODEL_IMAGE: str = "llamaindex/vdr-2b-multi-v1"
    CLAP_EMBED_MODEL_AUDIO: Literal[
        "effect_short", "effect_varlen", "music", "speech", "general"
    ] = "effect_varlen"

    # ingest
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    USER_AGENT: str = PROJECT_NAME
    UPLOAD_DIR: str = "upload"

    # rerank
    FLAGEMBEDDING_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    COHERE_RERANK_MODEL: str = "rerank-multilingual-v3.0"
    TOPK: int = 10
