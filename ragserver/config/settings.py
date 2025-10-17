from __future__ import annotations

import os
from enum import StrEnum
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class VectorStoreProvider(StrEnum):
    CHROMA = "chroma"
    PGVECTOR = "pgvector"


class EmbedProvider(StrEnum):
    CLIP = "clip"
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


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
    RERANK_PROVIDER: RerankProvider = RerankProvider.FLAGEMBEDDING
    DEVICE: str = "cuda"
    LOG_LEVEL: str = "DEBUG"

    # vector store
    LOAD_LIMIT: int = 10000
    CHECK_UPDATE: bool = False
    CHROMA_PERSIST_DIR: str = f"{PROJECT_NAME}_db"
    CHROMA_HOST: Optional[str] = None
    CHROMA_PORT: Optional[int] = None
    CHROMA_API_KEY: Optional[str] = None
    CHROMA_TENANT: Optional[str] = None
    CHROMA_DATABASE: Optional[str] = None
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DATABASE: str = PROJECT_NAME
    PGVECTOR_USER: str = PROJECT_NAME
    PGVECTOR_PASSWORD: Optional[str] = os.getenv("PG_PASSWORD")

    # embedding
    OPENAI_EMBED_MODEL_TEXT: str = "text-embedding-3-small"
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = None
    COHERE_EMBED_MODEL_TEXT: str = "embed-v4.0"
    COHERE_EMBED_MODEL_IMAGE: str = "embed-v4.0"
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    CLIP_EMBED_MODEL_TEXT: str = "ViT-B/32"
    CLIP_EMBED_MODEL_IMAGE: str = "ViT-B/32"
    HUGGINGFACE_EMBED_MODEL_TEXT: str = "intfloat/multilingual-e5-base"
    # HUGGINGFACE_EMBED_MODEL_IMAGE: str = "openai/clip-vit-base-patch32"

    # ingest
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    USER_AGENT: str = PROJECT_NAME
    UPLOAD_DIR: str = "upload"

    # rerank
    FLAGEMBEDDING_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    COHERE_RERANK_MODEL: str = "rerank-multilingual-v3.0"
    TOPK: int = 10
