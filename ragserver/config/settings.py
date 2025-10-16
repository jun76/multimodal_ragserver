import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    ## ラベル
    PROJECT_NAME = "ragserver"
    CHROMA = "chroma"
    PGVECTOR = "pgvector"
    CLIP = "clip"
    OPENAI = "openai"
    COHERE = "cohere"
    FLAGEMBEDDING = "flagembedding"
    COHERE = "cohere"

    ## パラメータ
    # vector store
    VECTOR_STORE: str = CHROMA
    LOAD_LIMIT: int = 10000
    CHECK_UPDATE: bool = False
    KNOWLEDGEBASE_NAME: str = "default"
    CHROMA_PERSIST_DIR: str = "chroma_db"
    CHROMA_HOST: Optional[str] = None
    CHROMA_PORT: Optional[int] = None
    CHROMA_API_KEY: Optional[str] = None
    CHROMA_TENANT: Optional[str] = None
    CHROMA_DATABASE: Optional[str] = None
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DATABASE: str = PROJECT_NAME
    PGVECTOR_USER: str = PROJECT_NAME
    PGVECTOR_PASSWORD: str = PROJECT_NAME

    # embedding
    TEXT_EMBED_PROVIDER: str = CLIP
    IMAGE_EMBED_PROVIDER: str = CLIP
    OPENAI_EMBED_MODEL_TEXT: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "dummy")
    OPENAI_BASE_URL: Optional[str] = None
    COHERE_EMBED_MODEL_TEXT: str = "embed-v4.0"
    COHERE_EMBED_MODEL_IMAGE: str = "embed-v4.0"
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    CLIP_EMBED_MODEL_TEXT: str = "ViT-B/32"
    CLIP_EMBED_MODEL_IMAGE: str = "ViT-B/32"

    # ingest
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    USER_AGENT: str = PROJECT_NAME
    UPLOAD_DIR: str = "upload"

    # rerank
    RERANK_PROVIDER: str = FLAGEMBEDDING
    FLAGEMBEDDING_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    COHERE_RERANK_MODEL: str = "rerank-multilingual-v3.0"
    TOPK: int = 10
