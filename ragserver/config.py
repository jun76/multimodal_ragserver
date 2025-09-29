from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from ragserver.core.names import (
    CHROMA_STORE_NAME,
    COHERE_EMBED_NAME,
    COHERE_RERANK_NAME,
    LOCAL_EMBED_NAME,
    LOCAL_RERANK_NAME,
    OPENAI_EMBED_NAME,
    PGVECTOR_STORE_NAME,
    PROJECT_NAME,
)
from ragserver.logger import logger

__all__ = ["get_config"]


@dataclass
class Config:
    # vector store
    vector_store: str
    load_limit: int
    check_update: bool

    # Chroma
    chroma_persist_dir: str
    chroma_host: str | None
    chroma_port: int | None
    chroma_api_key: str | None
    chroma_tenant: str | None
    chroma_database: str | None

    # PgVector
    pg_host: str
    pg_port: int
    pg_database: str
    pg_user: str
    pg_password: str

    # Embeddings
    emped_provider: str  # local|openai|cohere
    openai_embed_model_text: str
    openai_api_key: str
    openai_base_url: str | None
    cohere_embed_model_text: str
    cohere_embed_model_image: str
    cohere_api_key: str | None
    local_embed_model_text: str
    local_embed_model_image: str
    local_embed_base_url: str

    # Ingestion
    chunk_size: int
    chunk_overlap: int
    user_agent: str

    # Retrieval/Rerank
    rerank_provider: str  # local|cohere|none
    local_rerank_model: str
    local_rerank_base_url: str
    cohere_rerank_model: str
    topk: int
    topk_rerank_scale: int
    upload_dir: str


def _to_int_none(key: str) -> Optional[int]:
    """int 型の環境変数に None (未指定) を許すためのフィルタ。

    Args:
        key (str): 環境変数名

    Raises:
        ValueError: int 値以外が設定されている

    Returns:
        Optional[int]: int 値または None
    """
    # logger.debug("trace")

    s = os.getenv(key)

    if s is None or s == "":
        return None
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(f"{key} must be an integer, got: {s!r}") from e


def _to_int(key: str, default: int) -> int:
    """int 型の環境変数に None (未指定) を許さないためのフィルタ。
    ユーザによって空文字列が指定された場合を想定。デフォルト値にフォールバックする。

    Args:
        key (str): 環境変数名
        default (int): デフォルト値

    Raises:
        ValueError: int 値以外が設定されている

    Returns:
        int: int 値
    """
    # logger.debug("trace")

    s = os.getenv(key, default)

    if s is None or s == "":
        return default
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(f"{key} must be an integer, got: {s!r}") from e


def _to_bool(key: str, default: bool) -> bool:
    """bool 型の環境変数を解釈し、未指定時はデフォルト値を返す。

    Args:
        key (str): 環境変数名
        default (bool): デフォルト値

    Raises:
        ValueError: 真偽値として解釈できない文字列が指定された場合

    Returns:
        bool: 解釈した真偽値
    """
    logger.debug("trace")

    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default

    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{key} must be a boolean string")


def _validate_config(cfg: Config) -> Config:
    """生成した設定オブジェクトを検証する。

    Args:
        cfg (Config): 検証対象

    Raises:
        ValueError: 設定値がサポート外または不正な場合

    Returns:
        Config: 検証済み設定
    """
    logger.debug("trace")

    allowed_stores = {CHROMA_STORE_NAME, PGVECTOR_STORE_NAME}
    if cfg.vector_store not in allowed_stores:
        raise ValueError("vector_store must be chroma or pgvector")

    allowed_embeds = {LOCAL_EMBED_NAME, OPENAI_EMBED_NAME, COHERE_EMBED_NAME}
    if cfg.emped_provider not in allowed_embeds:
        raise ValueError("unsupported embed provider")

    allowed_rerank = {LOCAL_RERANK_NAME, COHERE_RERANK_NAME, "none"}
    if cfg.rerank_provider not in allowed_rerank:
        raise ValueError("unsupported rerank provider")

    if cfg.load_limit <= 0:
        raise ValueError("load_limit must be positive")

    if cfg.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if cfg.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive")

    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if cfg.topk <= 0:
        raise ValueError("topk must be positive")

    if cfg.topk_rerank_scale <= 0:
        raise ValueError("topk_rerank_scale must be positive")

    if cfg.vector_store == PGVECTOR_STORE_NAME:
        for key, value in {
            "pg_host": cfg.pg_host,
            "pg_database": cfg.pg_database,
            "pg_user": cfg.pg_user,
            "pg_password": cfg.pg_password,
        }.items():
            if not value:
                raise ValueError(f"{key} must not be empty")

    if cfg.emped_provider == LOCAL_EMBED_NAME and not cfg.local_embed_base_url:
        raise ValueError("local_embed_base_url must not be empty")

    if cfg.rerank_provider == LOCAL_RERANK_NAME and not cfg.local_rerank_base_url:
        raise ValueError("local_rerank_base_url must not be empty")

    if cfg.user_agent.strip() == "":
        raise ValueError("user_agent must not be empty")

    return cfg


def get_config() -> Config:
    """呼び出し時点の環境変数を読み取り、設定オブジェクトを生成する。

    Returns:
        Config: 設定オブジェクト

    Raises:
        RuntimeError: 環境変数の読み込みに失敗した場合
        ValueError: 設定値の検証に失敗した場合
    """
    logger.debug("trace")

    try:
        load_dotenv(override=True)
    except Exception as e:
        raise RuntimeError("failed to load environment variables") from e

    cfg = Config(
        # vector store
        vector_store=os.getenv("VECTOR_STORE", CHROMA_STORE_NAME),
        load_limit=_to_int("LOAD_LIMIT", 10000),
        check_update=_to_bool("CHECK_UPDATE", False),
        # Chroma
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
        chroma_host=os.getenv("CHROMA_HOST"),
        chroma_port=_to_int_none("CHROMA_PORT"),
        chroma_api_key=os.getenv("CHROMA_API_KEY"),
        chroma_tenant=os.getenv("CHROMA_TENANT"),
        chroma_database=os.getenv("CHROMA_DATABASE"),
        # PgVector
        pg_host=os.getenv("PG_HOST", "localhost"),
        pg_port=_to_int("PG_PORT", 5432),
        pg_database=os.getenv("PG_DATABASE", PROJECT_NAME),
        pg_user=os.getenv("PG_USER", PROJECT_NAME),
        pg_password=os.getenv("PG_PASSWORD", PROJECT_NAME),
        # Embeddings
        emped_provider=os.getenv("EMBED_PROVIDER", LOCAL_EMBED_NAME),
        openai_embed_model_text=os.getenv(
            "OPENAI_EMBED_MODEL_TEXT", "text-embedding-3-small"
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        cohere_embed_model_text=os.getenv("COHERE_EMBED_MODEL_TEXT", "embed-v4.0"),
        cohere_embed_model_image=os.getenv("COHERE_EMBED_MODEL_IMAGE", "embed-v4.0"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        local_embed_model_text=os.getenv(
            "LOCAL_EMBED_MODEL_TEXT", "openai/clip-vit-base-patch32"
        ),
        local_embed_model_image=os.getenv(
            "LOCAL_EMBED_MODEL_IMAGE", "openai/clip-vit-base-patch32"
        ),
        local_embed_base_url=os.getenv(
            "LOCAL_EMBED_BASE_URL", "http://localhost:8001/v1"
        ),
        # Ingestion
        chunk_size=_to_int("CHUNK_SIZE", 500),
        chunk_overlap=_to_int("CHUNK_OVERLAP", 50),
        user_agent=os.getenv("USER_AGENT", PROJECT_NAME),
        # Retrieval/Rerank
        rerank_provider=os.getenv("RERANK_PROVIDER", LOCAL_RERANK_NAME),
        local_rerank_model=os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        local_rerank_base_url=os.getenv(
            "LOCAL_RERANK_BASE_URL", "http://localhost:8002/v1"
        ),
        cohere_rerank_model=os.getenv(
            "COHERE_RERANK_MODEL", "rerank-multilingual-v3.0"
        ),
        topk=_to_int("TOPK", 10),
        topk_rerank_scale=_to_int("TOPK_RERANK_SCALE", 5),
        upload_dir=os.getenv("UPLOAD_DIR", "upload"),
    )

    return _validate_config(cfg)
