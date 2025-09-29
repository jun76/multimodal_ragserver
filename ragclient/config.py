from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

__all__ = ["get_config"]

CHROMA_STORE_NAME = "chroma"
PGVECTOR_STORE_NAME = "pgvector"
LOCAL_EMBED_NAME = "local"
LOCAL_RERANK_NAME = "local"


@dataclass
class Config:
    # ragserver
    ragserver_base_url: str

    # vector store
    vector_store: str
    check_update: bool

    # Embeddings
    embed_provider: str  # local|openai|cohere
    local_embed_base_url: str

    # Retrieval/Rerank
    rerank_provider: str  # local|cohere|none
    local_rerank_base_url: str


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
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default

    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{key} must be a boolean string")


def get_config() -> Config:
    """呼び出し時点の環境変数を読み取り、設定オブジェクトを生成する。

    Returns:
        Config: 設定オブジェクト

    Raises:
        RuntimeError: 環境変数の読み込みに失敗した場合
        ValueError: 設定値の検証に失敗した場合
    """
    try:
        load_dotenv(override=True)
    except Exception as e:
        raise RuntimeError("failed to load environment variables") from e

    cfg = Config(
        # ragserver
        ragserver_base_url=os.getenv("RAGSERVER_BASE_URL", "http://localhost:8000/v1"),
        # vector store
        vector_store=os.getenv("VECTOR_STORE", CHROMA_STORE_NAME),
        check_update=_to_bool("CHECK_UPDATE", False),
        # Embeddings
        embed_provider=os.getenv("EMBED_PROVIDER", LOCAL_EMBED_NAME),
        local_embed_base_url=os.getenv(
            "LOCAL_EMBED_BASE_URL", "http://localhost:8001/v1"
        ),
        # Retrieval/Rerank
        rerank_provider=os.getenv("RERANK_PROVIDER", LOCAL_RERANK_NAME),
        local_rerank_base_url=os.getenv(
            "LOCAL_RERANK_BASE_URL", "http://localhost:8002/v1"
        ),
    )

    return cfg
