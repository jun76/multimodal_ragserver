from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

__all__ = ["get_config"]

CHROMA_STORE_NAME = "chroma"
PGVECTOR_STORE_NAME = "pgvector"
HFCLIP_EMBED_NAME = "hfclip"
HF_RERANK_NAME = "hf"


@dataclass
class Config:
    # ragserver
    ragserver_base_url: str

    # vector store
    vector_store: str

    # Embeddings
    hfclip_embed_base_url: str

    # Retrieval/Rerank
    hf_rerank_base_url: str

    # LLM
    llm_provider: str  # local|openai
    llm_local_model: str
    llm_local_base_url: str
    llm_openai_model: str
    openai_api_key: str


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
        # Embeddings
        hfclip_embed_base_url=os.getenv(
            "HFCLIP_EMBED_BASE_URL", "http://localhost:8001/v1"
        ),
        # Retrieval/Rerank
        hf_rerank_base_url=os.getenv("HF_RERANK_BASE_URL", "http://localhost:8002/v1"),
        # LLM
        llm_provider=os.getenv("LLM_PROVIDER", "local"),
        llm_local_model=os.getenv("LLM_LOCAL_MODEL", "unsloth/gpt-oss-20b"),
        llm_local_base_url=os.getenv("LLM_LOCAL_BASE_URL", "http://localhost:1234/v1"),
        llm_openai_model=os.getenv("LLM_OPENAI_MODEL", "gpt-3.5-turbo"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    return cfg
