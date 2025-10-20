from __future__ import annotations

import os
from enum import StrEnum, auto
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


class VectorStoreProvider(StrEnum):
    CHROMA = auto()
    PGVECTOR = auto()


class LLMProvider(StrEnum):
    OPENAI = auto()
    LOCAL = auto()


class Settings:
    """各種設定値の指定用クラス

    以下の値を設定ファイルのように書き換えて使用する想定。
    API キーやパスワード等は予め .env ファイルに記述しておく。
    """

    RAGSERVER_BASE_URL: str = "http://localhost:8000/v1"
    VECTOR_STORE_PROVIDER: VectorStoreProvider = VectorStoreProvider.CHROMA
    LLM_PROVIDER: LLMProvider = LLMProvider.OPENAI
    LLM_LOCAL_BASE_URL: str = "http://localhost:1234/v1"
    LLM_LOCAL_MODEL: str = "unsloth/gpt-oss-20b"
    LLM_OPENAI_MODEL: str = "gpt-4-turbo"
    OPENAI_API_KEY: Optional[SecretStr] = (
        SecretStr(os.getenv("OPENAI_API_KEY", "")) or None
    )
    DEVICE: Literal["cpu", "cuda", "mps"] = "cuda"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
