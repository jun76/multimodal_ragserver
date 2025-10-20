from __future__ import annotations

from ..logger import logger
from .structured.sqlite_structured import SQLiteStructured
from .structured.structured import Structured

__all__ = ["create_meta_store"]


def create_meta_store() -> Structured:
    """メタデータ用ストアのインスタンスを生成する。

    Raises:
        RuntimeError: インスタンス生成に失敗

    Returns:
        Structured: メタデータ用ストア
    """

    try:
        meta_store = SQLiteStructured()
    except Exception as e:
        raise RuntimeError(f"failed to prepare metadata store: {e}") from e

    return meta_store
