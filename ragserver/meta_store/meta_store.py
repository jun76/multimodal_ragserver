from __future__ import annotations

from ragserver.logger import logger
from ragserver.meta_store.structured.sqlite_structured import SQLiteStructured
from ragserver.meta_store.structured.structured import Structured

__all__ = ["create_meta_store"]


def create_meta_store() -> Structured:
    """メタデータ用ストアのインスタンスを生成する。

    Raises:
        RuntimeError: インスタンス生成に失敗

    Returns:
        Structured: メタデータ用ストア
    """
    logger.debug("trace")

    try:
        meta_store = SQLiteStructured()
    except Exception as e:
        raise RuntimeError(f"failed to prepare metadata store: {e}") from e

    return meta_store
