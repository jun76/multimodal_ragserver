from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import SecretStr

from ragserver.config.settings import Settings


@dataclass(kw_only=True, frozen=True)
class VectorStoreConfig:
    """ベクトルストア関連の設定用データクラス"""

    load_limit: int = Settings.LOAD_LIMIT
    check_update: bool = Settings.CHECK_UPDATE
    chroma_persist_dir: str = Settings.CHROMA_PERSIST_DIR
    chroma_host: Optional[str] = Settings.CHROMA_HOST
    chroma_port: Optional[int] = Settings.CHROMA_PORT
    chroma_api_key: Optional[SecretStr] = Settings.CHROMA_API_KEY
    chroma_tenant: Optional[str] = Settings.CHROMA_TENANT
    chroma_database: Optional[str] = Settings.CHROMA_DATABASE
    pgvector_host: str = Settings.PGVECTOR_HOST
    pgvector_port: int = Settings.PGVECTOR_PORT
    pgvector_database: str = Settings.PGVECTOR_DATABASE
    pgvector_user: str = Settings.PGVECTOR_USER
    pgvector_password: Optional[SecretStr] = Settings.PGVECTOR_PASSWORD
