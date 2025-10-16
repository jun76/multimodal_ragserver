from dataclasses import dataclass, field
from typing import Optional

from ragserver.config.settings import Settings


@dataclass(kw_only=True)
class VectorStoreConfig:
    load_limit: int = field(default=Settings.LOAD_LIMIT)
    check_update: bool = field(default=Settings.CHECK_UPDATE)
    chroma_persist_dir: str = field(default=Settings.CHROMA_PERSIST_DIR)
    chroma_host: Optional[str] = field(default=Settings.CHROMA_HOST)
    chroma_port: Optional[int] = field(default=Settings.CHROMA_PORT)
    chroma_api_key: Optional[str] = field(default=Settings.CHROMA_API_KEY)
    chroma_tenant: Optional[str] = field(default=Settings.CHROMA_TENANT)
    chroma_database: Optional[str] = field(default=Settings.CHROMA_DATABASE)
    pgvector_host: str = field(default=Settings.PGVECTOR_HOST)
    pgvector_port: int = field(default=Settings.PGVECTOR_PORT)
    pgvector_database: str = field(default=Settings.PGVECTOR_DATABASE)
    pgvector_user: str = field(default=Settings.PGVECTOR_USER)
    pgvector_password: str = field(default=Settings.PGVECTOR_PASSWORD)
