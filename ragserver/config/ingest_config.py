from dataclasses import dataclass, field

from ragserver.config.settings import Settings


@dataclass(kw_only=True)
class IngestConfig:
    chunk_size: int = field(default=Settings.CHUNK_SIZE)
    chunk_overlap: int = field(default=Settings.CHUNK_OVERLAP)
    user_agent: str = field(default=Settings.USER_AGENT)
    upload_dir: str = field(default=Settings.UPLOAD_DIR)
