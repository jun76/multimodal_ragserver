from __future__ import annotations

from dataclasses import dataclass

from .settings import Settings


@dataclass(kw_only=True, frozen=True)
class IngestConfig:
    """ドキュメント取り込み処理関連の設定用データクラス"""

    chunk_size: int = Settings.CHUNK_SIZE
    chunk_overlap: int = Settings.CHUNK_OVERLAP
    user_agent: str = Settings.USER_AGENT
    upload_dir: str = Settings.UPLOAD_DIR
