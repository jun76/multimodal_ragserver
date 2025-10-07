from __future__ import annotations

from dataclasses import dataclass

from ragserver.logger import logger


@dataclass
class Exts:
    TEXT_FILE_EXTS = {".txt"}
    MARKDOWN_FILE_EXTS = {".md"}
    IMAGE_FILE_EXTS = {".jpg", ".jpeg", ".png", ".gif"}
    PDF_FILE_EXTS = {".pdf"}
    SUPPORTED_EXTS = (
        TEXT_FILE_EXTS | MARKDOWN_FILE_EXTS | IMAGE_FILE_EXTS | PDF_FILE_EXTS
    )


class Loader:
    def __init__(self) -> None:
        """ローダー基底クラス。"""
        logger.debug("trace")

    def _read_sources_from_file(self, path: str) -> list[str]:
        """空行・コメントを除外して source リストを読み込む。

        Args:
            path: source 列挙ファイルのパス

        Returns:
            list[str]: source のリスト

        Raises:
            RuntimeError: ソースリストの読み込みに失敗した場合
        """
        logger.debug("trace")

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return [
                    ln.strip()
                    for ln in f
                    if ln.strip() and not ln.strip().startswith("#")
                ]
        except OSError as e:
            raise RuntimeError("failed to read source list") from e
