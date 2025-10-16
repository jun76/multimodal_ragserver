from __future__ import annotations

from ragserver.logger import logger


class Loader:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """ローダー基底クラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
        """
        logger.debug("trace")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # 最上位の load_from_*_list() がループを回している間は一度もストアに書き出されないので
        # 同一ソースに対して何度もフェッチがかかる場合がある。それを避けるため、
        # Loader クラス内にも独自のキャッシュを持つ。
        self._source_cache: set[str] = set()

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
            with open(path, "r", encoding="utf-8") as f:
                return [
                    stripped
                    for ln in f
                    if (stripped := ln.strip()) and not stripped.startswith("#")
                ]
        except OSError as e:
            raise RuntimeError(f"failed to read source list from {path}") from e
