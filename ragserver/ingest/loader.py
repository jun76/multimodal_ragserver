from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Set

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        self._source_cache: Set[str] = set()

    def _required_ext(self, uri: str, exts: set) -> bool:
        """URI に指定の拡張子が含まれるか。

        Args:
            uri (str): ファイルパス、URL
            exts (set): 拡張子セット

        Returns:
            bool: 含まれる場合 True
        """
        # logger.debug("trace")

        # suffix を比較（大小無視）
        suffix = Path(uri).suffix.lower()

        return suffix in {e.lower() for e in exts}

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

    def _split_text(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Document の page_content を複数に分割し、metadata も複製する。

        Args:
            documents (list[Document]): 分割対象ドキュメント

        Returns:
            list[Document]: 分割後のドキュメント
        """
        logger.debug("trace")

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError("failed to split documents") from e
