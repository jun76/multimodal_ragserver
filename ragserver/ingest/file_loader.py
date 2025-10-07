from __future__ import annotations

from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from starlette.concurrency import run_in_threadpool

from ragserver.ingest.loader import Loader
from ragserver.logger import logger


class FileLoader(Loader):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """ローカルファイルを読み込み、ドキュメントを生成するためのクラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
        """
        logger.debug("trace")

        Loader.__init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def load_from_path(
        self,
        root: str,
    ) -> list[Document]:
        """ローカルパス（ディレクトリ、ファイル）からコンテンツを取り込み、ドキュメントを生成する。
        ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

        Args:
            root (str): 対象パス

        Returns:
            list[Document]: ドキュメント（テキスト、画像混在）
        """
        logger.debug("trace")

        # テキスト分割の設定（グローバル）
        Settings.text_splitter = SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        # TODO: メタ整理
        try:
            path = Path(root)
            reader = SimpleDirectoryReader(
                input_dir=str(path) if path.is_dir() else None,
                input_files=[str(path)] if path.is_file() else None,
                recursive=True,
            )

            docs = await run_in_threadpool(reader.load_data)
            logger.info(f"Ingested {len(docs)} docs from {path}")
        except Exception as e:
            logger.exception(e)

        return docs

    async def load_from_path_list(
        self,
        list_path: str,
    ) -> list[Document]:
        """path リストに記載の複数パスからコンテンツを取得し、ドキュメントを生成する。

        Args:
            list_path (str): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

        Returns:
            list[Document]: ドキュメント（テキスト、画像混在）
        """
        logger.debug("trace")

        paths = self._read_sources_from_file(list_path)

        docs = []
        for path in paths:
            try:
                temp, _ = await self.load_from_path(root=path)
                docs.extend(temp)
            except Exception as e:
                logger.exception(e)
                continue

        return docs
