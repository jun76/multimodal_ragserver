from __future__ import annotations

import time
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.schema import BaseNode

from ragserver.core.exts import Exts
from ragserver.core.metadata import BasicMetaData
from ragserver.ingest.loader.loader import Loader
from ragserver.ingest.loader.reader.pdf_reader import MultiPDFReader
from ragserver.logger import logger
from ragserver.vector_store.vector_store_manager import VectorStoreManager


class FileLoader(Loader):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        store: VectorStoreManager,
    ) -> None:
        """ローカルファイルを読み込み、ノードを生成するためのクラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
            store (VectorStoreManager): 登録済みソースの判定に使用
        """
        logger.debug("trace")

        Loader.__init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._store = store

        # 独自 reader の辞書。後段で SimpleDirectoryReader に渡す
        self._readers: dict[str, BaseReader] = {Exts.PDF: MultiPDFReader()}

    async def _aload_from_file(
        self,
        path: str,
        reader: SimpleDirectoryReader,
        splitter: MetadataAwareTextSplitter,
    ) -> list[BaseNode]:
        """ファイルからノードを生成する。

        Args:
            path (str): ファイルパス
            reader (SimpleDirectoryReader): ファイルリーダ
            splitter (MetadataAwareTextSplitter): テキストスプリッタ

        Raises:
            RuntimeError: ノードの生成に失敗

        Returns:
            list[BaseNode]: 生成したノード
        """
        # logger.debug("trace")

        try:
            docs = await reader.aload_file(
                input_file=Path(path),
                file_metadata=reader.file_metadata,
                file_extractor=reader.file_extractor,
            )

            all_nodes = []
            for doc in docs:
                nodes = await splitter.aget_nodes_from_documents([doc])
                for i, node in enumerate(nodes):
                    meta = BasicMetaData().from_dict(node.metadata)
                    meta.chunk_no = i
                    meta.node_lastmod_at = time.time()
                    node.metadata = meta.to_dict()

                all_nodes.extend(nodes)
        except Exception as e:
            raise RuntimeError(f"failed to generate nodes from file: {path}") from e

        logger.info(f"loaded {len(all_nodes)} nodes from {path}")

        return all_nodes

    async def aload_from_path(
        self,
        root: str,
    ) -> list[BaseNode]:
        """ローカルパス（ディレクトリ、ファイル）からコンテンツを取り込み、ノードを生成する。
        ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

        Args:
            root (str): 対象パス

        Returns:
            list[BaseNode]: 生成したノード
        """
        logger.debug("trace")

        try:
            path = Path(root)
            reader = SimpleDirectoryReader(
                input_dir=root if path.is_dir() else None,
                input_files=[root] if path.is_file() else None,
                recursive=True,
                file_extractor=self._readers,
            )

            splitter = SentenceSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                include_metadata=True,
            )

            paths = reader.list_resources()
        except Exception as e:
            logger.exception(e)
            return []

        # 最上位ループ内で複数ソースをまたいで _source_cache を共有したいため
        # ここでは _source_cache.clear() しないこと。
        nodes = []
        for path in paths:
            try:
                if path in self._source_cache:
                    continue

                if self._store.skip_update(path):
                    logger.info(f"skip loading: source exists ({path})")
                    continue

                nodes.extend(
                    await self._aload_from_file(
                        path=path, reader=reader, splitter=splitter
                    )
                )

                # 取得済みキャッシュに追加
                self._source_cache.add(path)
            except Exception as e:
                logger.exception(e)
                continue

        logger.info(f"loaded {len(nodes)} nodes from {root}")

        return nodes

    async def aload_from_path_list(
        self,
        list_path: str,
    ) -> list[BaseNode]:
        """path リストに記載の複数パスからコンテンツを取得し、ノードを生成する。

        Args:
            list_path (str): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

        Returns:
            list[BaseNode]: 生成したノード
        """
        logger.debug("trace")

        paths = self._read_sources_from_file(list_path)

        # 最上位ループ。キャッシュを空にしてから使う。
        self._source_cache.clear()
        nodes = []
        for path in paths:
            try:
                temp = await self.aload_from_path(path)
                nodes.extend(temp)
            except Exception as e:
                logger.exception(e)
                continue

        return nodes
