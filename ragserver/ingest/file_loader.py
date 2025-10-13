from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from ragserver.core.metadata import META_KEYS_FROM as MKF
from ragserver.core.metadata import BasicMetaData
from ragserver.ingest.loader import Loader
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

    async def load_from_path(
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
            # TODO
            # 例えば pdf ファイル内の埋め込み画像はデフォルトの PDFReader では
            # 抽出してくれない（OCR の流れはあるみたい）ので、抽出する場合は
            # Reader を自作して file_extractor に渡す必要がある
            path = Path(root)
            reader = SimpleDirectoryReader(
                input_dir=root if path.is_dir() else None,
                input_files=[root] if path.is_file() else None,
                recursive=True,
            )

            # 最上位ループ内で複数ソースをまたいで _source_cache を共有したいため
            # ここでは _source_cache.clear() しないこと。
            paths = reader.list_resources()
            docs = []
            for path in paths:
                if path in self._source_cache:
                    continue

                if self._store.skip_update(path):
                    logger.info(f"skip loading: source exists ({path})")
                    continue

                docs.extend(
                    await reader.aload_file(
                        input_file=Path(path),
                        file_metadata=reader.file_metadata,
                        file_extractor=reader.file_extractor,
                    )
                )
                # 取得済みキャッシュに追加
                self._source_cache.add(path)

            splitter = SentenceSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                include_metadata=True,
            )
        except Exception as e:
            logger.exception(e)
            return []

        all_nodes = []
        for doc in docs:
            try:
                nodes = splitter.get_nodes_from_documents([doc])

                for i, node in enumerate(nodes):
                    meta = BasicMetaData(node.metadata)
                    meta.chunk_no = i
                    meta.node_lastmod_at = time.time()
                    node.metadata = meta.to_dict()

                all_nodes.extend(nodes)
            except Exception as e:
                logger.exception(e)
                continue

        logger.info(f"loaded {len(all_nodes)} nodes from {root}")

        return all_nodes

    async def load_from_path_list(
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
                temp = await self.load_from_path(path)
                nodes.extend(temp)
            except Exception as e:
                logger.exception(e)
                continue

        return nodes
