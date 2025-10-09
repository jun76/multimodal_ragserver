from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from ragserver.core.metadata import META_KEYS_FROM as MKF
from ragserver.core.metadata import BasicMetaData
from ragserver.ingest.loader import Loader
from ragserver.logger import logger


class FileLoader(Loader):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """ローカルファイルを読み込み、ノードを生成するためのクラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
        """
        logger.debug("trace")

        Loader.__init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def load_from_path(
        self,
        root: str,
    ) -> list[BaseNode]:
        """ローカルパス（ディレクトリ、ファイル）からコンテンツを取り込み、ノードを生成する。
        ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

        Args:
            root (str): 対象パス

        Returns:
            list[BaseNode]: テキストノードまたは画像ノード
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
                exclude=list(self._source_cache),
                recursive=True,
            )
            docs = await reader.aload_data(show_progress=True)

            splitter = SentenceSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                include_metadata=True,
            )

            # 最上位ループ内で複数ソースをまたいで _source_cache を共有したいため
            # ここでは _source_cache.clear() しないこと。
            all_nodes = []
            for doc in docs:
                nodes = splitter.get_nodes_from_documents([doc])

                for i, node in enumerate(nodes):
                    meta = node.metadata
                    file_path = meta.get(MKF.FILE_PATH) or ""
                    node.metadata = BasicMetaData(
                        file_path=file_path,
                        file_type=meta.get(MKF.FILE_TYPE) or "",
                        file_size=meta.get(MKF.FILE_SIZE) or "",
                        creation_date=meta.get(MKF.CREATION_DATE) or "",
                        last_modified_date=meta.get(MKF.LAST_MODIFIED_DATE) or "",
                        chunk_no=str(i),
                    ).to_dict()

                    # 取得済みキャッシュに追加
                    self._source_cache.add(file_path)

                all_nodes.extend(nodes)
        except Exception as e:
            logger.exception(e)
            return []

        logger.info(f"Ingested {len(all_nodes)} nodes from {root}")

        return all_nodes

    async def load_from_path_list(
        self,
        list_path: str,
    ) -> list[BaseNode]:
        """path リストに記載の複数パスからコンテンツを取得し、ノードを生成する。

        Args:
            list_path (str): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

        Returns:
            list[BaseNode]: テキストノードまたは画像ノード
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
