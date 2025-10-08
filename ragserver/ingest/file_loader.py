from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from starlette.concurrency import run_in_threadpool

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
        """ローカルファイルを読み込み、テキストノードを生成するためのクラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
        """
        logger.debug("trace")

        Loader.__init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def load_from_path(
        self,
        root: str,
    ) -> list[TextNode]:
        """ローカルパス（ディレクトリ、ファイル）からコンテンツを取り込み、テキストノードを生成する。
        ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

        Args:
            root (str): 対象パス

        Returns:
            list[TextNode]: テキストノード（画像パス含む）
        """
        logger.debug("trace")

        try:
            path = Path(root)
            reader = SimpleDirectoryReader(
                input_dir=root if path.is_dir() else None,
                input_files=[root] if path.is_file() else None,
                recursive=True,
            )
            docs = await run_in_threadpool(reader.load_data)

            # ここで取れるのはテキストノードのみ。後段で画像をフェッチする際に
            # 画像ノードに分化
            splitter = SentenceSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                include_metadata=True,
            )
            nodes = splitter.get_nodes_from_documents(docs)

            text_nodes = []
            for i, node in enumerate(nodes):
                meta = node.metadata
                file_path = meta.get(MKF.FILE_PATH) or ""

                text_node = TextNode(
                    text=node.get_content(),
                    metadata=BasicMetaData(
                        file_path=file_path,
                        file_type=meta.get(MKF.FILE_TYPE) or "",
                        file_size=meta.get(MKF.FILE_SIZE) or "",
                        creation_date=meta.get(MKF.CREATION_DATE) or "",
                        last_modified_date=meta.get(MKF.LAST_MODIFIED_DATE) or "",
                        ref_doc_id=file_path,
                        chunk_no=str(i),
                    ).to_dict(),
                )
                text_nodes.append(text_node)
        except Exception as e:
            logger.exception(e)
            return []

        logger.info(f"Ingested {len(text_nodes)} text nodes from {root}")

        return text_nodes

    async def load_from_path_list(
        self,
        list_path: str,
    ) -> list[TextNode]:
        """path リストに記載の複数パスからコンテンツを取得し、テキストノードを生成する。

        Args:
            list_path (str): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

        Returns:
            list[TextNode]: テキストノード（画像パス含む）
        """
        logger.debug("trace")

        paths = self._read_sources_from_file(list_path)

        docs = []
        for path in paths:
            try:
                temp, _ = await self.load_from_path(path)
                docs.extend(temp)
            except Exception as e:
                logger.exception(e)
                continue

        return docs
