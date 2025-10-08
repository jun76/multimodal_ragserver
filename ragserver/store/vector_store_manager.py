from __future__ import annotations

import os
from abc import ABC
from typing import Optional

from llama_index.core import StorageContext
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode, TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from ragserver.core.metadata import META_KEYS as MK
from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.ingest.loader import Exts
from ragserver.logger import logger


class VectorStoreManager(ABC):
    def __init__(self, embed: EmbeddingManager, check_update: bool = True) -> None:
        """ベクトルストア管理クラスの抽象クラス

        Args:
            embed (EmbeddingManager): 埋め込み管理
            check_update (bool, optional): ファイルの更新チェック要否。 Defaults to True.
        """
        logger.debug("trace")

        self._embed = embed
        self._check_update = check_update

        self._text_store: Optional[BasePydanticVectorStore] = None
        self._image_store: Optional[BasePydanticVectorStore] = None
        self._index: Optional[MultiModalVectorStoreIndex] = None

    def _split_nodes_modality(
        self, nodes: list[TextNode]
    ) -> tuple[list[TextNode], list[ImageNode]]:
        """ノードをテキスト用と画像用に分ける。

        Args:
            nodes (list[TextNode]): テキストノード（画像パス、URL 含む）

        Returns:
            tuple[list[TextNode], list[ImageNode]]: テキストノード、画像ノード
        """
        logger.debug("trace")

        text_nodes = []
        image_nodes = []
        for node in nodes:
            if self._has_image_source(node):
                image_nodes.append(ImageNode(text=node.text, metadata=node.metadata))
            else:
                text_nodes.append(node)

        return text_nodes, image_nodes

    def _has_image_source(self, node: TextNode) -> bool:
        """ノードが画像のファイルパスや URL を持っているか。

        Args:
            node (TextNode): 対象ノード

        Returns:
            bool: 持っていれば True
        """
        logger.debug("trace")

        meta = node.metadata
        path = (
            meta.get(MK.FILE_PATH)
            or meta.get(MK.TEMP_FILE_PATH)
            or meta.get(MK.URL)
            or ""
        ).lower()

        return any(path.endswith(ext) for ext in Exts.IMAGE_FILE_EXTS)

    async def upsert_text(self, nodes: list[TextNode]) -> None:
        """テキストを埋め込み、ストアに格納する。

        Args:
            nodes (list[TextNode]): 対象ノード
        """
        logger.debug("trace")

        if self._text_store is None:
            logger.warning("text store is not initialized")
            return

        nodes = []
        vecs = []
        for node in nodes:
            vec = await self._embed.embed_text(node.text)
            nodes.append(node)
            vecs.append(vec)

        self._text_store.add(
            nodes=nodes,
            embeddings=vecs,
        )

    async def upsert_image(self, nodes: list[ImageNode]) -> None:
        """画像を埋め込み、ストアに格納する。

        Args:
            nodes (list[ImageNode]): 対象ノード
        """
        logger.debug("trace")

        if not isinstance(self._embed, MultiModalEmbeddingManager):
            logger.warning("multimodal embed model is required")
            return

        if self._image_store is None:
            logger.warning("image store is not initialized")
            return

        nodes = []
        vecs = []
        for node in nodes:
            meta = node.metadata

            file_path = meta.get(MK.FILE_PATH)
            if file_path:
                # ローカルファイル
                vec = await self._embed.embed_image(file_path)
            else:
                temp_file_path = meta.get(MK.TEMP_FILE_PATH)
                if temp_file_path:
                    # フェッチした一時ファイル
                    vec = await self._embed.embed_image(temp_file_path)
                    os.remove(temp_file_path)
                else:
                    logger.warning("image is not found, skipped")
                    continue

            nodes.append(node)
            vecs.append(vec)

        self._image_store.add(
            nodes=nodes,
            embeddings=vecs,
        )

    def create_index(self) -> MultiModalVectorStoreIndex:
        """インデックスを作成する。

        Raises:
            RuntimeError: ストア未初期化

        Returns:
            MultiModalVectorStoreIndex: インデックス
        """
        logger.debug("trace")

        if self._text_store is None or self._image_store is None:
            raise RuntimeError("text/image stores are not initialized")

        storage_context = StorageContext.from_defaults(
            vector_store=self._text_store,
            image_store=self._image_store,
        )

        return MultiModalVectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
        )

    async def upsert_index(self, nodes: list[TextNode] = []) -> None:
        """インデックスを追加する。

        Args:
            nodes (list[TextNode], optional): ノードのリスト。 Defaults to [].
        """
        logger.debug("trace")

        if self._index is None:
            self._index = self.create_index()

        await self._index.ainsert_nodes(nodes)

    def skip_update(self, source: str) -> bool:
        """ソースが登録済みであり、更新処理が不要か。

        Args:
            source (str): 対象ソース

        Returns:
            bool: 更新処理不要の場合に True
        """
        logger.debug("trace")

        # TODO: 更新チェック
        # check_update 指定がなく、かつソースが登録済み
        # return (not self._check_update) and (
        #     self._fingerprint_cache.get(source) is not None
        # )
        return False
