from __future__ import annotations

from abc import ABC
from typing import Optional

from llama_index.core import StorageContext
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import Document, ImageNode, TextNode
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

    def _split_docs_modality(
        self, docs: list[Document]
    ) -> tuple[list[Document], list[Document]]:
        """ドキュメントをテキスト用と画像用に分ける。

        Args:
            docs (list[Document]): ドキュメント（テキスト、画像混在）

        Returns:
            tuple[list[Document], list[Document]]: テキスト用、画像用ドキュメント
        """
        logger.debug("trace")

        text_docs = []
        image_docs = []
        for doc in docs:
            if self._is_image_doc(doc):
                image_docs.append(doc)
            else:
                text_docs.append(doc)

        return text_docs, image_docs

    def _is_image_doc(self, doc: Document) -> bool:
        """ドキュメントは画像用か。

        Args:
            doc (Document): 対象ドキュメント

        Returns:
            bool: 画像用なら True
        """
        logger.debug("trace")

        # TODO: メタ整理
        path = (doc.metadata.get("file_path") or doc.metadata.get("url") or "").lower()

        return any(path.endswith(ext) for ext in Exts.IMAGE_FILE_EXTS)

    async def _upsert_text(self, docs: list[Document]) -> None:
        """テキストを埋め込み、ストアに格納する。

        Args:
            docs (list[Document]): 対象ドキュメント
        """
        logger.debug("trace")

        if self._text_store is None:
            logger.warning("text store is not initialized")
            return

        # TODO: メタ整理
        nodes = []
        vecs = []
        for doc in docs:
            node = TextNode(content=doc.text, metadata=doc.metadata)
            vec = await self._embed.embed_text(text=doc.text)

            nodes.append(node)
            vecs.append(vec)

        self._text_store.add(
            nodes=nodes,
            embeddings=vecs,
        )

    async def _upsert_image(self, docs: list[Document]) -> None:
        """画像を埋め込み、ストアに格納する。

        Args:
            docs (list[Document]): 対象ドキュメント
        """
        logger.debug("trace")

        if not isinstance(self._embed, MultiModalEmbeddingManager):
            logger.warning("multimodal embed model is required")
            return

        if self._image_store is None:
            logger.warning("image store is not initialized")
            return

        # TODO: メタ整理
        nodes = []
        vecs = []
        for doc in docs:
            # 画像のパス or 一時保存したローカルパスを作る
            path = doc.metadata.get("file_path")
            if not path and doc.metadata.get("url"):
                path = download_to_tmp(doc.text)

            if path is None:
                continue

            node = ImageNode(content=doc.text, metadata=doc.metadata)
            vec = await self._embed.embed_image(path)

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

    async def upsert_index(self, docs: list[Document] = []) -> None:
        """インデックスを追加する。

        Args:
            docs (list[Document], optional): ドキュメントのリスト。 Defaults to [].
        """
        logger.debug("trace")

        if self._index is None:
            self._index = self.create_index()

        for doc in docs:
            await self._index.ainsert(document=doc)

    async def upsert_docs(self, docs: list[Document]) -> None:
        """ドキュメント（テキスト、画像混在）を埋め込み、ストアに格納する。

        Args:
            docs (list[Document]): 対象ドキュメント
        """
        logger.debug("trace")

        text_docs, image_docs = self._split_docs_modality(docs)
        await self._upsert_text(text_docs)
        await self._upsert_image(image_docs)

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
