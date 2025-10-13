from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import BaseNode, ImageNode, TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import BasicMetaData
from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.logger import logger
from ragserver.structured_store.structured_store_manager import StructuredStoreManager


class VectorStoreManager(ABC):
    def __init__(self, check_update: bool = True) -> None:
        """ベクトルストア管理クラスの抽象

        空間キーごとにテーブルを一つ割り当て、ノードを管理する想定。

        Args:
            check_update (bool, optional): ファイルの更新チェック要否。Defaults to True.
        """
        logger.debug("trace")

        self._check_update = check_update

        self._text_store: Optional[BasePydanticVectorStore] = None
        self._image_store: Optional[BasePydanticVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        self._embed: Optional[EmbeddingManager] = None
        self._meta_store: Optional[StructuredStoreManager] = None

        # 各ストア（空間キー毎）の初期化時に同期し、以降は fingerprint チェックの度に追加
        self._fp_cache: dict[str, str] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """

    @property
    def index(self) -> Optional[VectorStoreIndex]:
        """ストレージから生成したインデックス。

        Returns:
            Optional[VectorStoreIndex]: インデックス
        """
        return self._index

    @abstractmethod
    def prepare_with(
        self, embed: EmbeddingManager, meta_store: StructuredStoreManager, limit: int
    ) -> None:
        """埋め込み管理に合わせてストアを初期化する。

        Args:
            embed (EmbeddingManager): 埋め込み管理
            meta_store (StructuredStoreManager): メタデータ管理
            limit (int): メタデータ読み込み件数上限

        Raises:
            RuntimeError: ストア初期化失敗
        """

    def _load_fp_cache(self, limit: int) -> dict[str, str]:
        logger.debug("trace")

        if self._meta_store is None:
            logger.warning("metadata store is not initialized")
            return {}

        rows = self._meta_store.select(
            cols=[MK.FILE_PATH, MK.URL, MK.FINGERPRINT], limit=limit
        )

        fp_cache = {}
        for row in rows:
            fp_cache[row[0] or row[1]] = row[2]

        return fp_cache

    def _split_nodes_modality(
        self,
        nodes: list[BaseNode],
    ) -> tuple[list[TextNode], list[ImageNode]]:
        """ノードをテキスト用と画像用に分ける。

        Args:
            nodes (list[BaseNode]): テキストノードまたは画像ノード

        Returns:
            tuple[list[TextNode], list[ImageNode]]: テキストノード、画像ノード
        """
        logger.debug("trace")

        text_nodes = []
        image_nodes = []
        for node in nodes:
            if isinstance(node, TextNode):
                text_nodes.append(node)
            elif isinstance(node, ImageNode):
                image_nodes.append(node)
            else:
                logger.warning(f"unexpected node type {type(node)}, skipped")

        return text_nodes, image_nodes

    async def _upsert_text(self, nodes: list[TextNode]) -> None:
        """テキストを埋め込み、ストアに格納する。

        Args:
            nodes (list[TextNode]): 対象ノード
        """
        logger.debug("trace")

        if len(nodes) == 0:
            logger.warning("empty list")
            return

        if self._text_store is None:
            logger.warning("text store is not initialized")
            return

        if self._embed is None:
            logger.warning("embedding is not initialized")
            return

        if self._meta_store is None:
            logger.warning("metadata store is not initialized")
            return

        texts = []
        ids = []
        metas = []
        fps = []
        for node in nodes:
            texts.append(node.text)
            ids.append(node.node_id)
            meta = BasicMetaData(node.metadata)
            metas.append(meta)
            fps.append(self._get_lazy_fp(meta))

        try:
            vecs = await self._embed.embed_text(texts)
            for node, vec in zip(nodes, vecs):
                node.embedding = vec

            await self._text_store.adelete_nodes(ids)
            await self._text_store.async_add(nodes)
            self._meta_store.upsert_text_metas(metas=metas, fingerprints=fps)
        except Exception as e:
            logger.exception(e)

        logger.info(f"{len(nodes)} nodes are upserted")

    async def _upsert_image(self, nodes: list[ImageNode]) -> None:
        """画像を埋め込み、ストアに格納する。

        Args:
            nodes (list[ImageNode]): 対象ノード
        """
        logger.debug("trace")

        if len(nodes) == 0:
            logger.warning("empty list")
            return

        if not isinstance(self._embed, MultiModalEmbeddingManager):
            logger.warning("multimodal embed model is required")
            return

        if self._image_store is None:
            logger.warning("image store is not initialized")
            return

        if self._meta_store is None:
            logger.warning("metadata store is not initialized")
            return

        file_paths = []
        temp_file_paths = []
        ids = []
        metas = []
        fps = []
        for node in nodes:
            meta = BasicMetaData(node.metadata)

            temp_file_path = meta.temp_file_path
            if temp_file_path and temp_file_path != "":
                # フェッチした一時ファイル
                file_paths.append(temp_file_path)
                temp_file_paths.append(temp_file_path)
                meta.temp_file_path = ""
            else:
                file_path = meta.file_path
                if file_path:
                    # ローカルファイル
                    file_paths.append(file_path)
                else:
                    logger.warning("image is not found, skipped")
                    continue

            ids.append(node.node_id)
            metas.append(meta)
            fps.append(self._get_lazy_fp(meta))

        try:
            vecs = await self._embed.embed_image(file_paths)
            for node, vec in zip(nodes, vecs):
                node.embedding = vec

            await self._image_store.adelete_nodes(ids)
            await self._image_store.async_add(nodes)
            self._meta_store.upsert_image_metas(metas=metas, fingerprints=fps)
        except Exception as e:
            logger.exception(e)
        finally:
            for path in temp_file_paths:
                os.remove(path)

        logger.info(f"{len(nodes)} nodes are upserted")

    def _create_index(
        self,
        text_store: BasePydanticVectorStore,
        image_store: Optional[BasePydanticVectorStore] = None,
    ) -> Optional[VectorStoreIndex]:
        """インデックスを作成する。

        Args:
            text_store (BasePydanticVectorStore): テキストストア
            image_store (Optional[BasePydanticVectorStore], optional): 画像ストア。Defaults to None.

        Raises:
            RuntimeError: テキスト埋め込み管理に対してマルチモーダルストアの要求

        Returns:
            Optional[VectorStoreIndex]: _description_
        """
        logger.debug("trace")

        if self._embed is None:
            logger.warning("embedding is not initialized")
            return

        if image_store:
            if not isinstance(self._embed, MultiModalEmbeddingManager):
                raise RuntimeError("multimodal embed model is required")

            return MultiModalVectorStoreIndex.from_vector_store(
                vector_store=text_store,
                embed_model=self._embed.embedding,
                image_vector_store=image_store,
                image_embed_model=self._embed.embedding_multi,
            )

        return VectorStoreIndex.from_vector_store(
            vector_store=text_store,
            embed_model=self._embed.embedding,
        )

    def _add_fp_cache(self, nodes: list[BaseNode]) -> None:
        """ノードを fingerprint キャッシュに追加する。

        Args:
            nodes (list[BaseNode]): 追加するノード
        """
        logger.debug("trace")

        for node in nodes:
            meta = BasicMetaData(node.metadata)
            source = meta.file_path or meta.url

            if source is None:
                logger.warning("no source info")
                continue

            # fingerprint キャッシュになければ追加（＝次回以降スキップ）
            if source not in self._fp_cache:
                self._fp_cache[source] = self._get_lazy_fp(meta)

    def _get_lazy_fp(self, meta: BasicMetaData) -> str:
        """fingerprint を取得する。

        ingest スキップ用。
        スキップできなくても再 ingest されるだけなので、厳密な fingerprint は取らない。

        Args:
            meta (BasicMetaData): メタデータの辞書

        Returns:
            str: fingerprint 文字列
        """
        logger.debug("trace")

        # Web ページの場合、現状 URL しかチェックしない
        fp_data = {
            MK.FILE_PATH: meta.file_path,
            MK.FILE_SIZE: meta.file_size,
            MK.FILE_LASTMOD_AT: meta.file_lastmod_at,
            MK.CHUNK_NO: meta.chunk_no,
            MK.URL: meta.url,
        }
        return hashlib.md5(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()

    def _filter_nodes_by_fp(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """fingerprint に基づき既存ノードを除外したリストを返す。

        Args:
            nodes (list[BaseNode]): 登録候補のノード

        Returns:
            list[BaseNode]: フィルター後のノード
        """
        logger.debug("trace")

        filtered: list[BaseNode] = []

        for node in nodes:
            meta = BasicMetaData(node.metadata)
            source = meta.file_path or meta.url

            if source is None:
                logger.warning("no source info")
                continue

            # fingerprint キャッシュになければ新規扱い
            if source not in self._fp_cache:
                filtered.append(node)
                continue

            existing_fp = self._fp_cache.get(source)

            # None が登録されていた場合（ないはず。型チェックの都合）
            if existing_fp is None:
                filtered.append(node)
                continue

            fp = self._get_lazy_fp(meta)
            if existing_fp == fp:
                logger.info(f"skip document: identical fingerprint for {source}")
                continue

            filtered.append(node)

        return filtered

    async def upsert_nodes(self, nodes: list[BaseNode]) -> None:
        """ノードを埋め込み、ストアに格納する。

        Args:
            nodes (list[BaseNode]): 対象ノード
        """
        logger.debug("trace")

        # fingerprint が既存・同一のノードは upsert しない
        nodes = self._filter_nodes_by_fp(nodes)
        if len(nodes) == 0:
            logger.info("skip upsert: all nodes already exist")
            return

        text_nodes, image_nodes = self._split_nodes_modality(nodes)
        await self._upsert_text(text_nodes)
        await self._upsert_image(image_nodes)

        # キャッシュ登録
        self._add_fp_cache(nodes)

    def skip_update(self, source: str) -> bool:
        """ソースが登録済みであり、更新処理が不要か。

        Args:
            source (str): 対象ソース

        Returns:
            bool: 更新処理不要の場合に True
        """
        logger.debug("trace")

        # check_update 指定がなく、かつソースが登録済み
        return (not self._check_update) and (self._fp_cache.get(source) is not None)
