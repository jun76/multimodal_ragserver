from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import BaseNode, ImageNode, TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from ragserver.core.exts import Exts
from ragserver.core.metadata import META_KEYS
from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import BasicMetaData
from ragserver.embed.embedding_manager import EmbeddingManager, Modality
from ragserver.logger import logger
from ragserver.structured_store.structured_store_manager import StructuredStoreManager


@dataclass
class VectorStoreContainer:
    """モダリティ毎のベクトルストア関連パラメータを集約"""

    modality: Modality
    provider_name: str
    store: BasePydanticVectorStore


class VectorStoreManager:
    def __init__(
        self,
        stores: list[VectorStoreContainer],
        embed: EmbeddingManager,
        meta_store: StructuredStoreManager,
        load_limit: int,
        check_update: bool = True,
    ) -> None:
        """ベクトルストア管理クラスの抽象

        空間キーごとにテーブルを一つ割り当て、ノードを管理する想定。

        Args:
            stores (list[VectorStoreContainer]): ベクトルストアコンテナのリスト
            embed (EmbeddingManager): 埋め込み管理
            meta_store (StructuredStoreManager): メタデータ管理
            load_limit (int): キャッシュロード件数上限
            check_update (bool, optional): ファイルの更新チェック要否。Defaults to True.
        """
        logger.debug("trace")

        self._vector_store_text: Optional[VectorStoreContainer] = None
        self._vector_store_image: Optional[VectorStoreContainer] = None
        self._modality: set[Modality] = set()

        for store in stores:
            match store.modality:
                case Modality.TEXT:
                    self._vector_store_text = store
                case Modality.IMAGE:
                    self._vector_store_image = store
                case _:
                    raise ValueError(f"unexpected modality: {store.modality}")

            self._modality.add(store.modality)

        self._embed = embed
        self._check_update = check_update
        self._meta_store = meta_store
        self._index = self._create_index()

        # メタデータ専用ストアから fingerprint キャッシュを復元
        self._fp_cache = self._load_fp_cache(load_limit)

    @property
    def index(self) -> VectorStoreIndex:
        """ストレージから生成したインデックス。

        Returns:
            VectorStoreIndex: インデックス
        """
        return self._index

    @property
    def modality(self) -> set[Modality]:
        """このベクトルストアがサポートするモダリティ一覧。

        Returns:
            set[Modality]: モダリティ一覧
        """
        return self._modality

    def get_container(self, modality: Modality) -> VectorStoreContainer:
        """モダリティ別のベクトルストアコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            ValueError: 予期せぬモダリティ
            RuntimeError: 未初期化

        Returns:
            VectorStoreContainer: ベクトルストアコンテナ
        """
        logger.debug("trace")

        match modality:
            case Modality.TEXT:
                if self._vector_store_text:
                    return self._vector_store_text
            case Modality.IMAGE:
                if self._vector_store_image:
                    return self._vector_store_image
            case _:
                raise ValueError(f"unexpected modality: {modality}")

        raise RuntimeError(f"store {modality} is not initialized")

    async def aupsert_nodes(self, nodes: list[BaseNode]) -> None:
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
        await self._aupsert_text(text_nodes)
        await self._aupsert_image(image_nodes)

        # キャッシュ登録
        self._add_fp_cache(nodes)

    def skip_update(self, source: str) -> bool:
        """ソースが登録済みであり、更新処理が不要か。

        Args:
            source (str): 対象ソース

        Returns:
            bool: 更新処理不要の場合に True
        """
        # logger.debug("trace")

        # check_update 指定がなく、かつソースが登録済み
        return (not self._check_update) and (self._fp_cache.get(source) is not None)

    def _load_fp_cache(self, load_limit: int) -> dict[str, str]:
        """メタデータ用ストアから fingerprint のキャッシュを読み込む。

        Args:
            load_limit (int): メタデータ読み込み件数上限

        Returns:
            dict[str, str]: ソース情報対 fingerprint の KVS
        """
        logger.debug("trace")

        if self._meta_store is None:
            logger.warning("metadata store is not initialized")
            return {}

        rows = self._meta_store.select(
            cols=[MK.FILE_PATH, MK.URL, MK.FINGERPRINT], limit=load_limit
        )

        fp_cache = {}
        for row in rows:
            path, url, fp = row
            source = path or url
            if source and fp:
                fp_cache[path or url] = fp

        return fp_cache

    def _split_nodes_modality(
        self,
        nodes: list[BaseNode],
    ) -> tuple[list[TextNode], list[ImageNode]]:
        """ノードをモダリティ別に分ける。

        Args:
            nodes (list[BaseNode]): 入力ノード

        Returns:
            tuple[list[TextNode], list[ImageNode]]: テキストノード、画像ノード
        """
        logger.debug("trace")

        text_nodes = []
        image_nodes = []
        for node in nodes:
            if isinstance(node, TextNode) and self._is_image_node(node):
                image_nodes.append(ImageNode(text=node.text, metadata=node.metadata))
            elif isinstance(node, TextNode):
                text_nodes.append(node)
            else:
                logger.warning(f"unexpected node type {type(node)}, skipped")

        return text_nodes, image_nodes

    def _is_image_node(self, node: BaseNode) -> bool:
        """画像ノードか。

        Args:
            node (BaseNode): 対象ノード

        Returns:
            bool: 画像ノードなら True
        """
        logger.debug("trace")

        # ファイルパスか URL の末尾に画像ファイルの拡張子が含まれるものを画像ノードとする
        return Exts.is_image_file(
            node.metadata.get(META_KEYS.FILE_PATH, "")
        ) or Exts.is_image_file(node.metadata.get(META_KEYS.URL, ""))

    async def _aupsert_text(self, nodes: list[TextNode]) -> None:
        """テキストを埋め込み、ストアに格納する。

        Raises:
            RuntimeError: upsert 失敗

        Args:
            nodes (list[TextNode]): 対象ノード
        """
        logger.debug("trace")

        if len(nodes) == 0:
            logger.warning("empty list")
            return

        texts = []
        ids = []
        metas = []
        fps = []
        for node in nodes:
            if not node.text:
                logger.warning(f"empty text for node {node.node_id}, skipped")
                continue

            texts.append(node.text)
            ids.append(node.node_id)
            meta = BasicMetaData(node.metadata)
            metas.append(meta)
            fps.append(self._get_lazy_fp(meta))

        try:
            # TODO: バッチだと効率は良いが一件でも失敗すると全滅扱いになる。
            # 最初バッチで実行してみて、失敗したら一件ずつのループにして被害局所化？
            vecs = await self._embed.aembed_text(texts)
            if len(vecs) != len(nodes):
                raise RuntimeError(
                    f"embedding count mismatch: expected {len(nodes)}, got {len(vecs)}"
                )

            for node, vec in zip(nodes, vecs):
                node.embedding = vec

            await self.get_container(Modality.TEXT).store.adelete_nodes(ids)
            await self.get_container(Modality.TEXT).store.async_add(nodes)
            await self._meta_store.aupsert_text_metas(metas=metas, fingerprints=fps)
        except Exception as e:
            raise RuntimeError("failed to upsert text") from e

        logger.info(f"{len(nodes)} nodes are upserted")

    async def _aupsert_image(self, nodes: list[ImageNode]) -> None:
        """画像を埋め込み、ストアに格納する。

        Raises:
            RuntimeError: upsert 失敗

        Args:
            nodes (list[ImageNode]): 対象ノード
        """
        logger.debug("trace")

        if len(nodes) == 0:
            logger.warning("empty list")
            return

        file_paths = []
        temp_file_paths = []
        ids = []
        metas = []
        fps = []
        for node in nodes:
            meta = BasicMetaData(node.metadata)

            temp = meta.temp_file_path
            if temp:
                # フェッチした一時ファイル
                file_paths.append(temp)
                temp_file_paths.append(temp)
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
            vecs = await self._embed.aembed_image(file_paths)
            if len(vecs) != len(nodes):
                raise RuntimeError(
                    f"embedding count mismatch: expected {len(nodes)}, got {len(vecs)}"
                )

            for node, vec in zip(nodes, vecs):
                node.embedding = vec

            await self.get_container(Modality.IMAGE).store.adelete_nodes(ids)
            await self.get_container(Modality.IMAGE).store.async_add(nodes)
            await self._meta_store.aupsert_image_metas(metas=metas, fingerprints=fps)
        except Exception as e:
            raise RuntimeError("failed to upsert text") from e
        finally:
            for path in temp_file_paths:
                os.remove(path)

        logger.info(f"{len(nodes)} nodes are upserted")

    def _create_index(self) -> VectorStoreIndex:
        """インデックスを生成する。

        Raises:
            RuntimeError: コンテナ未初期化

        Returns:
            VectorStoreIndex: 生成したインデックス
        """
        logger.debug("trace")

        if Modality.IMAGE in self.modality:
            return MultiModalVectorStoreIndex.from_vector_store(
                vector_store=self.get_container(Modality.TEXT).store,
                embed_model=self._embed.get_container(Modality.TEXT).embedding,
                image_vector_store=self.get_container(Modality.IMAGE).store,
                image_embed_model=self._embed.get_container(Modality.IMAGE).embedding,
            )

        return VectorStoreIndex.from_vector_store(
            vector_store=self.get_container(Modality.TEXT).store,
            embed_model=self._embed.get_container(Modality.TEXT).embedding,
        )

    def _add_fp_cache(self, nodes: list[BaseNode]) -> None:
        """ノードを fingerprint キャッシュに追加する。

        Args:
            nodes (list[BaseNode]): 追加するノード
        """
        logger.debug("trace")

        for node in nodes:
            meta = BasicMetaData(node.metadata)

            # MultiModalVectorStoreIndex 参照用に画像の一時ファイルを file_path に
            # 入れている場合は URL が正ソースとなるため、この or 順序が重要
            source = meta.url or meta.file_path

            if source == "":
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
            source = meta.url or meta.file_path

            if source is None:
                logger.warning("no source info")
                continue

            # fingerprint キャッシュになければ新規扱い
            if source not in self._fp_cache:
                filtered.append(node)
                continue

            existing_fp = self._fp_cache.get(source, "")
            fp = self._get_lazy_fp(meta)
            if existing_fp == fp:
                logger.info(f"skip document: identical fingerprint for {source}")
                continue

            filtered.append(node)

        return filtered
