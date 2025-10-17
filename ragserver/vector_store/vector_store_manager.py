from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import BaseNode, ImageNode, MediaResource, TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from ragserver.core.exts import Exts
from ragserver.core.metadata import META_KEYS
from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import BasicMetaData
from ragserver.embed.embed_manager import EmbedManager, Modality
from ragserver.llama.core.schema import AudioNode
from ragserver.logger import logger
from ragserver.meta_store.structured.structured import Structured


@dataclass
class VectorStoreContainer:
    """モダリティ毎のベクトルストア関連パラメータを集約"""

    provider_name: str
    store: BasePydanticVectorStore
    table_name: str
    index: Optional[VectorStoreIndex] = None


class VectorStoreManager:
    """ベクトルストアの管理クラス。

    空間キーごとにテーブルを一つ割り当て、ノードを管理する想定。"""

    def __init__(
        self,
        conts: dict[Modality, VectorStoreContainer],
        embed: EmbedManager,
        meta_store: Structured,
        load_limit: int,
        check_update: bool = True,
    ) -> None:
        """コンストラクタ

        Args:
            conts (dict[Modality, VectorStoreContainer]): ベクトルストアコンテナの辞書
            embed (EmbeddingManager): 埋め込み管理
            meta_store (StructuredStoreManager): メタデータ管理
            load_limit (int): キャッシュロード件数上限
            check_update (bool, optional): ファイルの更新チェック要否。Defaults to True.
        """
        logger.debug("trace")

        self._conts = conts
        self._embed = embed
        self._meta_store = meta_store
        self._check_update = check_update

        for modality, cont in self._conts.items():
            cont.index = self._create_index(modality)

        # メタデータ専用ストアから fingerprint キャッシュを復元
        self._fp_cache = self._load_fp_cache(load_limit)

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def modality(self) -> set[Modality]:
        """このベクトルストアがサポートするモダリティ一覧。

        Returns:
            set[Modality]: モダリティ一覧
        """
        return set(self._conts.keys())

    @property
    def table_names(self) -> list[str]:
        """このベクトルストアが保持するテーブル名一覧。

        Returns:
            list[str]: テーブル名一覧
        """
        return [cont.table_name for cont in self._conts.values()]

    def get_index(self, modality: Modality) -> VectorStoreIndex:
        """ストレージから生成したインデックス。

        Raises:
            RuntimeError: 未初期化

        Returns:
            VectorStoreIndex: インデックス
        """
        logger.debug("trace")

        index = self.get_container(modality).index
        if index is None:
            raise RuntimeError(f"index for {modality} is not initialized")

        return index

    def get_container(self, modality: Modality) -> VectorStoreContainer:
        """モダリティ別のベクトルストアコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: 未初期化

        Returns:
            VectorStoreContainer: ベクトルストアコンテナ
        """
        logger.debug("trace")

        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"store {modality} is not initialized")

        return cont

    async def aupsert_nodes(self, nodes: list[BaseNode]) -> None:
        """ノードを埋め込み、ストアに格納する。

        Args:
            nodes (list[BaseNode]): 対象ノード
        """
        logger.debug("trace")

        # fingerprint が既存・同一のノードは upsert しない
        nodes = self._filter_nodes_by_fp(nodes)
        if len(nodes) == 0:
            logger.info("skip upsert: no new nodes")
            return

        text_nodes, image_nodes, audio_nodes = self._split_nodes_modality(nodes)
        await self._aupsert_text(text_nodes)
        await self._aupsert_image(image_nodes)
        await self._aupsert_audio(audio_nodes)

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

        rows = self._meta_store.select(
            cols=[MK.FILE_PATH, MK.URL, MK.FINGERPRINT],
            table_names=self.table_names,
            limit=load_limit,
        )

        fp_cache = {}
        for row in rows:
            path, url, fp = row
            source = path or url
            if source and fp:
                fp_cache[path or url] = fp

        logger.info(f"loaded {len(fp_cache)} fingerprint caches")

        return fp_cache

    def _split_nodes_modality(
        self,
        nodes: list[BaseNode],
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
        """ノードをモダリティ別に分ける。

        Args:
            nodes (list[BaseNode]): 入力ノード

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
                テキストノード、画像ノード、音声ノード
        """
        logger.debug("trace")

        text_nodes = []
        image_nodes = []
        audio_nodes = []
        for node in nodes:
            if isinstance(node, TextNode) and self._is_image_node(node):
                image_nodes.append(ImageNode(text=node.text, metadata=node.metadata))
            if isinstance(node, TextNode) and self._is_audio_node(node):
                audio_nodes.append(
                    AudioNode(
                        audio_resource=MediaResource(text=node.text),
                        extra_info=node.metadata,
                    )
                )
            elif isinstance(node, TextNode):
                text_nodes.append(node)
            else:
                logger.warning(f"unexpected node type {type(node)}, skipped")

        return text_nodes, image_nodes, audio_nodes

    def _is_image_node(self, node: BaseNode) -> bool:
        """画像ノードか。

        Args:
            node (BaseNode): 対象ノード

        Returns:
            bool: 画像ノードなら True
        """
        # logger.debug("trace")

        # ファイルパスか URL の末尾に画像ファイルの拡張子が含まれるものを画像ノードとする
        path = node.metadata.get(META_KEYS.FILE_PATH, "")
        url = node.metadata.get(META_KEYS.URL, "")

        # 独自 reader を使用し、temp_file_path に画像ファイルの拡張子が含まれるものも抽出
        temp_file_path = node.metadata.get(META_KEYS.TEMP_FILE_PATH, "")

        return (
            Exts.endswith_exts(path, Exts.IMAGE)
            or Exts.endswith_exts(url, Exts.IMAGE)
            or Exts.endswith_exts(temp_file_path, Exts.IMAGE)
        )

    def _is_audio_node(self, node: BaseNode) -> bool:
        """音声ノードか。

        Args:
            node (BaseNode): 対象ノード

        Returns:
            bool: 音声ノードなら True
        """
        # logger.debug("trace")

        path = node.metadata.get(META_KEYS.FILE_PATH, "")
        url = node.metadata.get(META_KEYS.URL, "")
        temp_file_path = node.metadata.get(META_KEYS.TEMP_FILE_PATH, "")

        return (
            Exts.endswith_exts(path, Exts.AUDIO)
            or Exts.endswith_exts(url, Exts.AUDIO)
            or Exts.endswith_exts(temp_file_path, Exts.AUDIO)
        )

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
            vecs = await self._embed.aembed_text(texts)
            if len(vecs) != len(nodes):
                raise RuntimeError(
                    f"embedding count mismatch: expected {len(nodes)}, got {len(vecs)}"
                )

            for node, vec in zip(nodes, vecs):
                node.embedding = vec

            cont = self.get_container(Modality.TEXT)
            await cont.store.adelete_nodes(ids)
            await cont.store.async_add(nodes)
            await self._meta_store.aupsert(
                metas=metas, fingerprints=fps, table_name=cont.table_name
            )
        except Exception as e:
            raise RuntimeError("failed to upsert text") from e

        logger.info(f"{len(nodes)} nodes are upserted")

    async def _aupsert_fetched_content(
        self, nodes: Sequence[BaseNode], modality: Modality, aembed_func: Callable
    ) -> None:
        """一時ファイルに保存されたコンテンツを埋め込み、ストアに格納する。

        Raises:
            RuntimeError: upsert 失敗

        Args:
            nodes (Itarable[BaseNode]): 対象ノード
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

                # ファイルパスはベースソースで上書き
                # （空になるか、PDF 等の独自 reader が退避していた元パスが復元されるか）
                meta.file_path = meta.base_source

                # 一時ファイルパスは消去
                meta.temp_file_path = ""
                node.metadata = meta.to_dict()
            else:
                file_path = meta.file_path
                if file_path:
                    # ローカルファイル
                    file_paths.append(file_path)
                else:
                    logger.warning(f"{modality} is not found, skipped")
                    continue

            ids.append(node.node_id)
            metas.append(meta)
            fps.append(self._get_lazy_fp(meta))

        try:
            vecs = await aembed_func(file_paths)
            if len(vecs) != len(nodes):
                raise RuntimeError(
                    f"embedding count mismatch: expected {len(nodes)}, got {len(vecs)}"
                )

            for node, vec in zip(nodes, vecs):
                node.embedding = vec

            cont = self.get_container(modality)
            await cont.store.adelete_nodes(ids)
            await cont.store.async_add(nodes)
            await self._meta_store.aupsert(
                metas=metas, fingerprints=fps, table_name=cont.table_name
            )
        except Exception as e:
            raise RuntimeError(f"failed to upsert {modality}") from e
        finally:
            for path in temp_file_paths:
                os.remove(path)

        logger.info(f"{len(nodes)} nodes are upserted")

    async def _aupsert_image(self, nodes: list[ImageNode]) -> None:
        """画像を埋め込み、ストアに格納する。

        Raises:
            RuntimeError: upsert 失敗

        Args:
            nodes (list[ImageNode]): 対象ノード
        """
        logger.debug("trace")

        await self._aupsert_fetched_content(
            nodes=nodes, modality=Modality.IMAGE, aembed_func=self._embed.aembed_image
        )

    async def _aupsert_audio(self, nodes: list[AudioNode]) -> None:
        """音声を埋め込み、ストアに格納する。

        Raises:
            RuntimeError: upsert 失敗

        Args:
            nodes (list[AudioNode]): 対象ノード
        """
        logger.debug("trace")

        await self._aupsert_fetched_content(
            nodes=nodes, modality=Modality.AUDIO, aembed_func=self._embed.aembed_audio
        )

    def _create_index(self, modality: Modality) -> VectorStoreIndex:
        """インデックスを生成する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: コンテナ未初期化

        Returns:
            VectorStoreIndex: 生成したインデックス
        """
        logger.debug("trace")

        match modality:
            case Modality.TEXT:
                return VectorStoreIndex.from_vector_store(
                    vector_store=self.get_container(Modality.TEXT).store,
                    embed_model=self._embed.get_container(Modality.TEXT).embed,
                )
            case Modality.IMAGE:
                return MultiModalVectorStoreIndex.from_vector_store(
                    vector_store=self.get_container(Modality.TEXT).store,
                    embed_model=self._embed.get_container(Modality.TEXT).embed,
                    image_vector_store=self.get_container(Modality.IMAGE).store,
                    image_embed_model=self._embed.get_container(Modality.IMAGE).embed,
                )
            case Modality.AUDIO:
                return VectorStoreIndex.from_vector_store(
                    vector_store=self.get_container(Modality.AUDIO).store,
                    embed_model=self._embed.get_container(Modality.AUDIO).embed,
                )
            case _:
                raise RuntimeError("unexpected modality")

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
        # logger.debug("trace")

        # Web ページの場合、現状 URL しかチェックしない
        fp_data = {
            MK.FILE_PATH: meta.file_path,
            MK.FILE_SIZE: meta.file_size,
            MK.FILE_LASTMOD_AT: meta.file_lastmod_at,
            MK.CHUNK_NO: meta.chunk_no,
            MK.PAGE_NO: meta.page_no,
            MK.ASSET_NO: meta.asset_no,
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
