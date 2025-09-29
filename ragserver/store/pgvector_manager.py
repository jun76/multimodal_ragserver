from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from sqlalchemy import select

from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.names import PROJECT_NAME
from ragserver.embed.multimodal_embeddings_manager import MultimodalEmbeddings
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


class PgVectorManager(VectorStoreManager):
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        table_prefix: str = f"{PROJECT_NAME}__",
        load_limit: int = 10000,
        check_update: bool = True,
    ) -> None:
        """PgVector 管理クラス

        Args:
            host (str): 接続先ホスト名
            port (int): 接続先ポート番号
            dbname (str): 接続先データベース名
            user (str): 接続ユーザー名
            password (str): 接続パスワード
            table_prefix (str, optional): 各 space テーブルの接頭辞。Defaults to f"{PROJECT_NAME}__".
            load_limit (int, optional): ストアからの読み込み件数上限。 Defaults to 10000.
            check_update (bool, optional): ファイルの更新チェック要否。 Defaults to True.
        """
        logger.debug("trace")

        VectorStoreManager.__init__(
            self, name="pgvector", load_limit=load_limit, check_update=check_update
        )
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self._table_prefix = table_prefix

        self._active_store: PGVector | None = None

    def _select_from(self, cols: set[str], store: VectorStore) -> dict[str, list[Any]]:
        """特定ストアに対する select。

        Args:
            cols (set[str]): 取得対象のメタ情報列
            store (VectorStore): 対象ストア

        Returns:
            dict[str, list[Any]]: 取得したメタ情報列のリスト

        Raises:
            RuntimeError: PgVector からの取得に失敗した場合
        """
        # logger.debug("trace")

        if not isinstance(store, PGVector):
            return {}

        values: dict[str, list[Any]] = {}
        for col in cols:
            values[col] = []

        try:
            with store._make_sync_session() as session:
                collection = store.get_collection(session)
                if not collection:
                    return values

                stmt = (
                    select(store.EmbeddingStore.id, store.EmbeddingStore.cmetadata)
                    .where(store.EmbeddingStore.collection_id == collection.uuid)
                    .order_by(store.EmbeddingStore.id)
                    .limit(self._load_limit)
                )

                results = session.execute(stmt).all()
        except Exception as e:
            raise RuntimeError("failed to select from pgvector") from e

        for store_id, metadata in results:
            meta_dict = metadata or {}
            for col in cols:
                if col == "ids":
                    values[col].append(store_id)
                else:
                    values[col].append(meta_dict.get(col))

        return values

    def load_store_by_space_key(self, space_key: str, embed: Embeddings) -> VectorStore:
        """空間キーに対応する store を作成。

        Args:
            space_key (str): 空間キー
            embed (Embeddings): 埋め込み管理

        Raises:
            e: 初期化周りの例外
            RuntimeError: 保存先の指定漏れ

        Returns:
            VectorStore: ベクトルストア
        """
        logger.debug("trace")

        try:
            uri = (
                "postgresql+psycopg://"
                f"{self._user}"
                f":{self._password}"
                f"@{self._host}"
                f":{self._port}"
                f"/{self._dbname}"
            )
            self._connection = uri

            self._active_store = PGVector(
                embeddings=embed,
                collection_name=space_key,
                connection=self._connection,
                use_jsonb=True,
                distance_strategy=DistanceStrategy.COSINE,
            )
            logger.info(f"PGVector started at {uri}")
        except Exception as e:
            raise RuntimeError("failed to initialize pgvector") from e

        if self._active_store is None:
            raise RuntimeError("no create option specified")

        self._stores[space_key] = self._active_store
        self._active_space = space_key
        self._embed = embed
        try:
            self._load_fingerprint_cache(space_key)
        except Exception as e:
            raise RuntimeError("failed to load fingerprint cache") from e

        logger.info(f"now {len(self._fingerprint_cache)} docs in fp cache")

        return self._active_store

    def activate_space(self, space_key: str) -> None:
        """登録済みストアをアクティブに切り替える。

        Args:
            space_key (str): 空間キー
        """
        logger.debug("trace")

        if self._active_space == space_key:
            return

        store = self._stores.get(space_key)
        if store is None:
            logger.error(f"space [{space_key}] is not loaded")
            return

        if not isinstance(store, PGVector):
            return

        self._active_space = space_key
        self._active_store = store

    def get_active_store(self) -> PGVector:
        """現在使用中の store を返す。

        Raises:
            RuntimeError: store 未初期化の場合

        Returns:
            PGVector: PGVector store
        """
        logger.debug("trace")

        if self._active_store is None:
            raise RuntimeError("store is not initialized")

        return self._active_store

    def upsert_multi(
        self, docs: list[Document], space_key: Optional[str] = None
    ) -> list[str]:
        """マルチモーダル（画像等）の Document を埋め込み、upsert する。

        Args:
            docs (list[Document]): メタ情報入りの Document リスト
            space_key (Optional[str], optional): 空間切り替えも行う場合に指定。 Defaults to None.

        Returns:
            list[str]: 登録済み ids
        """
        logger.debug("trace")

        if len(docs) == 0:
            logger.info("empty docs")
            return []

        if space_key:
            self.activate_space(space_key)

        if self._active_store is None:
            logger.error("active store is not initialized")
            return []

        docs = self._filter_docs_by_fingerprint(docs)
        if len(docs) == 0:
            logger.info("skip upsert_multi: all documents already exist")
            return []

        uris: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        for doc in docs:
            uris.append(doc.page_content)  # 画像 URI（ローカルパス可）
            meta = dict(doc.metadata)
            metadatas.append(meta)
            ids.append(doc.metadata[MK.ID])

        try:
            self._active_store.delete(ids)
            ids = self.add_images(uris, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(docs)
            raise RuntimeError("failed to upsert multimodal documents") from e
        finally:
            for uri in uris:
                # 一時ファイルを削除
                basename = os.path.basename(uri)
                if os.path.isfile(uri) and basename.startswith(f"{PROJECT_NAME}_"):
                    try:
                        os.remove(uri)
                    except OSError as e:
                        logger.exception(e)

        if len(ids) == 0:
            logger.warning("empty ids")
            return []

        # キャッシュ登録
        self._regist_fingerprint_cache(docs)

        return ids

    def add_images(
        self,
        uris: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Chroma と階層を合わせてここで画像の埋め込みとストア格納を行う。

        Args:
            uris (list[str]): 画像の URI
            metadatas (list[dict[str, Any]] | None, optional): 画像のメタ情報。 Defaults to None.
            ids (list[str] | None, optional): 発行済み ID。 Defaults to None.

        Returns:
            list[str]: 今回新規発行の ID
        """
        logger.debug("trace")

        if self._active_store is None:
            logger.error("active store is not initialized")
            return []

        if not isinstance(self._embed, MultimodalEmbeddings):
            logger.error("not supported embed_image")
            return []

        try:
            vecs = self._embed.embed_image(uris)
        except Exception as e:
            raise RuntimeError("failed to embed images") from e

        if len(vecs) == 0:
            logger.warning("empty vecs")
            return []

        try:
            return self._active_store.add_embeddings(
                uris, embeddings=vecs, metadatas=metadatas, ids=ids
            )
        except Exception as e:
            raise RuntimeError("failed to add embeddings") from e
