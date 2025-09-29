from __future__ import annotations

import os
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ragserver.core.names import PROJECT_NAME
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


class ChromaManager(VectorStoreManager):
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
        database: Optional[str] = None,
        load_limit: int = 10000,
        check_update: bool = True,
    ) -> None:
        """Chroma 管理クラス

        Args:
            persist_directory (Optional[str], optional): ローカル利用時の保存先ディレクトリ。Defaults to None.
            host (Optional[str], optional): リモートサーバ利用時のホスト。Defaults to None.
            port (Optional[int], optional): リモートサーバ利用時のポート番号。Defaults to None.
            api_key (Optional[str], optional): Chroma Cloud 利用時の API キー。Defaults to None.
            tenant (Optional[str], optional): Chroma Cloud 利用時のテナント。Defaults to None.
            database (Optional[str], optional): Chroma Cloud 利用時の DB 名 。Defaults to None.
            load_limit (int, optional): ストアからの読み込み件数上限。 Defaults to 10000.
            check_update (bool, optional): ファイルの更新チェック要否。 Defaults to True.
        """
        logger.debug("trace")

        VectorStoreManager.__init__(
            self, name="chroma", load_limit=load_limit, check_update=check_update
        )
        self._persist_directory = persist_directory
        self._host = host
        self._port = port
        self._api_key = api_key
        self._tenant = tenant
        self._database = database

        self._active_store: Chroma | None = None

    def _select_from(self, cols: set[str], store: VectorStore) -> dict[str, list[Any]]:
        """特定ストアに対する select。

        Args:
            cols (set[str]): 取得対象のメタ情報列
            store (VectorStore): 対象ストア

        Returns:
            dict[str, list[Any]]: 取得したメタ情報列のリスト

        Raises:
            RuntimeError: Chroma からの取得に失敗した場合
        """
        # logger.debug("trace")

        if not isinstance(store, Chroma):
            return {}

        # res は dict: {"ids": [...], "embeddings": [...],
        #   "documents": [...], "metadatas": [...]}
        try:
            res = store.get(include=["metadatas"], limit=self._load_limit)
        except Exception as e:
            raise RuntimeError("failed to select from chroma") from e

        # ids は必須
        cols = cols.copy()
        cols.add("ids")

        values: dict[str, list[Any]] = {}
        for col in cols:
            values[col] = []

        for metadata in res.get("metadatas") or []:
            if not isinstance(metadata, dict):
                continue

            for col in cols:
                val = metadata.get(col)
                values[col].append(val)

        # 例：values["source"] で全件分の source リスト
        return values

    def load_store_by_space_key(self, space_key: str, embed: Embeddings) -> VectorStore:
        """空間キーに対応する store を作成。

        Args:
            space_key (str): 空間キー
            embed (Embeddings): 埋め込み管理

        Raises:
            RuntimeError: 初期化周り、fingerprint キャッシュロードエラー等

        Returns:
            VectorStore: ベクトルストア
        """
        logger.debug("trace")

        # マルチモーダルの場合は add_images() で埋め込む。
        # Chroma はダックタイピングで embed_image() を利用。
        try:
            # ローカルディレクトリ
            if self._persist_directory:
                self._active_store = Chroma(
                    collection_name=space_key,
                    embedding_function=embed,
                    persist_directory=self._persist_directory,
                    collection_metadata={"hnsw:space": "cosine"},
                )
                logger.info("Chroma started: local directory mode")
            # リモートサーバ
            elif self._host:
                self._active_store = Chroma(
                    collection_name=space_key,
                    embedding_function=embed,  # type: ignore
                    host=self._host,
                    port=self._port,
                    collection_metadata={"hnsw:space": "cosine"},
                )
                logger.info("Chroma started: remote server mode")
            # Chroma Cloud
            else:
                self._active_store = Chroma(
                    collection_name=space_key,
                    embedding_function=embed,  # type: ignore
                    chroma_cloud_api_key=self._api_key,
                    tenant=self._tenant,
                    database=self._database,
                    collection_metadata={"hnsw:space": "cosine"},
                )
                logger.info("Chroma started: cloud mode")
        except Exception as e:
            raise RuntimeError("failed to initialize chroma") from e

        if self._active_store is None:
            raise RuntimeError("no create option specified")

        self._stores[space_key] = self._active_store
        self._active_space = space_key
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

        if not isinstance(store, Chroma):
            return

        self._active_space = space_key
        self._active_store = store

    def get_active_store(self) -> Chroma:
        """現在使用中の store を返す。

        Raises:
            RuntimeError: store 未初期化の場合

        Returns:
            Chroma: Chroma store
        """
        logger.debug("trace")

        if self._active_store is None:
            raise RuntimeError("store is not initialized")

        return self._active_store

    def upsert_multi(
        self, docs: list[Document], space_key: Optional[str] = None
    ) -> list[str]:
        """マルチモーダル（画像等）の Document を埋め込み、upsert する。

        Chroma はダックタイピングで embed_image() を呼び出す。

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
            ids.append(doc.metadata["id"])

        try:
            self._active_store.delete(ids)
            # add_images() 内で embed_image() 呼び出し
            self._active_store.add_images(uris=uris, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(uris)
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
