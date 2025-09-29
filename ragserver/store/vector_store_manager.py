from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ragserver.core.metadata import DUMMY_FINGERPRINT, FINGERPRINT_KEYS
from ragserver.core.metadata import META_KEYS as MK
from ragserver.logger import logger


class VectorStoreManager(ABC):
    def __init__(
        self, name: str, load_limit: int = 10000, check_update: bool = True
    ) -> None:
        """ベクトルストア管理クラスの抽象クラス

        Args:
            name (str): プロバイダ名
            load_limit (int, optional): ストアからの読み込み件数上限。 Defaults to 10000.
            check_update (bool, optional): ファイルの更新チェック要否。 Defaults to True.
        """
        logger.debug("trace")

        self._name = name
        self._active_space: str = ""
        self._stores: dict[str, VectorStore] = {}
        self._load_limit = load_limit
        self._check_update = check_update

        # 各ストア（空間キー毎）の初期化時に同期し、以降は fingerprint チェックの度に追加
        # fingerprint が存在しない場合（html 等）でもダミーを登録することで
        # ソースの存在チェックに併用する。
        self._fingerprint_cache: dict[str, Optional[dict[str, Any]]] = {}

    def get_name(self) -> str:
        """プロバイダ名を取得する。
        クライアント側での状態確認用途を想定。

        Returns:
            str: プロバイダ名
        """
        logger.debug("trace")

        return self._name

    @abstractmethod
    def _select_from(self, cols: set[str], store: VectorStore) -> dict[str, list[Any]]:
        """特定ストアに対する select。

        Args:
            cols (set[str]): 取得対象のメタ情報列
            store (VectorStore): 対象ストア

        Returns:
            dict[str, list[Any]]: 取得したメタ情報列のリスト
        """
        # logger.debug("trace")
        ...

    def _select(
        self, cols: set[str], space_key: Optional[str] = None
    ) -> dict[str, list[Any]]:
        """メタ情報の列を取得。

        Args:
            cols (set[str]): 取得対象のメタ情報列
            space_key (Optional[str], optional): 指定なしの場合全空間。 Defaults to None.

        Returns:
            dict[str, list[Any]]: 取得したメタ情報列のリスト
        """
        # logger.debug("trace")

        if space_key:
            self.activate_space(space_key)
            if self.get_active_store() is None:
                logger.error("active store is not initialized")
                return {}
            return self._select_from(cols=cols, store=self.get_active_store())

        values: dict[str, list[Any]] = {}
        for col in cols:
            values[col] = []

        temp: dict[str, list[Any]] = {}
        for store in self._stores.values():
            temp = self._select_from(cols, store)
            for col in cols:
                values[col].extend(temp.get(col, []))

        return values

    def _load_fingerprint_cache(self, space_key: str) -> None:
        """ストア内の fingerprint をキャッシュに登録する。
        ドキュメント数が増えると重い処理なので limit で調整。

        Args:
            space_key (str): 対象ストアの空間キー
        """
        logger.debug("trace")

        cols = self._select(cols=({MK.SOURCE} | FINGERPRINT_KEYS), space_key=space_key)
        sources = cols[MK.SOURCE]

        for i, source in enumerate(sources):
            # ある source について、キャッシュ未登録なら
            if self._fingerprint_cache.get(source) is None:
                fp = {}
                for key in FINGERPRINT_KEYS:
                    fp[key] = cols[key][i]
                # 登録
                self._fingerprint_cache[source] = fp

    def _get_fingerprint_by_source(self, source: str) -> Optional[dict[str, Any]]:
        """指定ソースの fingerprint を 1 件（代表値）取得する。

        複数ドキュメントに分かれる場合（テキスト、PDF 等）でも元がファイルであれば
        同一 fingerprint を持つ前提のため、いずれか 1 件を返せば十分とする。

        Args:
            source (str): 対象ソース

        Returns:
            Optional[dict[str, Any]]: fingerprint の辞書。未登録または取得失敗時は None
        """
        # logger.debug("trace")

        cols = self._select(cols=({MK.SOURCE} | FINGERPRINT_KEYS))
        sources = cols[MK.SOURCE]

        fp = {}
        for i, src in enumerate(sources):
            if src == source:
                for key in FINGERPRINT_KEYS:
                    fp[key] = cols[key][i]
                # 1 件で return
                return fp

        return None

    def _filter_docs_by_fingerprint(self, docs: list[Document]) -> list[Document]:
        """fingerprint に基づき既存ドキュメントを除外したリストを返す。

        Args:
            docs (list[Document]): 登録候補のドキュメント

        Returns:
            list[Document]: フィルター後のドキュメント
        """
        logger.debug("trace")

        filtered: list[Document] = []

        for doc in docs:
            metadata = doc.metadata or {}
            source = metadata.get(MK.SOURCE)

            # source が None ならスキップできないので新規扱い
            if source is None:
                filtered.append(doc)
                continue

            # 計算済み fingerprint キャッシュになければ新規扱い
            if source not in self._fingerprint_cache:
                filtered.append(doc)
                continue

            existing_fp = self._fingerprint_cache[source]

            # None が登録されていた場合（ないはず。型チェックの都合）
            if existing_fp is None:
                filtered.append(doc)
                continue

            fp = self._extract_fingerprint(metadata)
            if self._fingerprint_equals(existing_fp, fp):
                logger.info("skip document: identical fingerprint for %s", source)
                continue

            filtered.append(doc)

        return filtered

    def _extract_fingerprint(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """metadata から fingerprint 情報を抽出する。
        URL 等、fingerprint が存在しない場合はダミーを返却し、source の存在だけは
        マークできるようにする。

        Args:
            metadata (dict[str, Any]): 対象のメタ情報

        Returns:
            dict[str, Any]: fingerprint 情報
        """
        # logger.debug("trace")

        fp: dict[str, Any] = {}
        for key in FINGERPRINT_KEYS:
            value = metadata.get(key)
            if value is None:
                return DUMMY_FINGERPRINT

            fp[key] = value

        return fp

    def _fingerprint_equals(
        self, stored: dict[str, Any], current: dict[str, Any]
    ) -> bool:
        """取得済み fingerprint と現在の fingerprint を比較する。

        Args:
            stored (dict[str, Any]): 取得済み fingerprint
            current (dict[str, Any]): 現在の fingerprint

        Returns:
            bool: 完全一致で True
        """
        # logger.debug("trace")

        for key in FINGERPRINT_KEYS:
            if current.get(key) is None:
                return False
            if stored.get(key) != current.get(key):
                return False

        return True

    def _regist_fingerprint_cache(self, docs: list[Document]) -> None:
        """ドキュメントを計算済み fingerprint キャッシュに追加する。

        Args:
            docs (list[Document]): 追加するドキュメント
        """
        logger.debug("trace")

        for doc in docs:
            metadata = doc.metadata or {}
            source = metadata.get(MK.SOURCE)

            if source is None:
                continue

            # 計算済み fingerprint キャッシュになければ追加（＝次回以降スキップ）
            if source not in self._fingerprint_cache:
                self._fingerprint_cache[source] = self._get_fingerprint_by_source(
                    source
                )

    @abstractmethod
    def load_store_by_space_key(self, space_key: str, embed: Embeddings) -> VectorStore:
        """空間キーに対応するストアを新規作成またはロードする。

        Args:
            space_key (str): 空間キー
            embed (Embeddings): 埋め込み管理

        Returns:
            VectorStore: ベクトルストア
        """
        logger.debug("trace")
        ...

    @abstractmethod
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

        # 多態のために self._active_store は上位 Manager クラスで宣言したいため、
        # 本関数のオーバーライド後、続きは以下のように実装すること。
        #
        # if not isinstance(store, sub_class_of_vector_store):
        #     return

        # self._active_space = space_key
        # self._active_store = store

    @abstractmethod
    def get_active_store(self) -> VectorStore:
        """現在使用中のストアを返す。

        Returns:
            VectorStore: store
        """
        logger.debug("trace")
        ...

    def skip_update(self, source: str) -> bool:
        """ソースが登録済みであり、更新処理が不要か。

        Args:
            source (str): 対象ソース

        Returns:
            bool: 更新処理不要の場合に True
        """
        logger.debug("trace")

        # check_update 指定がなく、かつソースが登録済み
        return (not self._check_update) and (
            self._fingerprint_cache.get(source) is not None
        )

    def upsert(
        self, docs: list[Document], space_key: Optional[str] = None
    ) -> list[str]:
        """テキストの Document を埋め込み、upsert する。

        Args:
            docs (list[Document]): Document リスト
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

        if self.get_active_store() is None:
            logger.error("active store is not initialized")
            return []

        docs = self._filter_docs_by_fingerprint(docs)
        if len(docs) == 0:
            logger.info("skip upsert: all documents already exist")
            return []

        ids = []
        for doc in docs:
            ids.append(doc.metadata[MK.ID])

        try:
            self.get_active_store().delete(ids)
            ids = self.get_active_store().add_documents(docs, ids=ids)
        except Exception as e:
            logger.error(docs)
            logger.exception(e)
            return []

        return ids

    @abstractmethod
    def upsert_multi(
        self, docs: list[Document], space_key: Optional[str] = None
    ) -> list[str]:
        """マルチモーダル（画像等）のドキュメントを埋め込み、upsert する。

        Args:
            docs (list[Document]): メタ情報入りのドキュメントリスト
            space_key (Optional[str], optional): 空間切り替えも行う場合に指定。 Defaults to None.

        Returns:
            list[str]: 登録済み ids
        """
        logger.debug("trace")
        ...

    def query(
        self,
        query_vec: list[float],
        topk: int = 10,
        filter: Optional[dict[str, str]] = None,
        space_key: Optional[str] = None,
    ) -> list[Document]:
        """埋め込み済み query による検索実行。

        Args:
            query_vec (list[float]): クエリベクトル
            topk (int, optional): 取得件数。Defaults to 10.
            filter (Optional[dict[str, str]], optional): metadata 用絞り込みフィルタ。Defaults to None.
            space_key (Optional[str], optional): 空間切り替えも行う場合に指定。 Defaults to None.

        Returns:
            list[Document]: 類似度上位ドキュメント
        """
        logger.debug("trace")

        if space_key:
            self.activate_space(space_key)

        if self.get_active_store() is None:
            logger.error("active store is not initialized")
            return []

        if len(query_vec) == 0:
            logger.warning("empty query_vec")
            return []

        return self.get_active_store().similarity_search_by_vector(
            embedding=query_vec, k=topk, filter=filter
        )
