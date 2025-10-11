from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import BasicMetaData
from ragserver.core.names import PROJECT_NAME
from ragserver.logger import logger


class StructuredStoreManager(ABC):
    def __init__(
        self,
        knowledgebase_name: str = "default",
    ) -> None:
        """構造化ストア管理クラスの抽象

        空間キーごとにテーブルを一つ割り当て、メタ情報を管理する想定。

        Args:
            knowledgebase_name (str, optional): ナレッジベース（用途）名。Defaults to "default".
        """
        logger.debug("trace")

        self._knowledgebase_name = knowledgebase_name
        self._space_key_text = None
        self._space_key_multi = None

    @abstractmethod
    def _exec_query(self, query: str, params: list[Any]) -> dict[str, str]:
        """クエリを実行する。

        Args:
            query (str): クエリ
            params (list[Any]): パラメータのリスト

        Returns:
            dict[str, str]: 取得したレコード群
        """

    @abstractmethod
    def prepare_with(
        self, space_key_text: str, space_key_multi: Optional[str] = None
    ) -> None:
        """空間キーに合わせてストアを初期化する。

        Args:
            space_key_text (str): テキストベクトルの空間キー
            space_key_multi (Optional[str], optional): 画像ベクトルの空間キー。Defaults to None.

        Raises:
            RuntimeError: ストア初期化失敗
        """

    def select(self, cols: list[str], limit: int) -> dict[str, str]:
        """select 文を実行する。

        Args:
            cols (list[str]): 取得する列
            limit (int): 件数上限

        Returns:
            dict[str, str]: 取得したレコード群
        """
        logger.debug("trace")

        if self._space_key_text is None or self._space_key_multi is None:
            logger.warning("space key is not initialized")
            return {}

        params = []
        query = f"SELECT ? FROM ?"
        params.append(" ".join(cols))
        params.append(
            f"{PROJECT_NAME}_{self._knowledgebase_name}_{self._space_key_text} "
        )
        params.append(
            f"{PROJECT_NAME}_{self._knowledgebase_name}_{self._space_key_multi} "
        )

        query += f"ORDER BY {MK.NODE_LASTMOD_AT} DESC LIMIT ?"
        params.append(limit)

        return self._exec_query(query, params)

    @abstractmethod
    def upsert_text_metas(
        self, metas: list[BasicMetaData], fingerprints: list[str]
    ) -> None:
        """テキストノードのメタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
        """

    @abstractmethod
    def upsert_image_metas(
        self, metas: list[BasicMetaData], fingerprints: list[str]
    ) -> None:
        """画像ノードのメタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
        """
