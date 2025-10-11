from __future__ import annotations

from abc import ABC, abstractmethod

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
            space_key (str): 空間キー
            knowledgebase_name (str, optional): ナレッジベース（用途）名。Defaults to "default".
        """
        logger.debug("trace")

        self._knowledgebase_name = knowledgebase_name

    @abstractmethod
    def _exec_query(self, query: str, params: list[str]) -> dict[str, str]:
        """クエリを実行する。

        Args:
            query (str): クエリ
            params (list[str]): パラメータのリスト
        """

    @abstractmethod
    def activate_with(self, space_key: str):
        """空間キーに合わせてストアを初期化する。

        Args:
            space_key (str): 空間キー

        Raises:
            RuntimeError: ストア初期化失敗
        """

    def select_from(
        self, space_key: str, cols: list[str], limit: int
    ) -> dict[str, str]:
        """select 文を実行する。

        Args:
            space_key (str): 空間キー
            cols (list[str]): 取得する列
            limit (int): 件数上限

        Returns:
            dict[str, str]: 取得したレコード群
        """
        logger.debug("trace")

        params = []
        query = f"SELECT ? FROM ?"
        params.append(" ".join(cols))
        params.append(f"{PROJECT_NAME}_{self._knowledgebase_name}_{space_key}")

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        return self._exec_query(query, params)
