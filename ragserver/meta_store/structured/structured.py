from __future__ import annotations

from abc import ABC, abstractmethod

from ...core.metadata import BasicMetaData


class Structured(ABC):
    """構造化ストア管理クラスの抽象

    空間キーごとにテーブルを一つ割り当て、メタ情報を管理する想定。
    """

    @abstractmethod
    def _prepare_with(self, table_name: str) -> None:
        """指定のテーブルが存在しない場合、予め作成する。

        Args:
            table_name (str): テーブル名

        Raises:
            RuntimeError: テーブル作成失敗
        """
        ...

    @abstractmethod
    async def aupsert(
        self, metas: list[BasicMetaData], fingerprints: list[str], table_name: str
    ) -> None:
        """メタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
            table_name (str): テーブル名

        Raises:
            RuntimeError: upsert 失敗
        """
        ...

    @abstractmethod
    def select(
        self, cols: list[str], table_names: list[str], limit: int
    ) -> list[tuple]:
        """select 文を実行する。

        Args:
            cols (list[str]): 取得する列
            table_names (list[str]): テーブル名のリスト
            limit (int): 件数上限

        Returns:
            list[tuple]: 取得したレコード群
        """
        ...
