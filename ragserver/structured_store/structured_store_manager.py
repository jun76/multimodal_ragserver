from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ragserver.core.metadata import BasicMetaData
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
        self._space_key_image = None

    @abstractmethod
    def prepare_with(
        self, space_key_text: str, space_key_image: Optional[str] = None
    ) -> None:
        """空間キーに合わせてストアを初期化する。

        Args:
            space_key_text (str): テキストベクトルの空間キー
            space_key_image (Optional[str], optional): 画像ベクトルの空間キー。Defaults to None.

        Raises:
            RuntimeError: ストア初期化失敗
        """

    @abstractmethod
    async def aupsert_text_metas(
        self, metas: list[BasicMetaData], fingerprints: list[str]
    ) -> None:
        """テキストノードのメタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
        """

    @abstractmethod
    async def aupsert_image_metas(
        self, metas: list[BasicMetaData], fingerprints: list[str]
    ) -> None:
        """画像ノードのメタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
        """

    @abstractmethod
    def select(self, cols: list[str], limit: int) -> list[tuple]:
        """select 文を実行する。

        Args:
            cols (list[str]): 取得する列
            limit (int): 件数上限

        Returns:
            list[tuple]: 取得したレコード群
        """
