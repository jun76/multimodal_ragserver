from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from ragserver.logger import logger


@dataclass
class RerankContainer:
    """リランク関連パラメータを集約"""

    provider_name: str
    rerank: BaseNodePostprocessor


class RerankManager:
    """リランクの管理クラス。"""

    def __init__(self, cont: Optional[RerankContainer] = None) -> None:
        """コンストラクタ

        Args:
            cont (RerankContainer): リランクコンテナ
        """

        self._cont = cont

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return self._cont.provider_name if self._cont else "none"

    async def arerank(
        self, nodes: list[NodeWithScore], query: str
    ) -> list[NodeWithScore]:
        """クエリに基づきリランカーで結果を並べ替える。

        Args:
            nodes (list[NodeWithScore]): 並べ替え対象ノード
            query (str): クエリ文字列

        Returns:
            list[NodeWithScore]: 並べ替え済みのノード

        Raises:
            RuntimeError: リランカーが処理に失敗した場合
        """

        if self._cont is None:
            logger.warning("rerank provider is not specified")
            return nodes

        try:
            return await self._cont.rerank.apostprocess_nodes(
                nodes=nodes, query_str=query
            )
        except Exception as e:
            raise RuntimeError("failed to rerank documents") from e
