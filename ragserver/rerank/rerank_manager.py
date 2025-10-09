from __future__ import annotations

from abc import ABC, abstractmethod

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from ragserver.logger import logger


class RerankManager(ABC):
    def __init__(self) -> None:
        """リランカー管理の抽象インタフェース

        現状、画像のキャプション（テキスト）をつけることで langchain のリランカが
        そのまま使えるため MultimodalRerankManager は不要だが、マルチモーダル対応の
        リランカを提供するプロバイダが出てきた場合は別途実装のこと。

        Args:
            name (str): プロバイダ名
        """
        logger.debug("trace")

        self._rerank: BaseNodePostprocessor

    @property
    @abstractmethod
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        logger.debug("trace")
        ...

    async def rerank(
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
        logger.debug("trace")
        logger.info("start reranking...")

        try:
            return await self._rerank.apostprocess_nodes(nodes=nodes, query_str=query)
        except Exception as e:
            raise RuntimeError("failed to rerank documents") from e
