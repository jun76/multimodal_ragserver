from __future__ import annotations

from typing import Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from ragserver.config.general_config import GeneralConfig
from ragserver.config.rerank_config import RerankConfig
from ragserver.config.settings import RerankProvider
from ragserver.logger import logger


class RerankProviderManager:
    def __init__(self) -> None:
        """リランカーのプロバイダ管理クラス。

        Args:
            name (str): プロバイダ名
        """
        logger.debug("trace")

        self._rerank: Optional[BaseNodePostprocessor] = None
        self._provider_name: str = "none"

    def create(self) -> RerankProviderManager:
        logger.debug("trace")

        match GeneralConfig.rerank_provider:
            case RerankProvider.COHERE:
                self._provider_name = RerankProvider.COHERE
                self._rerank = CohereRerank(
                    model=RerankConfig.cohere_rerank_model, top_n=RerankConfig.topk
                )
            case RerankProvider.FLAGEMBEDDING:
                self._provider_name = RerankProvider.FLAGEMBEDDING
                self._rerank = FlagEmbeddingReranker(
                    model=RerankConfig.flagembedding_rerank_model,
                    top_n=RerankConfig.topk,
                )
            case _:
                self._provider_name = "none"
                self._rerank = None

        return self

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return self._provider_name

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
        logger.debug("trace")

        if self._rerank is None:
            logger.warning("rerank provider is not specified")
            return []

        try:
            return await self._rerank.apostprocess_nodes(nodes=nodes, query_str=query)
        except Exception as e:
            raise RuntimeError("failed to rerank documents") from e
