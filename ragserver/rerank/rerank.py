from __future__ import annotations

from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from ragserver.config.general_config import GeneralConfig
from ragserver.config.rerank_config import RerankConfig
from ragserver.config.settings import RerankProvider
from ragserver.logger import logger
from ragserver.rerank.rerank_manager import RerankContainer, RerankManager

__all__ = ["create_rerank_manager"]


def create_rerank_manager() -> RerankManager:
    """リランク管理インスタンスを生成する。

    Raises:
        RuntimeError: インスタンス生成に失敗

    Returns:
        RerankManager: リランク管理
    """

    try:
        match GeneralConfig.rerank_provider:
            case RerankProvider.COHERE:
                rerank = _cohere()
            case RerankProvider.FLAGEMBEDDING:
                rerank = _flagembedding()
            case _:
                rerank = None

        return RerankManager(rerank)
    except Exception as e:
        raise RuntimeError(f"failed to create rerank: {e}") from e


def _cohere() -> RerankContainer:
    """リランク管理生成ヘルパー

    Returns:
        RerankContainer: コンテナ
    """

    return RerankContainer(
        provider_name=RerankProvider.COHERE,
        rerank=CohereRerank(
            model=RerankConfig.cohere_rerank_model, top_n=RerankConfig.topk
        ),
    )


def _flagembedding() -> RerankContainer:
    """リランク管理生成ヘルパー

    Returns:
        RerankContainer: コンテナ
    """

    return RerankContainer(
        provider_name=RerankProvider.FLAGEMBEDDING,
        rerank=FlagEmbeddingReranker(
            model=RerankConfig.flagembedding_rerank_model, top_n=RerankConfig.topk
        ),
    )
