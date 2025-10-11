from __future__ import annotations

from typing import Optional

from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import NodeWithScore

from ragserver.logger import logger
from ragserver.rerank.rerank_manager import RerankManager
from ragserver.vector_store.vector_store_manager import VectorStoreManager

__all__ = ["query_text", "query_text_multi", "query_image"]


async def query_text(
    query: str,
    store: VectorStoreManager,
    topk: int = 10,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列によるテキストドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (VectorStoreManager): ベクトルストア
        topk (int, optional): 取得件数。Defaults to 10.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    logger.debug("trace")

    if store.index is None:
        logger.error("store is not initialized")
        return []

    retriever_engine = store.index.as_retriever(similarity_top_k=topk)
    nwss = await retriever_engine.aretrieve(query)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    if rerank is None:
        return nwss

    nwss = await rerank.rerank(nodes=nwss, query=query)
    logger.info(f"reranked {len(nwss)} nodes")

    return nwss


async def query_text_multi(
    query: str,
    store: VectorStoreManager,
    topk: int = 10,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列によるマルチモーダルドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (VectorStoreManager): ベクトルストア
        topk (int, optional): 取得件数。Defaults to 10.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    logger.debug("trace")

    if store.index is None:
        logger.error("store is not initialized")
        return []

    if not isinstance(store.index, MultiModalVectorStoreIndex):
        logger.error("multimodal index is required")
        return []

    retriever_engine = store.index.as_retriever(
        similarity_top_k=topk, image_similarity_top_k=topk
    )
    nwss = await retriever_engine.atext_retrieve(query)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    if rerank is None:
        return nwss

    nwss = await rerank.rerank(nodes=nwss, query=query)
    logger.info(f"reranked {len(nwss)} nodes")

    return nwss


async def query_image(
    path: str,
    store: VectorStoreManager,
    topk: int = 10,
) -> list[NodeWithScore]:
    """クエリ画像によるマルチモーダルドキュメント検索。

    Args:
        path (str): クエリ画像の ローカルパス
        store (VectorStoreManager): ベクトルストア
        topk (int, optional): 取得件数。Defaults to 10.

    Returns:
        list[NodeWithScore]: 検索結果のリスト

    Raises:
        RuntimeError: ストア操作や検索処理に失敗した場合
    """
    logger.debug("trace")

    if store.index is None:
        logger.error("store is not initialized")
        return []

    if not isinstance(store.index, MultiModalVectorStoreIndex):
        logger.error("multimodal index is required")
        return []

    retriever_engine = store.index.as_retriever(
        similarity_top_k=topk, image_similarity_top_k=topk
    )
    nwss = await retriever_engine.aimage_to_image_retrieve(path)

    if len(nwss) == 0:
        logger.warning("empty nodes")

    return nwss
