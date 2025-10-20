from __future__ import annotations

from typing import Optional

from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import NodeWithScore

from ..llama.core.indices.multi_modal.retriever import AudioRetriever
from ..llama.core.schema import Modality
from ..logger import logger
from ..rerank.rerank_manager import RerankManager
from ..vector_store.vector_store_manager import VectorStoreManager

__all__ = [
    "aquery_text_text",
    "aquery_text_image",
    "aquery_image_image",
    "aquery_text_audio",
    "aquery_audio_audio",
]


async def aquery_text_text(
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

    index = store.get_index(Modality.TEXT)
    if index is None:
        logger.error("store is not initialized")
        return []

    retriever_engine = index.as_retriever(similarity_top_k=topk)
    nwss = await retriever_engine.aretrieve(query)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query)
    logger.info(f"reranked {len(nwss)} nodes")

    return nwss


async def aquery_text_image(
    query: str,
    store: VectorStoreManager,
    topk: int = 10,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による画像ドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (VectorStoreManager): ベクトルストア
        topk (int, optional): 取得件数。Defaults to 10.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 画像埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """

    index = store.get_index(Modality.IMAGE)
    if index is None:
        logger.error("store is not initialized")
        return []

    if not isinstance(index, MultiModalVectorStoreIndex):
        logger.error("multimodal index is required")
        return []

    retriever_engine = index.as_retriever(
        similarity_top_k=topk, image_similarity_top_k=topk
    )

    try:
        nwss = await retriever_engine.atext_to_image_retrieve(query)
    except Exception as e:
        raise RuntimeError(
            "this embed model may not support text --> image embedding"
        ) from e

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query)
    logger.info(f"reranked {len(nwss)} nodes")

    return nwss


async def aquery_image_image(
    path: str,
    store: VectorStoreManager,
    topk: int = 10,
) -> list[NodeWithScore]:
    """クエリ画像による画像ドキュメント検索。

    Args:
        path (str): クエリ画像の ローカルパス
        store (VectorStoreManager): ベクトルストア
        topk (int, optional): 取得件数。Defaults to 10.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """

    index = store.get_index(Modality.IMAGE)
    if index is None:
        logger.error("store is not initialized")
        return []

    if not isinstance(index, MultiModalVectorStoreIndex):
        logger.error("multimodal index is required")
        return []

    retriever_engine = index.as_retriever(
        similarity_top_k=topk, image_similarity_top_k=topk
    )
    nwss = await retriever_engine.aimage_to_image_retrieve(path)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    logger.info(f"got {len(nwss)} nodes")

    return nwss


async def aquery_text_audio(
    query: str,
    store: VectorStoreManager,
    topk: int = 10,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による音声ドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (VectorStoreManager): ベクトルストア
        topk (int, optional): 取得件数。Defaults to 10.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 音声埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """

    index = store.get_index(Modality.AUDIO)
    if index is None:
        logger.error("store is not initialized")
        return []

    retriever_engine = AudioRetriever(index=index, top_k=topk)
    try:
        nwss = await retriever_engine.atext_to_audio_retrieve(query)
    except Exception as e:
        raise RuntimeError(
            "this embed model may not support text --> audio embedding"
        ) from e

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query)
    logger.info(f"reranked {len(nwss)} nodes")

    return nwss


async def aquery_audio_audio(
    path: str,
    store: VectorStoreManager,
    topk: int = 10,
) -> list[NodeWithScore]:
    """クエリ音声による音声ドキュメント検索。

    Args:
        path (str): クエリ音声の ローカルパス
        store (VectorStoreManager): ベクトルストア
        topk (int, optional): 取得件数。Defaults to 10.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """

    index = store.get_index(Modality.AUDIO)
    if index is None:
        logger.error("store is not initialized")
        return []

    retriever_engine = AudioRetriever(index=index, top_k=topk)
    nwss = await retriever_engine.aaudio_to_audio_retrieve(path)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    logger.info(f"got {len(nwss)} nodes")

    return nwss
