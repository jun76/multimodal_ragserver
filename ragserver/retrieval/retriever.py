from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document

from ragserver.core.metadata import META_KEYS as MK
from ragserver.embed.embeddings_manager import EmbeddingsManager
from ragserver.embed.multimodal_embeddings_manager import MultimodalEmbeddingsManager
from ragserver.logger import logger
from ragserver.rerank.rerank_manager import RerankManager
from ragserver.store.vector_store_manager import VectorStoreManager

__all__ = ["query_text", "query_text_multi", "query_image"]


def _review_page_content(
    docs: list[Document], content: Optional[str] = None
) -> list[Document]:
    """マルチモーダル埋め込みの場合、 page_content に画像データの Base64 文字列が
    入るが、使用しないため画像の説明文で上書き。

    Args:
        docs (list[Document]): 対象ドキュメント
        content (Optional[str], optional): 画像の説明文。なければ最低限 source で上書き。 Defaults to None.

    Returns:
        list[Document]: 上書き後のドキュメント
    """
    logger.debug("trace")

    if content is None:
        for doc in docs:
            doc.page_content = doc.metadata[MK.SOURCE]
    else:
        for doc in docs:
            doc.page_content = content

    return docs


def query_text(
    query: str,
    store: VectorStoreManager,
    embed: EmbeddingsManager,
    topk: int = 10,
    rerank: Optional[RerankManager] = None,
    topk_rerank_scale: int = 5,
) -> list[Document]:
    """クエリ文字列によるテキストドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (VectorStoreManager): ベクトルストア
        embed (EmbeddingsManager): 埋め込み管理
        topk (int, optional): 取得件数。 Defaults to 10.
        rerank (Optional[RerankManager], optional): リランカー管理。 Defaults to None.
        topk_rerank_scale (int, optional): リランキング前の取得倍率。 Defaults to 5.

    Returns:
        list[Document]: 検索結果のリスト
    """
    logger.debug("trace")

    space_key = embed.space_key_text()
    store.load_store_by_space_key(space_key, embed.get_embeddings())

    qvec = embed.embed_query(query)
    topk_scaled = topk * max(1, topk_rerank_scale)
    docs = store.query(query_vec=qvec, topk=topk_scaled, space_key=space_key)

    if len(docs) == 0:
        logger.warning("empty docs")
        return []

    if rerank is None:
        return docs[: min(topk, len(docs))]

    scaled_len = len(docs)
    docs = rerank.rerank(docs=docs, query=query)[: min(scaled_len, topk)]
    logger.info(f"finished reranking. {scaled_len} --[ rerank ]--> {len(docs)} docs")

    return docs


def query_text_multi(
    query: str,
    store: VectorStoreManager,
    embed: MultimodalEmbeddingsManager,
    topk: int = 10,
    rerank: Optional[RerankManager] = None,
    topk_rerank_scale: int = 5,
) -> list[Document]:
    """クエリ文字列によるマルチモーダルドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (VectorStoreManager): ベクトルストア
        embed (MultimodalEmbeddingsManager): 埋め込み管理
        topk (int, optional): 取得件数。 Defaults to 10.
        rerank (Optional[RerankManager], optional): リランカー管理。 Defaults to None.
        topk_rerank_scale (int, optional): リランキング前の取得倍率。 Defaults to 5.

    Returns:
        list[Document]: 検索結果のリスト
    """
    logger.debug("trace")

    space_key_multi = embed.space_key_multi()
    store.load_store_by_space_key(space_key_multi, embed.get_embeddings())

    qvec = embed.embed_text_for_image_query(query)
    topk_scaled = topk * max(1, topk_rerank_scale)
    docs = store.query(
        query_vec=qvec,
        topk=topk_scaled,
        space_key=space_key_multi,
    )

    if len(docs) == 0:
        logger.warning("empty docs")
        return []

    if rerank is None:
        return docs[: min(topk, len(docs))]

    docs = _review_page_content(docs)
    scaled_len = len(docs)
    docs = rerank.rerank(docs=docs, query=query)[: min(scaled_len, topk)]
    logger.info(f"finished reranking. {scaled_len} --[ rerank ]--> {len(docs)} docs")

    return docs


def query_image(
    path: str,
    store: VectorStoreManager,
    embed: MultimodalEmbeddingsManager,
    topk: int = 10,
) -> list[Document]:
    """クエリ画像によるマルチモーダルドキュメント検索。

    Args:
        path (str): クエリ画像の ローカルパス
        store (VectorStoreManager): ベクトルストア
        embed (MultimodalEmbeddingsManager): 埋め込み管理
        topk (int, optional): 取得件数。 Defaults to 10.

    Returns:
        list[Document]: 検索結果のリスト

    Raises:
        RuntimeError: ストア操作や検索処理に失敗した場合
    """
    logger.debug("trace")

    space_key_multi = embed.space_key_multi()
    try:
        store.load_store_by_space_key(space_key_multi, embed.get_embeddings())
    except Exception as e:
        raise RuntimeError("failed to load store for multimodal space") from e

    qvec = embed.embed_image([path])
    if not qvec or not qvec[0]:
        logger.warning("empty image embedding")
        return []

    try:
        docs = store.query(query_vec=qvec[0], topk=topk, space_key=space_key_multi)
    except Exception as e:
        raise RuntimeError("failed to query store") from e
    docs = _review_page_content(docs)

    return docs
