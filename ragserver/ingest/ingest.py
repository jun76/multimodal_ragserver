from __future__ import annotations

from ragserver.embed.embeddings_manager import EmbeddingsManager
from ragserver.embed.multimodal_embeddings_manager import MultimodalEmbeddingsManager
from ragserver.ingest.file_loader import FileLoader
from ragserver.ingest.html_loader import HTMLLoader
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


def ingest_from_path(
    path: str,
    store: VectorStoreManager,
    embed: EmbeddingsManager,
    file_loader: FileLoader,
) -> None:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        store (VectorStoreManager): ベクトルストア
        embed (EmbeddingsManager): 埋め込み管理
        file_loader (FileLoader): ファイル読み込み用

    Raises:
        RuntimeError: ストアの初期化またはドキュメント登録に失敗した場合
    """
    logger.debug("trace")

    space_key = embed.space_key_text()
    try:
        store.load_store_by_space_key(space_key, embed.get_embeddings())
    except Exception as e:
        raise RuntimeError("failed to load store for text space") from e

    if isinstance(embed, MultimodalEmbeddingsManager):
        space_key_multi = embed.space_key_multi()
        try:
            store.load_store_by_space_key(space_key_multi, embed.get_embeddings())
        except Exception as e:
            raise RuntimeError("failed to load store for multimodal space") from e

        text_docs, image_docs = file_loader.load_from_path(
            root=path, space_key=space_key, space_key_multi=space_key_multi
        )
        if image_docs:
            try:
                store.upsert_multi(image_docs, space_key_multi)
            except Exception as e:
                raise RuntimeError("failed to upsert multimodal documents") from e
    else:
        text_docs, _ = file_loader.load_from_path(root=path, space_key=space_key)

    if text_docs:
        try:
            store.upsert(text_docs, space_key)
        except Exception as e:
            raise RuntimeError("failed to upsert text documents") from e


def ingest_from_path_list(
    list_path: str,
    store: VectorStoreManager,
    embed: EmbeddingsManager,
    file_loader: FileLoader,
) -> None:
    """path リストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）
        store (VectorStoreManager): ベクトルストア
        embed (EmbeddingsManager): 埋め込み管理
        file_loader (FileLoader): ファイル読み込み用

    Raises:
        RuntimeError: ストアの初期化またはドキュメント登録に失敗した場合
    """
    logger.debug("trace")

    space_key = embed.space_key_text()
    try:
        store.load_store_by_space_key(space_key, embed.get_embeddings())
    except Exception as e:
        raise RuntimeError("failed to load store for text space") from e

    if isinstance(embed, MultimodalEmbeddingsManager):
        space_key_multi = embed.space_key_multi()
        try:
            store.load_store_by_space_key(space_key_multi, embed.get_embeddings())
        except Exception as e:
            raise RuntimeError("failed to load store for multimodal space") from e

        text_docs, image_docs = file_loader.load_from_path_list(
            list_path=list_path,
            space_key=space_key,
            space_key_multi=space_key_multi,
        )
        if image_docs:
            try:
                store.upsert_multi(image_docs, space_key_multi)
            except Exception as e:
                raise RuntimeError("failed to upsert multimodal documents") from e
    else:
        text_docs, _ = file_loader.load_from_path_list(
            list_path=list_path, space_key=space_key
        )

    if text_docs:
        try:
            store.upsert(text_docs, space_key)
        except Exception as e:
            raise RuntimeError("failed to upsert text documents") from e


def ingest_from_url(
    url: str,
    store: VectorStoreManager,
    embed: EmbeddingsManager,
    html_loader: HTMLLoader,
) -> None:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        store (VectorStoreManager): ベクトルストア
        embed (EmbeddingsManager): 埋め込み管理
        html_loader (HTMLLoader): HTML 読み込み用

    Raises:
        RuntimeError: ストアの初期化またはドキュメント登録に失敗した場合
    """
    logger.debug("trace")

    space_key = embed.space_key_text()
    try:
        store.load_store_by_space_key(space_key, embed.get_embeddings())
    except Exception as e:
        raise RuntimeError("failed to load store for text space") from e

    if isinstance(embed, MultimodalEmbeddingsManager):
        space_key_multi = embed.space_key_multi()
        try:
            store.load_store_by_space_key(space_key_multi, embed.get_embeddings())
        except Exception as e:
            raise RuntimeError("failed to load store for multimodal space") from e

        text_docs, image_docs = html_loader.load_from_url(
            url=url, space_key=space_key, space_key_multi=space_key_multi
        )
        if image_docs:
            try:
                store.upsert_multi(image_docs, space_key_multi)
            except Exception as e:
                raise RuntimeError("failed to upsert multimodal documents") from e
    else:
        text_docs, _ = html_loader.load_from_url(url=url, space_key=space_key)

    if text_docs:
        try:
            store.upsert(text_docs, space_key)
        except Exception as e:
            raise RuntimeError("failed to upsert text documents") from e


def ingest_from_url_list(
    list_path: str,
    store: VectorStoreManager,
    embed: EmbeddingsManager,
    html_loader: HTMLLoader,
) -> None:
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）
        store (VectorStoreManager): ベクトルストア
        embed (EmbeddingsManager): 埋め込み管理
        html_loader (HTMLLoader): HTML 読み込み用

    Raises:
        RuntimeError: ストアの初期化またはドキュメント登録に失敗した場合
    """
    logger.debug("trace")

    space_key = embed.space_key_text()
    try:
        store.load_store_by_space_key(space_key, embed.get_embeddings())
    except Exception as e:
        raise RuntimeError("failed to load store for text space") from e

    if isinstance(embed, MultimodalEmbeddingsManager):
        space_key_multi = embed.space_key_multi()
        try:
            store.load_store_by_space_key(space_key_multi, embed.get_embeddings())
        except Exception as e:
            raise RuntimeError("failed to load store for multimodal space") from e

        text_docs, image_docs = html_loader.load_from_url_list(
            list_path=list_path,
            space_key=space_key,
            space_key_multi=space_key_multi,
        )
        if image_docs:
            try:
                store.upsert_multi(image_docs, space_key_multi)
            except Exception as e:
                raise RuntimeError("failed to upsert multimodal documents") from e
    else:
        text_docs, _ = html_loader.load_from_url_list(
            list_path=list_path,
            space_key=space_key,
        )

    if text_docs:
        try:
            store.upsert(text_docs, space_key)
        except Exception as e:
            raise RuntimeError("failed to upsert text documents") from e
