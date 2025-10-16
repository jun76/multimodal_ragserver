from __future__ import annotations

from ragserver.ingest.file_loader import FileLoader
from ragserver.ingest.html_loader import HTMLLoader
from ragserver.logger import logger
from ragserver.vector_store.vector_store_modality_manager import (
    VectorStoreModalityManager,
)

__all__ = [
    "aingest_from_path",
    "aingest_from_path_list",
    "aingest_from_url",
    "aingest_from_url_list",
]


async def aingest_from_path(
    path: str,
    store: VectorStoreModalityManager,
    file_loader: FileLoader,
) -> None:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        store (VectorStoreManager): ベクトルストア
        file_loader (FileLoader): ファイル読み込み用
    """
    logger.debug("trace")

    nodes = await file_loader.aload_from_path(path)
    await store.aupsert_nodes(nodes)


async def aingest_from_path_list(
    list_path: str,
    store: VectorStoreModalityManager,
    file_loader: FileLoader,
) -> None:
    """path リストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）
        store (VectorStoreManager): ベクトルストア
        file_loader (FileLoader): ファイル読み込み用
    """
    logger.debug("trace")

    nodes = await file_loader.aload_from_path_list(list_path)
    await store.aupsert_nodes(nodes)


async def aingest_from_url(
    url: str,
    store: VectorStoreModalityManager,
    html_loader: HTMLLoader,
) -> None:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        store (VectorStoreManager): ベクトルストア
        html_loader (HTMLLoader): HTML 読み込み用
    """
    logger.debug("trace")

    nodes = await html_loader.aload_from_url(url)
    await store.aupsert_nodes(nodes)


async def aingest_from_url_list(
    list_path: str,
    store: VectorStoreModalityManager,
    html_loader: HTMLLoader,
) -> None:
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）
        store (VectorStoreManager): ベクトルストア
        html_loader (HTMLLoader): HTML 読み込み用
    """
    logger.debug("trace")

    nodes = await html_loader.aload_from_url_list(list_path)
    await store.aupsert_nodes(nodes)
