from __future__ import annotations

from ragserver.ingest.file_loader import FileLoader
from ragserver.ingest.html_loader import HTMLLoader
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


async def ingest_from_path(
    path: str,
    store: VectorStoreManager,
    file_loader: FileLoader,
) -> None:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        store (VectorStoreManager): ベクトルストア
        file_loader (FileLoader): ファイル読み込み用

    Raises:
        RuntimeError: ストアの初期化またはドキュメント登録に失敗した場合
    """
    logger.debug("trace")

    docs = await file_loader.load_from_path(path)
    await store.upsert_docs(docs)


async def ingest_from_path_list(
    list_path: str,
    store: VectorStoreManager,
    file_loader: FileLoader,
) -> None:
    """path リストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）
        store (VectorStoreManager): ベクトルストア
        file_loader (FileLoader): ファイル読み込み用

    Raises:
        RuntimeError: ストアの初期化またはドキュメント登録に失敗した場合
    """
    logger.debug("trace")

    docs = await file_loader.load_from_path_list(list_path)
    await store.upsert_docs(docs)


async def ingest_from_url(
    url: str,
    store: VectorStoreManager,
    html_loader: HTMLLoader,
) -> None:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        store (VectorStoreManager): ベクトルストア
        html_loader (HTMLLoader): HTML 読み込み用

    Raises:
        RuntimeError: ストアの初期化またはドキュメント登録に失敗した場合
    """
    logger.debug("trace")

    docs = await html_loader.load_from_url(url)
    await store.upsert_docs(docs)


async def ingest_from_url_list(
    list_path: str,
    store: VectorStoreManager,
    html_loader: HTMLLoader,
) -> None:
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list_path (str): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）
        store (VectorStoreManager): ベクトルストア
        html_loader (HTMLLoader): HTML 読み込み用

    Raises:
        RuntimeError: ストアの初期化またはドキュメント登録に失敗した場合
    """
    logger.debug("trace")

    docs = await html_loader.load_from_url_list(list_path)
    await store.upsert_docs(docs)
