from __future__ import annotations

from llama_index.core.schema import ImageNode, TextNode

from ragserver.core.metadata import META_KEYS as MK
from ragserver.ingest.file_loader import FileLoader
from ragserver.ingest.html_loader import HTMLLoader
from ragserver.ingest.loader import Exts
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


def _split_nodes_modality(
    nodes: list[TextNode],
) -> tuple[list[TextNode], list[ImageNode]]:
    """ノードをテキスト用と画像用に分ける。

    Args:
        nodes (list[TextNode]): テキストノード（画像パス、URL 含む）

    Returns:
        tuple[list[TextNode], list[ImageNode]]: テキストノード、画像ノード
    """
    logger.debug("trace")

    text_nodes = []
    image_nodes = []
    for node in nodes:
        if _has_image_source(node):
            image_nodes.append(ImageNode(text=node.text, metadata=node.metadata))
        else:
            text_nodes.append(node)

    return text_nodes, image_nodes


def _has_image_source(node: TextNode) -> bool:
    """ノードが画像のファイルパスや URL を持っているか。

    Args:
        node (TextNode): 対象ノード

    Returns:
        bool: 持っていれば True
    """
    logger.debug("trace")

    meta = node.metadata
    path = (
        meta.get(MK.FILE_PATH) or meta.get(MK.TEMP_FILE_PATH) or meta.get(MK.URL) or ""
    ).lower()

    return any(path.endswith(ext) for ext in Exts.IMAGE_FILE_EXTS)


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

    nodes = await file_loader.load_from_path(path)
    await store.upsert_nodes(docs)


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
    await store.upsert_nodes(docs)


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
    await store.upsert_nodes(docs)


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
    await store.upsert_nodes(docs)
