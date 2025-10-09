from __future__ import annotations

from llama_index.core.schema import BaseNode, ImageNode, TextNode

from ragserver.ingest.file_loader import FileLoader
from ragserver.ingest.html_loader import HTMLLoader
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager

__all__ = [
    "ingest_from_path",
    "ingest_from_path_list",
    "ingest_from_url",
    "ingest_from_url_list",
]


def _split_nodes_modality(
    nodes: list[BaseNode],
) -> tuple[list[TextNode], list[ImageNode]]:
    """ノードをテキスト用と画像用に分ける。

    Args:
        nodes (list[BaseNode]): テキストノードまたは画像ノード

    Returns:
        tuple[list[TextNode], list[ImageNode]]: テキストノード、画像ノード
    """
    logger.debug("trace")

    text_nodes = []
    image_nodes = []
    for node in nodes:
        if isinstance(node, TextNode):
            text_nodes.append(node)
        elif isinstance(node, ImageNode):
            image_nodes.append(node)
        else:
            logger.warning(f"unexpected node type {type(node)}, skipped")

    return text_nodes, image_nodes


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
    """
    logger.debug("trace")

    nodes = await file_loader.load_from_path(path)
    text_nodes, image_nodes = _split_nodes_modality(nodes)
    await store.upsert_text(text_nodes)
    await store.upsert_image(image_nodes)


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
    """
    logger.debug("trace")

    nodes = await file_loader.load_from_path_list(list_path)
    text_nodes, image_nodes = _split_nodes_modality(nodes)
    await store.upsert_text(text_nodes)
    await store.upsert_image(image_nodes)


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
    """
    logger.debug("trace")

    nodes = await html_loader.load_from_url(url)
    text_nodes, image_nodes = _split_nodes_modality(nodes)
    await store.upsert_text(text_nodes)
    await store.upsert_image(image_nodes)


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
    """
    logger.debug("trace")

    nodes = await html_loader.load_from_url_list(list_path)
    text_nodes, image_nodes = _split_nodes_modality(nodes)
    await store.upsert_text(text_nodes)
    await store.upsert_image(image_nodes)
