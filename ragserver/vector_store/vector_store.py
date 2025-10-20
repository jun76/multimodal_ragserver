from __future__ import annotations

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.postgres import PGVectorStore

from ragserver.config.general_config import GeneralConfig
from ragserver.config.settings import VectorStoreProvider
from ragserver.config.vector_store_config import VectorStoreConfig
from ragserver.embed.embed_manager import EmbedManager
from ragserver.llama.core.schema import Modality
from ragserver.logger import logger
from ragserver.meta_store.structured.structured import Structured
from ragserver.vector_store.vector_store_manager import (
    VectorStoreContainer,
    VectorStoreManager,
)

__all__ = ["create_vector_store_manager"]


def create_vector_store_manager(
    embed: EmbedManager,
    meta_store: Structured,
) -> VectorStoreManager:
    """ベクトルストアのインスタンスを生成する。

    Args:
        embed (EmbedManager): 埋め込み管理
        meta_store (Structured): メタデータ管理

    Raises:
        RuntimeError: インスタンス生成に失敗

    Returns:
        VectorStoreManager: ベクトルストア
    """

    try:
        conts: dict[Modality, VectorStoreContainer] = {}
        if GeneralConfig.text_embed_provider:
            conts[Modality.TEXT] = _create_container(embed.space_key_text)

        if GeneralConfig.image_embed_provider:
            conts[Modality.IMAGE] = _create_container(embed.space_key_image)

        if GeneralConfig.audio_embed_provider:
            conts[Modality.AUDIO] = _create_container(embed.space_key_audio)
    except Exception as e:
        raise RuntimeError(f"failed to create vector store: {e}") from e

    if not conts:
        raise RuntimeError("no embedding providers are specified")

    return VectorStoreManager(
        conts=conts,
        embed=embed,
        meta_store=meta_store,
        cache_load_limit=VectorStoreConfig.cache_load_limit,
        check_update=VectorStoreConfig.check_update,
    )


def _create_container(space_key: str) -> VectorStoreContainer:
    """空間キー毎のベクトルストアコンテナを生成する。

    Args:
        space_key (str): 空間キー

    Raises:
        RuntimeError: サポート外のプロバイダ

    Returns:
        VectorStoreContainer: コンテナ
    """

    table_name = _generate_table_name(space_key)
    match GeneralConfig.vector_store_provider:
        case VectorStoreProvider.PGVECTOR:
            cont = _pgvector(table_name)
        case VectorStoreProvider.CHROMA:
            cont = _chroma(table_name)
        case _:
            raise RuntimeError(
                f"unsupported vector store: {GeneralConfig.vector_store_provider}"
            )

    return cont


def _generate_table_name(space_key: str) -> str:
    """テーブル名を生成する。

    Args:
        space_key (str): 空間キー

    Returns:
        str: テーブル名
    """

    return (
        f"{GeneralConfig.project_name}__{GeneralConfig.knowledgebase_name}__{space_key}"
    )


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _pgvector(table_name: str) -> VectorStoreContainer:
    """ベクトルストアコンテナ生成ヘルパー

    Args:
        table_name (str): テーブル名

    Raise:
        ValueError: パスワード未指定

    Returns:
        VectorStoreContainer: コンテナ
    """

    sec = VectorStoreConfig.pgvector_password
    if sec is None:
        raise ValueError("pgvector_password must be specified")

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.PGVECTOR,
        store=PGVectorStore.from_params(
            host=VectorStoreConfig.pgvector_host,
            port=str(VectorStoreConfig.pgvector_port),
            database=VectorStoreConfig.pgvector_database,
            user=VectorStoreConfig.pgvector_user,
            password=sec.get_secret_value(),
            table_name=table_name,
        ),
        table_name=table_name,
    )


def _chroma(table_name: str) -> VectorStoreContainer:
    """ベクトルストアコンテナ生成ヘルパー

    Args:
        table_name (str): テーブル名

    Raises:
        RuntimeError: パラメータ指定漏れ

    Returns:
        VectorStoreContainer: コンテナ
    """

    if (
        VectorStoreConfig.chroma_host is not None
        and VectorStoreConfig.chroma_port is not None
    ):
        client = chromadb.HttpClient(
            host=VectorStoreConfig.chroma_host,
            port=VectorStoreConfig.chroma_port,
        )
    elif VectorStoreConfig.chroma_persist_dir:
        client = chromadb.PersistentClient(path=VectorStoreConfig.chroma_persist_dir)
    else:
        raise RuntimeError("persist_directory or host + port must be specified")

    collection = client.get_or_create_collection(table_name)

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.CHROMA,
        store=ChromaVectorStore(chroma_collection=collection),
        table_name=table_name,
    )
