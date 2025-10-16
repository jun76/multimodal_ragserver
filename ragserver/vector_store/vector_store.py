import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.postgres import PGVectorStore

from ragserver.config.general_config import GeneralConfig
from ragserver.config.settings import VectorStoreProvider
from ragserver.config.vector_store_config import VectorStoreConfig
from ragserver.core.metadata import Modality
from ragserver.embed.embed_manager import EmbedManager
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
        embed (EmbeddingManager): 埋め込み管理
        meta_store (StructuredStoreManager): メタデータ管理

    Raises:
        RuntimeError: インスタンス生成に失敗

    Returns:
        VectorStoreManager: ベクトルストア
    """
    logger.debug("trace")

    try:
        conts: dict[Modality, VectorStoreContainer] = {}
        table_name_text = _generate_table_name(embed.space_key_text)
        match GeneralConfig.vector_store:
            case VectorStoreProvider.PGVECTOR:
                text_store = _pgvector(table_name_text)
            case VectorStoreProvider.CHROMA:
                text_store = _chroma(table_name_text)
            case _:
                raise RuntimeError(
                    f"unsupported vector store: {GeneralConfig.vector_store}"
                )
        conts[Modality.TEXT] = text_store

        if GeneralConfig.image_embed_provider:
            table_name_image = _generate_table_name(embed.space_key_image)
            match GeneralConfig.vector_store:
                case VectorStoreProvider.PGVECTOR:
                    image_store = _pgvector(table_name_image)
                case VectorStoreProvider.CHROMA:
                    image_store = _chroma(table_name_image)
                case _:
                    raise RuntimeError(
                        f"unsupported vector store: {GeneralConfig.vector_store}"
                    )
            conts[Modality.IMAGE] = image_store
    except Exception as e:
        raise RuntimeError(f"failed to create vector store: {e}") from e

    return VectorStoreManager(
        conts=conts,
        embed=embed,
        meta_store=meta_store,
        load_limit=VectorStoreConfig.load_limit,
        check_update=VectorStoreConfig.check_update,
    )


def _generate_table_name(space_key: str) -> str:
    """テーブル名を生成する。

    Args:
        space_key (str): 空間キー

    Returns:
        str: テーブル名
    """
    logger.debug("trace")

    return (
        f"{GeneralConfig.project_name}__{GeneralConfig.knowledgebase_name}__{space_key}"
    )


def _pgvector(table_name: str) -> VectorStoreContainer:
    """ベクトルストアコンテナ生成ヘルパー

    Args:
        table_name (str): テーブル名

    Returns:
        EmbeddingContainer: コンテナ
    """
    logger.debug("trace")

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.PGVECTOR,
        store=PGVectorStore.from_params(
            host=VectorStoreConfig.pgvector_host,
            port=str(VectorStoreConfig.pgvector_port),
            database=VectorStoreConfig.pgvector_database,
            user=VectorStoreConfig.pgvector_user,
            password=VectorStoreConfig.pgvector_password,
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
    logger.debug("trace")

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
