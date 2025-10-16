import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.postgres import PGVectorStore

from ragserver.config.general_config import GeneralConfig
from ragserver.config.settings import VectorStoreProvider
from ragserver.config.vector_store_config import VectorStoreConfig
from ragserver.core.metadata import Modality
from ragserver.embed.embedding_modality_manager import EmbeddingModalityManager
from ragserver.logger import logger
from ragserver.structured_store.structured_store_abst import StructuredStoreAbst
from ragserver.vector_store.vector_store_modality_manager import (
    VectorStoreContainer,
    VectorStoreModalityManager,
)


class VectorStoreProviderManager:
    """ベクトルストアプロバイダの管理クラス。"""

    def create(
        self,
        embed: EmbeddingModalityManager,
        meta_store: StructuredStoreAbst,
    ) -> VectorStoreModalityManager:
        """ベクトルストアのインスタンスを生成する。

        Args:
            embed (EmbeddingManager): 埋め込み管理
            meta_store (StructuredStoreManager): メタデータ管理
            knowledgebase_name (str, optional): ナレッジベース（用途）名。Defaults to "default".

        Raises:
            RuntimeError: インスタンス生成に失敗

        Returns:
            VectorStoreManager: ベクトルストア
        """
        logger.debug("trace")

        try:
            stores = []
            table_name_text = self._generate_table_name(embed.space_key_text)
            match GeneralConfig.vector_store:
                case VectorStoreProvider.PGVECTOR:
                    text_store = self._pgvector(
                        table_name=table_name_text, modality=Modality.TEXT
                    )
                case VectorStoreProvider.CHROMA:
                    text_store = self._chroma(
                        table_name=table_name_text, modality=Modality.TEXT
                    )
                case _:
                    raise RuntimeError(
                        f"unsupported vector store: {VectorStoreConfig.vector_store}"
                    )
            stores.append(text_store)

            if GeneralConfig.image_embed_provider:
                table_name_image = self._generate_table_name(embed.space_key_image)
                match GeneralConfig.vector_store:
                    case VectorStoreProvider.PGVECTOR:
                        image_store = self._pgvector(
                            table_name=table_name_image, modality=Modality.IMAGE
                        )
                    case VectorStoreProvider.CHROMA:
                        image_store = self._chroma(
                            table_name=table_name_image, modality=Modality.IMAGE
                        )
                    case _:
                        raise RuntimeError(
                            f"unsupported vector store: {VectorStoreConfig.vector_store}"
                        )
                stores.append(image_store)
        except Exception as e:
            raise RuntimeError(f"failed to prepare vector store: {e}") from e

        return VectorStoreModalityManager(
            stores=stores,
            embed=embed,
            meta_store=meta_store,
            load_limit=VectorStoreConfig.load_limit,
            check_update=VectorStoreConfig.check_update,
        )

    def _generate_table_name(self, space_key: str) -> str:
        """テーブル名を生成する。

        Args:
            space_key (str): 空間キー

        Returns:
            str: テーブル名
        """
        logger.debug("trace")

        return f"{GeneralConfig.project_name}__{GeneralConfig.knowledgebase_name}__{space_key}"

    def _pgvector(self, table_name: str, modality: Modality) -> VectorStoreContainer:
        """ベクトルストアコンテナ生成ヘルパー

        Args:
            table_name (str): テーブル名
            modality (Modality): モダリティ

        Returns:
            EmbeddingContainer: コンテナ
        """
        logger.debug("trace")

        return VectorStoreContainer(
            modality=modality,
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

    def _chroma(self, table_name: str, modality: Modality) -> VectorStoreContainer:
        """ベクトルストアコンテナ生成ヘルパー

        Args:
            table_name (str): テーブル名
            modality (Modality): モダリティ

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
            client = chromadb.PersistentClient(
                path=VectorStoreConfig.chroma_persist_dir
            )
        else:
            raise RuntimeError("persist_directory or host + port must be specified")

        collection = client.get_or_create_collection(table_name)

        return VectorStoreContainer(
            modality=modality,
            provider_name=VectorStoreProvider.PGVECTOR,
            store=ChromaVectorStore(chroma_collection=collection),
            table_name=table_name,
        )
