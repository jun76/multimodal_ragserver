from __future__ import annotations

from llama_index.vector_stores.postgres import PGVectorStore

from ragserver.core.names import PGVECTOR_STORE_NAME, PROJECT_NAME
from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


class PgVectorManager(VectorStoreManager):
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        embed: EmbeddingManager,
        check_update: bool = True,
        table_prefix: str = f"{PROJECT_NAME}__",
    ) -> None:
        """PgVector 管理クラス

        Raises:
            RuntimeError: ストア生成失敗

        Args:
            host (str): 接続先ホスト名
            port (int): 接続先ポート番号
            dbname (str): 接続先データベース名
            user (str): 接続ユーザー名
            password (str): 接続パスワード
            embed (EmbeddingManager): 埋め込み管理
            check_update (bool, optional): ファイルの更新チェック要否。Defaults to True.
            table_prefix (str, optional): 各 space テーブルの接頭辞。Defaults to f"{PROJECT_NAME}__".
        """
        logger.debug("trace")

        VectorStoreManager.__init__(
            self,
            embed=embed,
            check_update=check_update,
        )

        try:
            text_store = PGVectorStore.from_params(
                host=host,
                port=str(port),
                database=dbname,
                user=user,
                password=password,
                table_name=table_prefix + embed.space_key_text,
            )

            if isinstance(embed, MultiModalEmbeddingManager):
                image_store = PGVectorStore.from_params(
                    host=host,
                    port=str(port),
                    database=dbname,
                    user=user,
                    password=password,
                    table_name=table_prefix + embed.space_key_multi,
                )
                self._create_index(text_store, image_store)
            else:
                self._create_index(text_store)

        except Exception as e:
            raise RuntimeError("failed to create stores") from e

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        logger.debug("trace")

        return PGVECTOR_STORE_NAME
