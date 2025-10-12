from __future__ import annotations

from llama_index.vector_stores.postgres import PGVectorStore

from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.names import PGVECTOR_STORE_NAME, PROJECT_NAME
from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.logger import logger
from ragserver.stractured_store.structured_store_manager import StructuredStoreManager
from ragserver.vector_store.vector_store_manager import VectorStoreManager


class PgVectorManager(VectorStoreManager):
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        check_update: bool = True,
        knowledgebase_name: str = "default",
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
            check_update (bool, optional): ファイルの更新チェック要否。Defaults to True.
            knowledgebase_name (str, optional): ナレッジベース（用途）名。Defaults to "default".
        """
        logger.debug("trace")

        super().__init__(check_update)
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self._knowledgebase_name = knowledgebase_name

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return PGVECTOR_STORE_NAME

    def prepare_with(
        self, embed: EmbeddingManager, meta_store: StructuredStoreManager, limit: int
    ) -> None:
        """埋め込み管理に合わせてストアを初期化する。

        Args:
            embed (EmbeddingManager): 埋め込み管理
            meta_store (StructuredStoreManager): メタデータ管理
            limit (int): メタデータ読み込み件数上限

        Raises:
            RuntimeError: ストア初期化失敗
        """
        logger.debug("trace")

        self._embed = embed
        self._meta_store = meta_store

        try:
            self._text_store = PGVectorStore.from_params(
                host=self._host,
                port=str(self._port),
                database=self._dbname,
                user=self._user,
                password=self._password,
                table_name=f"{PROJECT_NAME}_{self._knowledgebase_name}_{embed.space_key_text}",
            )

            if isinstance(embed, MultiModalEmbeddingManager):
                self._image_store = PGVectorStore.from_params(
                    host=self._host,
                    port=str(self._port),
                    database=self._dbname,
                    user=self._user,
                    password=self._password,
                    table_name=f"{PROJECT_NAME}_{self._knowledgebase_name}_{embed.space_key_multi}",
                )
                self._index = self._create_index(
                    text_store=self._text_store,
                    image_store=self._image_store,
                )
                self._meta_store.prepare_with(
                    space_key_text=embed.space_key_text,
                    space_key_multi=embed.space_key_multi,
                )
            else:
                self._index = self._create_index(text_store=self._text_store)
                self._meta_store.prepare_with(
                    space_key_text=embed.space_key_text,
                )

            # メタデータ専用ストアから fingerprint キャッシュを復元
            self._fp_cache = self._load_fp_cache(limit)
        except Exception as e:
            raise RuntimeError("failed to initialize stores") from e
