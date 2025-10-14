from __future__ import annotations

from typing import Optional

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from ragserver.core.names import CHROMA_STORE_NAME, PROJECT_NAME
from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.logger import logger
from ragserver.structured_store.structured_store_manager import StructuredStoreManager
from ragserver.vector_store.vector_store_manager import VectorStoreManager


class ChromaManager(VectorStoreManager):
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        check_update: bool = True,
        knowledgebase_name: str = "default",
    ) -> None:
        """Chroma 管理クラス

        Args:
            persist_directory (Optional[str], optional): ローカル利用時の保存先ディレクトリ。Defaults to None.
            host (Optional[str], optional): リモートサーバ利用時のホスト。Defaults to None.
            port (Optional[int], optional): リモートサーバ利用時のポート番号。Defaults to None.
            check_update (bool, optional): ファイルの更新チェック要否。Defaults to True.
            knowledgebase_name (str, optional): ナレッジベース（用途）名。Defaults to "default".

        Raises:
            RuntimeError: Chroma クライアント生成失敗
        """
        logger.debug("trace")

        super().__init__(check_update)
        self._knowledgebase_name = knowledgebase_name

        try:
            if host is not None and port is not None:
                # リモートモード
                client = chromadb.HttpClient(
                    host=host,
                    port=port,
                )
            elif persist_directory:
                # ローカルモード
                client = chromadb.PersistentClient(path=persist_directory)
            else:
                raise RuntimeError("persist_directory or host + port must be specified")

            self._client = client
        except Exception as e:
            raise RuntimeError("failed to create chroma client") from e

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return CHROMA_STORE_NAME

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
            text_collection = self._client.get_or_create_collection(
                name=f"{PROJECT_NAME}_{self._knowledgebase_name}_{embed.space_key_text}"
            )
            self._text_store = ChromaVectorStore(chroma_collection=text_collection)

            if isinstance(embed, MultiModalEmbeddingManager):
                image_collection = self._client.get_or_create_collection(
                    name=f"{PROJECT_NAME}_{self._knowledgebase_name}_{embed.space_key_multi}"
                )
                self._image_store = ChromaVectorStore(
                    chroma_collection=image_collection
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
