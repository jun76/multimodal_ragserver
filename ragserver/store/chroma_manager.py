from __future__ import annotations

from typing import Optional

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from ragserver.core.names import CHROMA_STORE_NAME
from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


class ChromaManager(VectorStoreManager):
    def __init__(
        self,
        embed: EmbeddingManager,
        check_update: bool = True,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """Chroma 管理クラス

        Raises:
            RuntimeError: ストア生成失敗

        Args:
            embed (EmbeddingManager): 埋め込み管理
            check_update (bool, optional): ファイルの更新チェック要否。Defaults to True.
            persist_directory (Optional[str], optional): ローカル利用時の保存先ディレクトリ。Defaults to None.
            host (Optional[str], optional): リモートサーバ利用時のホスト。Defaults to None.
            port (Optional[int], optional): リモートサーバ利用時のポート番号。Defaults to None.
        """
        logger.debug("trace")

        VectorStoreManager.__init__(
            self,
            embed=embed,
            check_update=check_update,
        )

        try:
            if host and port:
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

            text_collection = client.get_or_create_collection(embed.space_key_text)
            text_store = ChromaVectorStore(chroma_collection=text_collection)

            if isinstance(embed, MultiModalEmbeddingManager):
                image_collection = client.get_or_create_collection(
                    embed.space_key_multi
                )
                image_store = ChromaVectorStore(chroma_collection=image_collection)
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

        return CHROMA_STORE_NAME
