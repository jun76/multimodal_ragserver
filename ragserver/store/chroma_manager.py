from __future__ import annotations

from typing import Optional

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


class ChromaManager(VectorStoreManager):
    def __init__(
        self,
        space_key_text: str,
        space_key_multi: str,
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
            space_key_text (str): テキストベクトルの空間キー
            space_key_multi (str): 画像ベクトルの空間キー
            embed (EmbeddingManager): 埋め込み管理
            check_update (bool, optional): ファイルの更新チェック要否。 Defaults to True.
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

            # テキスト用と画像用のコレクションを作成
            text_collection = client.get_or_create_collection(space_key_text)
            image_collection = client.get_or_create_collection(space_key_multi)

            # ストアを作成
            self._text_store = ChromaVectorStore(chroma_collection=text_collection)
            self._image_store = ChromaVectorStore(chroma_collection=image_collection)

            # インデックス（空）を作成
            self.create_index()

        except Exception as e:
            raise RuntimeError("failed to create stores") from e
