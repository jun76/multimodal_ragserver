from __future__ import annotations

from abc import abstractmethod
from types import CoroutineType
from typing import Any

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType

from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.logger import logger


class MultiModalEmbeddingManager(EmbeddingManager):
    def __init__(self, model_text: str, model_image: str) -> None:
        """埋め込み管理の抽象（マルチモーダル）

        Args:
            model_text (str): テキスト埋め込みモデル名
            model_image (str): 画像埋め込みモデル名
        """
        logger.debug("trace")

        EmbeddingManager.__init__(self, model_text=model_text)
        self._model_image = model_image

    @property
    @abstractmethod
    def embedding_multi(self) -> MultiModalEmbedding:
        """マルチモーダル対応の埋め込みモデル。

        Returns:
            MultiModalEmbedding: 埋め込みモデル
        """

    @property
    @abstractmethod
    def space_key_multi(self) -> str:
        """画像（インデックス用）ベクトルの空間キー。

        Returns:
            str: この埋め込み実装が生成する画像用ベクトルの空間キー
        """

    async def embed_image(self, paths: list[ImageType]) -> list[Embedding]:
        """画像の埋め込みベクトルを取得する。

        Args:
            paths (list[ImageType]): 画像のパス（または base64 画像の直渡しでも OK）

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        logger.debug("trace")

        return await self.embedding_multi.aget_image_embedding_batch(
            img_file_paths=paths
        )
