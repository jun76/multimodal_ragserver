from __future__ import annotations

from abc import ABC, abstractmethod

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

from ragserver.embed.util import EMBTYPE_TEXT, generate_space_key
from ragserver.logger import logger


class EmbeddingManager(ABC):
    def __init__(self, model_text: str) -> None:
        """埋め込み管理の抽象（テキスト）

        Args:
            model_text (str): テキスト埋め込みモデル名
        """
        logger.debug("trace")

        self._model_text = model_text

    @property
    @abstractmethod
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """

    @property
    @abstractmethod
    def embedding(self) -> BaseEmbedding:
        """埋め込みモデル。

        Returns:
            BaseEmbedding: 埋め込みモデル
        """

    @property
    def space_key_text(self) -> str:
        """ローカル CLIP テキストベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        return generate_space_key(self.name, self._model_text, EMBTYPE_TEXT)

    async def aembed_text(self, texts: list[str]) -> list[Embedding]:
        """テキストの埋め込みベクトルを取得する。

        Args:
            texts (list[str]): テキスト

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        logger.debug("trace")

        return await self.embedding.aget_text_embedding_batch(texts)
