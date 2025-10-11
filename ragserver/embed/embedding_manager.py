from __future__ import annotations

from abc import ABC, abstractmethod
from types import CoroutineType
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

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
    @abstractmethod
    def space_key_text(self) -> str:
        """テキスト文書（インデックス用）ベクトルの空間キー。

        Returns:
            str: この埋め込み実装が生成する文書用ベクトルの空間キー
        """

    async def embed_text(self, texts: list[str]) -> list[Embedding]:
        """テキストの埋め込みベクトルを取得する。

        Args:
            texts (list[str]): テキスト

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        logger.debug("trace")

        return await self.embedding.aget_text_embedding_batch(texts)
