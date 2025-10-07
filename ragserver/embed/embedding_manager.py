from __future__ import annotations

from abc import ABC, abstractmethod
from types import CoroutineType
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

from ragserver.logger import logger


class EmbeddingManager(ABC):
    def __init__(self, model_text: str) -> None:
        """埋め込み管理の抽象インターフェース（テキスト）

        Args:
            name (str): プロバイダ名
            model_text (str): テキスト埋め込みモデル名
        """
        logger.debug("trace")

        self._model_text = model_text

    @abstractmethod
    def get_embedding(self) -> BaseEmbedding:
        """埋め込みモデルを返す。

        Returns:
            BaseEmbedding: 埋め込みモデル
        """
        logger.debug("trace")
        ...

    @abstractmethod
    def space_key_text(self) -> str:
        """テキスト文書（インデックス用）ベクトルの空間キー。

        Returns:
            str: この埋め込み実装が生成する文書用ベクトルの空間キー
        """
        logger.debug("trace")
        ...

    async def embed_text(self, text: str) -> CoroutineType[Any, Any, Embedding]:
        """テキストの埋め込みベクトルを取得する。

        Args:
            text (str): テキスト

        Returns:
            Embedding: 埋め込みベクトル
        """
        logger.debug("trace")

        return self.get_embedding().aget_text_embedding(text)
