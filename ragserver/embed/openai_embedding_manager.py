from __future__ import annotations

from typing import Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai.base import OpenAIEmbedding

from ragserver.core.names import OPENAI_EMBED_NAME
from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.embed.util import EMBTYPE_TEXT, generate_space_key
from ragserver.logger import logger


class OpenAIEmbeddingManager(EmbeddingManager):
    def __init__(
        self,
        model_text: str,
        base_url: Optional[str] = None,
    ) -> None:
        """OpenAI の埋め込みモデル管理クラス

        Args:
            model_text (str, optional): テキスト埋め込みモデル名
            base_url (Optional[str], optional): ローカルモデルのエンドポイント。Defaults to None.
        """
        logger.debug("trace")

        super().__init__(model_text)
        self._embed = OpenAIEmbedding(
            model=model_text,
            api_base=base_url,
        )

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        logger.debug("trace")

        return OPENAI_EMBED_NAME

    @property
    def embedding(self) -> BaseEmbedding:
        """埋め込みモデル。

        Returns:
            BaseEmbedding: 埋め込みモデル
        """
        logger.debug("trace")

        return self._embed

    @property
    def space_key_text(self) -> str:
        """OpenAI テキストベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        logger.debug("trace")

        return generate_space_key(OPENAI_EMBED_NAME, self._model_text, EMBTYPE_TEXT)
