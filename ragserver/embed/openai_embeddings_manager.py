from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ragserver.core.metadata import EMBTYPE_TEXT
from ragserver.core.names import OPENAI_EMBED_NAME
from ragserver.embed.embeddings_manager import EmbeddingsManager
from ragserver.embed.util import generate_space_key
from ragserver.logger import logger


class OpenAIEmbeddingsManager(EmbeddingsManager):
    def __init__(
        self,
        model_text: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        need_norm: bool = True,
    ) -> None:
        """OpenAI の埋め込みモデル管理クラス

        Args:
            model_text (str, optional): テキスト埋め込みモデル名
            base_url (Optional[str], optional): ローカルモデルのエンドポイント。 Defaults to None.
            api_key (Optional[str], optional): API キー。 Defaults to None.
            need_norm (bool, optional): L2 正規化要否。 Defaults to True.
        """
        logger.debug("trace")

        EmbeddingsManager.__init__(
            self, name="openai", model_text=model_text, need_norm=need_norm
        )
        self._embed = OpenAIEmbeddings(
            check_embedding_ctx_length=False,
            model=model_text,
            base_url=base_url,
            api_key=api_key,  # type: ignore
        )

    def get_embeddings(self) -> Embeddings:
        """埋め込みモデルを返す。

        Returns:
            Embeddings: 埋め込みモデル
        """
        logger.debug("trace")

        return self._embed

    def space_key_text(self) -> str:
        """OpenAI テキスト文書用ベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        logger.debug("trace")

        return generate_space_key(OPENAI_EMBED_NAME, self._model_text, EMBTYPE_TEXT)
