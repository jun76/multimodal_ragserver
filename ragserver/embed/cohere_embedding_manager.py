from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.embeddings.cohere.base import CohereEmbedding

from ragserver.core.metadata import EMBTYPE_IMAGE, EMBTYPE_TEXT
from ragserver.core.names import COHERE_EMBED_NAME
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.embed.util import generate_space_key
from ragserver.logger import logger


class CohereEmbeddingManager(MultiModalEmbeddingManager):
    def __init__(
        self,
        model_text: str,
        model_image: str,
    ) -> None:
        """Cohere の埋め込みモデル管理クラス

        Args:
            model_text (str): テキスト埋め込みモデル名
            model_image (str): 画像埋め込みモデル名
        """
        logger.debug("trace")

        MultiModalEmbeddingManager.__init__(
            self,
            model_text=model_text,
            model_image=model_image,
        )
        self._embed_text = CohereEmbedding(model_name=model_text)
        self._embed_image = CohereEmbedding(model_name=model_image)

    def get_embedding(self) -> BaseEmbedding:
        """埋め込みモデルを返す。

        Returns:
            BaseEmbedding: 埋め込みモデル
        """
        logger.debug("trace")

        return self._embed_text

    def get_embedding_multi(self) -> MultiModalEmbedding:
        """マルチモーダル対応の埋め込みモデルを返す。

        Returns:
            MultiModalEmbedding: 埋め込みモデル
        """
        logger.debug("trace")

        return self._embed_image

    def space_key_text(self) -> str:
        """Cohere テキストベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        logger.debug("trace")

        return generate_space_key(COHERE_EMBED_NAME, self._model_text, EMBTYPE_TEXT)

    def space_key_multi(self) -> str:
        """Cohere 画像ベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        logger.debug("trace")

        return generate_space_key(COHERE_EMBED_NAME, self._model_image, EMBTYPE_IMAGE)
