from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.embeddings.clip import ClipEmbedding

from ragserver.core.names import CLIP_EMBED_NAME
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.logger import logger


class ClipEmbeddingManager(MultiModalEmbeddingManager):
    def __init__(
        self,
        model_text: str,
        model_image: str,
    ) -> None:
        """ローカル CLIP の埋め込みモデル管理クラス

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
        self._embed_text = ClipEmbedding(model_name=model_text)
        self._embed_image = ClipEmbedding(model_name=model_image)

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return CLIP_EMBED_NAME

    @property
    def embedding(self) -> BaseEmbedding:
        """埋め込みモデル。

        Returns:
            BaseEmbedding: 埋め込みモデル
        """
        return self._embed_text

    @property
    def embedding_multi(self) -> MultiModalEmbedding:
        """マルチモーダル対応の埋め込みモデル。

        Returns:
            MultiModalEmbedding: 埋め込みモデル
        """
        return self._embed_image
