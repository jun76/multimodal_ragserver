from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.cohere.base import CohereEmbedding
from llama_index.embeddings.openai.base import OpenAIEmbedding

from ragserver.config.embed_config import EmbedConfig
from ragserver.config.general_config import GeneralConfig
from ragserver.config.settings import EmbedProvider
from ragserver.core.metadata import Modality
from ragserver.embed.embedding_modality_manager import (
    EmbeddingContainer,
    EmbeddingModalityManager,
)
from ragserver.logger import logger


class EmbeddingProviderManager:
    """埋め込みプロバイダの管理クラス。"""

    def create(self) -> EmbeddingModalityManager:
        """埋め込み管理インスタンスを生成する。

        Raises:
            RuntimeError: インスタンス生成に失敗

        Returns:
            EmbeddingsManager: 埋め込み管理
        """
        logger.debug("trace")

        try:
            embeds = []
            match GeneralConfig.text_embed_provider:
                case EmbedProvider.OPENAI:
                    embed_text = self._openai_text()
                case EmbedProvider.COHERE:
                    embed_text = self._cohere_text()
                case EmbedProvider.CLIP:
                    embed_text = self._clip_text()
                case _:
                    raise ValueError(
                        f"unsupported text embed provider: {EmbedConfig.text_embed_provider}"
                    )
            embeds.append(embed_text)

            if GeneralConfig.image_embed_provider:
                match GeneralConfig.image_embed_provider:
                    case EmbedProvider.COHERE:
                        embed_image = self._cohere_image()
                    case EmbedProvider.CLIP:
                        embed_image = self._clip_image()
                    case _:
                        raise ValueError(
                            f"unsupported image embed provider: {GeneralConfig.image_embed_provider}"
                        )
                embeds.append(embed_image)
        except Exception as e:
            raise RuntimeError(f"failed to prepare embedding: {e}") from e

        return EmbeddingModalityManager(embeds)

    def _openai_text(self) -> EmbeddingContainer:
        """埋め込みコンテナ生成ヘルパー

        Returns:
            EmbeddingContainer: コンテナ
        """
        logger.debug("trace")

        return EmbeddingContainer(
            modality=Modality.TEXT,
            provider_name=EmbedProvider.OPENAI,
            embedding=OpenAIEmbedding(
                model=EmbedConfig.openai_embed_model_text,
                api_base=EmbedConfig.openai_base_url,
            ),
        )

    def _cohere_text(self) -> EmbeddingContainer:
        """埋め込みコンテナ生成ヘルパー

        Returns:
            EmbeddingContainer: コンテナ
        """
        logger.debug("trace")

        return EmbeddingContainer(
            modality=Modality.TEXT,
            provider_name=EmbedProvider.COHERE,
            embedding=CohereEmbedding(
                model_name=EmbedConfig.cohere_embed_model_text,
            ),
        )

    def _cohere_image(self) -> EmbeddingContainer:
        """埋め込みコンテナ生成ヘルパー

        Returns:
            EmbeddingContainer: コンテナ
        """
        logger.debug("trace")

        return EmbeddingContainer(
            modality=Modality.IMAGE,
            provider_name=EmbedProvider.COHERE,
            embedding=CohereEmbedding(
                model_name=EmbedConfig.cohere_embed_model_image,
            ),
        )

    def _clip_text(self) -> EmbeddingContainer:
        """埋め込みコンテナ生成ヘルパー

        Returns:
            EmbeddingContainer: コンテナ
        """
        logger.debug("trace")

        return EmbeddingContainer(
            modality=Modality.TEXT,
            provider_name=EmbedProvider.CLIP,
            embedding=ClipEmbedding(
                model_name=EmbedConfig.clip_embed_model_text,
            ),
        )

    def _clip_image(self) -> EmbeddingContainer:
        """埋め込みコンテナ生成ヘルパー

        Returns:
            EmbeddingContainer: コンテナ
        """
        logger.debug("trace")

        return EmbeddingContainer(
            modality=Modality.IMAGE,
            provider_name=EmbedProvider.CLIP,
            embedding=ClipEmbedding(
                model_name=EmbedConfig.clip_embed_model_image,
            ),
        )
