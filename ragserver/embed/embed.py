from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.cohere.base import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai.base import OpenAIEmbedding

from ragserver.config.embed_config import EmbedConfig
from ragserver.config.general_config import GeneralConfig
from ragserver.config.settings import EmbedProvider
from ragserver.core.metadata import Modality
from ragserver.embed.embed_manager import EmbedContainer, EmbedManager
from ragserver.logger import logger

__all__ = ["create_embed_manager"]


def create_embed_manager() -> EmbedManager:
    """埋め込み管理インスタンスを生成する。

    Raises:
        RuntimeError: インスタンス生成に失敗

    Returns:
        EmbeddingsManager: 埋め込み管理
    """
    logger.debug("trace")

    try:
        conts: dict[Modality, EmbedContainer] = {}
        match GeneralConfig.text_embed_provider:
            case EmbedProvider.OPENAI:
                cont = _openai_text()
            case EmbedProvider.COHERE:
                cont = _cohere_text()
            case EmbedProvider.CLIP:
                cont = _clip_text()
            case EmbedProvider.HUGGINGFACE:
                cont = _huggingface_text()
            case _:
                raise ValueError(
                    f"unsupported text embed provider: {GeneralConfig.text_embed_provider}"
                )
        conts[Modality.TEXT] = cont

        if GeneralConfig.image_embed_provider:
            match GeneralConfig.image_embed_provider:
                case EmbedProvider.COHERE:
                    cont = _cohere_image()
                case EmbedProvider.CLIP:
                    cont = _clip_image()
                # case EmbedProvider.HUGGINGFACE:
                #     cont = _huggingface_image()
                case _:
                    raise ValueError(
                        f"unsupported image embed provider: {GeneralConfig.image_embed_provider}"
                    )
            conts[Modality.IMAGE] = cont
    except Exception as e:
        raise RuntimeError(f"failed to create embedding: {e}") from e

    return EmbedManager(conts)


def _openai_text() -> EmbedContainer:
    """埋め込みコンテナ生成ヘルパー

    Returns:
        EmbedContainer: コンテナ
    """
    logger.debug("trace")

    return EmbedContainer(
        provider_name=EmbedProvider.OPENAI,
        embed=OpenAIEmbedding(
            model=EmbedConfig.openai_embed_model_text,
            api_base=EmbedConfig.openai_base_url,
            # device=GeneralConfig.device,
        ),
    )


def _cohere_text() -> EmbedContainer:
    """埋め込みコンテナ生成ヘルパー

    Returns:
        EmbeddingContainer: コンテナ
    """
    logger.debug("trace")

    return EmbedContainer(
        provider_name=EmbedProvider.COHERE,
        embed=CohereEmbedding(
            model_name=EmbedConfig.cohere_embed_model_text,
            # device=GeneralConfig.device,
        ),
    )


def _cohere_image() -> EmbedContainer:
    """埋め込みコンテナ生成ヘルパー

    Returns:
        EmbeddingContainer: コンテナ
    """
    logger.debug("trace")

    return EmbedContainer(
        provider_name=EmbedProvider.COHERE,
        embed=CohereEmbedding(
            model_name=EmbedConfig.cohere_embed_model_image,
            device=GeneralConfig.device,
        ),
    )


def _clip_text() -> EmbedContainer:
    """埋め込みコンテナ生成ヘルパー

    Returns:
        EmbeddingContainer: コンテナ
    """
    logger.debug("trace")

    return EmbedContainer(
        provider_name=EmbedProvider.CLIP,
        embed=ClipEmbedding(
            model_name=EmbedConfig.clip_embed_model_text,
            device=GeneralConfig.device,
        ),
    )


def _clip_image() -> EmbedContainer:
    """埋め込みコンテナ生成ヘルパー

    Returns:
        EmbeddingContainer: コンテナ
    """
    logger.debug("trace")

    return EmbedContainer(
        provider_name=EmbedProvider.CLIP,
        embed=ClipEmbedding(
            model_name=EmbedConfig.clip_embed_model_image,
            device=GeneralConfig.device,
        ),
    )


def _huggingface_text() -> EmbedContainer:
    """埋め込みコンテナ生成ヘルパー

    Returns:
        EmbeddingContainer: コンテナ
    """
    logger.debug("trace")

    return EmbedContainer(
        provider_name=EmbedProvider.HUGGINGFACE,
        embed=HuggingFaceEmbedding(
            model_name=EmbedConfig.huggingface_embed_model_text,
            device=GeneralConfig.device,
        ),
    )


# https://github.com/run-llama/llama_index/issues/15519?utm_source=chatgpt.com
#
# def _huggingface_image() -> EmbedContainer:
#     """埋め込みコンテナ生成ヘルパー

#     Returns:
#         EmbeddingContainer: コンテナ
#     """
#     logger.debug("trace")

#     return EmbedContainer(
#         provider_name=EmbedProvider.HUGGINGFACE,
#         embed=HuggingFaceEmbedding(
#             model_name=EmbedConfig.huggingface_embed_model_image,
#             device=GeneralConfig.device,
#         ),
#     )
