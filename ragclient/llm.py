from __future__ import annotations

from langchain_openai import ChatOpenAI

from ragclient.config import get_config
from ragclient.logger import logger

__all__ = ["get_chat_model"]


def get_chat_model(provider: str) -> ChatOpenAI:
    """LLM プロバイダ設定に応じたチャットモデルを生成する。

    Args:
        provider (str): 利用する LLM プロバイダ名

    Returns:
        ChatOpenAI: LangChain 互換のチャットモデル

    Raises:
        ValueError: 未対応のプロバイダが指定された場合
    """
    logger.debug("trace")

    cfg = get_config()
    normalized = provider.strip().lower()
    if normalized == "openai":
        logger.info(f"llm model = {cfg.llm_openai_model}")
        return ChatOpenAI(model=cfg.llm_openai_model, timeout=30, max_retries=3)

    if normalized == "local":
        logger.info(f"llm model = {cfg.llm_local_model}")
        return ChatOpenAI(
            model=cfg.llm_local_model,
            base_url=cfg.llm_local_base_url.rstrip("/"),
            timeout=30,
            max_retries=3,
        )

    raise ValueError(f"unsupported llm provider: {provider}")
