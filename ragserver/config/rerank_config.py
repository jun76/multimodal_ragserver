from __future__ import annotations

from dataclasses import dataclass

from .settings import Settings


@dataclass(kw_only=True, frozen=True)
class RerankConfig:
    """リランク関連の設定用データクラス"""

    flagembedding_rerank_model: str = Settings.FLAGEMBEDDING_RERANK_MODEL
    cohere_rerank_model: str = Settings.COHERE_RERANK_MODEL
    topk: int = Settings.TOPK
