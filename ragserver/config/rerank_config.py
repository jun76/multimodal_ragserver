from dataclasses import dataclass, field

from ragserver.config.settings import Settings


@dataclass(kw_only=True)
class RerankConfig:
    flagembedding_rerank_model: str = field(default=Settings.FLAGEMBEDDING_RERANK_MODEL)
    cohere_rerank_model: str = field(default=Settings.COHERE_RERANK_MODEL)
    topk: int = field(default=Settings.TOPK)
