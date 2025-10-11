from __future__ import annotations

from llama_index.postprocessor.cohere_rerank import CohereRerank

from ragserver.core.names import COHERE_RERANK_NAME
from ragserver.logger import logger
from ragserver.rerank.rerank_manager import RerankManager


class CohereRerankManager(RerankManager):
    """Cohere の提供するリランカーの管理クラス"""

    def __init__(self, model: str, topk: int = 10) -> None:
        """コンストラクタ

        Args:
            model (str): リランカーモデル名
            topk (int, optional): 取得件数。Defaults to 10.
        """
        logger.debug("trace")

        RerankManager.__init__(self)
        self._rerank = CohereRerank(model=model, top_n=topk)

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return COHERE_RERANK_NAME
