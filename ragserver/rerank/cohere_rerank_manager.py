from __future__ import annotations

from langchain_cohere import CohereRerank

from ragserver.logger import logger
from ragserver.rerank.rerank_manager import RerankManager


class CohereRerankManager(RerankManager):
    """Cohere の提供するリランカーの管理クラス"""

    def __init__(self, model: str, topk: int = 10) -> None:
        """コンストラクタ

        Args:
            model (str): リランカーモデル名
            topk (int, optional): 取得件数。 Defaults to 10.
        """
        logger.debug("trace")

        RerankManager.__init__(self, "cohere")
        self._rerank = CohereRerank(model=model, top_n=topk)
