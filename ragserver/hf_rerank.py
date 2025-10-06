"""HuggingFace Rerank Server との連携を行う LlamaIndex カスタムリランククラス。

このモジュールは、外部の HF Rerank Server（rerank_server）と通信し、
検索結果のリランキングを行う LlamaIndex 用のカスタム Node Postprocessor を提供します。
"""

from __future__ import annotations

from typing import List, Optional

import requests
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from ragserver.logger import logger


class HFRerank(BaseNodePostprocessor):
    """HuggingFace Rerank Server と連携する LlamaIndex カスタムリランククラス。

    外部の rerank_server にリクエストを送信し、検索結果をリランキングします。
    LlamaIndex の BaseNodePostprocessor を継承し、標準的なポストプロセッサーインターフェースを提供します。

    Attributes:
        base_url (str): rerank_server のベース URL
        model (str): リランクに使用するモデル名
        top_n (int): 返却する上位ノード数
        timeout (int): リクエストのタイムアウト（秒）
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        top_n: int = 5,
        timeout: int = 30,
    ) -> None:
        """HFRerank を初期化する。

        Args:
            base_url (str): rerank_server のベース URL
            model (str): リランクモデル名
            top_n (int, optional): 返却する上位ノード数. Defaults to 5.
            timeout (int, optional): タイムアウト秒数. Defaults to 30.
        """
        logger.debug("trace")

        super().__init__()
        self._base_url = base_url
        self._model = model
        self._top_n = top_n
        self._timeout = timeout
        self._session = requests.Session()

    @classmethod
    def class_name(cls) -> str:
        """クラス名を返す。

        Returns:
            str: クラス名
        """
        return "HFRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """ノードをリランキングする（内部メソッド）。

        Args:
            nodes (List[NodeWithScore]): リランク対象のノードリスト
            query_bundle (Optional[QueryBundle], optional): クエリ情報. Defaults to None.

        Returns:
            List[NodeWithScore]: リランク済みノードリスト

        Raises:
            ValueError: query_bundle が指定されていない場合
            RuntimeError: rerank_server との通信に失敗した場合
        """
        logger.debug("trace")

        if query_bundle is None:
            raise ValueError("query_bundle is required for HFRerank")

        if len(nodes) == 0:
            logger.debug("No nodes to rerank")
            return []

        query_str = query_bundle.query_str

        # ノードのテキストを抽出
        documents = [node.node.get_content() for node in nodes]

        # rerank_server にリクエスト
        url = f"{self._base_url}/rerank"
        payload = {
            "query": query_str,
            "documents": documents,
            "model": self._model,
            "top_n": min(self._top_n, len(documents)),
        }

        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()

            # Cohere API 互換のレスポンス形式を想定
            # results: [{"index": int, "relevance_score": float}, ...]
            results = data.get("results", [])

            # インデックスとスコアを使ってノードを並び替え
            reranked_nodes = []
            for result in results:
                idx = result["index"]
                score = result["relevance_score"]

                if 0 <= idx < len(nodes):
                    node = nodes[idx]
                    # スコアを更新
                    node.score = score
                    reranked_nodes.append(node)

            logger.debug(f"Reranked {len(reranked_nodes)} nodes")
            return reranked_nodes[: self._top_n]

        except requests.RequestException as e:
            logger.error(f"Failed to rerank: {e}")
            raise RuntimeError(f"Failed to rerank with {url}") from e
