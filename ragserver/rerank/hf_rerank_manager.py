from __future__ import annotations

from typing import Any, Optional, Sequence

import requests
from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

from ragserver.core.util import cool_down
from ragserver.logger import logger
from ragserver.rerank.rerank_manager import RerankManager


class HFRerank(BaseDocumentCompressor):
    def __init__(
        self,
        model: str,
        base_url: str,
        topk: int,
    ):
        """OpenAI 互換（風）の HuggingFace モデルリランカークラス

        Args:
            model (str): リランカーモデル名
            base_url (str): HuggingFace モデルのエンドポイント
            topk (int): 取得件数。 Defaults to 10.
        """
        logger.debug("trace")

        self._model = model
        self._base_url = base_url
        self._topk = topk

    def _get_filtered_docs(
        self, docs: Sequence[Document]
    ) -> tuple[list[str], list[int]]:
        """空でない page_content を持つドキュメントだけを抽出する。

        Args:
            docs (Sequence[Document]): 抽出対象のドキュメント列

        Returns:
            tuple[list[str], list[int]]: 抽出した本文リストと元インデックスの対応表
        """
        logger.debug("trace")

        filtered_docs: list[str] = []
        index_map: list[int] = []
        for idx, doc in enumerate(docs):
            content = (doc.page_content or "").strip()
            if not content:
                continue

            filtered_docs.append(content)
            index_map.append(idx)

        return filtered_docs, index_map

    def _request_post(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int = 30,
    ) -> dict[str, Any]:
        """エンドポイントへ POST を実行し、JSON 形式の応答を取得。

        Args:
            url (str): エンドポイント
            headers (dict[str, str]): HTTP ヘッダ
            payload (dict[str, Any]): リクエストボディ
            timeout (int, optional): タイムアウト（秒）。Defaults to 30.

        Raises:
            requests.HTTPError: HTTPError 例外

        Returns:
            dict[str, Any]: レスポンス JSON
        """
        logger.debug("trace")

        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            text = response.text if response is not None else ""
            raise requests.HTTPError(f"{e} | response={text}") from e
        except requests.RequestException as e:
            raise RuntimeError("failed to post rerank request") from e
        finally:
            cool_down()

        try:
            res = response.json()
        except ValueError as e:
            raise ValueError("response is not valid json") from e

        return res

    def _get_selected_indices(
        self,
        results: list[dict[str, Any]],
        index_map: list[int],
        docs: Sequence[Document],
        limit: int,
    ) -> list[int]:
        """リランカー応答から有効なインデックスを再構成する。

        Args:
            results (list[dict[str, Any]]): リランカー応答（json）の results ブロック
            index_map (list[int]): 抽出時に構築した元インデックス対応表
            docs (Sequence[Document]): 元のドキュメント列（補完用途）
            limit (int): 返却上限件数

        Returns:
            list[int]: 元ドキュメントのインデックス一覧
        """
        logger.debug("trace")

        selected_indices: list[int] = []
        for item in results:
            try:
                mapped_idx = index_map[item["index"]]
            except (KeyError, IndexError, TypeError):
                continue

            if mapped_idx not in selected_indices:
                selected_indices.append(mapped_idx)

        if not selected_indices:
            return []

        if len(selected_indices) < limit:
            for idx in range(len(docs)):
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) >= limit:
                    break

        return selected_indices

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """BaseDocumentCompressor 既定の要求するリランク用仮想関数のオーバーライド。

        Args:
            documents (Sequence[Document]): 対象ドキュメント
            query (str): クエリ文字列
            callbacks (Optional[Callbacks], optional): 不使用。 Defaults to None.

        Returns:
            Sequence[Document]: リランク後のドキュメント
        """
        logger.debug("trace")

        if len(documents) == 0:
            logger.warning("empty documents")
            return []

        limit = self._topk if (self._topk and self._topk > 0) else len(documents)
        filtered_docs, index_map = self._get_filtered_docs(documents)

        if not filtered_docs:
            logger.warning("all documents are empty")
            return documents[:limit]

        topk = min(limit, len(filtered_docs))

        url = f"{self._base_url.rstrip('/')}/rerank"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self._model,
            "query": query,
            "documents": filtered_docs,
            "topk": topk,
        }

        try:
            res_json = self._request_post(url=url, headers=headers, payload=payload)
        except Exception as e:
            logger.exception(e)
            return documents[:limit]

        results = res_json.get("results") or []
        selected_indices = self._get_selected_indices(
            results=results, index_map=index_map, docs=documents, limit=limit
        )

        if len(selected_indices) == 0:
            logger.warning("empty selected_indices")
            return documents[:limit]

        return [documents[idx] for idx in selected_indices[:limit]]


class HFRerankManager(RerankManager):
    """OpenAI 互換（風）の HuggingFace モデルリランカーの管理クラス"""

    def __init__(self, model: str, base_url: str, topk: int = 10) -> None:
        """コンストラクタ

        Args:
            model (str): リランカーモデル名
            base_url (str):  HuggingFace モデルのエンドポイント
            topk (int, optional): 取得件数。 Defaults to 10.
        """
        logger.debug("trace")

        RerankManager.__init__(self, "hf")
        self._rerank = HFRerank(model=model, base_url=base_url, topk=topk)
