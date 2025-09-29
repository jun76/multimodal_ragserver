from __future__ import annotations

from abc import ABC

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

from ragserver.logger import logger


class RerankManager(ABC):
    def __init__(self, name: str) -> None:
        """リランカー管理の抽象インタフェース

        現状、画像のキャプション（テキスト）をつけることで langchain のリランカが
        そのまま使えるため MultimodalRerankManager は不要だが、マルチモーダル対応の
        リランカを提供するプロバイダが出てきた場合は別途実装のこと。

        Args:
            name (str): プロバイダ名
        """
        logger.debug("trace")

        self._name = name
        self._rerank: BaseDocumentCompressor

    def get_name(self) -> str:
        """プロバイダ名を取得する。
        クライアント側での状態確認用途を想定。

        Returns:
            str: プロバイダ名
        """
        logger.debug("trace")

        return self._name

    def rerank(self, docs: list[Document], query: str) -> list[Document]:
        """クエリに基づきリランカーで結果を並べ替える。

        Args:
            docs (list[Document]): 並べ替え対象ドキュメント
            query (str): クエリ文字列

        Returns:
            list[Document]: 並べ替え済みのドキュメント

        Raises:
            RuntimeError: リランカーが処理に失敗した場合
        """
        logger.debug("trace")
        logger.info("start reranking...")

        try:
            return list(self._rerank.compress_documents(documents=docs, query=query))
        except Exception as e:
            raise RuntimeError("failed to rerank documents") from e
