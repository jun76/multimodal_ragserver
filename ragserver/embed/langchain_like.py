from __future__ import annotations

from abc import abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings

from ragserver.logger import logger

# ここにあるのは langchain 側で対応してくれたら不要になるかもしれないクラス、関数たち。


class MultimodalEmbeddings(Embeddings):
    def __init__(self) -> None:
        """langchain 系列に embed_image() を組み込むための抽象クラス"""
        logger.debug("trace")

        super().__init__()

    def _response_to_float_vecs(self, response: Any) -> list[list[float]]:
        """langchain ツールセット外での各 embeddings による直接埋め込みの
        応答オブジェクトから、 list[list[float]] 形式の埋め込みベクトルを抽出

        Args:
            response (Any): 応答オブジェクト

        Returns:
            list[list[float]]: 埋め込みベクトル
        """
        logger.debug("trace")

        if response is None:
            logger.warning("empty response")
            return []

        embeddings = getattr(response, "embeddings", None)

        if embeddings is None:
            logger.error("embed response has no 'embeddings'")
            return []
        else:
            # 典型: response.embeddings.float が list[list[float]]
            if hasattr(embeddings, "float"):
                vecs = getattr(embeddings, "float") or []
            # 辞書形式で返る場合
            elif isinstance(embeddings, dict):
                vecs = embeddings.get("float", []) or []
            # まれに直接 list[list[float]] のことも想定
            elif isinstance(embeddings, list):
                vecs = embeddings  # type: ignore[assignment]
            else:
                logger.error("unexpected embeddings format: %s", type(embeddings))
                return []

        return vecs

    @abstractmethod
    def embed_image(self, paths: list[str]) -> list[list[float]]:
        """画像のローカルパスを受け取り、埋め込み行列を返す。
        Chroma の embedding_function がダックタイピングで受け取る関数になるので、
        シグネチャを崩さないように注意。

        Args:
            paths (list[str]): 画像のローカルパスのリスト

        Returns:
            list[list[float]]: 画像の埋め込み行列
        """
        logger.debug("trace")
        ...
