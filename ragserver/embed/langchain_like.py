from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from openai import OpenAI

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


class MyOpenAIEmbeddings(Embeddings):
    def __init__(
        self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None
    ):
        """OpenAIEmbeddings 代替独自ラッパークラス

        langchain_openai.OpenAIEmbeddings だと base_url 使用時に何故か request body の
        フォーマットが崩れる？（ローカル埋め込みモデル側に受理されない）ので
        独自にラッパークラスを定義して使用

        Args:
            model (str): テキスト埋め込みモデル名
            base_url (Optional[str], optional): ローカルモデルのエンドポイント Defaults to None.
            api_key (Optional[str], optional): API キー Defaults to None.

        Raises:
            RuntimeError: OpenAI クライアントの初期化に失敗した場合
        """
        logger.debug("trace")

        self._model = model

        if base_url:
            logger.info(f"base_url specified: {base_url}")

        try:
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        except Exception as e:
            raise RuntimeError("failed to initialize OpenAI client") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """代替 embed_documents

        Args:
            texts (list[str]): 埋め込み対象テキスト

        Returns:
            list[list[float]]: 埋め込みベクトル

        Raises:
            RuntimeError: 埋め込みの生成に失敗した場合
        """
        logger.debug("trace")

        if len(texts) == 0:
            return []

        try:
            res = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as e:
            raise RuntimeError("failed to create embeddings") from e

        return [d.embedding for d in res.data]

    def embed_query(self, query: str) -> list[float]:
        """代替 embed_query

        Args:
            query (str): 埋め込み対象クエリ

        Returns:
            list[float]: 埋め込みベクトル

        Raises:
            RuntimeError: 埋め込みの生成に失敗した場合
        """
        logger.debug("trace")

        try:
            res = self._client.embeddings.create(model=self._model, input=[query])
        except Exception as e:
            raise RuntimeError("failed to create embedding") from e

        return res.data[0].embedding
