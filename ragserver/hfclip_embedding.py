"""HuggingFace CLIP Embedding Server との連携を行う LlamaIndex カスタム埋め込みクラス。

このモジュールは、外部の HFCLIP Embedding Server（embed_server）と通信し、
テキストと画像の埋め込みベクトルを取得する LlamaIndex 用のカスタム埋め込みクラスを提供します。
"""

from __future__ import annotations

from typing import Any, List

import requests
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from ragserver.logger import logger


class HFCLIPEmbedding(BaseEmbedding):
    """HuggingFace CLIP Embedding Server と連携する LlamaIndex カスタム埋め込みクラス。

    外部の embed_server にリクエストを送信し、テキストと画像の埋め込みベクトルを取得します。
    LlamaIndex の BaseEmbedding を継承し、標準的な埋め込みインターフェースを提供します。

    Attributes:
        base_url (str): embed_server のベース URL
        text_model (str): テキスト埋め込みに使用するモデル名
        image_model (str): 画像埋め込みに使用するモデル名
        timeout (int): リクエストのタイムアウト（秒）
    """

    base_url: str = Field(description="Base URL of the HFCLIP embedding server")
    text_model: str = Field(description="Model name for text embeddings")
    image_model: str = Field(description="Model name for image embeddings")
    timeout: int = Field(default=30, description="Request timeout in seconds")

    _session: requests.Session = PrivateAttr()

    def __init__(
        self,
        base_url: str,
        text_model: str,
        image_model: str,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        """HFCLIPEmbedding を初期化する。

        Args:
            base_url (str): embed_server のベース URL
            text_model (str): テキスト埋め込みモデル名
            image_model (str): 画像埋め込みモデル名
            timeout (int, optional): タイムアウト秒数. Defaults to 30.
            **kwargs: BaseEmbedding に渡す追加パラメータ
        """
        logger.debug("trace")

        super().__init__(
            base_url=base_url,
            text_model=text_model,
            image_model=image_model,
            timeout=timeout,
            **kwargs,
        )
        self._session = requests.Session()

    @classmethod
    def class_name(cls) -> str:
        """クラス名を返す。

        Returns:
            str: クラス名
        """
        return "HFCLIPEmbedding"

    def _get_query_embedding(self, query: str) -> Embedding:
        """クエリテキストの埋め込みベクトルを取得する（内部メソッド）。

        Args:
            query (str): クエリテキスト

        Returns:
            Embedding: 埋め込みベクトル

        Raises:
            RuntimeError: embed_server との通信に失敗した場合
        """
        logger.debug("trace")

        return self._embed_text(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """クエリテキストの埋め込みベクトルを非同期で取得する（内部メソッド）。

        現在は同期版を呼び出すだけの実装。

        Args:
            query (str): クエリテキスト

        Returns:
            Embedding: 埋め込みベクトル
        """
        logger.debug("trace")

        # 非同期実装は将来的に追加可能
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """テキストの埋め込みベクトルを取得する（内部メソッド）。

        Args:
            text (str): テキスト

        Returns:
            Embedding: 埋め込みベクトル
        """
        logger.debug("trace")

        return self._embed_text(text)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """テキストの埋め込みベクトルを非同期で取得する（内部メソッド）。

        現在は同期版を呼び出すだけの実装。

        Args:
            text (str): テキスト

        Returns:
            Embedding: 埋め込みベクトル
        """
        logger.debug("trace")

        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """複数のテキストの埋め込みベクトルを取得する（内部メソッド）。

        Args:
            texts (List[str]): テキストのリスト

        Returns:
            List[Embedding]: 埋め込みベクトルのリスト
        """
        logger.debug("trace")

        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> Embedding:
        """テキストを埋め込みベクトルに変換する。

        embed_server の /v1/embeddings エンドポイントにリクエストを送信します。

        Args:
            text (str): 埋め込み対象のテキスト

        Returns:
            Embedding: 埋め込みベクトル

        Raises:
            RuntimeError: embed_server との通信に失敗した場合
        """
        logger.debug("trace")

        url = f"{self.base_url}/embeddings"
        payload = {
            "input": text,
            "model": self.text_model,
        }

        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # OpenAI API 互換のレスポンス形式を想定
            embedding = data["data"][0]["embedding"]
            return embedding

        except requests.RequestException as e:
            logger.error(f"Failed to get text embedding: {e}")
            raise RuntimeError(f"Failed to get text embedding from {url}") from e

    def get_image_embedding(self, image_path: str) -> Embedding:
        """画像の埋め込みベクトルを取得する。

        embed_server の /v1/embeddings エンドポイントに画像パスを送信します。

        Args:
            image_path (str): 画像ファイルのパス

        Returns:
            Embedding: 埋め込みベクトル

        Raises:
            RuntimeError: embed_server との通信に失敗した場合
        """
        logger.debug("trace")

        url = f"{self.base_url}/embeddings"
        payload = {
            "input": image_path,
            "model": self.image_model,
        }

        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # OpenAI API 互換のレスポンス形式を想定
            embedding = data["data"][0]["embedding"]
            return embedding

        except requests.RequestException as e:
            logger.error(f"Failed to get image embedding: {e}")
            raise RuntimeError(f"Failed to get image embedding from {url}") from e
