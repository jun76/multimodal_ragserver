from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings

from ragserver.core.util import cool_down
from ragserver.embed.util import l2_normalize
from ragserver.logger import logger


class EmbeddingsManager(ABC):
    def __init__(self, name: str, model_text: str, need_norm: bool) -> None:
        """埋め込み管理の抽象インターフェース（テキスト）

        Args:
            name (str): プロバイダ名
            model_text (str): テキスト埋め込みモデル名
            need_norm: (bool): L2 正規化要否
        """
        logger.debug("trace")

        self._name = name
        self._model_text = model_text
        self._need_norm = need_norm

    def get_name(self) -> str:
        """プロバイダ名を取得する。
        クライアント側での状態確認用途を想定。

        Returns:
            str: プロバイダ名
        """
        logger.debug("trace")

        return self._name

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """埋め込みモデルを返す。

        Returns:
            Embeddings: 埋め込みモデル
        """
        logger.debug("trace")
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """検索対象のテキストを埋め込む。

        Args:
            texts (list[str]): 埋め込み対象のテキスト

        Returns:
            list[list[float]]: 埋め込み行列
        """
        logger.debug("trace")

        if len(texts) == 0:
            logger.warning("empty texts")
            return []

        try:
            embs = self.get_embeddings().embed_documents(texts)
            embs = l2_normalize(embs) if self._need_norm else embs
        except Exception as e:
            logger.exception(e)
            return []
        finally:
            cool_down()

        return embs

    def embed_query(self, query: str) -> list[float]:
        """クエリ文字列を埋め込む。

        Args:
            query (str): 埋め込み対象のクエリ

        Returns:
            list[float]: 埋め込みベクトル
        """
        logger.debug("trace")

        try:
            emb = self.get_embeddings().embed_query(query)
            emb = l2_normalize([emb])[0] if self._need_norm else emb
        except Exception as e:
            logger.exception(e)
            return []
        finally:
            cool_down()

        return emb

    @abstractmethod
    def space_key_text(self) -> str:
        """テキスト文書（インデックス用）ベクトルの空間キー。

        Returns:
            str: この埋め込み実装が生成する文書用ベクトルの空間キー
        """
        logger.debug("trace")
        ...
