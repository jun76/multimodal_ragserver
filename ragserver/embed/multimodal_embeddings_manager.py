from __future__ import annotations

from abc import abstractmethod

from ragserver.core.util import cool_down
from ragserver.embed.embeddings_manager import EmbeddingsManager
from ragserver.embed.langchain_like import MultimodalEmbeddings
from ragserver.embed.util import l2_normalize
from ragserver.logger import logger


class MultimodalEmbeddingsManager(EmbeddingsManager):
    def __init__(
        self, name: str, model_text: str, model_image: str, need_norm: bool
    ) -> None:
        """埋め込み管理の抽象インターフェース（マルチモーダル）

        Args:
            name (str): プロバイダ名
            model_text (str): テキスト埋め込みモデル名
            model_image (str): 画像埋め込みモデル名
            need_norm (bool): L2 正規化要否
        """
        logger.debug("trace")

        EmbeddingsManager.__init__(
            self, name=name, model_text=model_text, need_norm=need_norm
        )
        self._model_image = model_image

    @abstractmethod
    def get_embeddings(self) -> MultimodalEmbeddings:
        """マルチモーダル対応の埋め込みモデルを返す。

        Returns:
            MultimodalEmbeddings: 埋め込みモデル
        """
        logger.debug("trace")
        ...

    def embed_image(self, paths: list[str]) -> list[list[float]]:
        """検索対象の画像を埋め込む。

        Args:
            paths (list[str]): 画像のローカルパスのリスト

        Returns:
            list[list[float]]: 画像の埋め込み行列
        """
        logger.debug("trace")

        try:
            embs = self.get_embeddings().embed_image(paths)
            embs = l2_normalize(embs) if self._need_norm else embs
        except Exception as e:
            logger.exception(e)
            return []
        finally:
            cool_down()

        return embs

    @abstractmethod
    def embed_text_for_image_query(self, query: str) -> list[float]:
        """検索クエリ（照会用）のテキストを埋め込む。

        Args:
            query (str): 埋め込み対象のクエリ

        Returns:
            list[float]: 埋め込みベクトル
        """
        logger.debug("trace")
        ...

    @abstractmethod
    def space_key_multi(self) -> str:
        """画像（インデックス用）ベクトルの空間キー。

        Returns:
            str: この埋め込み実装が生成する画像用ベクトルの空間キー
        """
        logger.debug("trace")
        ...
