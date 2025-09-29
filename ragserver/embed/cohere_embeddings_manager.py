from __future__ import annotations

import cohere
from langchain_cohere import CohereEmbeddings

from ragserver.core.metadata import EMBTYPE_IMAGE, EMBTYPE_TEXT
from ragserver.core.names import COHERE_EMBED_NAME
from ragserver.core.util import cool_down
from ragserver.embed.multimodal_embeddings_manager import (
    MultimodalEmbeddings,
    MultimodalEmbeddingsManager,
)
from ragserver.embed.util import generate_space_key, image_to_data_uri
from ragserver.logger import logger


class CohereMultimodalEmbeddings(CohereEmbeddings, MultimodalEmbeddings):
    def __init__(self, model_text: str, model_image: str) -> None:
        """MultimodalEmbeddings の embed_image() 抽象に対する実装を与えるクラス。
        テキスト埋め込みの場合は MyOpenAIEmbeddings で完結。

        Args:
            model_text (str): テキスト埋め込みモデル名
            model_image (str): 画像埋め込みモデル名

        Raises:
            RuntimeError: Cohere クライアントの初期化に失敗した場合
        """
        logger.debug("trace")

        CohereEmbeddings.__init__(self, model=model_text)  # type: ignore
        MultimodalEmbeddings.__init__(self)
        self._model_image = model_image

        # 画像埋め込み用に専用クライアントを利用
        # （langchain の embeddings がマルチモーダル未対応のため）
        try:
            self._client = cohere.ClientV2()
        except Exception as e:
            raise RuntimeError("failed to initialize Cohere client") from e

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        """画像のローカルパスを受け取り、埋め込み行列を返す。
        Chroma の embedding_function がダックタイピングで受け取る関数になるので、
        シグネチャを崩さないように注意。

        Args:
            uris (list[str]): 画像のローカルパスのリスト

        Returns:
            list[list[float]]: 画像の埋め込み行列
        """
        logger.debug("trace")

        # シグネチャが崩せないが意味的にはパスしか扱えないのでここで改名
        paths = uris

        if len(paths) == 0:
            logger.warning("empty paths")
            return []

        inputs = []
        for path in paths:
            data_uri = image_to_data_uri(path)
            if data_uri is None:
                continue

            inputs.append(
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        }
                    ]
                }
            )

        if len(inputs) == 0:
            logger.warning("empty inputs")
            return []

        try:
            res = self._client.embed(
                model=self._model_image,
                input_type="image",
                inputs=inputs,
                embedding_types=["float"],
            )
            # Cohere SDK v2 の応答から float 埋め込みを抽出
            vecs = self._response_to_float_vecs(res)
        except Exception as e:
            logger.exception(e)
            return []
        finally:
            cool_down()

        return vecs


class CohereEmbeddingsManager(MultimodalEmbeddingsManager):
    def __init__(
        self,
        model_text: str,
        model_image: str,
        need_norm: bool = True,
    ) -> None:
        """Cohere の埋め込みモデル管理クラス

        Args:
            model_text (str): テキスト埋め込みモデル名
            model_image (str): 画像埋め込みモデル名
            need_norm (bool, optional): L2 正規化要否。 Defaults to True.
        """
        logger.debug("trace")

        MultimodalEmbeddingsManager.__init__(
            self,
            name="cohere",
            model_text=model_text,
            model_image=model_image,
            need_norm=need_norm,
        )
        self._embed = CohereMultimodalEmbeddings(
            model_text=model_text, model_image=model_image
        )

    def get_embeddings(self) -> MultimodalEmbeddings:
        """マルチモーダル対応の埋め込みモデルを返す。

        Returns:
            MultimodalEmbeddings: 埋め込みモデル
        """
        logger.debug("trace")

        return self._embed

    def space_key_text(self) -> str:
        """Cohere テキスト文書用ベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        logger.debug("trace")

        return generate_space_key(COHERE_EMBED_NAME, self._model_text, EMBTYPE_TEXT)

    def embed_text_for_image_query(self, query: str) -> list[float]:
        """検索クエリ（照会用）のテキストを埋め込む。

        Args:
            query (str): 埋め込み対象のクエリ

        Returns:
            list[float]: 埋め込みベクトル
        """
        logger.debug("trace")

        try:
            return self._embed.embed([query], input_type="search_query")[0]
        except Exception as e:
            logger.exception(e)
            return []

    def space_key_multi(self) -> str:
        """Cohere 画像ベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        logger.debug("trace")

        return generate_space_key(COHERE_EMBED_NAME, self._model_image, EMBTYPE_IMAGE)
