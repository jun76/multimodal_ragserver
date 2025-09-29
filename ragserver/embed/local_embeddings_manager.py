from __future__ import annotations

from ragserver.core.metadata import EMBTYPE_IMAGE, EMBTYPE_TEXT
from ragserver.core.names import LOCAL_EMBED_NAME
from ragserver.core.util import cool_down
from ragserver.embed.langchain_like import MyOpenAIEmbeddings
from ragserver.embed.multimodal_embeddings_manager import (
    MultimodalEmbeddings,
    MultimodalEmbeddingsManager,
)
from ragserver.embed.util import generate_space_key, image_to_data_uri
from ragserver.logger import logger


class LocalMultimodalEmbeddings(MyOpenAIEmbeddings, MultimodalEmbeddings):
    def __init__(
        self, model_text: str, model_image: str, base_url: str, api_key: str
    ) -> None:
        """MultimodalEmbeddings の embed_image() 抽象に対する実装を与えるクラス。
        テキスト埋め込みの場合は MyOpenAIEmbeddings で完結。

        Args:
            model_text (str): テキスト埋め込みモデル名
            model_image (str): 画像埋め込みモデル名
            base_url (str): ローカルモデルのエンドポイント
            api_key (str): API キー（ダミー想定）
        """
        logger.debug("trace")

        MyOpenAIEmbeddings.__init__(
            self, model=model_text, base_url=base_url, api_key=api_key
        )
        MultimodalEmbeddings.__init__(self)
        self._model_image = model_image

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

            inputs.append(data_uri)

        if len(inputs) == 0:
            logger.warning("empty inputs")
            return []

        try:
            # テキスト用埋め込みで OK
            vecs = self.embed_documents(inputs)
        except Exception as e:
            logger.exception(e)
            return []
        finally:
            cool_down()

        return vecs


class LocalEmbeddingsManager(MultimodalEmbeddingsManager):
    def __init__(
        self,
        model_text: str,
        model_image: str,
        base_url: str,
        api_key: str,
        need_norm: bool = True,
    ) -> None:
        """OpenAI 互換の REST API + CLIP モデル使用の埋め込みモデル管理クラス

        前提として、画像の埋め込みは Data URI をテキスト埋め込み用インタフェースで
        リクエストすること。CLIP モデル側で埋め込みデータが "data:image" で
        始まれば画像と判断して埋め込むことを期待している。

        Args:
            model_text (str): テキスト埋め込みモデル名
            model_image (str): 画像埋め込みモデル名
            base_url (str): ローカルモデルのエンドポイント
            api_key (str): API キー（ダミー想定）
            need_norm (bool, optional): L2 正規化要否。 Defaults to True.
        """
        logger.debug("trace")

        MultimodalEmbeddingsManager.__init__(
            self,
            name="local",
            model_text=model_text,
            model_image=model_image,
            need_norm=need_norm,
        )
        self._embed = LocalMultimodalEmbeddings(
            model_text=model_text,
            model_image=model_image,
            base_url=base_url,
            api_key=api_key,
        )

    def get_embeddings(self) -> MultimodalEmbeddings:
        """マルチモーダル対応の埋め込みモデルを返す。

        Returns:
            MultimodalEmbeddings: 埋め込みモデル
        """
        logger.debug("trace")

        return self._embed

    def space_key_text(self) -> str:
        """Local CLIP テキスト文書用ベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        logger.debug("trace")

        return generate_space_key(LOCAL_EMBED_NAME, self._model_text, EMBTYPE_TEXT)

    def embed_text_for_image_query(self, query: str) -> list[float]:
        """検索クエリ（照会用）のテキストを埋め込む。

        Args:
            query (str): 埋め込み対象のクエリ

        Returns:
            list[float]: 埋め込みベクトル
        """
        logger.debug("trace")

        try:
            # テキスト用埋め込みで OK
            return self._embed.embed_query(query)
        except Exception as e:
            logger.exception(e)
            return []

    def space_key_multi(self) -> str:
        """Local CLIP 画像ベクトルの空間キー。

        Returns:
            str: 空間キー
        """
        logger.debug("trace")

        return generate_space_key(LOCAL_EMBED_NAME, self._model_image, EMBTYPE_IMAGE)
