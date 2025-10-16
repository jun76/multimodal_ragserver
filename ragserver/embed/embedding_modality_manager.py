from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType

from ragserver.core.metadata import Modality
from ragserver.logger import logger


@dataclass
class EmbeddingContainer:
    """モダリティ毎の埋め込み器関連パラメータを集約"""

    modality: Modality
    provider_name: str
    embedding: BaseEmbedding
    space_key: str = ""


class EmbeddingModalityManager:
    """埋め込みモダリティの管理クラス。"""

    def __init__(self, embeds: list[EmbeddingContainer]) -> None:
        """コンストラクタ

        Args:
            embeds (list[EmbeddingContainer]): 埋め込みコンテナのリスト

        Raises:
            ValueError: 予期せぬモダリティ
        """
        logger.debug("trace")

        self._embed_text: Optional[EmbeddingContainer] = None
        self._embed_image: Optional[EmbeddingContainer] = None
        self._modality: set[Modality] = set()

        for embed in embeds:
            embed.space_key = self._generate_space_key(
                provider=embed.provider_name,
                model=embed.embedding.model_name,
                modality=embed.modality,
            )
            match embed.modality:
                case Modality.TEXT:
                    self._embed_text = embed
                case Modality.IMAGE:
                    self._embed_image = embed
                case _:
                    raise ValueError(f"unexpected modality: {embed.modality}")

            self._modality.add(embed.modality)

    @property
    def modality(self) -> set[Modality]:
        """この埋め込み管理がサポートするモダリティ一覧。

        Returns:
            set[Modality]: モダリティ一覧
        """
        return self._modality

    @property
    def space_key_text(self) -> str:
        """テキスト埋め込みの空間キー。

        Raises:
            RuntimeError: 未初期化

        Returns:
            str: 空間キー
        """
        return self.get_container(Modality.TEXT).space_key

    @property
    def space_key_image(self) -> str:
        """画像埋め込みの空間キー。

        Raises:
            RuntimeError: 未初期化

        Returns:
            str: 空間キー
        """
        return self.get_container(Modality.IMAGE).space_key

    def get_container(self, modality: Modality) -> EmbeddingContainer:
        """モダリティ別の埋め込みコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            ValueError: 予期せぬモダリティ
            RuntimeError: 未初期化

        Returns:
            EmbeddingContainer: 埋め込みコンテナ
        """
        logger.debug("trace")

        match modality:
            case Modality.TEXT:
                if self._embed_text:
                    return self._embed_text
            case Modality.IMAGE:
                if self._embed_image:
                    return self._embed_image
            case _:
                raise ValueError(f"unexpected modality: {modality}")

        raise RuntimeError(f"embed {modality} is not initialized")

    async def aembed_text(self, texts: list[str]) -> list[Embedding]:
        """テキストの埋め込みベクトルを取得する。

        Args:
            texts (list[str]): テキスト

        Raises:
            RuntimeError: 未初期化

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        logger.debug("trace")

        return await self.get_container(
            Modality.TEXT
        ).embedding.aget_text_embedding_batch(texts)

    async def aembed_image(self, paths: list[ImageType]) -> list[Embedding]:
        """画像の埋め込みベクトルを取得する。

        Args:
            paths (list[ImageType]): 画像のパス（または base64 画像の直渡しでも OK）

        Raises:
            RuntimeError: 未初期化または画像埋め込み器でない

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        logger.debug("trace")

        embed = self.get_container(Modality.IMAGE).embedding
        if not isinstance(embed, MultiModalEmbedding):
            raise RuntimeError("multimodal embed model is required")

        return await embed.aget_image_embedding_batch(img_file_paths=paths)

    def _sanitize_space_key(self, space_key: str) -> str:
        """制約にマッチするよう space_key 文字列を整形する。

        制約（AND）：
            Chroma
                containing 3-512 characters from [a-zA-Z0-9._-],
                starting and ending with a character in [a-zA-Z0-9]

            SQLite
                念のため英数とアンダースコア以外は '_'

        Args:
            space_key (str): 整形前の space_key

        Returns:
            str: 整形後の space_key
        """
        logger.debug("trace")

        allowed = frozenset(
            "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789" "_"
        )

        # 許可されない文字は '_' に置換
        chars = [ch if ch in allowed else "_" for ch in space_key]

        # 長すぎる場合は 512 にトリム
        if len(chars) > 512:
            chars = chars[:512]

        # 先頭・末尾を英数字にする（英数字でなければ '0' に置換）
        def is_alnum(ch: str) -> bool:
            return ("0" <= ch <= "9") or ("a" <= ch <= "z") or ("A" <= ch <= "Z")

        if not chars:
            chars = list("000")
        else:
            if not is_alnum(chars[0]):
                chars[0] = "0"
            if not is_alnum(chars[-1]):
                chars[-1] = "0"

        return "".join(chars)

    def _generate_space_key(self, provider: str, model: str, modality: Modality) -> str:
        """空間キー文字列を生成する。

        Args:
            provider (str): プロバイダ名
            model (str): モデル名
            modality (Modality): モダリティ

        Returns:
            str: 空間キー文字列
        """
        logger.debug("trace")

        space_key = self._sanitize_space_key(f"{provider}_{model}_{modality}")
        logger.info(f"space_key [{space_key}] generated")

        return space_key
