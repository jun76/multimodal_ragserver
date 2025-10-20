from abc import abstractmethod
from io import BytesIO
from typing import Any, Union

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding

AudioType = Union[str, BytesIO]


class AudioEmbedding(BaseEmbedding):
    """
    音声埋め込みクラスの抽象

    MultiModalEmbedding 自身に音声埋め込みサポートがあれば良いが未だ無いので
    このクラスを音声埋め込みの抽象として一段噛ませる。
    """

    @abstractmethod
    async def aget_audio_embedding_batch(
        self, audio_file_paths: list[AudioType], show_progress: bool = False
    ) -> list[Embedding]:
        """音声埋め込みの非同期バッチインタフェース。

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス
            show_progress (bool, optional): 進捗の表示。Defaults to False.

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        ...
