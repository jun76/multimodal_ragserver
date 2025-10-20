from __future__ import annotations

import asyncio
from enum import StrEnum, auto
from typing import Coroutine

import laion_clap
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.utils import get_tqdm_iterable

from .multi_modal_base import AudioEmbedding, AudioType
from ...logger import logger


class ModelName(StrEnum):
    EFFECT_SHORT = auto()
    EFFECT_VARLEN = auto()
    MUSIC = auto()
    SPEECH = auto()
    GENERAL = auto()


class AudioEncoderModel(StrEnum):
    HTSAT_TINY = "HTSAT-tiny"
    HTSAT_BASE = "HTSAT-base"


class TextEncoderModel(StrEnum):
    ROBERTA = auto()


class ClapEmbedding(AudioEmbedding):
    """
    LAION-AI CLAP 埋め込み専用クラス

    MultiModalEmbedding を参考に実装。
    MultiModalEmbedding 自身に音声埋め込みサポートがあれば良いが
    未だ無いので BaseEmbedding --> AudioEmbedding を基底として実装する。
    """

    @classmethod
    def class_name(cls) -> str:
        """クラス名

        Returns:
            str: クラス名
        """
        return "ClapEmbedding"

    def __init__(
        self,
        model_name: str = ModelName.EFFECT_VARLEN,
        device: str = "cuda",
        embed_batch_size: int = 8,
    ) -> None:
        """コンストラクタ

        Args:
            model_name (str, optional): モデル名。未整備のため、ModelName として独自定義。Defaults to "general".
            device (str, optional): 埋め込みデバイス。Defaults to "cuda".
        """

        super().__init__(
            model_name=f"clap/{model_name}",
            embed_batch_size=embed_batch_size,
        )

        enable_fusion = False
        tmodel = TextEncoderModel.ROBERTA
        match model_name:
            case ModelName.EFFECT_SHORT:
                amodel = AudioEncoderModel.HTSAT_TINY
                model_id = 1
            case ModelName.EFFECT_VARLEN:
                enable_fusion = True
                amodel = AudioEncoderModel.HTSAT_TINY
                model_id = 3
            case ModelName.MUSIC | ModelName.SPEECH | ModelName.GENERAL:
                amodel = AudioEncoderModel.HTSAT_BASE
                raise NotImplementedError("loading local .pt is not implemented")
            case _:
                raise RuntimeError(f"unexpected model name: {model_name}")

        self._model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion, device=device, amodel=amodel, tmodel=tmodel
        )
        self._model.load_ckpt(model_id=model_id)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """クエリ文字列の非同期埋め込みを行う。

        Args:
            query (str): クエリ文字列

        Returns:
            Embedding: 埋め込みベクトル
        """

        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """単一テキストの同期埋め込みを行う。

        Args:
            text (str): テキスト

        Returns:
            Embedding: 埋め込みベクトル
        """

        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """複数テキストの同期埋め込みを行う。

        Args:
            texts (list[str]): テキスト

        Returns:
            list[Embedding]: 埋め込みベクトル
        """

        vecs = self._model.get_text_embedding(x=texts)

        return [vec.tolist() for vec in vecs]

    def _get_query_embedding(self, query: str) -> Embedding:
        """クエリ文字列の同期埋め込みを行う。

        Args:
            query (str): クエリ文字列

        Returns:
            Embedding: 埋め込みベクトル
        """

        return self._get_text_embedding(query)

    def _get_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """LAION-AI CLAP の同期 API 呼び出しラッパー。

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス

        Returns:
            list[Embedding]: 埋め込みベクトル
        """

        vecs = self._model.get_audio_embedding_from_filelist(x=audio_file_paths)

        return [vec.tolist() for vec in vecs]

    async def aget_audio_embedding_batch(
        self, audio_file_paths: list[AudioType], show_progress: bool = False
    ) -> list[Embedding]:
        """音声埋め込みの非同期バッチインタフェース。

        MultiModalEmbedding の aget_image_embedding_batch がベース。

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス
            show_progress (bool, optional): 進捗の表示。Defaults to False.

        Returns:
            list[Embedding]: 埋め込みベクトル
        """

        cur_batch: list[AudioType] = []
        callback_payloads: list[tuple[str, list[AudioType]]] = []
        result_embeddings: list[Embedding] = []
        embeddings_coroutines: list[Coroutine] = []
        for idx, audio_file_path in enumerate(audio_file_paths):
            cur_batch.append(audio_file_path)
            if (
                idx == len(audio_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(self._aget_audio_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio

                nested_embeddings = await tqdm_asyncio.gather(
                    *embeddings_coroutines,
                    total=len(embeddings_coroutines),
                    desc="Generating embeddings",
                )
            except ImportError:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)
        else:
            nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, audio_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: audio_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings

    async def _aget_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """LAION-AI CLAP の非同期 API 呼び出しラッパー。

        この関数の実装時点で、LAION-AI CLAP には未だ同期インタフェースしかない

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス

        Returns:
            list[Embedding]: 埋め込みベクトル
        """

        return await asyncio.to_thread(self._get_audio_embeddings, audio_file_paths)
