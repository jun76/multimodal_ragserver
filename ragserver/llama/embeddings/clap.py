from __future__ import annotations

import asyncio
from enum import StrEnum, auto
from typing import Coroutine

import laion_clap
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.utils import get_tqdm_iterable

from ragserver.llama.embeddings.multi_modal_base import AudioEmbedding, AudioType
from ragserver.logger import logger


class SubModality(StrEnum):
    EFFECT_SHORT = auto()
    EFFECT_VARLEN = auto()
    MUSIC = auto()
    SPEECH = auto()
    GENERAL = auto()


class AudioEncoderModel:
    HTSAT_TINY = "HTSAT-tiny"
    HTSAT_BASE = "HTSAT-base"


class TextEncoderModel:
    ROBERTA = "roberta"


class AvailCkpt:
    K630_AUDIOSET_BEST = (
        "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt"
    )
    K630_AUDIOSET_FUSION_BEST = "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt"
    MUSIC_AUDIOSET_EPOCH_15_ESC_90_14 = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
    MUSIC_SPEECH_EPOCH_15_ESC_89_25 = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt"
    MUSIC_SPEECH_AUDIOSET_EPOCH_15_ESC_89_98 = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt"


class ClapEmbedding(AudioEmbedding):
    """
    LAION-AI CLAP 埋め込み専用クラス

    MultiModalEmbedding を参考に実装。
    MultiModalEmbedding 自身に音声埋め込みサポートがあれば良いが
    未だ無いので BaseEmbedding を基底として実装する。
    """

    @classmethod
    def class_name(cls) -> str:
        return "ClapEmbedding"

    def __init__(
        self,
        model_name: str = SubModality.GENERAL,
        device: str = "cuda",
    ):
        """コンストラクタ

        Args:
            model_name (str, optional): モデル名。未整備のため、SubModality として独自定義。Defaults to "general".
            device (str, optional): 埋め込みデバイス。Defaults to "cuda".
        """
        logger.debug("trace")

        enable_fusion = False
        tmodel = TextEncoderModel.ROBERTA
        match model_name:
            case SubModality.EFFECT_SHORT:
                amodel = AudioEncoderModel.HTSAT_TINY
                ckpt = AvailCkpt.K630_AUDIOSET_BEST
            case SubModality.EFFECT_VARLEN:
                enable_fusion = True
                amodel = AudioEncoderModel.HTSAT_TINY
                ckpt = AvailCkpt.K630_AUDIOSET_FUSION_BEST
            case SubModality.MUSIC:
                amodel = AudioEncoderModel.HTSAT_BASE
                ckpt = AvailCkpt.MUSIC_AUDIOSET_EPOCH_15_ESC_90_14
            case SubModality.SPEECH:
                amodel = AudioEncoderModel.HTSAT_BASE
                ckpt = AvailCkpt.MUSIC_SPEECH_EPOCH_15_ESC_89_25
            case SubModality.GENERAL:
                amodel = AudioEncoderModel.HTSAT_BASE
                ckpt = AvailCkpt.MUSIC_SPEECH_AUDIOSET_EPOCH_15_ESC_89_98

        self._model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion, device=device, amodel=amodel, tmodel=tmodel
        )

        self._model.load_ckpt(ckpt)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """クエリ文字列の非同期埋め込みを行う。

        Args:
            query (str): クエリ文字列

        Returns:
            Embedding: 埋め込みベクトル
        """
        logger.debug("trace")

        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """単一テキストの同期埋め込みを行う。

        Args:
            text (str): テキスト

        Returns:
            Embedding: 埋め込みベクトル
        """
        logger.debug("trace")

        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """複数テキストの同期埋め込みを行う。

        Args:
            texts (list[str]): テキスト

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        logger.debug("trace")

        return self._model.get_text_embedding(x=texts)

    def _get_query_embedding(self, query: str) -> Embedding:
        """クエリ文字列の同期埋め込みを行う。

        Args:
            query (str): クエリ文字列

        Returns:
            Embedding: 埋め込みベクトル
        """
        logger.debug("trace")

        return self._get_text_embedding(query)

    def get_audio_embedding_batch(
        self, audio_file_paths: list[AudioType], show_progress: bool = False
    ) -> list[Embedding]:
        """音声埋め込みの同期バッチインタフェース。

        MultiModalEmbedding の get_image_embedding_batch がベース。

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス
            show_progress (bool, optional): 進捗の表示。Defaults to False.

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        logger.debug("trace")

        cur_batch: list[AudioType] = []
        result_embeddings: list[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(
                audio_file_paths, show_progress, "Generating audio embeddings"
            )
        )

        for idx, img_file_path in queue_with_progress:
            cur_batch.append(img_file_path)
            if (
                idx == len(audio_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    embeddings = self._get_audio_embeddings(cur_batch)
                    result_embeddings.extend(embeddings)
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: cur_batch,
                            EventPayload.EMBEDDINGS: embeddings,
                        },
                    )
                cur_batch = []

        return result_embeddings

    def _get_audio_embeddings(
        self, audio_file_paths: list[AudioType]
    ) -> list[Embedding]:
        """LAION-AI CLAP の API 呼び出しラッパー。

        Args:
            audio_file_paths (list[AudioType]): 音声ファイルパス

        Returns:
            list[Embedding]: 埋め込みベクトル
        """
        logger.debug("trace")

        vecs = self._model.get_audio_embedding_from_filelist(
            x=audio_file_paths, use_tensor=False
        )

        return [v.tolist() for v in vecs]

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
        logger.debug("trace")

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
        """この関数の実装時点で、LAION-AI CLAP には未だ同期インタフェースしかない"""
        logger.debug("trace")

        return self.get_audio_embedding_batch(audio_file_paths)
