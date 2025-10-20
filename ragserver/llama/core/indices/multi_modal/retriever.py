from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional, Sequence, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

Embeddings = Sequence[float]


@dataclass
class AudioEncoders:
    """音声検索用エンコーダ群。

    text_encoder / audio_encoder には、それぞれクエリのリストを受け取り
    埋め込みベクトルのリストを返す非同期関数を渡す。
    """

    text_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = None
    audio_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = None

    @classmethod
    def from_embed_model(cls, embed_model: Optional[BaseEmbedding]) -> "AudioEncoders":
        """埋め込みモデルから利用可能なエンコーダを生成する。"""
        if embed_model is None:
            return cls()

        text_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = (
            None
        )
        audio_encoder: Optional[Callable[[list[str]], Awaitable[list[Embeddings]]]] = (
            None
        )

        if hasattr(embed_model, "aget_text_embedding_batch"):

            async def encode_text(queries: list[str]) -> list[Embeddings]:
                return await embed_model.aget_text_embedding_batch(texts=queries)  # type: ignore[attr-defined]

            text_encoder = encode_text

        if hasattr(embed_model, "aget_audio_embedding_batch"):

            async def encode_audio(paths: list[str]) -> list[Embeddings]:
                return await embed_model.aget_audio_embedding_batch(  # type: ignore[attr-defined]
                    audio_file_paths=paths
                )

            audio_encoder = encode_audio

        return cls(text_encoder=text_encoder, audio_encoder=audio_encoder)

    async def aencode_text(self, queries: list[str]) -> list[Embeddings]:
        if self.text_encoder is None:
            raise RuntimeError("text encoder for audio retrieval is not available")
        return await self.text_encoder(queries)

    async def aencode_audio(self, paths: list[str]) -> list[Embeddings]:
        if self.audio_encoder is None:
            raise RuntimeError("audio encoder for audio retrieval is not available")
        return await self.audio_encoder(paths)


class AudioRetriever(BaseRetriever):
    """音声モダリティ専用リトリーバー。"""

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 10,
        encoders: Optional[AudioEncoders] = None,
        *,
        filters: Optional[MetadataFilters] = None,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        node_ids: Optional[list[str]] = None,
        doc_ids: Optional[list[str]] = None,
        vector_store_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize retriever."""

        self._index = index
        self._vector_store = index.vector_store
        self._docstore = index.docstore
        self._top_k = top_k
        self._filters = filters
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._mode = VectorStoreQueryMode(vector_store_query_mode)
        self._kwargs = vector_store_kwargs or {}

        if encoders is None:
            # NOTE: VectorStoreIndex keeps the embedding model on _embed_model
            embed_model = getattr(index, "_embed_model", None)
            encoders = AudioEncoders.from_embed_model(embed_model)

        self._encoders = encoders

    # BaseRetriever インタフェース（同期版）は今回利用しないため、利用者に警告を出す
    def _retrieve(
        self, query_bundle: QueryBundle
    ) -> list[NodeWithScore]:  # pragma: no cover - sync API not used
        raise NotImplementedError("AudioRetriever only supports async retrieval APIs")

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if query_bundle.embedding is None:
            raise RuntimeError("embedding is required for async retrieval")
        return await self._aquery_with_embedding(
            embedding=query_bundle.embedding,
            query_str=query_bundle.query_str,
        )

    async def atext_to_audio_retrieve(
        self, query: Union[str, QueryBundle]
    ) -> list[NodeWithScore]:
        if isinstance(query, QueryBundle):
            query_str = query.query_str
            embedding = query.embedding
            if embedding is None:
                if query.embedding_strs:
                    texts = list(query.embedding_strs)
                else:
                    texts = [query.query_str]
                embedding = (await self._encoders.aencode_text(texts))[0]  # type: ignore
            return await self._aquery_with_embedding(
                embedding=embedding, query_str=query_str
            )

        embedding = (await self._encoders.aencode_text([query]))[0]  # type: ignore
        return await self._aquery_with_embedding(embedding=embedding, query_str=query)

    async def aaudio_to_audio_retrieve(self, audio_path: str) -> list[NodeWithScore]:
        embedding = (await self._encoders.aencode_audio([audio_path]))[0]  # type: ignore
        return await self._aquery_with_embedding(embedding=embedding, query_str="")

    async def _aquery_with_embedding(
        self,
        embedding: Sequence[float],
        query_str: str,
    ) -> list[NodeWithScore]:
        query = VectorStoreQuery(
            query_embedding=list(embedding),
            similarity_top_k=self._top_k,
            node_ids=self._node_ids,
            doc_ids=self._doc_ids,
            query_str=query_str,
            mode=self._mode,
            filters=self._filters,
        )

        query_result = await self._vector_store.aquery(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)

    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> list[NodeWithScore]:
        nodes: Iterable[BaseNode] = query_result.nodes or []
        nodes = list(nodes)

        # docstore を利用可能なら node を再取得
        for idx, node in enumerate(nodes):
            if node is None:
                continue
            node_id = node.node_id
            if self._docstore.document_exists(node_id):
                nodes[idx] = self._docstore.get_node(node_id)  # type: ignore[assignment]

        node_with_scores: list[NodeWithScore] = []
        for idx, node in enumerate(nodes):
            score: Optional[float] = None
            if query_result.similarities is not None and idx < len(
                query_result.similarities
            ):
                score = query_result.similarities[idx]
            node_with_scores.append(NodeWithScore(node=node, score=score))

        return node_with_scores
