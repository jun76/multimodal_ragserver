from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
)


def _nodes_with_scores_from_result(res) -> List[NodeWithScore]:
    out: List[NodeWithScore] = []

    nodes: List[BaseNode] = getattr(res, "nodes", []) or []
    sims: Optional[List[float]] = getattr(res, "similarities", None)
    dists: Optional[List[float]] = getattr(res, "distances", None)

    # 距離のみ返す実装に対しては擬似類似度を生成（単調減少ならOK）
    if (not sims or len(sims) == 0) and dists:
        sims = [1.0 / (1.0 + float(d)) for d in dists]

    # それでも無ければ None にしておく（後段で扱えるように）
    if sims is None:
        sims = [None] * len(nodes)  # type: ignore

    for i, node in enumerate(nodes):
        score = float(sims[i]) if i < len(sims) and sims[i] is not None else None  # type: ignore
        out.append(NodeWithScore(node=node, score=score))
    return out


class AudioEncoders:
    def __init__(self, audio_encoder, text_to_audio_encoder):
        self.audio_encoder = audio_encoder  # (wav)->np.ndarray[d]
        self.text_to_audio_encoder = text_to_audio_encoder  # (text)->np.ndarray[d]


class AudioRetriever(BaseRetriever):
    def __init__(
        self,
        index: VectorStoreIndex,
        enc: AudioEncoders,
        top_k: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self._index = index
        self._vs: BasePydanticVectorStore = index.storage_context.vector_store
        self._enc = enc
        self._top_k = top_k
        self._filters = metadata_filters
        self._normalize = normalize

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return super()._retrieve(query_bundle)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        if not self._normalize:
            return v
        n = np.linalg.norm(v)
        return v if n == 0.0 else v / n  # type: ignore

    def _query_vs(self, qvec: np.ndarray):
        q = VectorStoreQuery(
            query_embedding=self._norm(qvec).tolist(),
            similarity_top_k=self._top_k,
            filters=self._filters,
        )
        return self._vs.query(q)

    async def _aquery_vs(self, qvec: np.ndarray):
        q = VectorStoreQuery(
            query_embedding=self._norm(qvec).tolist(),
            similarity_top_k=self._top_k,
            filters=self._filters,
        )
        return await self._vs.aquery(q)

    def _search_by_embedding(self, qvec: np.ndarray) -> List[NodeWithScore]:
        res = self._query_vs(qvec)
        return _nodes_with_scores_from_result(res)

    async def _asearch_by_embedding(self, qvec: np.ndarray) -> List[NodeWithScore]:
        res = await self._aquery_vs(qvec)
        return _nodes_with_scores_from_result(res)

    # BaseRetriever の抽象メソッド
    async def aretrieve(self, query: Union[str, QueryBundle]) -> List[NodeWithScore]:
        text = query.query_str if isinstance(query, QueryBundle) else str(query)
        qvec = self._enc.text_to_audio_encoder(text)
        return await self._asearch_by_embedding(qvec)

    async def atext_to_audio_retrieve(self, text: str) -> List[NodeWithScore]:
        qvec = self._enc.text_to_audio_encoder(text)
        return await self._asearch_by_embedding(qvec)

    async def aaudio_to_audio_retrieve(
        self, waveform_16k: np.ndarray
    ) -> List[NodeWithScore]:
        qvec = self._enc.audio_encoder(waveform_16k)
        return await self._asearch_by_embedding(qvec)
