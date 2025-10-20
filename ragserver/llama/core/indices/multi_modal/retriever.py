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


from ragserver.logger import logger


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
        """埋め込みモデルから利用可能なエンコーダを生成する。

        Args:
            embed_model (Optional[BaseEmbedding]): 埋め込みモデル

        Returns:
            AudioEncoders: テキスト・音声エンコーダを含むインスタンス
        """

        if embed_model is None:
            return cls()

        if hasattr(embed_model, "aget_text_embedding_batch"):

            async def encode_text(queries: list[str]) -> list[Embeddings]:
                return await embed_model.aget_text_embedding_batch(texts=queries)  # type: ignore[attr-defined]

        if hasattr(embed_model, "aget_audio_embedding_batch"):

            async def encode_audio(paths: list[str]) -> list[Embeddings]:
                return await embed_model.aget_audio_embedding_batch(  # type: ignore[attr-defined]
                    audio_file_paths=paths
                )

        return cls(text_encoder=encode_text, audio_encoder=encode_audio)

    async def aencode_text(self, queries: list[str]) -> list[Embeddings]:
        """テキストクエリ群を埋め込みベクトルへ変換する。

        Args:
            queries (list[str]): テキストクエリのリスト

        Returns:
            list[Embeddings]: テキスト埋め込みベクトルのリスト
        """

        if self.text_encoder is None:
            raise RuntimeError("text encoder for audio retrieval is not available")

        return await self.text_encoder(queries)

    async def aencode_audio(self, paths: list[str]) -> list[Embeddings]:
        """音声ファイル群を埋め込みベクトルへ変換する。

        Args:
            paths (list[str]): 音声ファイルパスのリスト

        Returns:
            list[Embeddings]: 音声埋め込みベクトルのリスト
        """

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
        """リトリーバーを初期化する。

        Args:
            index (VectorStoreIndex): ベクトルストアインデックス
            top_k (int, optional): 類似ドキュメントの最大取得件数。Defaults to 10.
            encoders (Optional[AudioEncoders], optional): 事前構築済みエンコーダ。Defaults to None.
            filters (Optional[MetadataFilters], optional): メタデータフィルタ条件。Defaults to None.
            vector_store_query_mode (VectorStoreQueryMode, optional): クエリモード。Defaults to VectorStoreQueryMode.DEFAULT.
            node_ids (Optional[list[str]], optional): 対象ノード ID の制限。Defaults to None.
            doc_ids (Optional[list[str]], optional): 対象ドキュメント ID の制限。Defaults to None.
            vector_store_kwargs (Optional[dict], optional): ベクトルストアへ渡す追加パラメータ。Defaults to None.
        """

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

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """埋め込み済みクエリを利用して同期検索を実施する。

        Args:
            query_bundle (QueryBundle): クエリ情報

        Raises:
            NotImplementedError: 未実装

        Returns:
            list[NodeWithScore]: 類似ノードのリスト
        """
        raise NotImplementedError("AudioRetriever only supports async retrieval APIs")

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """埋め込み済みクエリを利用して非同期検索を実施する。

        Args:
            query_bundle (QueryBundle): クエリ情報

        Returns:
            list[NodeWithScore]: 類似ノードのリスト
        """

        if query_bundle.embedding is None:
            raise RuntimeError("embedding is required for async retrieval")

        return await self._aquery_with_embedding(
            embedding=query_bundle.embedding,
            query_str=query_bundle.query_str,
        )

    async def atext_to_audio_retrieve(
        self, query: Union[str, QueryBundle]
    ) -> list[NodeWithScore]:
        """テキストクエリから音声モダリティを検索する。

        Args:
            query (Union[str, QueryBundle]): テキストクエリまたは QueryBundle

        Returns:
            list[NodeWithScore]: 類似ノードのリスト
        """

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
        """音声ファイルをクエリとして検索する。

        Args:
            audio_path (str): クエリ音声ファイルパス

        Returns:
            list[NodeWithScore]: 類似ノードのリスト
        """

        embedding = (await self._encoders.aencode_audio([audio_path]))[0]  # type: ignore

        return await self._aquery_with_embedding(embedding=embedding, query_str="")

    async def _aquery_with_embedding(
        self,
        embedding: Sequence[float],
        query_str: str,
    ) -> list[NodeWithScore]:
        """埋め込みベクトルを用いてベクトルストアを検索する。

        Args:
            embedding (Sequence[float]): クエリ埋め込みベクトル
            query_str (str): クエリ文字列

        Returns:
            list[NodeWithScore]: 類似ノードのリスト
        """

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
        """検索結果を NodeWithScore のリストへ変換する。

        Args:
            query_result (VectorStoreQueryResult): ベクトルストアの検索結果

        Returns:
            list[NodeWithScore]: 変換後のノード一覧
        """

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
