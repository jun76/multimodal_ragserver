"""LlamaIndex ベースの multimodal_ragserver メインモジュール。

このモジュールは、LlamaIndex を使用してマルチモーダルRAGシステムを構築し、
FastAPI経由でREST APIとして公開します。

既存のFastAPI仕様を維持しつつ、内部実装をLlamaIndexに最適化しています。
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

import aiofiles
import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi_mcp.server import FastApiMCP
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ragserver.config import get_config
from ragserver.core import names
from ragserver.hf_rerank import HFRerank
from ragserver.hfclip_embedding import HFCLIPEmbedding
from ragserver.logger import logger

__all__ = ["app"]

# ログレベルの設定
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("unstructured.trace").setLevel(logging.WARNING)


class ReloadRequest(BaseModel):
    """リロードリクエストのモデル。"""

    target: str
    name: str


class QueryTextRequest(BaseModel):
    """テキストクエリリクエストのモデル。"""

    query: str
    topk: Optional[int] = None


class QueryImageRequest(BaseModel):
    """画像クエリリクエストのモデル。"""

    path: str
    topk: Optional[int] = None


class PathRequest(BaseModel):
    """パスリクエストのモデル。"""

    path: str


class URLRequest(BaseModel):
    """URLリクエストのモデル。"""

    url: str


# FastAPI アプリケーション
app = FastAPI(title=names.PROJECT_NAME, version="2.0-llamaindex")

# グローバル変数
_index: Optional[MultiModalVectorStoreIndex] = None
_storage_context: Optional[StorageContext] = None
_embed_model: Any = None
_rerank: Any = None
_request_lock = threading.Lock()


def _create_embed_model(name: Optional[str] = None) -> Any:
    """埋め込みモデルを生成する。

    Args:
        name (Optional[str], optional): 埋め込みプロバイダ名。 Defaults to None.

    Returns:
        Any: 埋め込みモデルインスタンス

    Raises:
        RuntimeError: 設定の読み込み、インスタンス生成に失敗した場合
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to load configuration: {e}") from e

    if name:
        cfg.emped_provider = name

    match cfg.emped_provider:
        case names.OPENAI_EMBED_NAME:
            logger.info(
                f"Creating OpenAI embedding model: {cfg.openai_embed_model_text}"
            )
            return OpenAIEmbedding(
                model=cfg.openai_embed_model_text,
                api_key=cfg.openai_api_key,
                api_base=cfg.openai_base_url if cfg.openai_base_url else None,
            )
        case names.COHERE_EMBED_NAME:
            logger.info(
                f"Creating Cohere embedding model: {cfg.cohere_embed_model_text}"
            )
            return CohereEmbedding(
                model_name=cfg.cohere_embed_model_text,
                api_key=cfg.cohere_api_key,
            )
        case names.HFCLIP_EMBED_NAME:
            logger.info(
                f"Creating HFCLIP embedding model: {cfg.hfclip_embed_model_text}"
            )
            return HFCLIPEmbedding(
                base_url=cfg.hfclip_embed_base_url,
                text_model=cfg.hfclip_embed_model_text,
                image_model=cfg.hfclip_embed_model_image,
            )
        case _:
            raise RuntimeError(f"Unsupported embedding provider: {cfg.emped_provider}")


def _create_vector_stores(name: Optional[str] = None) -> tuple[Any, Any]:
    """テキストと画像用のベクトルストアを生成する。

    Args:
        name (Optional[str], optional): ベクトルストア名。 Defaults to None.

    Returns:
        tuple[Any, Any]: (text_store, image_store)

    Raises:
        RuntimeError: 設定の読み込み、インスタンス生成に失敗した場合
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to load configuration: {e}") from e

    if name:
        cfg.vector_store = name

    match cfg.vector_store:
        case names.CHROMA_STORE_NAME:
            logger.info("Creating Chroma vector stores")
            # Chroma クライアントの作成
            if cfg.chroma_host and cfg.chroma_port:
                # リモートモード
                client = chromadb.HttpClient(
                    host=cfg.chroma_host,
                    port=cfg.chroma_port,
                    ssl=False,
                )
            else:
                # ローカルモード
                client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)

            # テキスト用と画像用のコレクションを作成
            text_collection = client.get_or_create_collection("text_collection")
            image_collection = client.get_or_create_collection("image_collection")

            text_store = ChromaVectorStore(chroma_collection=text_collection)
            image_store = ChromaVectorStore(chroma_collection=image_collection)

            return text_store, image_store

        case names.PGVECTOR_STORE_NAME:
            logger.info("Creating PGVector vector stores")
            # テキスト用と画像用のテーブルを作成
            text_store = PGVectorStore.from_params(
                database=cfg.pg_database,
                host=cfg.pg_host,
                password=cfg.pg_password,
                port=str(cfg.pg_port),
                user=cfg.pg_user,
                table_name="text_embeddings",
            )

            image_store = PGVectorStore.from_params(
                database=cfg.pg_database,
                host=cfg.pg_host,
                password=cfg.pg_password,
                port=str(cfg.pg_port),
                user=cfg.pg_user,
                table_name="image_embeddings",
            )

            return text_store, image_store

        case _:
            raise RuntimeError(f"Unsupported vector store: {cfg.vector_store}")


def _create_rerank(name: Optional[str] = None) -> Optional[Any]:
    """リランクモデルを生成する。

    Args:
        name (Optional[str], optional): リランクプロバイダ名。 Defaults to None.

    Returns:
        Optional[Any]: リランクモデルインスタンス

    Raises:
        RuntimeError: 設定の読み込みに失敗した場合
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to load configuration: {e}") from e

    if name:
        cfg.rerank_provider = name

    match cfg.rerank_provider:
        case names.COHERE_RERANK_NAME:
            logger.info(f"Creating Cohere rerank model: {cfg.cohere_rerank_model}")
            return CohereRerank(
                model=cfg.cohere_rerank_model,
                api_key=cfg.cohere_api_key,
                top_n=cfg.topk,
            )
        case names.HF_RERANK_NAME:
            logger.info(f"Creating HF rerank model: {cfg.hf_rerank_model}")
            return HFRerank(
                base_url=cfg.hf_rerank_base_url,
                model=cfg.hf_rerank_model,
                top_n=cfg.topk,
            )
        case _:
            logger.info("No rerank model specified")
            return None


def _initialize_index() -> None:
    """インデックスとストレージコンテキストを初期化する。

    Raises:
        RuntimeError: 初期化に失敗した場合
    """
    logger.debug("trace")

    global _index
    global _storage_context
    global _embed_model
    global _rerank

    try:
        cfg = get_config()

        # 埋め込みモデルの作成
        _embed_model = _create_embed_model()
        Settings.embed_model = _embed_model

        # テキスト分割の設定
        Settings.text_splitter = SentenceSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        # ベクトルストアの作成
        text_store, image_store = _create_vector_stores()

        # ストレージコンテキストの作成
        _storage_context = StorageContext.from_defaults(
            vector_store=text_store,
            image_store=image_store,
        )

        # インデックスの作成（空のインデックス）
        _index = MultiModalVectorStoreIndex.from_documents(
            [],
            storage_context=_storage_context,
        )

        # リランクモデルの作成
        _rerank = _create_rerank()

        logger.info("Index initialization completed")

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to initialize index: {e}") from e


# 起動時にインデックスを初期化
_initialize_index()


def _nodes_to_response(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    """NodeWithScore リストを JSON 返却可能な辞書リストへ変換する。

    Args:
        nodes (list[NodeWithScore]): 変換対象ノード

    Returns:
        list[dict[str, Any]]: JSON 変換済みノードリスト
    """
    logger.debug("trace")

    return [
        {
            "page_content": node.node.get_content(),
            "metadata": node.node.metadata,
            "score": node.score if node.score is not None else 0.0,
        }
        for node in nodes
    ]


@app.get("/v1/health")
async def health() -> dict[str, Any]:
    """ragserver の稼働状態を返却する。

    Returns:
        dict[str, Any]: 結果
    """
    logger.debug("trace")

    global _index
    global _embed_model
    global _rerank

    try:
        cfg = get_config()
        store_name = cfg.vector_store
        embed_name = cfg.emped_provider
        rerank_name = cfg.rerank_provider if _rerank else "none"
    except Exception as e:
        return {"status": f"{e}"}

    return {
        "status": "ok",
        "store": store_name,
        "embed": embed_name,
        "rerank": rerank_name,
        "framework": "llamaindex",
    }


@app.post("/v1/reload", operation_id="reload")
async def reload(payload: ReloadRequest) -> dict[str, Any]:
    """各種インスタンスをリロードする。

    Args:
        payload (ReloadRequest): リロード内容

    Raises:
        HTTPException(500): 初期化やファイル作成に失敗した場合

    Returns:
        dict[str, Any]: 結果
    """
    logger.debug("trace")

    global _index
    global _storage_context
    global _embed_model
    global _rerank

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            match payload.target:
                case "store":
                    # ベクトルストアを再作成
                    text_store, image_store = _create_vector_stores(payload.name)
                    _storage_context = StorageContext.from_defaults(
                        vector_store=text_store,
                        image_store=image_store,
                    )
                    _index = MultiModalVectorStoreIndex.from_documents(
                        [],
                        storage_context=_storage_context,
                    )
                case "embed":
                    # 埋め込みモデルを再作成
                    _embed_model = _create_embed_model(payload.name)
                    Settings.embed_model = _embed_model
                case "rerank":
                    # リランクモデルを再作成
                    _rerank = _create_rerank(payload.name)
                case _:
                    traceback.print_exc()
                    raise ValueError(
                        "invalid argument. specify store | embed | rerank ."
                    )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"reload failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok"}


@app.post("/v1/upload", operation_id="upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """ファイルを（クライアントから）アップロードする。

    Args:
        files (list[UploadFile], optional): ファイル群。 Defaults to File(...).

    Raises:
        HTTPException(500): 初期化やファイル作成に失敗した場合
        HTTPException(400): ファイル名が空の場合

    Returns:
        dict[str, Any]: 結果
    """
    logger.debug("trace")

    try:
        cfg = get_config()
        upload_dir = Path(cfg.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"upload init failure: {e}") from e

    await run_in_threadpool(_request_lock.acquire)
    try:
        results = []
        for f in files:
            if f.filename is None:
                traceback.print_exc()
                raise HTTPException(status_code=400, detail="filename is not specified")

            try:
                safe = Path(f.filename).name
                path = upload_dir / safe
                async with aiofiles.open(path, "wb") as buf:
                    while True:
                        chunk = await f.read(1024 * 1024)
                        if not chunk:
                            break
                        await buf.write(chunk)

            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500, detail=f"upload failure: {e}"
                ) from e
            finally:
                await f.close()

            results.append(
                {
                    "filename": safe,
                    "content_type": f.content_type,
                    "save_path": str(path),
                }
            )

        return {"files": results}
    finally:
        _request_lock.release()


@app.post("/v1/query/text", operation_id="query_text")
async def query_text(payload: QueryTextRequest) -> dict[str, Any]:
    """テキストクエリによる検索を実行する。

    Args:
        payload (QueryTextRequest): クエリ内容

    Returns:
        dict[str, Any]: 検索結果

    Raises:
        HTTPException: 設定の読み込みに失敗した場合
    """
    logger.debug("trace")

    global _index
    global _rerank

    if _index is None:
        raise HTTPException(status_code=500, detail="Index is not initialized")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"failed to load config: {e}"
        ) from e

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            # リトリーバーを作成
            topk = payload.topk or cfg.topk
            retriever = _index.as_retriever(
                similarity_top_k=topk * cfg.topk_rerank_scale if _rerank else topk,
            )

            # 検索実行
            nodes = await run_in_threadpool(
                retriever.retrieve,
                payload.query,
            )

            # リランクを適用
            if _rerank:
                query_bundle = QueryBundle(query_str=payload.query)
                nodes = await run_in_threadpool(
                    _rerank.postprocess_nodes,
                    nodes,
                    query_bundle,
                )
            else:
                # リランクなしの場合はtopkで切り詰め
                nodes = nodes[:topk]

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(nodes)}


@app.post("/v1/query/text_multi", operation_id="query_text_multi")
async def query_text_multi(payload: QueryTextRequest) -> dict[str, Any]:
    """テキストクエリでマルチモーダル検索を実行する（画像を検索）。

    Args:
        payload (QueryTextRequest): クエリ内容

    Returns:
        dict[str, Any]: 検索結果

    Raises:
        HTTPException: 設定の読み込みに失敗、または検索処理に失敗した場合
    """
    logger.debug("trace")

    global _index
    global _embed_model
    global _rerank

    if _index is None:
        raise HTTPException(status_code=500, detail="Index is not initialized")

    # HFCLIP または Cohere の場合のみマルチモーダル検索をサポート
    if not isinstance(_embed_model, (HFCLIPEmbedding, CohereEmbedding)):
        raise HTTPException(
            status_code=500,
            detail="Multimodal embeddings is not supported. Use HFCLIP or Cohere.",
        )

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"failed to load config: {e}"
        ) from e

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            # テキストクエリで画像を検索
            topk = payload.topk or cfg.topk

            # 画像用のリトリーバーを作成
            retriever = _index.as_retriever(
                image_similarity_top_k=(
                    topk * cfg.topk_rerank_scale if _rerank else topk
                ),
            )

            # 検索実行
            nodes = await run_in_threadpool(
                retriever.retrieve,
                payload.query,
            )

            # 画像ノードのみをフィルタリング
            image_nodes = [
                node
                for node in nodes
                if node.node.metadata.get("file_type", "").startswith("image/")
            ]

            # リランクを適用
            if _rerank and len(image_nodes) > 0:
                query_bundle = QueryBundle(query_str=payload.query)
                image_nodes = await run_in_threadpool(
                    _rerank.postprocess_nodes,
                    image_nodes,
                    query_bundle,
                )
            else:
                # リランクなしの場合はtopkで切り詰め
                image_nodes = image_nodes[:topk]

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(image_nodes)}


@app.post("/v1/query/image", operation_id="query_image")
async def query_image(payload: QueryImageRequest) -> dict[str, Any]:
    """画像クエリによる検索を実行する（類似画像を検索）。

    Args:
        payload (QueryImageRequest): クエリ内容

    Returns:
        dict[str, Any]: 検索結果

    Raises:
        HTTPException: 設定の読み込みに失敗、または検索処理に失敗した場合
    """
    logger.debug("trace")

    global _index
    global _embed_model

    if _index is None:
        raise HTTPException(status_code=500, detail="Index is not initialized")

    # HFCLIP または Cohere の場合のみマルチモーダル検索をサポート
    if not isinstance(_embed_model, (HFCLIPEmbedding, CohereEmbedding)):
        raise HTTPException(
            status_code=500,
            detail="Multimodal embeddings is not supported. Use HFCLIP or Cohere.",
        )

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"failed to load config: {e}"
        ) from e

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            # 画像の埋め込みベクトルを取得
            if isinstance(_embed_model, HFCLIPEmbedding):
                query_embedding = await run_in_threadpool(
                    _embed_model.get_image_embedding,
                    payload.path,
                )
            else:
                # Cohere の場合は画像パスを直接使用
                query_embedding = None

            # 画像ストアから検索
            topk = payload.topk or cfg.topk
            retriever = _index.as_retriever(
                image_similarity_top_k=topk,
            )

            # 検索実行（画像クエリの場合はリランクなし）
            if query_embedding:
                # カスタム埋め込みを使用
                nodes = await run_in_threadpool(
                    retriever.retrieve,
                    payload.path,
                )
            else:
                # Cohere の場合
                nodes = await run_in_threadpool(
                    retriever.retrieve,
                    payload.path,
                )

            # 画像ノードのみをフィルタリング
            image_nodes = [
                node
                for node in nodes
                if node.node.metadata.get("file_type", "").startswith("image/")
            ][:topk]

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(image_nodes)}


@app.post("/v1/ingest/path", operation_id="ingest_path")
async def ingest_path(payload: PathRequest) -> dict[str, str]:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        payload (PathRequest): 対象パス

    Returns:
        dict[str, str]: 実行結果

    Raises:
        HTTPException: インジェスト処理に失敗した場合
    """
    logger.debug("trace")

    global _index

    if _index is None:
        raise HTTPException(status_code=500, detail="Index is not initialized")

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            # SimpleDirectoryReader でドキュメントを読み込み
            path = Path(payload.path)
            reader = SimpleDirectoryReader(
                input_dir=str(path) if path.is_dir() else None,
                input_files=[str(path)] if path.is_file() else None,
                recursive=True,
            )

            documents = await run_in_threadpool(reader.load_data)

            # インデックスに追加
            for doc in documents:
                await run_in_threadpool(_index.insert, doc)

            logger.info(f"Ingested {len(documents)} documents from {payload.path}")

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok", "documents_count": f"{len(documents)}"}


@app.post("/v1/ingest/path_list", operation_id="ingest_path_list")
async def ingest_path_list(payload: PathRequest) -> dict[str, str]:
    """path リストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Returns:
        dict[str, str]: 実行結果

    Raises:
        HTTPException: インジェスト処理に失敗した場合
    """
    logger.debug("trace")

    global _index

    if _index is None:
        raise HTTPException(status_code=500, detail="Index is not initialized")

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            # パスリストを読み込み
            with open(payload.path, "r") as f:
                paths = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]

            total_docs = 0
            for path_str in paths:
                # SimpleDirectoryReader でドキュメントを読み込み
                path = Path(path_str)
                reader = SimpleDirectoryReader(
                    input_dir=str(path) if path.is_dir() else None,
                    input_files=[str(path)] if path.is_file() else None,
                    recursive=True,
                )

                documents = await run_in_threadpool(reader.load_data)

                # インデックスに追加
                for doc in documents:
                    await run_in_threadpool(_index.insert, doc)

                total_docs += len(documents)
                logger.info(f"Ingested {len(documents)} documents from {path_str}")

            logger.info(f"Total ingested {total_docs} documents")

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok", "documents_count": f"{total_docs}"}


@app.post("/v1/ingest/url", operation_id="ingest_url")
async def ingest_url(payload: URLRequest) -> dict[str, str]:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。

    注意: 現在は基本的な実装のみ。サイトマップ対応は今後追加予定。

    Args:
        payload (URLRequest): 対象 URL

    Returns:
        dict[str, str]: 実行結果

    Raises:
        HTTPException: インジェスト処理に失敗した場合
    """
    logger.debug("trace")

    global _index

    if _index is None:
        raise HTTPException(status_code=500, detail="Index is not initialized")

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            reader = SimpleWebPageReader()
            documents = await run_in_threadpool(reader.load_data, [payload.url])

            # インデックスに追加
            for doc in documents:
                await run_in_threadpool(_index.insert, doc)

            logger.info(f"Ingested {len(documents)} documents from {payload.url}")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok", "documents_count": f"{len(documents)}"}


@app.post("/v1/ingest/url_list", operation_id="ingest_url_list")
async def ingest_url_list(payload: PathRequest) -> dict[str, str]:
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Returns:
        dict[str, str]: 実行結果

    Raises:
        HTTPException: インジェスト処理に失敗した場合
    """
    logger.debug("trace")

    global _index

    if _index is None:
        raise HTTPException(status_code=500, detail="Index is not initialized")

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            # URLリストを読み込み
            with open(payload.path, "r") as f:
                urls = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]

            reader = SimpleWebPageReader()
            total_docs = 0

            for url in urls:
                documents = await run_in_threadpool(reader.load_data, [url])

                # インデックスに追加
                for doc in documents:
                    await run_in_threadpool(_index.insert, doc)

                total_docs += len(documents)
                logger.info(f"Ingested {len(documents)} documents from {url}")

            logger.info(f"Total ingested {total_docs} documents")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok", "documents_count": f"{total_docs}"}


# FastAPI アプリを MCP サーバとして公開
_mcp_server = FastApiMCP(
    app,
    name=names.PROJECT_NAME,
    include_operations=["query_text", "query_text_multi", "query_image"],
)
_mcp_server.mount_http()
