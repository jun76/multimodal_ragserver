from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi_mcp.server import FastApiMCP
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ragserver.config import get_config
from ragserver.core import names
from ragserver.embed.clip_embedding_manager import ClipEmbeddingManager
from ragserver.embed.cohere_embedding_manager import CohereEmbeddingManager
from ragserver.embed.embedding_manager import EmbeddingManager
from ragserver.embed.multimodal_embedding_manager import MultiModalEmbeddingManager
from ragserver.embed.openai_embedding_manager import OpenAIEmbeddingManager
from ragserver.ingest import ingest
from ragserver.ingest.file_loader import FileLoader
from ragserver.ingest.html_loader import HTMLLoader
from ragserver.logger import logger
from ragserver.rerank.cohere_rerank_manager import CohereRerankManager
from ragserver.rerank.flagembedding_rerank_manager import FlagEmbeddingRerankManager
from ragserver.rerank.rerank_manager import RerankManager
from ragserver.retrieval import retriever
from ragserver.stractured_store.sqlite_manager import SQLiteManager
from ragserver.stractured_store.structured_store_manager import StructuredStoreManager
from ragserver.vector_store.chroma_manager import ChromaManager
from ragserver.vector_store.pgvector_manager import PgVectorManager
from ragserver.vector_store.vector_store_manager import VectorStoreManager

__all__ = ["app"]

logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("unstructured.trace").setLevel(logging.WARNING)


class ReloadRequest(BaseModel):
    target: str
    name: str


class QueryTextRequest(BaseModel):
    query: str
    topk: Optional[int] = None


class QueryImageRequest(BaseModel):
    path: str
    topk: Optional[int] = None


class PathRequest(BaseModel):
    path: str


class URLRequest(BaseModel):
    url: str


# uvicorn ragserver.main:app --host 0.0.0.0 --port 8000
app = FastAPI(title=names.PROJECT_NAME, version="1.0")


def _create_embed(name: Optional[str] = None) -> EmbeddingManager:
    """埋め込み管理インスタンスを作成する。

    Args:
        name (Optional[str], optional): 埋め込み管理名。Defaults to None.

    Raises:
        RuntimeError: 設定の読み込み、インスタンス生成に失敗した場合

    Returns:
        EmbeddingsManager: 生成された埋め込み管理
    """
    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to load configuration: {e}") from e

    if name:
        cfg.embed_provider = name

    match cfg.embed_provider:
        case names.OPENAI_EMBED_NAME:
            return OpenAIEmbeddingManager(
                model_text=cfg.openai_embed_model_text,
                base_url=cfg.openai_base_url,
            )
        case names.COHERE_EMBED_NAME:
            return CohereEmbeddingManager(
                model_text=cfg.cohere_embed_model_text,
                model_image=cfg.cohere_embed_model_image,
            )
        case names.CLIP_EMBED_NAME:
            return ClipEmbeddingManager(
                model_text=cfg.clip_embed_model_text,
                model_image=cfg.clip_embed_model_image,
            )
        case _:
            raise RuntimeError(f"failed to create store")


_embed = _create_embed()


def _create_meta_store(embed: EmbeddingManager) -> StructuredStoreManager:
    """メタデータ専用ストアのインスタンスを作成する。

    Raises:
        RuntimeError: 設定の読み込み、インスタンス生成に失敗した場合

    Returns:
        StructuredStoreManager: 構造化ストア管理
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to load configuration: {e}") from e

    try:
        meta_store = SQLiteManager(knowledgebase_name=cfg.knowledgebase_name)
        if isinstance(embed, MultiModalEmbeddingManager):
            meta_store.prepare_with(
                space_key_text=embed.space_key_text,
                space_key_multi=embed.space_key_multi,
            )
        else:
            meta_store.prepare_with(
                space_key_text=embed.space_key_text,
            )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to prepare metadata store: {e}") from e

    return meta_store


_meta_store = _create_meta_store(_embed)


def _create_vector_store(
    embed: EmbeddingManager, name: Optional[str] = None
) -> VectorStoreManager:
    """ベクトルストアのインスタンスを作成する。

    Args:
        embed (EmbeddingManager): 埋め込み管理
        name (Optional[str], optional): ベクトルストア名。Defaults to None.

    Raises:
        RuntimeError: 設定の読み込み、インスタンス生成に失敗した場合

    Returns:
        VectorStoreManager: 新しいインスタンス
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to load configuration: {e}") from e

    if name:
        cfg.vector_store = name

    try:
        match cfg.vector_store:
            case names.PGVECTOR_STORE_NAME:
                vector_store = PgVectorManager(
                    host=cfg.pg_host,
                    port=cfg.pg_port,
                    dbname=cfg.pg_database,
                    user=cfg.pg_user,
                    password=cfg.pg_password,
                    check_update=cfg.check_update,
                    knowledgebase_name=cfg.knowledgebase_name,
                )
            case names.CHROMA_STORE_NAME:
                vector_store = ChromaManager(
                    persist_directory=cfg.chroma_persist_dir,
                    host=cfg.chroma_host,
                    port=cfg.chroma_port,
                    check_update=cfg.check_update,
                    knowledgebase_name=cfg.knowledgebase_name,
                )
            case _:
                raise RuntimeError(f"failed to create store")

        global _meta_store
        vector_store.prepare_with(
            embed=embed, meta_store=_meta_store, limit=cfg.load_limit
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to prepare vector store: {e}") from e

    return vector_store


_vector_store = _create_vector_store(_embed)


def _create_rerank(name: Optional[str] = None) -> Optional[RerankManager]:
    """リランク管理インスタンスを作成する。

    Args:
        name (Optional[str], optional): リランク管理名。Defaults to None.

    Raises:
        RuntimeError: 設定の読み込みに失敗した場合

    Returns:
        Optional[RerankManager]: 生成されたリランク管理
    """
    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"failed to load configuration: {e}") from e

    if name:
        cfg.rerank_provider = name

    match cfg.rerank_provider:
        case names.FLAGEMBEDDING_RERANK_NAME:
            return FlagEmbeddingRerankManager(
                model=cfg.flagembedding_rerank_model, topk=cfg.topk
            )
        case names.COHERE_RERANK_NAME:
            return CohereRerankManager(cfg.cohere_rerank_model, topk=cfg.topk)
        case _:
            return None


_rerank = _create_rerank()


_request_lock = threading.Lock()


def _nodes_to_response(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    """NodeWithScore リストを JSON 返却可能な辞書リストへ変換する。

    Args:
        nodes (list[NodeWithScore]): 変換対象ドキュメント

    Returns:
        list[dict[str, Any]]: JSON 変換済みドキュメントリスト
    """
    logger.debug("trace")

    return [
        {
            "text": node.text,
            "metadata": node.metadata,
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

    return {
        "status": "ok",
        "store": _vector_store.name,
        "embed": _embed.name,
        "rerank": _rerank.name if _rerank else "none",
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

    global _meta_store
    global _vector_store
    global _embed
    global _rerank
    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            match payload.target:
                case "store":
                    _vector_store = _create_vector_store(
                        embed=_embed, name=payload.name
                    )
                case "embed":
                    _embed = _create_embed(payload.name)
                    if isinstance(_embed, MultiModalEmbeddingManager):
                        _meta_store.prepare_with(
                            space_key_text=_embed.space_key_text,
                            space_key_multi=_embed.space_key_multi,
                        )
                    else:
                        _meta_store.prepare_with(
                            space_key_text=_embed.space_key_text,
                        )

                    cfg = get_config()
                    _vector_store.prepare_with(
                        embed=_embed, meta_store=_meta_store, limit=cfg.load_limit
                    )
                case "rerank":
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
        files (list[UploadFile], optional): ファイル群。Defaults to File(...).

    Raises:
        HTTPException(500): 初期化やファイル作成に失敗した場合
        HTTPException(400): ファイル名が空の場合

    Returns:
        dict[str, Any]: 結果
    """
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

    Raises:
        HTTPException: 設定の読み込みに失敗した場合

    Returns:
        dict[str, Any]: 検索結果
    """
    logger.debug("trace")

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
            docs = await retriever.query_text(
                query=payload.query,
                store=_vector_store,
                topk=payload.topk or cfg.topk,
                rerank=_rerank,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(docs)}


@app.post("/v1/query/text_multi", operation_id="query_text_multi")
async def query_text_multi(payload: QueryTextRequest) -> dict[str, Any]:
    """テキストクエリでマルチモーダル検索を実行する。

    Args:
        payload (QueryTextRequest): クエリ内容

    Raises:
        HTTPException: 設定の読み込みに失敗、または検索処理に失敗した場合

    Returns:
        dict[str, Any]: 検索結果
    """
    logger.debug("trace")

    if not isinstance(_embed, MultiModalEmbeddingManager):
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="multimodal embeddings is not supported",
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
            docs = await retriever.query_text_multi(
                query=payload.query,
                store=_vector_store,
                topk=payload.topk or cfg.topk,
                rerank=_rerank,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(docs)}


@app.post("/v1/query/image", operation_id="query_image")
async def query_image(payload: QueryImageRequest) -> dict[str, Any]:
    """画像クエリによる検索を実行する。

    Args:
        payload (QueryImageRequest): クエリ内容

    Returns:
        dict[str, Any]: 検索結果
    """
    logger.debug("trace")

    if not isinstance(_embed, MultiModalEmbeddingManager):
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="multimodal embeddings is not supported",
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
            docs = await retriever.query_image(
                path=payload.path,
                store=_vector_store,
                topk=payload.topk or cfg.topk,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(docs)}


@app.post("/v1/ingest/path", operation_id="ingest_path")
async def ingest_path(payload: PathRequest) -> dict[str, str]:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        payload (PathRequest): 対象パス

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"failed to load config: {e}"
        ) from e

    file_loader = FileLoader(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    await run_in_threadpool(_request_lock.acquire)
    try:
        await ingest.ingest_from_path(
            path=payload.path,
            store=_vector_store,
            file_loader=file_loader,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok"}


@app.post("/v1/ingest/path_list", operation_id="ingest_path_list")
async def ingest_path_list(payload: PathRequest) -> dict[str, str]:
    """path リストに記載の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"failed to load config: {e}"
        ) from e

    file_loader = FileLoader(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    await run_in_threadpool(_request_lock.acquire)
    try:
        await ingest.ingest_from_path_list(
            list_path=payload.path,
            store=_vector_store,
            file_loader=file_loader,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok"}


@app.post("/v1/ingest/url", operation_id="ingest_url")
async def ingest_url(payload: URLRequest) -> dict[str, str]:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        payload (URLRequest): 対象 URL

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"failed to load config: {e}"
        ) from e

    file_loader = FileLoader(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    html_loader = HTMLLoader(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        file_loader=file_loader,
        store=_vector_store,
        user_agent=cfg.user_agent,
    )

    await run_in_threadpool(_request_lock.acquire)
    try:
        await ingest.ingest_from_url(
            url=payload.url,
            store=_vector_store,
            html_loader=html_loader,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok"}


@app.post("/v1/ingest/url_list", operation_id="ingest_url_list")
async def ingest_url_list(payload: PathRequest) -> dict[str, str]:
    """URL リストに記載の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        payload (PathRequest): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("trace")

    try:
        cfg = get_config()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"failed to load config: {e}"
        ) from e

    file_loader = FileLoader(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    html_loader = HTMLLoader(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        file_loader=file_loader,
        store=_vector_store,
        user_agent=cfg.user_agent,
    )

    await run_in_threadpool(_request_lock.acquire)
    try:
        await ingest.ingest_from_url_list(
            list_path=payload.path,
            store=_vector_store,
            html_loader=html_loader,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok"}


# FastAPI アプリを MCP サーバとして公開
_mcp_server = FastApiMCP(
    app,
    name=names.PROJECT_NAME,
    include_operations=["query_text", "query_text_multi", "query_image"],
)
_mcp_server.mount_http()
