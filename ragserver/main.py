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

from ragserver.config.general_config import GeneralConfig
from ragserver.config.ingest_config import IngestConfig
from ragserver.config.rerank_config import RerankConfig
from ragserver.core.metadata import Modality
from ragserver.embed.embed import create_embed_manager
from ragserver.ingest import ingest
from ragserver.ingest.loader.file_loader import FileLoader
from ragserver.ingest.loader.html_loader import HTMLLoader
from ragserver.logger import logger
from ragserver.meta_store.meta_store import create_meta_store
from ragserver.rerank.rerank import create_rerank_manager
from ragserver.retrieve import retrieve
from ragserver.vector_store.vector_store import create_vector_store_manager

__all__ = ["app"]

logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("unstructured.trace").setLevel(logging.WARNING)


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
app = FastAPI(title=GeneralConfig.project_name, version=GeneralConfig.version)

_embed = create_embed_manager()
logger.info(f"{_embed.name} embed initialized")

_meta_store = create_meta_store()
logger.info("meta store initialized")

_vector_store = create_vector_store_manager(embed=_embed, meta_store=_meta_store)
logger.info(f"{_vector_store.name} vector store initialized")

_rerank = create_rerank_manager()
logger.info(f"{_rerank.name} rerank initialized")

_file_loader = FileLoader(
    chunk_size=IngestConfig.chunk_size,
    chunk_overlap=IngestConfig.chunk_overlap,
    store=_vector_store,
)
logger.info("file loader initialized")

_html_loader = HTMLLoader(
    chunk_size=IngestConfig.chunk_size,
    chunk_overlap=IngestConfig.chunk_overlap,
    file_loader=_file_loader,
    store=_vector_store,
    user_agent=IngestConfig.user_agent,
)
logger.info("html loader initialized")

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
        {"text": node.text, "metadata": node.metadata, "score": node.score}
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
        "rerank": _rerank.name,
    }


@app.post("/v1/upload", operation_id="upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """ファイルを（クライアントから）アップロードする。

    Args:
        files (list[UploadFile], optional): ファイル群。Defaults to File(...).

    Raises:
        HTTPException(500): 初期化やファイル作成に失敗
        HTTPException(400): ファイル名が空

    Returns:
        dict[str, Any]: 結果
    """
    logger.debug("trace")

    try:
        upload_dir = Path(IngestConfig.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"upload init failure: {e}") from e

    await run_in_threadpool(_request_lock.acquire)
    try:
        results = []
        for f in files:
            if f.filename is None:
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
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
    """
    logger.debug("trace")

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            nodes = await retrieve.aquery_text(
                query=payload.query,
                store=_vector_store,
                topk=payload.topk or RerankConfig.topk,
                rerank=_rerank,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(nodes)}


@app.post("/v1/query/text_multi", operation_id="query_text_multi")
async def query_text_multi(payload: QueryTextRequest) -> dict[str, Any]:
    """テキストクエリでマルチモーダル検索を実行する。

    Args:
        payload (QueryTextRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
    """
    logger.debug("trace")

    if Modality.IMAGE not in _embed.modality:
        raise HTTPException(
            status_code=500,
            detail="multimodal embeddings is not supported",
        )

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            nodes = await retrieve.aquery_text_multi(
                query=payload.query,
                store=_vector_store,
                topk=payload.topk or RerankConfig.topk,
                rerank=_rerank,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(nodes)}


@app.post("/v1/query/image", operation_id="query_image")
async def query_image(payload: QueryImageRequest) -> dict[str, Any]:
    """画像クエリによる検索を実行する。

    Args:
        payload (QueryImageRequest): クエリ内容

    Raises:
        HTTPException: 検索処理に失敗

    Returns:
        dict[str, Any]: 検索結果
    """
    logger.debug("trace")

    if Modality.IMAGE not in _embed.modality:
        raise HTTPException(
            status_code=500,
            detail="multimodal embeddings is not supported",
        )

    await run_in_threadpool(_request_lock.acquire)
    try:
        try:
            nodes = await retrieve.aquery_image(
                path=payload.path,
                store=_vector_store,
                topk=payload.topk or RerankConfig.topk,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"query failure: {e}") from e
    finally:
        _request_lock.release()

    return {"documents": _nodes_to_response(nodes)}


@app.post("/v1/ingest/path", operation_id="ingest_path")
async def ingest_path(payload: PathRequest) -> dict[str, str]:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        payload (PathRequest): 対象パス

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("trace")

    await run_in_threadpool(_request_lock.acquire)
    try:
        await ingest.aingest_from_path(
            path=payload.path,
            store=_vector_store,
            file_loader=_file_loader,
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

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("trace")

    await run_in_threadpool(_request_lock.acquire)
    try:
        await ingest.aingest_from_path_list(
            list_path=payload.path,
            store=_vector_store,
            file_loader=_file_loader,
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

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("trace")

    await run_in_threadpool(_request_lock.acquire)
    try:
        await ingest.aingest_from_url(
            url=payload.url,
            store=_vector_store,
            html_loader=_html_loader,
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

    Raises:
        HTTPException: 収集処理に失敗

    Returns:
        dict[str, str]: 実行結果
    """
    logger.debug("trace")

    await run_in_threadpool(_request_lock.acquire)
    try:
        await ingest.aingest_from_url_list(
            list_path=payload.path,
            store=_vector_store,
            html_loader=_html_loader,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ingest failure: {e}") from e
    finally:
        _request_lock.release()

    return {"status": "ok"}


# ログレベルを設定
log_level = getattr(logging, GeneralConfig.log_level.upper(), logging.INFO)
logger.setLevel(log_level)
logger.info("now mcp server is starting up...")

# FastAPI アプリを MCP サーバとして公開
_mcp_server = FastApiMCP(
    app,
    name=GeneralConfig.project_name,
    include_operations=["query_text", "query_text_multi", "query_image"],
)
_mcp_server.mount_http()
