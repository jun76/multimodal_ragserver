import logging
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("hf_rerank_server")

# バッチサイズと最大トークン長（必要に応じて調整）
BATCH_SIZE = 32
MAX_LENGTH = 512  # より長文を扱うならここを上げる

# 利用モデルの定義
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
SUPPORTED_MODELS = {MODEL_NAME}

# モデル・プロセッサのロード
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if (DEVICE.type == "cuda") else torch.float32  # 省メモリ&高速化

logger.info("loading reranker model: %s", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE if DEVICE.type == "cuda" else None,
)
model.to(DEVICE)
model.eval()
torch.set_grad_enabled(False)

# uvicorn hf_rerank_server:app --host 0.0.0.0 --port 8002
app = FastAPI(title="hf_rerank_server", version="1.0")


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    topk: Optional[int] = 10


class RerankResult(BaseModel):
    object: str = "reranking"
    model: str = MODEL_NAME
    query: str
    results: list[dict]  # {index, document, score} を格納
    usage: dict = {"prompt_tokens": 0, "total_tokens": 0}  # OpenAI風の互換ダミー


def _score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """(query, doc) のリストに対して関連度スコアを返す（0..1）。"""
    scores: list[float] = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[i : i + BATCH_SIZE]
        queries = [q for (q, _) in batch_pairs]
        docs = [d for (_, d) in batch_pairs]

        inputs = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits  # (batch, 1) 期待
            # スコアを 0..1 に落とす（bge-reranker は "higher is better" 想定）
            batch_scores = (
                torch.sigmoid(logits.squeeze(-1)).detach().float().cpu().tolist()
            )

        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    return scores


@app.get("/v1/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model": MODEL_NAME}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "hf",
            }
        ],
    }


@app.post("/v1/rerank", response_model=RerankResult)
async def rerank(req: RerankRequest):
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model mismatch (supported: {list(SUPPORTED_MODELS)})",
        )

    if not req.documents:
        return RerankResult(query=req.query, results=[])

    # (query, doc) のペアを作成
    pairs = [(req.query, doc) for doc in req.documents]
    scores = _score_pairs(pairs)

    # index, doc, score を束ねる
    indexed = [
        {"index": i, "document": req.documents[i], "score": float(scores[i])}
        for i in range(len(scores))
    ]
    # スコア降順でソート
    indexed.sort(key=lambda x: x["score"], reverse=True)

    topk = req.topk if req.topk is not None else len(indexed)
    results = indexed[:topk]

    return RerankResult(query=req.query, results=results)
