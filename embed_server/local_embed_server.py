import base64
import io
import logging

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger("local_embed_server")

# 利用モデルの定義
MODEL_NAME = "openai/clip-vit-base-patch32"
SUPPORTED_MODELS = {MODEL_NAME}

# モデル・プロセッサのロード
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

logger.info("loading CLIP model: %s", MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(
    MODEL_NAME, dtype=DTYPE if DEVICE.type == "cuda" else None
)
model.to(DEVICE)  # type: ignore
model.eval()
torch.set_grad_enabled(False)

# uvicorn local_embed_server:app --host 0.0.0.0 --port 8001
app = FastAPI(title="local_embed_server", version="1.0")


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str]  # テキスト or Data URI 形式の画像


def extract_text_embedding(text: str) -> list[float]:
    """テキスト埋め込みを取得する。"""

    logger.debug("trace")

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,  # type: ignore
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.squeeze(0).cpu().tolist()


def extract_image_embedding(image: Image.Image) -> list[float]:
    """画像埋め込みを取得する。"""

    logger.debug("trace")

    inputs = processor(images=image, return_tensors="pt")

    # pixel_values を fp16 に（CUDA時）。attention_mask等があればそのままでOK
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DEVICE, dtype=DTYPE)  # type: ignore
    inputs = {
        k: (v.to(DEVICE) if k != "pixel_values" else v) for k, v in inputs.items()
    }

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.squeeze(0).float().cpu().tolist()  # 返却はfloat32で安定化


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
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="model mismatch")

    embeddings = []

    for idx, item in enumerate(req.input):
        if item.startswith("data:image"):
            header, b64data = item.split(",", 1)
            img = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
            vec = extract_image_embedding(img)
        else:
            vec = extract_text_embedding(item)

        embeddings.append({"object": "embedding", "index": idx, "embedding": vec})

    return {
        "object": "list",
        "data": embeddings,
        "model": MODEL_NAME,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
