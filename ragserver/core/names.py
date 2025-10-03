from __future__ import annotations

PROJECT_NAME = "ragserver"

# store
CHROMA_STORE_NAME = "chroma"
PGVECTOR_STORE_NAME = "pgvector"

# embeddings
# ! 変更すると空間キーの字列が変わって別空間（ingest やり直し）になるので注意 !
HFCLIP_EMBED_NAME = "hfclip"
OPENAI_EMBED_NAME = "openai"
COHERE_EMBED_NAME = "cohere"

# rerank
HF_RERANK_NAME = "hf"
COHERE_RERANK_NAME = "cohere"
