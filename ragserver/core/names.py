from __future__ import annotations

PROJECT_NAME = "ragserver"

# store
CHROMA_STORE_NAME = "chroma"
PGVECTOR_STORE_NAME = "pgvector"

# embeddings
# ! 変更すると空間キーの字列が変わって別空間（ingest やり直し）になるので注意 !
LOCAL_EMBED_NAME = "local"
OPENAI_EMBED_NAME = "openai"
COHERE_EMBED_NAME = "cohere"

# rerank
LOCAL_RERANK_NAME = "local"
COHERE_RERANK_NAME = "cohere"
