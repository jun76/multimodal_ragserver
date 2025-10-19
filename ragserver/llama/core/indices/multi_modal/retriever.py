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


class AudioRetriever(BaseRetriever):
    pass
