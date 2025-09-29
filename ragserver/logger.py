from __future__ import annotations

import logging

from ragserver.core.names import PROJECT_NAME

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)7s - %(name)-30s %(pathname)-80s %(funcName)-30s: %(message)s",
)

logger = logging.getLogger(PROJECT_NAME)
