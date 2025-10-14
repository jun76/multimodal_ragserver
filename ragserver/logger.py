from __future__ import annotations

import logging

from ragserver.core.names import PROJECT_NAME

# ログレベルは他のパッケージによって上書きされる場合があるため、main での起動完了後にセットすること
logging.basicConfig(
    format="%(asctime)s - %(levelname)7s - %(name)-30s %(pathname)-80s %(funcName)-30s: %(message)s",
)

logger = logging.getLogger(PROJECT_NAME)
