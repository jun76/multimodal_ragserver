from __future__ import annotations

import logging

LOGGER_NAME = "ragclient"

# ログレベルは他のパッケージによって上書きされる場合があるため、main での起動完了後にセットすること
logging.basicConfig(
    format="%(asctime)s - %(levelname)7s - %(name)-20s %(pathname)s:%(lineno)d %(funcName)s: %(message)s",
)

logger = logging.getLogger(LOGGER_NAME)
