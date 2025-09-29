from __future__ import annotations

import logging

LOGGER_NAME = "ragclient"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)7s - %(name)-20s %(pathname)s:%(lineno)d %(funcName)s: %(message)s",
)

logger = logging.getLogger(LOGGER_NAME)
