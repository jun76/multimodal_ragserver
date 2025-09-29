from __future__ import annotations

import time

from ragserver.logger import logger


def cool_down(interval: float = 0.5) -> None:
    """連送防止用の sleep を挿入する。

    Args:
        interval (float, optional): sleep 時間（秒）。Defaults to 0.5.
    """
    logger.debug("trace")

    time.sleep(interval)
