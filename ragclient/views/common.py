from __future__ import annotations

from typing import Any, Optional

from ragclient.api_client import RagServerClient
from ragclient.logger import logger

__all__ = ["emojify_robot", "save_uploaded_files"]


def emojify_robot(s: str) -> str:
    """ロボットの絵文字がテキストとして表示されないように整形
    参考：https://github.com/streamlit/streamlit/issues/11390

    Args:
        s (str): ロボットの絵文字を含むかもしれない文字列

    Returns:
        str: 整形後の文字列
    """
    return s.replace("\U0001f916", "\U0001f916" + "\ufe0f")  # 🤖


def save_uploaded_files(client: RagServerClient, files: list[Any]) -> list[str]:
    """アップロード済みファイルを保存し、ragserver 上の保存パスを返す。

    Args:
        client (RagServerClient): ragserver API クライアント
        files (list[Any]): Streamlit のアップロードファイルオブジェクト

    Returns:
        list[str]: 保存したファイルパス一覧

    Raises:
        RuntimeError: 応答データが不正な場合
    """
    logger.debug("trace")

    payload: list[tuple[str, bytes, Optional[str]]] = []
    for uploaded in files:
        data = uploaded.getvalue()
        payload.append((uploaded.name, data, getattr(uploaded, "type", None)))

    if not payload:
        return []

    response = client.upload(payload)
    entries = response.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("ragserver upload response is invalid")

    saved: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            raise RuntimeError("ragserver upload response item is invalid")
        save_path = item.get("save_path")
        if not isinstance(save_path, str) or save_path == "":
            raise RuntimeError("ragserver upload save_path is invalid")
        saved.append(save_path)

    if len(saved) != len(payload):
        raise RuntimeError("ragserver upload file count mismatch")

    return saved
