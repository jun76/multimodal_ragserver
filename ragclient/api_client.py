from __future__ import annotations

from typing import Any, Optional

import requests

from ragclient.logger import logger


class RagServerClient:
    def __init__(self, base_url: str) -> None:
        """ragserver の REST API を呼び出すクライアント。

        Args:
            base_url (str): ragserver へのベース URL
        """
        logger.debug("trace")

        self._base_url = base_url.rstrip("/")

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST リクエストを送信し、JSON 応答を辞書で返す。

        Args:
            endpoint (str): ベース URL からの相対パス
            payload (dict[str, Any]): POST ボディ

        Raises:
            RuntimeError: リクエスト失敗または JSON 解析失敗時

        Returns:
            dict[str, Any]: JSON 応答
        """
        logger.debug("trace")

        url = f"{self._base_url}{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.exception(e)
            raise RuntimeError("failed to call ragserver endpoint") from e

        try:
            return response.json()
        except ValueError as e:
            raise RuntimeError("ragserver response is not json") from e

    def _post_form_data_json(
        self, endpoint: str, files: list[tuple[str, tuple[str, bytes, str]]]
    ) -> dict[str, Any]:
        """multipart/form-data POST を送信し、JSON 応答を辞書で返す。

        Args:
            endpoint (str): ベース URL からの相対パス
            files (list[tuple[str, tuple[str, bytes, str]]]): multipart/form-data 用ファイル情報

        Raises:
            RuntimeError: リクエスト失敗または JSON 解析失敗時

        Returns:
            dict[str, Any]: JSON 応答
        """
        logger.debug("trace")

        url = f"{self._base_url}{endpoint}"
        try:
            response = requests.post(url, files=files, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.exception(e)
            raise RuntimeError("failed to call ragserver endpoint") from e

        try:
            return response.json()
        except ValueError as e:
            raise RuntimeError("ragserver response is not json") from e

    def ingest_path(self, path: str) -> dict[str, Any]:
        """パス指定の取り込み API を呼び出す。

        Args:
            path (str): 取り込み対象パス

        Returns:
            dict[str, Any]: 応答データ
        """
        logger.debug("trace")

        return self._post_json("/ingest/path", {"path": path})

    def ingest_path_list(self, path: str) -> dict[str, Any]:
        """パスリスト指定の取り込み API を呼び出す。

        Args:
            path (str): パスリストのファイルパス

        Returns:
            dict[str, Any]: 応答データ
        """
        logger.debug("trace")

        return self._post_json("/ingest/path_list", {"path": path})

    def ingest_url(self, url: str) -> dict[str, Any]:
        """URL 指定の取り込み API を呼び出す。

        Args:
            url (str): 取り込み対象 URL

        Returns:
            dict[str, Any]: 応答データ
        """
        logger.debug("trace")

        return self._post_json("/ingest/url", {"url": url})

    def ingest_url_list(self, path: str) -> dict[str, Any]:
        """URL リスト指定の取り込み API を呼び出す。

        Args:
            path (str): URL リストファイルのパス

        Returns:
            dict[str, Any]: 応答データ
        """
        logger.debug("trace")

        return self._post_json("/ingest/url_list", {"path": path})

    def query_text(self, query: str, topk: Optional[int] = None) -> dict[str, Any]:
        """テキスト検索 API を呼び出す。

        Args:
            query (str): クエリ文字列
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        logger.debug("trace")

        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/text", payload)

    def query_text_multi(
        self, query: str, topk: Optional[int] = None
    ) -> dict[str, Any]:
        """マルチモーダル検索（テキスト） API を呼び出す。

        Args:
            query (str): クエリ文字列
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        logger.debug("trace")

        payload: dict[str, Any] = {"query": query}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/text_multi", payload)

    def query_image(self, path: str, topk: Optional[int] = None) -> dict[str, Any]:
        """画像検索 API を呼び出す。

        Args:
            path (str): クエリ画像パス
            topk (Optional[int]): 上限件数

        Returns:
            dict[str, Any]: 応答データ
        """
        logger.debug("trace")

        payload: dict[str, Any] = {"path": path}
        if topk is not None:
            payload["topk"] = topk

        return self._post_json("/query/image", payload)

    def reload(self, target: str, name: str) -> dict[str, Any]:
        """リロード API を呼び出す。

        Args:
            target (str): リロード対象
            name (str): 新規起動するプロバイダの名称

        Returns:
            dict[str, Any]: 応答データ
        """
        logger.debug("trace")

        return self._post_json("/reload", {"target": target, "name": name})

    def upload(self, files: list[tuple[str, bytes, Optional[str]]]) -> dict[str, Any]:
        """ファイルアップロード API を呼び出す。

        Args:
            files (list[tuple[str, bytes, Optional[str]]]): アップロードするファイル情報

        Returns:
            dict[str, Any]: 応答データ

        Raises:
            ValueError: 入力値が不正な場合
            RuntimeError: リクエスト失敗または JSON 解析失敗時
        """
        logger.debug("trace")

        if not files:
            raise ValueError("files must not be empty")

        files_payload: list[tuple[str, tuple[str, bytes, str]]] = []
        for name, data, content_type in files:
            if not isinstance(name, str) or name == "":
                raise ValueError("file name must be non-empty string")
            if not isinstance(data, bytes):
                raise ValueError("file data must be bytes")
            mime = content_type or "application/octet-stream"
            files_payload.append(("files", (name, data, mime)))

        return self._post_form_data_json("/upload", files_payload)
