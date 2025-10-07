from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from typing import Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from llama_index.core.schema import Document
from llama_index.readers.web.simple_web.base import SimpleWebPageReader
from llama_index.readers.web.sitemap.base import SitemapReader

from ragserver.core.metadata import EMBTYPE_TEXT
from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import (
    WebTextMetaData,
    assert_required_keys,
    stable_id_from,
)
from ragserver.core.names import PROJECT_NAME
from ragserver.core.util import cool_down
from ragserver.ingest.file_loader import FileLoader
from ragserver.ingest.loader import Exts, Loader
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


class HTMLLoader(Loader):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        file_loader: FileLoader,
        load_asset: bool = True,
        req_per_sec: int = 2,
        store: Optional[VectorStoreManager] = None,
        timeout: int = 30,
        user_agent: str = PROJECT_NAME,
        same_origin: bool = True,
    ):
        """HTML を読み込み、ドキュメントを生成するためのクラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
            file_loader (FileLoader): ファイル読み込み用
            load_asset (bool, optional): アセットを読み込むか。 Defaults to True.
            req_per_sec (int): 秒間リクエスト数。Defaults to 2.
            store (Optional[VectorStoreManager], opitonal): 登録済みソースの判定に使用。Defaults to None.
            timeout (int, optional): タイムアウト秒。 Defaults to 30.
            user_agent (str, optional): GET リクエスト時の user agent。 Defaults to PROJECT_NAME.
            same_origin (bool, optional): True なら同一オリジンのみ対象。 Defaults to True.
        """
        logger.debug("trace")

        Loader.__init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._file_loader = file_loader
        self._load_asset = load_asset
        self._req_per_sec = req_per_sec
        self._store = store
        self._timeout = timeout
        self._user_agent = user_agent
        self._same_origin = same_origin

    def _request_get(self, url: str) -> requests.Response:
        """request.get() のラッパー。

        Args:
            url (str): 対象 URL

        Raises:
            requests.HTTPError: GET 時の例外
            RuntimeError: フェッチ失敗

        Returns:
            requests.Response: 取得した Response データ
        """
        logger.debug("trace")

        # TODO: 非同期関数化
        try:
            # 追加のヘッダあればここ
            res = requests.get(
                url,
                timeout=self._timeout,
                headers={
                    "User-Agent": self._user_agent,
                    "Sec-Fetch-Site": "same-origin" if self._same_origin else "none",
                },
            )
            res.raise_for_status()
        except requests.HTTPError as e:
            text = res.text if res is not None else ""
            raise requests.HTTPError(f"response={text}") from e
        except requests.RequestException as e:
            raise RuntimeError("failed to fetch url") from e
        finally:
            cool_down(1 / self._req_per_sec)

        return res

    def _fetch_text(
        self,
        url: str,
    ) -> str:
        """HTML を取得し、テキストを返す。

        Args:
            url (str): 取得先 URL

        Returns:
            str: レスポンス本文
        """
        logger.debug("trace")

        try:
            res = self._request_get(url)
        except Exception as e:
            logger.exception(e)
            return ""

        return res.text

    def _is_file_url(self, url: str) -> bool:
        """ファイルへの直リンクか。

        Args:
            url (str): 対象 URL

        Returns:
            bool: True は直リンク
        """
        logger.debug("trace")

        parsed = urlparse(url)
        path = parsed.path or ""

        if not path:
            return False

        filename = path.rstrip("/").rsplit("/", maxsplit=1)[-1]

        return "." in filename and filename != ""

    def _gather_asset_links(
        self,
        html: str,
        base_url: str,
        allowed_exts: Set[str],
        limit: int = 20,
    ) -> list[str]:
        """HTML からアセット URL を収集する。

        Args:
            html (str): HTML 文字列
            base_url (str): 相対 URL 解決用の基準 URL
            allowed_exts (Set[str]): 許可される拡張子集合（ドット付き小文字）
            limit (int, optional): 返却する最大件数. Defaults to 20.

        Returns:
            list[str]: 収集した絶対 URL
        """
        logger.debug("trace")

        seen = set()
        out = []
        base = urlparse(base_url)

        def add(u: str) -> None:
            if not u:
                return

            try:
                absu = urljoin(base_url, u)
                if absu in seen:
                    return

                pu = urlparse(absu)
                if self._same_origin and (pu.scheme, pu.netloc) != (
                    base.scheme,
                    base.netloc,
                ):
                    return

                pth = pu.path.lower()
                if any(pth.endswith(ext) for ext in allowed_exts):
                    seen.add(absu)
                    out.append(absu)
            except Exception:
                return

        soup = BeautifulSoup(html, "html.parser")

        for img in soup.find_all("img"):
            add(img.get("src"))  # type: ignore

        for a in soup.find_all("a"):
            add(a.get("href"))  # type: ignore

        for src in soup.find_all("source"):
            ss = src.get("srcset")  # type: ignore
            if ss:
                cand = ss.split(",")[0].strip().split(" ")[0]  # type: ignore
                add(cand)

        return out[: max(0, int(limit))]

    def _download_direct_linked_file(
        self, url: str, max_asset_bytes: int = 100 * 1024 * 1024
    ) -> Optional[str]:
        """直リンクのファイルをダウンロードし、ローカルの一時ファイルパスを返す。

        Args:
            url (str): 対象 URL
            max_asset_bytes (int, optional): データサイズ上限。 Defaults to 100*1024*1024.

        Returns:
            Optional[str]: ローカルの一時ファイルパス
        """
        logger.debug("trace")

        try:
            res = self._request_get(url)
        except Exception as e:
            logger.exception(e)
            return None

        body = res.content or b""
        if len(body) > int(max_asset_bytes):
            logger.warning(
                f"skip asset (too large): {len(body)} Bytes > {int(max_asset_bytes)}"
            )
            return None

        ext = os.path.splitext(url.split("?")[0])[1].lower()
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, prefix=f"{PROJECT_NAME}_", suffix=ext
            ) as f:
                f.write(body)
                path = f.name
        except OSError as e:
            logger.exception(e)
            return None

        return path

    async def _load_direct_linked_file(
        self,
        url: str,
        base_url: Optional[str] = None,
    ) -> list[Document]:
        """ファイルへの直リンクからドキュメントを生成する。

        Args:
            url (str): 対象 URL
            base_url (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: ドキュメント（テキスト、画像混在）
        """
        logger.debug("trace")

        path = self._download_direct_linked_file(url)
        if path is None:
            logger.warning("downloading file failure")
            return []

        try:
            fl = FileLoader(
                chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
            )
            docs = await fl.load_from_path(path)
        except Exception as e:
            logger.exception(e)
            return []

        return docs

    def _load_html_text(
        self, url: str, base_url: Optional[str] = None
    ) -> list[Document]:
        """HTML を読み込み、テキスト部分からドキュメントを生成する。

        Args:
            url (str): 対象 URL
            base_url (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: 生成したドキュメントリスト
        """
        logger.debug("trace")

        # TODO: メタ整理
        reader = SimpleWebPageReader(html_to_text=True)

        return reader.load_data([url])

    def _load_html_asset_files(
        self,
        base_url: str,
    ) -> list[Document]:
        """HTML を読み込み、アセットファイルからドキュメントを生成する。

        Args:
            base_url (str): 対象 URL

        Returns:
            list[Document]: ドキュメント（テキスト、画像混在）
        """
        logger.debug("trace")

        html = self._fetch_text(base_url)
        urls = self._gather_asset_links(
            html=html, base_url=base_url, allowed_exts=Exts.SUPPORTED_EXTS
        )

        docs = []
        for url in urls:
            temp = self._load_direct_linked_file(
                url=url,
                base_url=base_url,
            )
            docs.extend(temp)

        return docs

    def _load_from_site(
        self,
        url: str,
    ) -> list[Document]:
        """単一サイトからコンテンツを取得し、ドキュメントを生成する。

        Args:
            url (str): 対象 URL

        Returns:
            list[Document]: ドキュメント（テキスト、画像混在）
        """
        logger.debug("trace")

        if urlparse(url).scheme not in {"http", "https"}:
            logger.error("invalid URL. expected http(s)://*")
            return []

        if self._store and self._store.skip_update(url):
            logger.info(f"skip loading: source exists ({url})")
            return []

        if self._is_file_url(url):
            return self._load_direct_linked_file(
                url=url,
                base_url=url,
            )

        docs = self._load_html_text(url)

        if self._load_asset:
            temp = self._load_html_asset_files(base_url=url)
            docs.extend(temp)

        logger.info(f"loaded {len(docs)} docs from {url}")

        return docs

    def load_from_url(
        self,
        url: str,
    ) -> list[Document]:
        """URL からコンテンツを取得し、ドキュメントを生成する。
        サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

        Args:
            url (str): 対象 URL

        Returns:
            list[Document]: ドキュメント（テキスト、画像混在）
        """
        logger.debug("trace")

        # .xml 以外は単一のサイトとして読み込み
        if not url.endswith(".xml"):
            return self._load_from_site(url)

        # 以下、サイトマップの解析と読み込み
        try:
            loader = SitemapReader()
            urls = loader._parse_sitemap(url)
        except Exception as e:
            logger.exception(e)
            return []

        docs = []
        for url in urls:
            temp = self._load_from_site(url)
            docs.extend(temp)

        return docs

    def load_from_url_list(
        self,
        list_path: str,
    ) -> list[Document]:
        """URL リストに記載の複数サイトからコンテンツを取得し、ドキュメントを生成する。

        Args:
            list_path (str): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

        Returns:
            list[Document]: ドキュメント（テキスト、画像混在）
        """
        logger.debug("trace")

        urls = self._read_sources_from_file(list_path)

        docs = []
        for url in urls:
            temp = self.load_from_url(url)
            docs.extend(temp)

        return docs
