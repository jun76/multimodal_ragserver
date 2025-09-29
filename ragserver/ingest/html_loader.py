from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from typing import Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_core.documents import Document

from ragserver.core.metadata import EMBTYPE_TEXT
from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import (
    WebTextMetaData,
    assert_required_keys,
    stable_id_from,
)
from ragserver.core.names import PROJECT_NAME
from ragserver.core.util import cool_down
from ragserver.ingest.file_loader import FileLoaderForWeb
from ragserver.ingest.loader import Exts, Loader
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


class HTMLLoader(Loader):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        file_loader: FileLoaderForWeb,
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
            file_loader (FileLoaderForWeb): ファイル読み込み用
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
    ) -> str:
        """直リンクのファイルをダウンロードし、ローカルの一時ファイルパスを返す。

        Args:
            url (str): 対象 URL
            max_asset_bytes (int, optional): データサイズ上限。 Defaults to 100*1024*1024.

        Returns:
            str: ローカルの一時ファイルパス
        """
        logger.debug("trace")

        try:
            res = self._request_get(url)
        except Exception as e:
            logger.exception(e)
            return ""

        body = res.content or b""
        if len(body) > int(max_asset_bytes):
            logger.warning(
                f"skip asset (too large): {len(body)} Bytes > {int(max_asset_bytes)}"
            )
            return ""

        ext = os.path.splitext(url.split("?")[0])[1].lower()
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, prefix=f"{PROJECT_NAME}_", suffix=ext
            ) as f:
                f.write(body)
                path = f.name
        except OSError as e:
            logger.exception(e)
            return ""

        return path

    def _load_direct_linked_file(
        self,
        url: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]:
        """ファイルへの直リンクからドキュメントを生成する。

        Args:
            url (str): 対象 URL
            space_key (str): テキスト用空間キー
            space_key_multi (Optional[str], optional): マルチモーダル用空間キー。 Defaults to None.
            base_url (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            tuple[list[Document], list[Document]]: テキストドキュメント, マルチモーダルドキュメント
        """
        logger.debug("trace")

        if not self._required_ext(url, Exts.SUPPORTED_EXTS):
            logger.warning(
                f"not supported ext. {' '.join(Exts.SUPPORTED_EXTS)} are supported"
            )
            return [], []

        path = self._download_direct_linked_file(url=url)
        if path == "":
            logger.warning("downloading file failure")
            return [], []

        try:
            if self._required_ext(path, Exts.TEXT_FILE_EXTS):
                return (
                    self._file_loader.load_text_file(
                        path=path, space_key=space_key, source=url, base_source=base_url
                    ),
                    [],
                )
            if self._required_ext(path, Exts.MARKDOWN_FILE_EXTS):
                return (
                    self._file_loader.load_markdown_file(
                        path=path, space_key=space_key, source=url, base_source=base_url
                    ),
                    [],
                )
            if space_key_multi and self._required_ext(path, Exts.IMAGE_FILE_EXTS):
                return [], self._file_loader.load_image_file(
                    path=path,
                    space_key=space_key_multi,
                    source=url,
                    base_source=base_url,
                )
            if self._required_ext(path, Exts.PDF_FILE_EXTS):
                return self._file_loader.load_pdf_file(
                    path=path,
                    space_key=space_key,
                    space_key_multi=space_key_multi,
                    source=url,
                    base_source=base_url,
                )
        except Exception as e:
            logger.exception(e)
            return [], []
        else:
            logger.info(f"load nothing from {url}")

        return [], []

    def _load_html_text(
        self, url: str, space_key: str, base_url: Optional[str] = None
    ) -> list[Document]:
        """HTML を読み込み、テキスト部分からドキュメントを生成する。

        Args:
            url (str): 対象 URL
            space_key (str): 空間キー
            base_url (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: 生成したドキュメントリスト
        """
        logger.debug("trace")

        try:
            # beautifulsoup のパラメータもここで指定可能
            # プロキシ設定もここ
            loader = WebBaseLoader(web_path=url, requests_per_second=self._req_per_sec)
            docs = loader.load()
        except Exception as e:
            logger.exception(e)
            return []

        # 一定長のチャンクに整形
        docs = self._split_text(docs)

        # メタ情報付与と ID 採番
        own_meta = asdict(
            WebTextMetaData(
                source=url,
                embed_type=EMBTYPE_TEXT,
                space_key=space_key,
                base_source=base_url if base_url else url,
            )
        )

        for i, d in enumerate(docs):
            d.metadata.update(own_meta)
            d.metadata[MK.CHUNK_NO] = i

            # 最後に id を生成・追加
            d.metadata[MK.ID] = stable_id_from(
                f"{d.metadata[MK.EMBED_TYPE]}"
                f"::{d.metadata[MK.SOURCE]}"
                f"::{d.metadata[MK.CHUNK_NO]}"
            )
            assert_required_keys(d.metadata, WebTextMetaData)

        return docs

    def _load_html_asset_files(
        self,
        base_url: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]:
        """HTML を読み込み、アセットファイルからドキュメントを生成する。

        Args:
            base_url (str): 対象 URL
            space_key (str): テキスト用空間キー
            space_key_multi (Optional[str], optional): マルチモーダル用空間キー。 Defaults to None.

        Returns:
            tuple[list[Document], list[Document]]: テキストドキュメント, マルチモーダルドキュメント
        """
        logger.debug("trace")

        html = self._fetch_text(base_url)
        urls = self._gather_asset_links(
            html=html, base_url=base_url, allowed_exts=Exts.SUPPORTED_EXTS
        )

        # 最上位ループ内で複数ソースをまたいで _source_cache を共有したいため
        # ここでは _source_cache.clear() しないこと。
        text_docs = []
        image_docs = []
        for url in urls:
            if url in self._source_cache:
                continue

            temp_text, temp_image = self._load_direct_linked_file(
                url=url,
                space_key=space_key,
                space_key_multi=space_key_multi,
                base_url=base_url,
            )
            text_docs.extend(temp_text)
            image_docs.extend(temp_image)
            self._source_cache.add(url)

        return text_docs, image_docs

    def _load_from_site(
        self,
        url: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]:
        """単一サイトからコンテンツを取得し、ドキュメントを生成する。

        Args:
            url (str): 対象 URL
            space_key (str): テキスト用空間キー
            space_key_multi (Optional[str], optional): マルチモーダル用空間キー。 Defaults to None.

        Returns:
            tuple[list[Document], list[Document]]: テキストドキュメント, マルチモーダルドキュメント
        """
        logger.debug("trace")

        if urlparse(url).scheme not in {"http", "https"}:
            logger.error("invalid URL. expected http(s)://*")
            return [], []

        if self._store and self._store.skip_update(url):
            logger.info(f"skip loading: source exists ({url})")
            return [], []

        if self._is_file_url(url):
            return self._load_direct_linked_file(
                url=url,
                space_key=space_key,
                space_key_multi=space_key_multi,
                base_url=url,
            )

        text_docs = self._load_html_text(url=url, space_key=space_key)

        if self._load_asset:
            buf, image_docs = self._load_html_asset_files(
                base_url=url,
                space_key=space_key,
                space_key_multi=space_key_multi,
            )
            text_docs.extend(buf)
        else:
            image_docs = []

        logger.info(
            f"loaded {len(text_docs)} text docs, {len(image_docs)} image docs from {url}"
        )

        return text_docs, image_docs

    def load_from_url(
        self,
        url: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]:
        """URL からコンテンツを取得し、ドキュメントを生成する。
        サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

        Args:
            url (str): 対象 URL
            space_key (str): テキスト用空間キー
            space_key_multi (Optional[str], optional): マルチモーダル用空間キー。 Defaults to None.

        Returns:
            tuple[list[Document], list[Document]]: テキストドキュメント, マルチモーダルドキュメント
        """
        logger.debug("trace")

        # .xml 以外は単一のサイトとして読み込み
        if not url.endswith(".xml"):
            return self._load_from_site(
                url=url,
                space_key=space_key,
                space_key_multi=space_key_multi,
            )

        # 以下、サイトマップの解析と読み込み
        try:
            loader = SitemapLoader(url)
            soup = loader.scrape(parser="xml")
        except Exception as e:
            logger.exception(e)
            return [], []

        entries = loader.parse_sitemap(soup)
        urls = [entry["loc"] for entry in entries if "loc" in entry]

        # 最上位ループの一つ。キャッシュを空にしてから使う。
        self._source_cache.clear()
        text_docs = []
        image_docs = []
        for url in urls:
            temp_text, temp_image = self._load_from_site(
                url=url,
                space_key=space_key,
                space_key_multi=space_key_multi,
            )
            text_docs.extend(temp_text)
            image_docs.extend(temp_image)

        return text_docs, image_docs

    def load_from_url_list(
        self,
        list_path: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]:
        """URL リストに記載の複数サイトからコンテンツを取得し、ドキュメントを生成する。

        Args:
            list_path (str): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）
            space_key (str): テキスト用空間キー
            space_key_multi (Optional[str], optional): マルチモーダル用空間キー。 Defaults to None.

        Returns:
            tuple[list[Document], list[Document]]: テキストドキュメント, マルチモーダルドキュメント
        """
        logger.debug("trace")

        urls = self._read_sources_from_file(list_path)

        # 最上位ループの一つ。キャッシュを空にしてから使う。
        self._source_cache.clear()
        text_docs = []
        image_docs = []
        for url in urls:
            temp_text, temp_image = self.load_from_url(
                url=url,
                space_key=space_key,
                space_key_multi=space_key_multi,
            )
            text_docs.extend(temp_text)
            image_docs.extend(temp_image)

        return text_docs, image_docs
