from __future__ import annotations

import asyncio
import tempfile
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.readers.web.simple_web.base import SimpleWebPageReader
from llama_index.readers.web.sitemap.base import SitemapReader

from ragserver.config.general_config import GeneralConfig
from ragserver.core.exts import Exts
from ragserver.core.metadata import BasicMetaData
from ragserver.ingest.loader.file_loader import FileLoader
from ragserver.ingest.loader.loader import Loader
from ragserver.logger import logger
from ragserver.vector_store.vector_store_manager import VectorStoreManager


class HTMLLoader(Loader):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        file_loader: FileLoader,
        store: VectorStoreManager,
        load_asset: bool = True,
        req_per_sec: int = 2,
        timeout: int = 30,
        user_agent: str = GeneralConfig.project_name,
        same_origin: bool = True,
    ):
        """HTML を読み込み、ノードを生成するためのクラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
            file_loader (FileLoader): ファイル読み込み用
            store (VectorStoreManager): 登録済みソースの判定に使用
            load_asset (bool, optional): アセットを読み込むか。Defaults to True.
            req_per_sec (int): 秒間リクエスト数。Defaults to 2.
            timeout (int, optional): タイムアウト秒。Defaults to 30.
            user_agent (str, optional): GET リクエスト時の user agent。Defaults to GeneralConfig.project_name.
            same_origin (bool, optional): True なら同一オリジンのみ対象。Defaults to True.
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

    async def _arequest_get(self, url: str) -> requests.Response:
        """HTTP GET を実行する非同期ラッパー。

        Args:
            url (str): 対象 URL

        Raises:
            requests.HTTPError: GET 時の例外
            RuntimeError: フェッチ失敗

        Returns:
            requests.Response: 取得した Response データ
        """
        logger.debug("trace")

        headers = {
            "User-Agent": self._user_agent,
            "Sec-Fetch-Site": "same-origin" if self._same_origin else "none",
        }
        res: Optional[requests.Response] = None

        try:
            res = await asyncio.to_thread(
                requests.get,
                url,
                timeout=self._timeout,
                headers=headers,
            )
            res.raise_for_status()
        except requests.HTTPError as e:
            status = res.status_code if res is not None else "unknown"
            raise requests.HTTPError(f"HTTP {status}") from e
        except requests.RequestException as e:
            raise RuntimeError("failed to fetch url") from e
        finally:
            await asyncio.sleep(1 / self._req_per_sec)

        return res

    async def _afetch_text(
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
            res = await self._arequest_get(url)
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
        allowed_exts: set[str],
        limit: int = 20,
    ) -> list[str]:
        """HTML からアセット URL を収集する。

        Args:
            html (str): HTML 文字列
            base_url (str): 相対 URL 解決用の基準 URL
            allowed_exts (set[str]): 許可される拡張子集合（ドット付き小文字）
            limit (int, optional): 返却する最大件数.Defaults to 20.

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

                path = pu.path.lower()
                if Exts.endswith_exts(path, allowed_exts):
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

    async def _adownload_direct_linked_file(
        self,
        url: str,
        allowed_exts: set[str],
        max_asset_bytes: int = 100 * 1024 * 1024,
    ) -> Optional[str]:
        """直リンクのファイルをダウンロードし、ローカルの一時ファイルパスを返す。

        Args:
            url (str): 対象 URL
            allowed_exts (set[str]): 許可される拡張子集合（ドット付き小文字）
            max_asset_bytes (int, optional): データサイズ上限。Defaults to 100*1024*1024.

        Returns:
            Optional[str]: ローカルの一時ファイルパス
        """
        logger.debug("trace")

        if not Exts.endswith_exts(url, allowed_exts):
            logger.warning(f"unsupported ext. {' '.join(allowed_exts)} are allowed.")
            return None

        try:
            res = await self._arequest_get(url)
        except Exception as e:
            logger.exception(e)
            return None

        body = res.content or b""
        if len(body) > int(max_asset_bytes):
            logger.warning(
                f"skip asset (too large): {len(body)} Bytes > {int(max_asset_bytes)}"
            )
            return None

        ext = Exts.get_ext(url)
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, prefix=f"{GeneralConfig.project_name}_", suffix=ext
            ) as f:
                f.write(body)
                path = f.name
        except OSError as e:
            logger.exception(e)
            return None

        return path

    async def _aload_direct_linked_file(
        self, url: str, base_url: Optional[str] = None
    ) -> Optional[BaseNode]:
        """直リンクのファイルからノードを生成する。

        Args:
            url (str): 対象 URL
            base_url (Optional[str], optional): source の取得元を指定する場合。Defaults to None.

        Returns:
            Optional[BaseNode]: 生成したノード
        """
        logger.debug("trace")

        temp_file_path = await self._adownload_direct_linked_file(
            url=url, allowed_exts=Exts.FETCH_TARGET
        )

        if temp_file_path is None:
            return None

        meta = BasicMetaData()
        meta.file_path = temp_file_path  # MultiModalVectorStoreIndex 参照用
        meta.url = url
        meta.base_source = base_url or ""
        meta.temp_file_path = temp_file_path  # 削除用
        meta.node_lastmod_at = time.time()

        return TextNode(text=url, metadata=meta.to_dict())

    async def _aload_html_text(
        self, url: str, base_url: Optional[str] = None
    ) -> list[BaseNode]:
        """HTML を読み込み、テキスト部分からノードを生成する。

        Args:
            url (str): 対象 URL
            base_url (Optional[str], optional): source の取得元を指定する場合。Defaults to None.

        Returns:
            list[BaseNode]: 生成したノード
        """
        logger.debug("trace")

        try:
            reader = SimpleWebPageReader(html_to_text=True)
            doc = await reader.aload_data([url])

            splitter = SentenceSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                include_metadata=True,
            )
            nodes = splitter.get_nodes_from_documents(doc)
        except Exception as e:
            logger.exception(e)
            return []

        for i, node in enumerate(nodes):
            meta = BasicMetaData()
            meta.chunk_no = i
            meta.url = url
            meta.base_source = base_url or ""
            meta.node_lastmod_at = time.time()
            node.metadata = meta.to_dict()

        return nodes

    async def _aload_html_asset_files(
        self,
        base_url: str,
    ) -> list[BaseNode]:
        """HTML を読み込み、アセットファイルからノードを生成する。

        Args:
            base_url (str): 対象 URL

        Returns:
            list[BaseNode]: 生成したノード
        """
        logger.debug("trace")

        html = await self._afetch_text(base_url)
        urls = self._gather_asset_links(
            html=html, base_url=base_url, allowed_exts=Exts.FETCH_TARGET
        )

        # 最上位ループ内で複数ソースをまたいで _source_cache を共有したいため
        # ここでは _source_cache.clear() しないこと。
        nodes = []
        for url in urls:
            if url in self._source_cache:
                continue

            node = await self._aload_direct_linked_file(url=url, base_url=base_url)
            if node is None:
                logger.warning(f"failed to fetch from {url}, skipped")
                continue

            nodes.append(node)

            # 取得済みキャッシュに追加
            self._source_cache.add(url)

        return nodes

    async def _aload_from_site(
        self,
        url: str,
    ) -> list[BaseNode]:
        """単一サイトからコンテンツを取得し、ノードを生成する。

        Args:
            url (str): 対象 URL

        Returns:
            list[BaseNode]: 生成したノード
        """
        logger.debug("trace")

        if urlparse(url).scheme not in {"http", "https"}:
            logger.error("invalid URL. expected http(s)://*")
            return []

        if self._store.skip_update(url):
            logger.info(f"skip loading: source exists ({url})")
            return []

        nodes = []
        if self._is_file_url(url):
            # 直リンクファイル
            node = await self._aload_direct_linked_file(url)
            if node is None:
                logger.warning(f"failed to fetch from {url}, skipped")
            else:
                nodes.append(node)
        else:
            # 本文テキスト
            nodes.extend(await self._aload_html_text(url))

            if self._load_asset:
                # アセットファイル
                nodes.extend(await self._aload_html_asset_files(base_url=url))

        logger.info(f"loaded {len(nodes)} nodes from {url}")

        return nodes

    async def aload_from_url(
        self,
        url: str,
    ) -> list[BaseNode]:
        """URL からコンテンツを取得し、ノードを生成する。
        サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

        Args:
            url (str): 対象 URL

        Returns:
            list[BaseNode]: 生成したノード
        """
        logger.debug("trace")

        # サイトマップ以外は単一のサイトとして読み込み
        if not Exts.endswith_exts(url, Exts.SITEMAP):
            return await self._aload_from_site(url)

        # 以下、サイトマップの解析と読み込み
        try:
            loader = SitemapReader()
            urls = loader._parse_sitemap(url)
        except Exception as e:
            logger.exception(e)
            return []

        # 最上位ループの一つ。キャッシュを空にしてから使う。
        self._source_cache.clear()
        nodes = []
        for url in urls:
            temp = await self._aload_from_site(url)
            nodes.extend(temp)

        return nodes

    async def aload_from_url_list(
        self,
        list_path: str,
    ) -> list[BaseNode]:
        """URL リストに記載の複数サイトからコンテンツを取得し、ノードを生成する。

        Args:
            list_path (str): URL リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）

        Returns:
            list[BaseNode]: 生成したノード
        """
        logger.debug("trace")

        urls = self._read_sources_from_file(list_path)

        # 最上位ループの一つ。キャッシュを空にしてから使う。
        self._source_cache.clear()
        nodes = []
        for url in urls:
            temp = await self.aload_from_url(url)
            nodes.extend(temp)

        return nodes
