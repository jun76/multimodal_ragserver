from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Iterable, Optional

import pymupdf as fitz
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

from ragserver.core.metadata import EMBTYPE_IMAGE, EMBTYPE_TEXT
from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import (
    ImageFileMetaData,
    PDFImageMetaData,
    PDFTextMetaData,
    TextFileMetaData,
    assert_required_keys,
    file_fingerprint,
    stable_id_from,
)
from ragserver.core.names import PROJECT_NAME
from ragserver.ingest.loader import Exts, Loader
from ragserver.logger import logger
from ragserver.store.vector_store_manager import VectorStoreManager


class FileLoaderForWeb(ABC):
    """HTMLLoader 向けに FileLoader の公開範囲を限定"""

    @abstractmethod
    def load_text_file(
        self,
        path: str,
        space_key: str,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> list[Document]: ...

    @abstractmethod
    def load_markdown_file(
        self,
        path: str,
        space_key: str,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> list[Document]: ...

    @abstractmethod
    def load_image_file(
        self,
        path: str,
        space_key: str,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> list[Document]: ...

    @abstractmethod
    def load_pdf_file(
        self,
        path: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]: ...


class FileLoader(Loader, FileLoaderForWeb):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        store: Optional[VectorStoreManager] = None,
    ) -> None:
        """ローカルファイルを読み込み、ドキュメントを生成するためのクラス。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): チャンク重複語数
            store (Optional[VectorStoreManager], opitonal): 登録済みソースの判定に使用。Defaults to None.
        """
        logger.debug("trace")

        Loader.__init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._store = store

    def _read_text_file(self, path: str) -> str:
        """テキストファイルを読み込む。

        Args:
            path (str): ファイルパス

        Returns:
            str: 読み込んだ文字列

        Raises:
            RuntimeError: ファイル読み込みに失敗した場合
        """
        logger.debug("trace")

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except OSError as e:
            raise RuntimeError("failed to read text file") from e

    def _build_text_docs(
        self,
        content: str,
        source: str,
        space_key: str,
        fp: dict,
        base_source: Optional[str] = None,
    ) -> list[Document]:
        """与えられたテキストから共通の分割・ID 付与処理を行い、ドキュメントリストを返す。

        Args:
            content (str): 入力テキスト
            source (str): メタデータに付与する source
            space_key (str): 空間キー
            fp (dict): fingerprint 情報の辞書（size, mtime, sha256_head）
            base_source (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: 分割・ID 付与済みドキュメントリスト
        """
        logger.debug("trace")

        metadata = asdict(
            TextFileMetaData(
                fingerprint_size=fp[MK.FP_SIZE],
                fingerprint_mtime=fp[MK.FP_MTIME],
                fingerprint_sha256_head=fp[MK.FP_SHA],
                source=source,
                embed_type=EMBTYPE_TEXT,
                space_key=space_key,
                base_source=base_source if base_source else source,
            )
        )

        base_doc = Document(page_content=content, metadata=metadata)
        docs = self._split_text([base_doc])

        for i, d in enumerate(docs):
            d.metadata[MK.CHUNK_NO] = i

            d.metadata[MK.ID] = stable_id_from(
                f"{d.metadata[MK.EMBED_TYPE]}"
                f"::{d.metadata[MK.SOURCE]}"
                f"::{d.metadata[MK.FP_SHA]}"
                f"::{d.metadata[MK.CHUNK_NO]}"
            )
            assert_required_keys(d.metadata, TextFileMetaData)

        return docs

    def _iter_files(self, root: str) -> Iterable[tuple[str, str]]:
        """ルート配下のファイルパスを列挙する。ファイルへの直パスも受理する。
        （yield）状態を持つので注意。

        Args:
            root (str): 走査するルートディレクトリ

        Yields:
            Iterator[Iterable[tuple[str, str]]]: （絶対パス, 拡張子）の Iterable
        """
        logger.debug("trace")

        if os.path.isfile(root):
            path = os.path.abspath(root)
            yield path, os.path.splitext(path)[1].lower()
            return

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                # 相対パス
                path = os.path.join(dirpath, fn)
                # 絶対パス
                path = os.path.abspath(path)
                yield path, os.path.splitext(path)[1].lower()

    def _load_pdf_text(
        self,
        path: str,
        space_key: str,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> list[Document]:
        """PDF ファイルを読み込み、テキスト部分からドキュメントを生成する。

        Args:
            path (str): ファイルパス
            space_key (str): 空間キー
            source (Optional[str], optional): path 以外の soruce を指定する場合。 Defaults to None.
            base_source (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: 生成したドキュメントリスト

        Raises:
            RuntimeError: PDF の読み込みに失敗した場合
        """
        logger.debug("trace")

        if not self._required_ext(path, Exts.PDF_FILE_EXTS):
            logger.warning(f"required {' '.join(Exts.PDF_FILE_EXTS)} file")
            return []

        if source is None:
            source = path

        try:
            pdf = fitz.open(path)
        except Exception as e:
            raise RuntimeError("failed to open pdf") from e

        fp = file_fingerprint(path)
        docs = []
        for page_no in range(pdf.page_count):
            try:
                page = pdf.load_page(page_no)
                content = page.get_text("text")  # type: ignore
            except Exception as e:
                logger.exception(e)
                continue
            if not content.strip():
                continue

            metadata = asdict(
                PDFTextMetaData(
                    fingerprint_size=fp[MK.FP_SIZE],
                    fingerprint_mtime=fp[MK.FP_MTIME],
                    fingerprint_sha256_head=fp[MK.FP_SHA],
                    source=source,
                    embed_type=EMBTYPE_TEXT,
                    space_key=space_key,
                    page=page_no,
                    base_source=base_source if base_source else source,
                )
            )

            base_doc = Document(page_content=content, metadata=metadata)
            splitted = self._split_text([base_doc])
            for i, d in enumerate(splitted):
                d.metadata[MK.CHUNK_NO] = i

                # 最後に id を生成・追加
                d.metadata[MK.ID] = stable_id_from(
                    f"{d.metadata[MK.EMBED_TYPE]}"
                    f"::{d.metadata[MK.SOURCE]}"
                    f"::{d.metadata[MK.FP_SHA]}"
                    f"::{d.metadata[MK.PAGE]}"
                    f"::{d.metadata[MK.CHUNK_NO]}"
                )
                assert_required_keys(d.metadata, PDFTextMetaData)

                docs.append(d)

        logger.info(f"loaded {len(docs)} docs from {source}")

        try:
            pdf.close()
        except Exception as e:
            logger.exception(e)

        return docs

    def _load_pdf_image(
        self,
        path: str,
        space_key: str,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> list[Document]:
        """PDF ファイルを読み込み、画像部分からドキュメントを生成する。

        Args:
            path (str): ファイルパス
            space_key (str): 空間キー
            source (Optional[str], optional): path 以外の soruce を指定する場合。 Defaults to None.
            base_source (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: 生成したドキュメントリスト

        Raises:
            RuntimeError: PDF の読み込みに失敗した場合
        """
        logger.debug("trace")

        if not self._required_ext(path, Exts.PDF_FILE_EXTS):
            logger.warning(f"required {' '.join(Exts.PDF_FILE_EXTS)} file")
            return []

        if source is None:
            source = path

        try:
            pdf = fitz.open(path)
        except Exception as e:
            raise RuntimeError("failed to open pdf") from e

        fp = file_fingerprint(path)
        docs = []
        for page_no in range(pdf.page_count):
            try:
                page = pdf.load_page(page_no)
                contents = page.get_images(full=True)  # type: ignore
            except Exception as e:
                logger.exception(e)
                continue

            metadata = asdict(
                PDFImageMetaData(
                    fingerprint_size=fp[MK.FP_SIZE],
                    fingerprint_mtime=fp[MK.FP_MTIME],
                    fingerprint_sha256_head=fp[MK.FP_SHA],
                    source=source,
                    embed_type=EMBTYPE_IMAGE,
                    space_key=space_key,
                    page=page_no,
                    base_source=base_source if base_source else source,
                )
            )

            for image_no, image in enumerate(contents):
                xref = image[0]  # 画像の参照番号
                try:
                    pix = fitz.Pixmap(pdf, xref)

                    if getattr(pix, "n", 0) >= 5:  # CMYK 等は RGB へ
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    with tempfile.NamedTemporaryFile(
                        delete=False, prefix=f"{PROJECT_NAME}_", suffix=".png"
                    ) as f:
                        pix.save(f.name)

                        doc = Document(page_content=f.name, metadata=metadata)
                        doc.metadata[MK.IMAGE_NO] = image_no
                except Exception as e:
                    logger.exception(e)
                    continue
                finally:
                    try:
                        pix = None
                    except Exception:
                        pass

                # 最後に id を生成・追加
                doc.metadata[MK.ID] = stable_id_from(
                    f"{doc.metadata[MK.EMBED_TYPE]}"
                    f"::{doc.metadata[MK.SOURCE]}"
                    f"::{doc.metadata[MK.FP_SHA]}"
                    f"::{doc.metadata[MK.PAGE]}"
                    f"::{doc.metadata[MK.IMAGE_NO]}"
                )
                assert_required_keys(doc.metadata, PDFImageMetaData)

                docs.append(doc)

        logger.info(f"loaded {len(docs)} docs from {source}")

        try:
            pdf.close()
        except Exception as e:
            logger.exception(e)

        return docs

    def load_text_file(
        self,
        path: str,
        space_key: str,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> list[Document]:
        """テキストファイルを読み込み、ドキュメントを生成する。

        Args:
            path (str): ファイルパス
            space_key (str): 空間キー
            source (Optional[str], optional): path 以外の soruce を指定する場合。 Defaults to None.
            base_source (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: 生成したドキュメントリスト
        """
        logger.debug("trace")

        if not self._required_ext(path, Exts.TEXT_FILE_EXTS):
            logger.warning(f"required {' '.join(Exts.TEXT_FILE_EXTS)} file")
            return []

        if source is None:
            source = path

        if self._store and self._store.skip_update(source):
            logger.info(f"skip loading: source exists ({source})")
            return []

        fp = file_fingerprint(path)
        content = self._read_text_file(path)
        docs = self._build_text_docs(
            content=content,
            source=source,
            space_key=space_key,
            fp=fp,
            base_source=base_source,
        )

        logger.info(f"loaded {len(docs)} docs from {source}")

        return docs

    def load_markdown_file(
        self,
        path: str,
        space_key: str,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> list[Document]:
        """マークダウンファイルを読み込み、ドキュメントを生成する。

        Args:
            path (str): ファイルパス
            space_key (str): 空間キー
            source (Optional[str], optional): path 以外の soruce を指定する場合。 Defaults to None.
            base_source (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: 生成したドキュメントリスト
        """
        logger.debug("trace")

        if not self._required_ext(path, Exts.MARKDOWN_FILE_EXTS):
            logger.warning(f"required {' '.join(Exts.MARKDOWN_FILE_EXTS)} file")
            return []

        if source is None:
            source = path

        if self._store and self._store.skip_update(source):
            logger.info(f"skip loading: source exists ({source})")
            return []

        fp = file_fingerprint(path)

        # Unstructured を優先利用し、失敗時のみプレーンテキストへフォールバック
        try:
            loader = UnstructuredMarkdownLoader(path, mode="elements")
            loaded = loader.load()
            content = "\n\n".join(d.page_content for d in loaded if d.page_content)
        except Exception as e:
            logger.warning(f"fallback to plain text: {e}")
            content = self._read_text_file(path)

        docs = self._build_text_docs(
            content=content,
            source=source,
            space_key=space_key,
            fp=fp,
            base_source=base_source,
        )

        logger.info(f"loaded {len(docs)} docs from {source}")

        return docs

    def load_image_file(
        self,
        path: str,
        space_key: str,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> list[Document]:
        """画像ファイルを読み込み、Document を生成する。

        Args:
            path (str): ファイルパス
            space_key (str): 空間キー
            source (Optional[str], optional): path 以外の soruce を指定する場合。 Defaults to None.
            base_source (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            list[Document]: 生成したドキュメント
        """
        logger.debug("trace")

        if not self._required_ext(path, Exts.IMAGE_FILE_EXTS):
            logger.warning(f"required {' '.join(Exts.IMAGE_FILE_EXTS)} file")
            return []

        if source is None:
            source = path

        if self._store and self._store.skip_update(source):
            logger.info(f"skip loading: source exists ({source})")
            return []

        fp = file_fingerprint(path)
        metadata = asdict(
            ImageFileMetaData(
                fingerprint_size=fp[MK.FP_SIZE],
                fingerprint_mtime=fp[MK.FP_MTIME],
                fingerprint_sha256_head=fp[MK.FP_SHA],
                source=source,
                embed_type=EMBTYPE_IMAGE,
                space_key=space_key,
                base_source=base_source if base_source else source,
            )
        )

        # page_content は画像の一時ファイルパス受け渡しに利用
        doc = Document(page_content=path, metadata=metadata)

        # 最後に id を生成・追加
        doc.metadata[MK.ID] = stable_id_from(
            f"{doc.metadata[MK.EMBED_TYPE]}"
            f"::{doc.metadata[MK.SOURCE]}"
            f"::{doc.metadata[MK.FP_SHA]}"
        )
        assert_required_keys(doc.metadata, ImageFileMetaData)

        # 現状、画像からは単一ドキュメントのみ生成するが他の load 系インタフェースと
        # 同様に list で返す
        docs = [doc]
        logger.info(f"loaded {len(docs)} docs from {source}")

        return docs

    def load_pdf_file(
        self,
        path: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
        source: Optional[str] = None,
        base_source: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]:
        """PDF ファイルを読み込み、テキストと画像のドキュメントをそれぞれ生成する。

        Args:
            path (str): ファイルパス
            space_key (str): テキスト用空間キー
            space_key_multi (Optional[str], optional): マルチモーダル用空間キー。 Defaults to None.
            source (Optional[str], optional): path 以外の soruce を指定する場合。 Defaults to None.
            base_source (Optional[str], optional): source の取得元を指定する場合。 Defaults to None.

        Returns:
            tuple[list[Document], list[Document]]: テキストドキュメント, マルチモーダルドキュメント
        """
        logger.debug("trace")

        if source is None:
            source = path

        if self._store and self._store.skip_update(source):
            logger.info(f"skip loading: source exists ({source})")
            return [], []

        text_docs = self._load_pdf_text(
            path=path, space_key=space_key, source=source, base_source=base_source
        )

        if space_key_multi:
            image_docs = self._load_pdf_image(
                path=path,
                space_key=space_key_multi,
                source=source,
                base_source=base_source,
            )
        else:
            image_docs = []

        logger.info(
            f"loaded {len(text_docs)} text docs, {len(image_docs)} image docs from {source}"
        )

        return text_docs, image_docs

    def load_from_path(
        self,
        root: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]:
        """ローカルパス（ディレクトリ、ファイル）からコンテンツを取り込み、ドキュメントを生成する。
        ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

        Args:
            root (str): 対象パス
            space_key (str): テキスト用空間キー
            space_key_multi (Optional[str], optional): マルチモーダル用空間キー。 Defaults to None.

        Returns:
            tuple[list[Document], list[Document]]: テキストドキュメント, マルチモーダルドキュメント

        Raises:
            RuntimeError: サポート外の拡張子を検出した場合
        """
        logger.debug("trace")

        # 最上位ループ内で複数ソースをまたいで _source_cache を共有したいため
        # ここでは _source_cache.clear() しないこと。
        text_docs = []
        image_docs = []
        for path, ext in self._iter_files(root):
            if path in self._source_cache:
                continue

            try:
                if self._required_ext(path, Exts.TEXT_FILE_EXTS):
                    buf_text, buf_image = (
                        self.load_text_file(path=path, space_key=space_key),
                        [],
                    )
                elif self._required_ext(path, Exts.MARKDOWN_FILE_EXTS):
                    buf_text, buf_image = (
                        self.load_markdown_file(path=path, space_key=space_key),
                        [],
                    )
                elif self._required_ext(path, Exts.IMAGE_FILE_EXTS) and space_key_multi:
                    buf_text, buf_image = [], self.load_image_file(
                        path=path, space_key=space_key_multi
                    )
                elif self._required_ext(path, Exts.PDF_FILE_EXTS):
                    buf_text, buf_image = self.load_pdf_file(
                        path=path,
                        space_key=space_key,
                        space_key_multi=space_key_multi,
                    )
                else:
                    logger.warning(
                        f"'{ext}' is not supported in this system(or model). "
                        f"{' '.join(Exts.SUPPORTED_EXTS)} are supported"
                    )
            except Exception as e:
                logger.exception(e)
                continue

            text_docs.extend(buf_text)
            image_docs.extend(buf_image)
            self._source_cache.add(path)

        return text_docs, image_docs

    def load_from_path_list(
        self,
        list_path: str,
        space_key: str,
        space_key_multi: Optional[str] = None,
    ) -> tuple[list[Document], list[Document]]:
        """path リストに記載の複数パスからコンテンツを取得し、ドキュメントを生成する。

        Args:
            list_path (str): path リストのパス（テキストファイル。# で始まるコメント行・空行はスキップ）
            space_key (str): テキスト用空間キー
            space_key_multi (Optional[str], optional): マルチモーダル用空間キー。 Defaults to None.

        Returns:
            tuple[list[Document], list[Document]]: テキストドキュメント, マルチモーダルドキュメント
        """
        logger.debug("trace")

        paths = self._read_sources_from_file(list_path)

        # 最上位ループ。キャッシュを空にしてから使う。
        self._source_cache.clear()
        text_docs = []
        image_docs = []
        for path in paths:
            try:
                temp_text, temp_image = self.load_from_path(
                    root=path,
                    space_key=space_key,
                    space_key_multi=space_key_multi,
                )
            except Exception as e:
                logger.exception(e)
                continue

            text_docs.extend(temp_text)
            image_docs.extend(temp_image)

        return text_docs, image_docs
