import os
import tempfile
import time
from typing import Any, Iterable

import pymupdf as fitz
from fitz import Document as FDoc
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....config.general_config import GeneralConfig
from ....core.exts import Exts
from ....core.metadata import BasicMetaData
from ....logger import logger


class MultiPDFReader(BaseReader):
    """画像抽出も行うための独自 PDF リーダー"""

    def lazy_load_data(self, path: str, extra_info: Any = None) -> Iterable[Document]:
        """PDF ファイルを読み込み、テキストと画像のドキュメントをそれぞれ生成する。

        Args:
            path (str): ファイルパス

        Returns:
            Iterable[Document]: テキストドキュメントと画像ドキュメント
        """

        path = os.path.abspath(path)
        if not os.path.exists(path):
            logger.warning(f"file not found: {path}")
            return []

        if not Exts.endswith_ext(path, Exts.PDF):
            logger.warning(f"unsupported ext. {' '.join(Exts.PDF)} is allowed.")
            return []

        try:
            pdf = fitz.open(path)
        except Exception as e:
            logger.exception(e)
            return []

        try:
            text_docs = self._load_pdf_text(pdf, path)
            image_docs = self._load_pdf_image(pdf, path)
        finally:
            pdf.close()

        logger.info(
            f"loaded {len(text_docs)} text docs, {len(image_docs)} image docs from {path}"
        )

        return text_docs + image_docs

    def _load_pdf_text(
        self,
        pdf: FDoc,
        path: str,
    ) -> list[Document]:
        """PDF ファイルを読み込み、テキスト部分からドキュメントを生成する。

        Args:
            pdf (FDoc): pdf インスタンス
            path (str): ファイルパス

        Returns:
            list[Document]: 生成したドキュメントリスト
        """

        docs = []
        for page_no in range(pdf.page_count):
            try:
                page = pdf.load_page(page_no)
                content = page.get_text("text")  # type: ignore
            except Exception as e:
                logger.exception(e)
                continue

            # 空ならスキップ
            if not content.strip():  # type: ignore
                continue

            meta = BasicMetaData()
            meta.file_path = path
            meta.node_lastmod_at = time.time()
            meta.page_no = page_no

            doc = Document(text=content, metadata=meta.to_dict())
            docs.append(doc)

        return docs

    def _load_pdf_image(
        self,
        pdf: FDoc,
        path: str,
    ) -> list[Document]:
        """PDF ファイルを読み込み、画像部分からドキュメントを生成する。

        Args:
            pdf (FDoc): pdf インスタンス
            path (str): ファイルパス

        Returns:
            list[Document]: 生成したドキュメントリスト
        """

        docs = []
        for page_no in range(pdf.page_count):
            try:
                page = pdf.load_page(page_no)
                contents = page.get_images(full=True)  # type: ignore
            except Exception as e:
                logger.exception(e)
                continue

            for image_no, image in enumerate(contents):
                xref = image[0]  # 画像の参照番号
                pix = None
                try:
                    pix = fitz.Pixmap(pdf, xref)

                    if (
                        pix.n - (1 if pix.alpha else 0) == 4
                    ):  # CMYK (アルファの有無に関わらず)
                        old_pix = pix
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        del old_pix

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        prefix=f"{GeneralConfig.project_name}_",
                        suffix=Exts.PNG,
                    ) as f:
                        pix.save(f.name)

                        meta = BasicMetaData()
                        meta.file_path = f.name  # MultiModalVectorStoreIndex 参照用
                        meta.temp_file_path = f.name  # 削除用
                        meta.base_source = path  # 元パスの復元用
                        meta.node_lastmod_at = time.time()
                        meta.page_no = page_no
                        meta.asset_no = image_no

                        doc = Document(text=f.name, metadata=meta.to_dict())
                        docs.append(doc)
                except Exception as e:
                    logger.exception(e)
                    continue
                finally:
                    if pix is not None:
                        del pix

        return docs
