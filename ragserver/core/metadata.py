from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ragserver.logger import logger


class META_KEYS_FROM:
    # ライブラリ側定義ラベル（字列変更不可）

    # SimpleDirectoryReader
    FILE_PATH = "file_path"
    FILE_TYPE = "file_type"
    FILE_SIZE = "file_size"
    CREATION_DATE = "creation_date"
    LAST_MODIFIED_DATE = "last_modified_date"


class META_KEYS(META_KEYS_FROM):
    # 正規化し、アプリ側で付与するラベル

    CHUNK_NO = "chunk_no"
    URL = "url"
    BASE_SOURCE = "base_source"
    TEMP_FILE_PATH = "temp_file_path"


@dataclass
class BasicMetaData:
    """ドキュメント、ノードの metadata フィールド用。
    Reader が自動付与するものを利用しつつ、アプリ側で明示的に挿入・利用するものはここで定義。

    参考
        SimpleDirectoryReader:
            file_path
            file_name
            file_type
            file_size
            creation_date
            last_modified_date
            last_accessed_date

    後段の各 Reader（基本、実装依存）
        PDFReader:
            page_label

        PptxReader:
            file_path
            page_label
            title
            extraction_errors
            extraction_warnings
            tables
            charts
            notes
            images
            text_sections

        ImageReader:
            下位 Reader 独自メタを合流する形のため色々

        等
    """

    file_path: str = ""  # 取得元ファイルパス
    file_type: str = ""  # ファイル種別（mimetype）
    file_size: str = ""  # ファイルサイズ
    creation_date: str = ""  # ファイル作成日時
    last_modified_date: str = ""  # 最終更新日時
    chunk_no: str = ""  # テキストのチャンク番号
    url: str = ""  # 取得元 URL
    base_source: str = ""  # 出典情報（直リンク画像の親ページ等）
    temp_file_path: str = ""  # ダウンロード画像等の一時ファイルパス

    def to_dict(self) -> dict[str, Any]:
        """メタデータの dict を返す。

        Returns:
            dict[str, Any]: メタデータの dict
        """
        # logger.debug("trace")

        return asdict(self)
