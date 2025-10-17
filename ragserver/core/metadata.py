from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from ragserver.logger import logger


class META_KEYS_FROM:
    # ライブラリ側定義ラベル（字列変更不可）
    ## SimpleDirectoryReader
    FILE_PATH = "file_path"
    FILE_TYPE = "file_type"
    FILE_SIZE = "file_size"
    FILE_CREATED_AT = "creation_date"
    FILE_LASTMOD_AT = "last_modified_date"


class META_KEYS(META_KEYS_FROM):
    # 正規化し、アプリ側で付与するラベル
    ## ノード内に保持する（＝BasicMetaData に含まれる）メタデータ
    CHUNK_NO = "chunk_no"
    URL = "url"
    BASE_SOURCE = "base_source"
    TEMP_FILE_PATH = "temp_file_path"
    NODE_LASTMOD_AT = "node_last_modified_date"
    PAGE_NO = "page_no"
    ASSET_NO = "asset_no"
    ## ノードの同一性確認用メタデータ
    FINGERPRINT = "fingerprint"


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

    # メタデータの中身
    # 追加・削除する場合、ノードのインスタンスを生成する loader 系の実装と
    # メタ情報を管理する meta_store 系の実装とも整合させること
    #
    file_path: str = ""  # 取得元ファイルパス
    file_type: str = ""  # ファイル種別（mimetype）
    file_size: int = 0  # ファイルサイズ
    file_created_at: str = ""  # ファイル作成日時
    file_lastmod_at: str = ""  # 最終更新日時
    chunk_no: int = 0  # テキストのチャンク番号
    url: str = ""  # 取得元 URL
    base_source: str = ""  # 出典情報（直リンク画像の親ページ等）
    temp_file_path: str = ""  # ダウンロード画像等の一時ファイルパス
    node_lastmod_at: float = 0  # ノードの最終更新時刻（epoch 秒）
    page_no: int = 0  # ページ番号
    asset_no: int = 0  # アセット番号（同一ページ内の画像等）

    def __init__(self, meta: Optional[dict[str, Any]] = None) -> None:
        """dict からメタデータインスタンスを生成する。

        Args:
            meta (Optional[dict[str, Any]], optional): メタデータの dict。 Defaults to None.
        """
        # logger.debug("trace")

        data = meta or {}
        self.file_path = data.get(META_KEYS.FILE_PATH, "")
        self.file_type = data.get(META_KEYS.FILE_TYPE, "")
        self.file_size = data.get(META_KEYS.FILE_SIZE, 0)
        self.file_created_at = data.get(META_KEYS.FILE_CREATED_AT, "")
        self.file_lastmod_at = data.get(META_KEYS.FILE_LASTMOD_AT, "")
        self.chunk_no = data.get(META_KEYS.CHUNK_NO, 0)
        self.url = data.get(META_KEYS.URL, "")
        self.base_source = data.get(META_KEYS.BASE_SOURCE, "")
        self.temp_file_path = data.get(META_KEYS.TEMP_FILE_PATH, "")
        self.node_lastmod_at = data.get(META_KEYS.NODE_LASTMOD_AT, 0)
        self.page_no = data.get(META_KEYS.PAGE_NO, 0)
        self.asset_no = data.get(META_KEYS.ASSET_NO, 0)

    def to_dict(self) -> dict[str, Any]:
        """メタデータの dict を返す。

        Returns:
            dict[str, Any]: メタデータの dict
        """
        # logger.debug("trace")

        return asdict(self)
