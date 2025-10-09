from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from ragserver.core.names import PROJECT_NAME
from ragserver.logger import logger


class META_KEYS_FROM:
    # ライブラリ側定義ラベル（字列変更不可）

    # SimpleDirectoryReader
    FILE_PATH = "file_path"
    FILE_TYPE = "file_type"
    FILE_SIZE = "file_size"
    CREATION_DATE = "creation_date"
    LAST_MODIFIED_DATE = "last_modified_date"


class META_KEYS:
    # 正規化し、アプリ側で付与するラベル

    FILE_PATH = "file_path"
    FILE_TYPE = "file_type"
    FILE_SIZE = "file_size"
    CREATION_DATE = "creation_date"
    LAST_MODIFIED_DATE = "last_modified_date"
    CHUNK_NO = "chunk_no"
    URL = "url"
    BASE_SOURCE = "base_source"
    TEMP_FILE_PATH = "temp_file_path"


@dataclass
class BasicMetaData:
    """Document の metadata フィールド用。
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

    # def _stable_id_from(self, key: str) -> str:
    #     """key から一意な UUIDv5 を生成する。

    #     Args:
    #         key (str): UUIDv5 生成用文字列

    #     Returns:
    #         str: 文字列化された UUIDv5
    #     """
    #     # logger.debug("trace")

    #     namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"https://{PROJECT_NAME}/namespace")

    #     return str(uuid.uuid5(namespace, key))

    def get_lazy_fingerprint(self) -> str:
        """ファイル更新スキップ用。
        スキップできなくても再 ingest されるだけなので、厳密な fingerprint は取らない。

        Returns:
            str: ファイルサイズ等を基にした緩い fingerprint
        """
        # logger.debug("trace")

        # Web ページの場合、現状 URL しかチェックしない
        return (
            f"{META_KEYS.FILE_PATH}:{self.file_path}_"
            + f"{META_KEYS.FILE_SIZE}:{self.file_size}_"
            + f"{META_KEYS.LAST_MODIFIED_DATE}:{self.last_modified_date}_"
            + f"{META_KEYS.CHUNK_NO}:{self.chunk_no}_"
            + f"{META_KEYS.URL}:{self.url}"
        )

    # def fix_id(self) -> None:
    #     """現状のメタデータから一意な ID を割り当てる。"""
    #     # logger.debug("trace")

    #     self.id = self._stable_id_from(self.get_lazy_fingerprint())

    # def to_json(self) -> str:
    #     """メタデータの json を返す。

    #     Returns:
    #         str: メタデータの json
    #     """
    #     # logger.debug("trace")

    #     self.fix_id()

    #     return json.dumps(asdict(self), ensure_ascii=False)

    # def from_json(self, jstr: str) -> BasicMetaData:
    #     """json から BasicMetaData コンテナを生成する。

    #     Args:
    #         jstr (str): メタデータの json 文字列

    #     Returns:
    #         BasicMetaData: 生成したコンテナ
    #     """
    #     # logger.debug("trace")

    #     return BasicMetaData(**json.loads(jstr))

    def to_dict(self) -> dict[str, Any]:
        """メタデータの dict を返す。

        Returns:
            dict[str, Any]: メタデータの dict
        """
        # logger.debug("trace")

        # self.fix_id()

        return asdict(self)


# def node_id(doc_id: str, suffix: str) -> str:
#     """ノードの id_ に格納する一意な文字列を作成する。

#     Reader(doc_id: 自動付与, ref_doc_id: don't care)
#       |
#       V
#     Document: doc_id, (ref_doc_id)
#       |
#     split(node_id: 自動付与, id_: = node_id, ref_doc_id: 親ドキュメントの doc_id を伝搬)
#       V
#     BaseNode: node_id, id_, ref_doc_id

#     Args:
#         doc_id (str): 親ドキュメントの doc_id
#         suffix (str): チャンク番号等

#     Returns:
#         str: id_ 文字列
#     """
#     # logger.debug("trace")

#     return f"{doc_id}_{suffix}"
