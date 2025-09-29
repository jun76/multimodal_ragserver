from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Set, Type

from ragserver.core.names import PROJECT_NAME
from ragserver.logger import logger

__all__ = ["stable_id_from", "file_fingerprint", "assert_required_keys"]

# 埋め込み種別
# ! 変更すると空間キーの字列が変わって別空間（ingest やり直し）になるので注意 !
EMBTYPE_TEXT = "text"
EMBTYPE_IMAGE = "image"


class META_KEYS:
    FP_SIZE = "fingerprint_size"
    FP_MTIME = "fingerprint_mtime"
    FP_SHA = "fingerprint_sha256_head"
    ID = "id"
    SOURCE = "source"
    SPACE_KEY = "space_key"
    EMBED_TYPE = "embed_type"
    PAGE = "page"
    CHUNK_NO = "chunk_no"
    IMAGE_NO = "image_no"
    BASE_URL = "base_url"


FINGERPRINT_KEYS = {
    META_KEYS.FP_SIZE,
    META_KEYS.FP_MTIME,
    META_KEYS.FP_SHA,
}

_INT_DEFAULT = -1
_FLOAT_DEFAULT = -1
_STR_DEFAULT = ""

# URL の ingest 等、fingerprint が取れない場合でも source は存在することをマーク
DUMMY_FINGERPRINT = {
    META_KEYS.FP_SIZE: _INT_DEFAULT,
    META_KEYS.FP_MTIME: _FLOAT_DEFAULT,
    META_KEYS.FP_SHA: _STR_DEFAULT,
}


@dataclass
class _BasicMetaData:
    """langchain Document の metadata フィールド用。
    必須の metadata 群。
    """

    id: str = _STR_DEFAULT  # Document 生成時に付与される ID
    source: str = _STR_DEFAULT  # データ取得元（ファイルパスや URL など）
    space_key: str = _STR_DEFAULT  # 空間キー
    embed_type: str = _STR_DEFAULT  # 埋め込み種別
    base_source: str = (
        _STR_DEFAULT  # データ取得元の親情報（任意。直リンク画像の元ページ等）
    )


@dataclass
class FileMetaData:
    """ファイルが要求する metadata。"""

    # ファイルサイズやハッシュ等の指紋情報
    # ストアがプリミティブしか受け付けない場合があるので展開しておく
    fingerprint_size: int = _INT_DEFAULT
    fingerprint_mtime: float = _FLOAT_DEFAULT
    fingerprint_sha256_head: str = _STR_DEFAULT


@dataclass
class _PagedFileMetaData(FileMetaData):
    """複数ページを持つファイルが要求する metadata。"""

    page: int = _INT_DEFAULT  # ページ番号


@dataclass
class _ChunkedTextMetaData:
    """複数チャンクを持つテキストが要求する metadata。"""

    chunk_no: int = _INT_DEFAULT  # 分割時のチャンク番号


@dataclass
class _PackedImageMetaData:
    """PDF, Web ページ等に複数個同梱された画像が要求する metadata。"""

    image_no: int = _INT_DEFAULT  # 画像の通し番号


@dataclass
class TextFileMetaData(_BasicMetaData, FileMetaData, _ChunkedTextMetaData):
    """テキストファイルが要求する metadata。"""

    pass


@dataclass
class ImageFileMetaData(_BasicMetaData, FileMetaData):
    """画像ファイルが要求する metadata。"""

    pass


@dataclass
class PDFTextMetaData(_BasicMetaData, _PagedFileMetaData, _ChunkedTextMetaData):
    """PDF ファイルのうち、テキスト部分が要求する metadata。"""

    pass


@dataclass
class PDFImageMetaData(_BasicMetaData, _PagedFileMetaData, _PackedImageMetaData):
    """PDF ファイルのうち、画像部分が要求する metadata。"""

    pass


@dataclass
class WebTextMetaData(_BasicMetaData, _ChunkedTextMetaData):
    """Web ページのうち、テキスト部分が要求する metadata。"""

    pass


@dataclass
class WebImageMetaData(_BasicMetaData, _PackedImageMetaData):
    """Web ページのうち、画像部分が要求する metadata。"""

    pass


def stable_id_from(key: str) -> str:
    """key から一意な UUIDv5 を生成する。

    Args:
        key (str): UUIDv5 生成用文字列

    Returns:
        str: 文字列化された UUIDv5
    """
    # logger.debug("trace")

    namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"https://{PROJECT_NAME}/namespace")

    return str(uuid.uuid5(namespace, key))


def file_fingerprint(path: str, head_bytes: int = 65536) -> dict[str, Any]:
    """ファイルの fingerprint を生成する。

    Args:
        path (str): ファイルパス
        head_bytes (int, optional): 先頭からハッシュ対象とするバイト数。 Defaults to 65536.

    Returns:
        dict[str, Any]: fingerprint 情報
    """
    logger.debug("trace")

    try:
        st = os.stat(path)
    except OSError as e:
        raise RuntimeError("failed to stat file") from e

    try:
        with open(path, "rb") as f:
            head = f.read(head_bytes)
    except (OSError, ValueError) as e:
        raise RuntimeError("failed to read file head") from e

    sha = hashlib.sha256(head).hexdigest()

    return {
        META_KEYS.FP_SIZE: st.st_size,  # バイトサイズ
        META_KEYS.FP_MTIME: st.st_mtime,  # 最終更新時刻（epoch）
        META_KEYS.FP_SHA: sha,  # 先頭 `head_bytes` のSHA-256ハッシュ
    }


def _required_keys_for(cls: Type[Any]) -> Set[str]:
    """指定した dataclass が要求するメタデータのキー集合を返す。

    Args:
        cls (Type[Any]): dataclass 型（BasicMetaData など）

    Returns:
        Set[str]: 要求されるキー名の集合
    """
    # logger.debug("trace")

    if not is_dataclass(cls):
        return set()

    return {f.name for f in fields(cls)}


def _still_default(val: Any) -> bool:
    """辞書データがデフォルト値のまま（設定漏れ）か

    Args:
        val (Any): 辞書データ

    Raises:
        TypeError:

    Returns:
        bool: デフォルト値のまま（設定漏れ）なら True
    """
    # logger.debug("trace")

    match val:
        case int():
            return val == _INT_DEFAULT
        case float():
            return val == _FLOAT_DEFAULT
        case str():
            return val == _STR_DEFAULT
        case _:
            raise TypeError("unsupported type for default detection")


def assert_required_keys(data: dict[str, Any], cls: Type[Any]) -> None:
    """辞書 `data` が、指定 dataclass `cls` の要求するキーをすべて含むか判定する。

    Args:
        data (dict[str, Any]): 対象のメタデータ辞書
        cls (Type[Any]): 検証対象の dataclass 型

    Raises:
        ValueError: 対象の metadata に要求されるキー値が欠落
        ValueError: 対象の metadata に要求されるキー値の値がデフォルトのまま
    """
    # logger.debug("trace")

    req = _required_keys_for(cls)
    present = set(data.keys())

    missing = req - present
    if missing:
        raise ValueError(
            f"missing required metadata keys: {', '.join(sorted(missing))}"
        )

    notset = []
    for key in req:
        try:
            if _still_default(data.get(key)):
                notset.append(key)
        except TypeError as e:
            raise ValueError(f"invalid metadata type for {key}") from e

    if notset:
        raise ValueError(f"metadata keys not set: {', '.join(notset)}")
