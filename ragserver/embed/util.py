from __future__ import annotations

import base64
import math
import mimetypes
import numbers
import os
from typing import Optional, Sequence

from ragserver.logger import logger

__all__ = ["generate_space_key", "l2_normalize", "image_to_data_uri"]


def _sanitize_space_key(space_key: str) -> str:
    """Chroma の制約にマッチするよう space_key 文字列を整形する。

    制約：
        containing 3-512 characters from [a-zA-Z0-9._-],
        starting and ending with a character in [a-zA-Z0-9]

    Args:
        space_key (str): 整形前の space_key

    Returns:
        str: 整形後の space_key
    """
    logger.debug("trace")

    if not space_key:
        # 空入力は最短かつ有効な "000" を返す
        return "000"

    allowed = set(
        "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789" "._-"
    )

    # 許可されない文字は '_' に置換
    chars = [ch if ch in allowed else "_" for ch in space_key]

    # 長すぎる場合は 512 にトリム
    if len(chars) > 512:
        chars = chars[:512]

    # 先頭・末尾を英数字にする（英数字でなければ '0' に置換）
    def is_alnum(ch: str) -> bool:
        return ("0" <= ch <= "9") or ("a" <= ch <= "z") or ("A" <= ch <= "Z")

    if not chars:
        chars = list("000")
    else:
        if not is_alnum(chars[0]):
            chars[0] = "0"
        if not is_alnum(chars[-1]):
            chars[-1] = "0"

    # 短すぎる場合は '0' でパディング（末尾を英数字に保てる）
    while len(chars) < 3:
        chars.append("0")

    return "".join(chars)


# バリデーション用の正規表現（要件の最終形）
# _VALID_RE: Final[re.Pattern] = re.compile(
#     r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$"
# )

# def _is_valid_space_key(s: str) -> bool:
#     """上記の制約に適合しているか（参考用の検証関数）。"""
#     return bool(_VALID_RE.match(s))


def generate_space_key(embed_name: str, model_name: str, embed_type: str) -> str:
    """space_key 文字列を生成する。

    Args:
        embed_name (str): 埋め込み管理名
        model_name (str): 埋め込みモデル名
        embed_type (str): 埋め込み用途

    Returns:
        str: 生成した space_key
    """
    logger.debug("trace")

    space_key = _sanitize_space_key(f"{embed_name}__{model_name}__{embed_type}")
    logger.info(f"space_key [{space_key}] generated")

    return space_key


def l2_normalize(
    vecs: Sequence[Sequence[float]], eps: float = 1e-12
) -> list[list[float]]:
    """embed 済みベクトルに対する L2 正規化用。

    Args:
        vecs (Sequence[Sequence[float]]): embed 系関数の戻りベクトル
        eps (float, optional): 誤差。 Defaults to 1e-12.

    Returns:
        list[list[float]]: L2 正規化済みのベクトル
    """
    logger.debug("trace")

    out: list[list[float]] = []
    for row in vecs:
        # 量子化（int8等）や非数値の行は素通し
        if not row or not isinstance(row[0], numbers.Real):
            out.append(list(row))
            continue

        s = math.fsum(x * x for x in row)
        n = math.sqrt(s)
        if n < eps:
            out.append(list(row))  # そのまま返す（全部0など）
        else:
            inv = 1.0 / n
            out.append([x * inv for x in row])

    return out


def image_to_data_uri(path: str, max_bytes: int = 10 * 1024 * 1024) -> Optional[str]:
    """画像を Data URI 形式に変換する。

    Args:
        path (str): 画像のパス
        max_bytes (int): 画像のサイズ上限

    Returns:
        Optional[str]: Data URI 文字列
    """
    # logger.debug("trace")

    # path は file_loader 側で正規化されている想定

    try:
        if os.path.getsize(path) > max_bytes:
            logger.warning(f"File too large: {path}")
            return None

        with open(path, "rb") as f:
            b = f.read()
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.exception(e)
        return None

    # WebP 等の一部の形式に対応するため strict=False を使用
    mime, _ = mimetypes.guess_type(path, strict=False)
    if mime is None:
        mime = "image/png"

    # 画像MIMEタイプの検証
    if not mime.startswith("image/"):
        logger.warning(f"Non-image MIME type detected: {mime}")
        return None

    b64 = base64.b64encode(b).decode("utf-8")

    return f"data:{mime};base64,{b64}"
