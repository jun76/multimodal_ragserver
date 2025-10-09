from __future__ import annotations

from ragserver.logger import logger

__all__ = ["EMBTYPE_TEXT", "EMBTYPE_IMAGE", "generate_space_key"]

# 埋め込み種別
# ! 変更すると空間キーの字列が変わって別空間（ingest やり直し）になるので注意 !
EMBTYPE_TEXT = "text"
EMBTYPE_IMAGE = "image"


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
