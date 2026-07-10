"""与原XGBoost训练脚本保持一致的文本清洗逻辑。"""

import re


_NON_CHINESE_PATTERN = re.compile(r"[^\u4e00-\u9fa5，。！？]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """保留中文字符和常用中文标点。"""
    if text is None:
        return ""
    cleaned = _NON_CHINESE_PATTERN.sub("", str(text).strip())
    return _WHITESPACE_PATTERN.sub("", cleaned)
