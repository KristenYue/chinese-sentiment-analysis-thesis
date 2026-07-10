"""TF-IDF使用的jieba分词函数。

该函数必须位于稳定模块中，否则joblib在反序列化时无法定位它。
"""

import jieba


def jieba_tokenizer(text: str) -> list[str]:
    return jieba.lcut(text)
