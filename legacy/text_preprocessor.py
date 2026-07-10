import os
import logging
import jieba
import pandas as pd
from pathlib import Path
from zhon.hanzi import punctuation
import emoji
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class TextPreprocessor:
    def __init__(self, stopwords_path=None, hownet_dir="hownet"):
        self.stopwords = self._load_stopwords(stopwords_path) if stopwords_path else set()
        self.hownet_dicts = self._load_hownet_dicts(hownet_dir)  # 加载 HowNet 词典
        self._init_jieba()

    def _init_jieba(self):
        """初始化 jieba 分词"""
        logging.info("Initializing Jieba...")
        jieba.initialize()

    def _load_stopwords(self, path):
        """加载停用词"""
        logging.info(f"Loading stopwords from {path}...")
        if not os.path.exists(path):
            logging.warning(f"未找到停用词文件 {path}，跳过加载")
            return set()
        with open(path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        return stopwords

    def _load_hownet_dicts(self, hownet_dir):
        """加载 HowNet 情感词典（积极/消极词）"""
        dicts = {'pos_emotion': set(), 'neg_emotion': set()}
        try:
            pos_path = Path(hownet_dir) / "pos_emotion.txt"
            neg_path = Path(hownet_dir) / "neg_emotion.txt"
            with open(pos_path, 'r', encoding='utf-8') as f:
                dicts['pos_emotion'] = {line.strip() for line in f if line.strip()}
            with open(neg_path, 'r', encoding='utf-8') as f:
                dicts['neg_emotion'] = {line.strip() for line in f if line.strip()}
            logging.info(f"Loaded HowNet词典: {len(dicts['pos_emotion'])} 积极词, {len(dicts['neg_emotion'])} 消极词")
        except Exception as e:
            logging.error(f"HowNet 词典加载失败: {e}")
        return dicts

    def clean_text(self, text):
        """清洗文本"""
        if pd.isna(text):
            return ""
        text = emoji.demojize(str(text))
        text = re.sub(r"<[^>]+>|http[s]?://\S+", "", text)
        text = re.sub(rf"[^\w\s{punctuation}]", "", text)
        return text.lower().strip()

    def tokenize_with_sentiment(self, text):
        """分词并返回单词列表"""
        text = self.clean_text(text)
        words = jieba.lcut(text)
        tokens = [{'word': word} for word in words if word not in self.stopwords and len(word) >= 1]
        return tokens

if __name__ == "__main__":
    # 测试代码
    preprocessor = TextPreprocessor(
        stopwords_path="stopwords.txt",  # 替换为实际路径
        hownet_dir="hownet"
    )
    print("积极词数量:", len(preprocessor.hownet_dicts['pos_emotion']))
    print("消极词数量:", len(preprocessor.hownet_dicts['neg_emotion']))