"""三分类XGBoost情感模型的稳定推理封装。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import __main__

import joblib
import numpy as np

from .preprocessing import clean_text
from .tokenizer import jieba_tokenizer


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SentimentClassifier:
    """加载毕设三分类模型，输出标签、置信度与各类概率。"""

    def __init__(self, artifacts_dir: str | Path) -> None:
        artifacts = Path(artifacts_dir).resolve()
        self.artifacts_dir = artifacts
        model_path = artifacts / "xgboost_sentiment_model.joblib"
        vectorizer_path = artifacts / "tfidf_vectorizer.joblib"
        encoder_path = artifacts / "label_encoder.joblib"

        missing = [str(path) for path in (model_path, vectorizer_path, encoder_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"缺少模型资产: {missing}")

        # 旧向量器把分词函数序列化为 __main__.jieba_tokenizer。
        # 该兼容层只为加载原毕设资产；重训后的向量器不再需要它。
        if not hasattr(__main__, "jieba_tokenizer"):
            setattr(__main__, "jieba_tokenizer", jieba_tokenizer)

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.encoder = joblib.load(encoder_path)
        self._validate_assets()

    def _validate_assets(self) -> None:
        model_features = int(self.model.n_features_in_)
        vectorizer_features = len(self.vectorizer.get_feature_names_out())
        if model_features != vectorizer_features:
            raise ValueError(
                f"模型需要{model_features}维特征，向量器输出{vectorizer_features}维"
            )
        if len(self.encoder.classes_) != len(self.model.classes_):
            raise ValueError("标签编码器与模型类别数不一致")

    @property
    def labels(self) -> list[str]:
        return [str(label) for label in self.encoder.classes_]

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: Iterable[str]) -> list[Prediction]:
        original = list(texts)
        if not original:
            return []
        cleaned = [clean_text(text) for text in original]
        if any(not text for text in cleaned):
            raise ValueError("清洗后文本为空")

        features = self.vectorizer.transform(cleaned)
        probabilities = self.model.predict_proba(features)
        encoded_predictions = np.asarray(self.model.predict(features), dtype=int)
        labels = self.encoder.inverse_transform(encoded_predictions)

        results: list[Prediction] = []
        for cleaned_text, label, row in zip(cleaned, labels, probabilities):
            probability_map = {
                class_name: float(probability)
                for class_name, probability in zip(self.labels, row)
            }
            results.append(
                Prediction(
                    label=str(label),
                    confidence=float(np.max(row)),
                    probabilities=probability_map,
                    cleaned_text=cleaned_text,
                )
            )
        return results
