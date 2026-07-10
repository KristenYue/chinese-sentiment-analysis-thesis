from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_system import SentimentClassifier  # noqa: E402


@pytest.fixture(scope="module")
def classifier() -> SentimentClassifier:
    return SentimentClassifier(PROJECT_ROOT / "artifacts")


def test_assets_are_three_class_and_dimensionally_compatible(classifier: SentimentClassifier) -> None:
    assert classifier.labels == ["Negative", "Neutral", "Positive"]
    assert classifier.model.n_features_in_ == len(classifier.vectorizer.get_feature_names_out()) == 2482


def test_prediction_contains_normalized_probabilities(classifier: SentimentClassifier) -> None:
    result = classifier.predict("这个政策非常好，我很支持")
    assert result.label in classifier.labels
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == set(classifier.labels)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_empty_text_is_rejected(classifier: SentimentClassifier) -> None:
    with pytest.raises(ValueError, match="清洗后文本为空"):
        classifier.predict("123 !!!")


def test_retrained_assets_when_available() -> None:
    retrained = PROJECT_ROOT / "artifacts" / "retrained"
    if not retrained.exists():
        pytest.skip("尚未生成重训模型")
    restored = SentimentClassifier(retrained)
    assert restored.labels == ["Negative", "Neutral", "Positive"]
    assert restored.model.n_features_in_ == len(restored.vectorizer.get_feature_names_out())
