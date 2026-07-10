"""在原始测试集上复现毕设三分类指标。"""

from pathlib import Path
import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_system import SentimentClassifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data" / "split" / "测试集.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation_results.json")
    args = parser.parse_args()

    data = pd.read_csv(args.test)
    required = {"content", "sentiment"}
    if not required.issubset(data.columns):
        raise ValueError(f"测试集必须包含列: {sorted(required)}")

    classifier = SentimentClassifier(PROJECT_ROOT / "artifacts")
    predictions = classifier.predict_many(data["content"].astype(str).tolist())
    y_true = data["sentiment"].astype(str).tolist()
    y_pred = [prediction.label for prediction in predictions]

    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    result = {
        "samples": len(data),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "classification_report": report,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
