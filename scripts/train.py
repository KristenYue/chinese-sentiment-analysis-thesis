"""可复现的XGBoost三分类训练脚本。"""

from pathlib import Path
import argparse
import json
import sys

import joblib
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_system.preprocessing import clean_text  # noqa: E402
from sentiment_system.tokenizer import jieba_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data" / "split" / "训练集.csv")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data" / "split" / "测试集.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "retrained")
    parser.add_argument("--quick", action="store_true", help="使用单组已知参数，跳过完整网格搜索")
    args = parser.parse_args()

    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)
    required = {"content", "sentiment"}
    for name, data in (("train", train_data), ("test", test_data)):
        if not required.issubset(data.columns):
            raise ValueError(f"{name}数据缺少列: {sorted(required - set(data.columns))}")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data["sentiment"].astype(str))
    y_test = encoder.transform(test_data["sentiment"].astype(str))

    vectorizer = TfidfVectorizer(max_features=5000, tokenizer=jieba_tokenizer, token_pattern=None)
    x_train = vectorizer.fit_transform(train_data["content"].map(clean_text))
    x_test = vectorizer.transform(test_data["content"].map(clean_text))
    x_resampled, y_resampled = ADASYN(random_state=42, sampling_strategy="not majority").fit_resample(x_train, y_train)

    estimator = xgb.XGBClassifier(random_state=42, eval_metric="mlogloss")
    parameter_grid = (
        {
            # 从已验证的原始三分类模型中读取的真实最优参数。
            "max_depth": [10],
            "learning_rate": [0.1],
            "n_estimators": [200],
            "subsample": [0.8],
            "colsample_bytree": [1.0],
        }
        if args.quick
        else {
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 200],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    )
    search = GridSearchCV(
        estimator,
        parameter_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1_macro",
        verbose=1,
    )
    search.fit(x_resampled, y_resampled)

    predictions = search.best_estimator_.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "best_params": search.best_params_,
        "labels": encoder.classes_.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            output_dict=True,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, args.output_dir / "xgboost_sentiment_model.joblib")
    joblib.dump(vectorizer, args.output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(encoder, args.output_dir / "label_encoder.joblib")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
