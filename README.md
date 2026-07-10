# 社交媒体评论情感分析系统（毕设恢复版）

本项目将原本分散的本科毕设资产整理为可运行、可评测、可重训的三分类情感分析系统。

## 已恢复的主链路

```text
中文评论
  -> 文本清洗
  -> jieba分词
  -> TF-IDF
  -> XGBoost
  -> Negative / Neutral / Positive + 置信度
```

已用原始143条测试集复现：

- Accuracy: 0.9021
- Macro-F1: 0.8782

使用从原模型读取的真实最优参数重训后：

- Accuracy: 0.9091
- Macro-F1: 0.8856
- 新向量器不再依赖`__main__.jieba_tokenizer`兼容层

## 目录说明

- `artifacts/`: 已验证的三分类模型、TF-IDF向量器和标签编码器
- `data/split/`: 本地保留的训练集与测试集，包含原始微博字段，公开仓库默认不上传
- `data/events/`: 本地保留的事件数据样例，包含原始微博字段，公开仓库默认不上传
- `dictionaries/`: HowNet正负面词典
- `src/sentiment_system/`: 重构后的推理与API代码
- `scripts/`: 评测和重训脚本
- `legacy/`: 原始脚本，仅供追溯

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 复现毕设指标

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\evaluate.py
```

评测结果会同时写入`evaluation_results.json`。公开仓库保留已复现的评测结果与模型资产；原始 CSV 数据因包含微博 ID、昵称和时间等字段，不随公开版本发布。

## 启动API

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m uvicorn sentiment_system.api:app --host 127.0.0.1 --port 8000
```

接口：

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- Swagger UI: `http://127.0.0.1:8000/docs`

API默认优先使用`artifacts/retrained/`中的完整重训模型。如需对照原始模型，可在启动前指定：

```powershell
$env:SENTIMENT_ARTIFACTS_DIR="artifacts"
```

`GET /health`会返回当前实际加载的资产路径和特征维度，避免模型版本混淆。

单条预测请求：

```json
{
  "text": "这个政策非常好，我很支持"
}
```

## 重新训练

快速验证训练流程：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\train.py --quick
```

完整GridSearchCV：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\train.py
```

重训资产输出至`artifacts/retrained/`，不会覆盖已验证的原始模型。

## 重要技术说明

1. 当前已验证模型为三分类，不与其他二分类`optimized`模型混用。
2. 原TF-IDF向量器序列化了`__main__.jieba_tokenizer`，推理封装包含兼容层。
3. 重训脚本使用稳定模块路径保存Tokenizer，重训后不再依赖兼容层。
4. 原XGBoost训练脚本实际使用TF-IDF，未将HowNet或Boson分数作为模型特征。未重新实现并评测前，不应宣称“词典特征融合训练”。

## 测试

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest
```
