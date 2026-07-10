import os
import re
import jieba
import joblib
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN

# ====================== 清洗 & 分词 ======================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\u4e00-\u9fa5，。！？、]', '', str(text).strip())
    return re.sub(r'\s+', '', text)

def jieba_tokenizer(text):
    return list(jieba.cut(text))

# ====================== 加载数据 ======================
train_df = pd.read_csv(r"C:\Users\29007\Desktop\PythonProject\划分结果\训练集.csv")
test_df = pd.read_csv(r"C:\Users\29007\Desktop\PythonProject\划分结果\测试集.csv")

train_df['content'] = train_df['content'].apply(clean_text)
test_df['content'] = test_df['content'].apply(clean_text)

X_train = train_df['content']
y_train = train_df['sentiment']
X_test = test_df['content']
y_test = test_df['sentiment']

# ====================== 标签编码 ======================
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# ====================== 特征提取 ======================
vectorizer = TfidfVectorizer(max_features=5000, tokenizer=jieba_tokenizer)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ====================== 过采样增强 ======================
adasyn = ADASYN(random_state=42, sampling_strategy='not majority')
X_train_res, y_train_res = adasyn.fit_resample(X_train_tfidf, y_train)

# ====================== 模型训练 ======================
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

try:
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_res, y_train_res)
    best_model = grid_search.best_estimator_
    print("✅ 模型训练完成")
    print("Best Params:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
except Exception as e:
    print("❌ 模型训练失败:", e)
    exit()

# ====================== 模型评估 ======================
y_pred = best_model.predict(X_test_tfidf)
print("\n✅ 分类报告:\n")
print(classification_report(y_test, y_pred, digits=4))

# 生成混淆矩阵并绘制
cm = confusion_matrix(y_test, y_pred)

# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels", fontsize=12)  # Label for x-axis (Predicted Labels)
plt.ylabel("True Labels", fontsize=12)  # Label for y-axis (True Labels)
plt.tight_layout()
plt.show()

# ====================== 保存模型（✅ 保存完整向量器） ======================
model_dir = r"C:\Users\29007\Desktop\PythonProject\model"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(best_model, os.path.join(model_dir, "xgboost_sentiment_model.joblib"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.joblib"))   # ✅ 保存完整对象
joblib.dump(encoder, os.path.join(model_dir, "label_encoder.joblib"))

print("✅ 模型、向量器和编码器保存成功！")
