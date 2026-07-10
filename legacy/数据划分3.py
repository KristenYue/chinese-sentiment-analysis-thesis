from sklearn.model_selection import train_test_split
import pandas as pd
import os
import time

# ============== 配置部分 ==============
data_path = r"C:\Users\29007\Desktop\PythonProject\情感分析结果.csv"  # 合并后的数据集路径
output_dir = r"C:\Users\29007\Desktop\PythonProject\划分结果"

# 根据实际数据调整列名（修改点1）
text_column = 'segmented'  # 文本列对应数据中的"segmented"
label_column = 'sentiment'  # 情感标签列对应数据中的"sentiment"


# ============== 主程序 ==============
def main():
    # 1. 创建输出目录（带重试机制）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            os.makedirs(output_dir, exist_ok=True)
            break
        except PermissionError:
            if attempt == max_retries - 1:
                print(f"错误：无法创建输出目录，请检查权限: {output_dir}")
                return
            time.sleep(1)  # 等待1秒后重试

    # 2. 加载数据
    try:
        df = pd.read_csv(data_path)
        print("数据加载成功！前3行预览:")
        print(df.head(3))
        print("\n可用列名:", df.columns.tolist())
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return

    # 3. 检查必要列（修改点2）
    missing_cols = []
    required_cols = [text_column, label_column]

    for col in required_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        print(f"\n错误：缺少必要列 -> {missing_cols}")
        print("请检查数据文件中的列名是否与配置匹配")
        return

    # 4. 检查空值并处理（修改点3）
    print("\n检查数据中的空值情况：")
    print(df.isnull().sum())  # 检查所有列的空值数量

    # 处理可能的空值
    df[label_column] = df[label_column].fillna('Neutral')

    # 5. 检查标签分布
    print(f"\n标签分布:\n{df[label_column].value_counts()}")

    # 6. 清理标签数据（修复标签映射问题）
    df[label_column] = df[label_column].astype(str).str.strip()

    # 输出原始唯一标签值
    print(f"\n原始唯一标签值: {df[label_column].unique().tolist()}")

    # 标签映射，确保正确处理中文标签
    label_mapping = {
        '积极': 'Positive',
        '中立': 'Neutral',
        '消极': 'Negative',
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral',
        '0': 'Neutral',
        '1': 'Positive',
        '-1': 'Negative',
    }

    df[label_column] = df[label_column].map(label_mapping)

    # 确保未映射的标签默认填充为 'Neutral'
    df[label_column] = df[label_column].fillna('Neutral')

    # 输出修改后的标签分布
    print(f"\n修改后的标签分布:\n{df[label_column].value_counts()}")

    # 7. 数据划分
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df[label_column]  # 保持类别分布一致
        )
    except Exception as e:
        print(f"\n数据划分失败: {str(e)}")
        return

    # 8. 保存结果（带错误处理和重试）
    train_path = os.path.join(output_dir, "训练集.csv")
    test_path = os.path.join(output_dir, "测试集.csv")

    # 定义需要保留的列（根据实际需求调整）
    keep_columns = [
        'weibo_id', 'comment_id', 'user_name',
        'content', 'segmented', 'sentiment',
        'created_at', 'like_count', 'retweets'
    ]

    for path, data in [(train_path, train_df), (test_path, test_df)]:
        for attempt in range(max_retries):
            try:
                # 筛选需要的列并保存
                data[keep_columns].to_csv(path, index=False, encoding='utf-8-sig')
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    print(f"错误：无法写入文件，请检查是否被其他程序占用: {path}")
                    return
                time.sleep(1)  # 等待1秒后重试

    # 9. 输出统计信息
    print("\n=== 数据划分结果 ===")
    print(f"总数据量: {len(df)} 条")
    print(f"训练集: {len(train_df)} 条 ({len(train_df) / len(df):.1%})")
    print(f"测试集: {len(test_df)} 条 ({len(test_df) / len(df):.1%})")

    print("\n=== 标签分布 ===")
    print("训练集:")
    print(train_df[label_column].value_counts().to_string())
    print("\n测试集:")
    print(test_df[label_column].value_counts().to_string())

    print(f"\n划分结果已保存至: {output_dir}")
    print(f"- 训练集: {train_path}")
    print(f"- 测试集: {test_path}")


if __name__ == "__main__":
    main()
