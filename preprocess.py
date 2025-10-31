import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM
import os

# --- 配置 ---
input_file = 'data/yeast.csv'
# 我们将保存到一个新文件，以保留原始编码文件（如果需要）
output_file = 'data/yeast_processed.csv'

# 1. 定义 8 个用于训练的特征列
feature_columns = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']

# 2. One-Class SVM 的关键参数
# 'nu' (nu) 参数大致是您“预估”数据中有百分之多少的异常值
# 0.05 意味着我们假设大约 5% 的数据是异常的
contamination_rate = 0.05 
# --- 结束配置 ---

# 确保输出目录存在
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    # 0. 读取 CSV 文件
    # 使用 on_bad_lines='skip' 来自动跳过第 408 行那样的格式错误行
    print(f"正在读取: {input_file}")
    df = pd.read_csv(input_file, on_bad_lines='skip')
    original_count = len(df)
    print(f"成功读取 {original_count} 行数据。")

    # --- 步骤 1: 类别标签编码 ---
    print("\n--- 步骤 1: 正在编码类别标签 ---")
    encoder = LabelEncoder()
    
    # 检查 'class_label' 列是否存在
    if 'class_label' not in df.columns:
        print(f"错误: 在CSV中找不到 'class_label' 列。")
        print(f"找到的列是: {df.columns.tolist()}")
    else:
        df['class_encoded'] = encoder.fit_transform(df['class_label'])
        
        print("类别到数字的映射关系:")
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        for label, code in mapping.items():
            print(f"{label}: {code}")

    # --- 步骤 2: 使用 One-Class SVM 检测异常值 ---
    print(f"\n--- 步骤 2: 使用 One-Class SVM 检测异常值 ---")
    
    # a. 提取用于检测的特征
    X_features = df[feature_columns]

    # b. !!重要!!: 对特征进行标准化 (SVM 对数据尺度非常敏感)
    print("正在对特征进行标准化 (StandardScaling)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # c. 初始化和训练 One-Class SVM
    print(f"正在训练 One-Class SVM... (nu={contamination_rate})")
    model_svm = OneClassSVM(nu=contamination_rate, kernel='rbf', gamma='auto')
    
    # d. 预测哪些是异常值 (-1) 哪些是正常值 (1)
    y_pred = model_svm.fit_predict(X_scaled)

    # --- 步骤 3: 过滤数据 ---
    print("\n--- 步骤 3: 过滤异常值 ---")
    
    # 将预测结果添加回 DataFrame
    df['is_outlier'] = y_pred

    # 筛选出所有被标记为 "正常" (1) 的行
    df_cleaned = df[df['is_outlier'] == 1].copy()

    # 报告结果
    removed_count = original_count - len(df_cleaned)
    print(f"原始数据: {original_count} 条")
    print(f"检测并移除了 {removed_count} 条异常值。")
    print(f"剩余数据: {len(df_cleaned)} 条")

    # --- 步骤 4: 保存最终的干净数据 ---
    
    # 准备要保存的列 (ID + 8个特征 + 编码后的标签)
    columns_to_save = ['seq_name'] + feature_columns + ['class_encoded']
    
    # 从清理后的数据中提取这些列
    df_final = df_cleaned[columns_to_save]
    
    # 保存到新文件，这次我们包含表头 (header=True)
    df_final.to_csv(output_file, header=True, index=False)
    
    print(f"\n✅ 成功! 处理后的干净数据已保存到: {output_file}")


except FileNotFoundError:
    print(f"错误: 找不到输入文件 '{input_file}'")
    print("请确保 'data/yeast.csv' 文件路径正确。")
except Exception as e:
    print(f"发生了一个意外错误: {e}")