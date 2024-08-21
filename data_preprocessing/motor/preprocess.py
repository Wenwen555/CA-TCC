import pandas as pd
import numpy as np
import os
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 配置文件路径和保存路径
file_path = '/home/shenyan/dataset/motor_dataset/fangzhen'  # CSV文件所在目录
save_dir = '/home/shenyan/workspace/TS-TCC/CA-TCC/data/motor'  # 保存目录

# 1. 加载数据
all_data = []
all_labels = []

# 读取所有CSV文件
csvs = list(Path(file_path).glob("**/*.csv"))
for filename in tqdm(csvs, total=len(csvs)):  
    df = pd.read_csv(filename, skiprows=2)
    all_data.append(df.values[:, :-1])  # 假设最后一列是标签
    all_labels.append(df.values[:, -1])  # 假设最后一列是标签

# 合并所有数据
X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)

# 2. 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. 数据预处理
# 数据标准化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4. 保存数据集到.pt文件
train_data = {
    'features': torch.tensor(X_train, dtype=torch.float32),
    'labels': torch.tensor(y_train, dtype=torch.long)
}
val_data = {
    'features': torch.tensor(X_val, dtype=torch.float32),
    'labels': torch.tensor(y_val, dtype=torch.long)
}
test_data = {
    'features': torch.tensor(X_test, dtype=torch.float32),
    'labels': torch.tensor(y_test, dtype=torch.long)
}

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 保存数据集为.pt文件
torch.save(train_data, os.path.join(save_dir, 'train.pt'))
torch.save(val_data, os.path.join(save_dir, 'val.pt'))
torch.save(test_data, os.path.join(save_dir, 'test.pt'))

print("Data preprocessing and saving to .pt files completed successfully.")


