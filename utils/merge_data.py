import os
import json
import random

# 定义文件夹路径和目标文件
source_folder = './ultravoice/'
target_file_template = 'ultravoice_emotion_train{}.json'

# 获取所有emotion相关的train JSON文件
emotion_train_files = [f for f in os.listdir(source_folder) if 'emotion' in f and 'train' in f and f.endswith('.json')]

# 初始化一个列表来存储所有数据
all_data = []

# 读取每个文件并合并数据
for file_name in emotion_train_files:
    file_path = os.path.join(source_folder, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        all_data.extend(data)

# 设置随机种子并打乱数据顺序
random.seed(42)
random.shuffle(all_data)

# 获取数据数量
data_count = len(all_data)
target_file = target_file_template.format(data_count)

# 将合并后的数据写入目标文件
with open(target_file, 'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)

print(f"合并完成，结果保存在 {target_file}")
