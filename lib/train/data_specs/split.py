import collections
import os

# 1. 读取原始数据
with open('/data/zjy/OSTrack_multi/lib/train/data_specs/lasot_train_split.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# 2. 按类别分组
class_dict = collections.defaultdict(list)
for line in lines:
    cls_name = line.split('-')[0]
    class_dict[cls_name].append(line)

train_list = []
val_list = []

# 3. 分层划分 (14个给train, 2个给val)
for cls_name, seqs in class_dict.items():
    train_list.extend(seqs[:-2])
    val_list.extend(seqs[-2:])

# 4. 写入新文件
with open('lasot_train_new_split.txt', 'w') as f:
    f.write('\n'.join(train_list) + '\n')

with open('lasot_val_split.txt', 'w') as f:
    f.write('\n'.join(val_list) + '\n')

print(f"✅ 划分完成！Train 数量: {len(train_list)}, Val 数量: {len(val_list)}")