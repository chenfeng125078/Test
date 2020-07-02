import numpy as np
import glob
import os
import shutil


# 从总数据集中随机抽取部分数据集作为测试数据集
data_path = os.path.join("./data/data_1/5", "*.bmp")
Formal = glob.glob(data_path)
test_number = 1000
result = np.random.choice(Formal, test_number, replace=False)
for item in result:
    shutil.copy(item, "./data/data_1/test_dir_5")
