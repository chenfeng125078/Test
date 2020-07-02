import os
import glob
import shutil
import numpy as np


data_path = os.path.join("./data/data_1/3", "*.bmp")
Formal = glob.glob(data_path)
image_number = len(Formal)
# 需测试的图片数量
test_number = 1000
i = 0
interval = int(np.floor(image_number / test_number))
print(interval)
# print()
for item in Formal:
    if  i % interval == 0:
        shutil.copy(item, os.path.join("./data/data_1","test_dir_3"))
        print(i)
    i += 1