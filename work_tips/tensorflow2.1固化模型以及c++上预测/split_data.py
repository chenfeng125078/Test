import glob
import os
import numpy as np
import shutil


def split_images(images, case):
    base_dir = os.path.join("./data", case)
    # print(target_dir)
    for item in images:
        # print(item)
        dir_name = item.split("\\")[-2]
        # print(dir_name)
        # print(file_name)
        # break
        target_dir = os.path.join(base_dir, dir_name)
        # print(target_dir)
        # break
        if os.path.exists(target_dir):
            pass
        else:
            os.mkdir(target_dir)
        # print(os.path.join(target_dir, file_name))
        shutil.copy(item, target_dir)


if __name__ == '__main__':
    data_path_1 = os.path.join("./data", "data_1")
    data_path_2 = os.path.join("./data", "data_2")
    for item in os.listdir(data_path_1):
        # print(item)
        data_1 = glob.glob(os.path.join(data_path_1, "%s/*.bmp" % item))
        data_2 = glob.glob(os.path.join(data_path_2, "%s/*.bmp" % item))
        data = data_1 + data_2
        # print(len(data_1))
        total_number = len(data)
        train_number = int(np.ceil(total_number * 0.7))
        train_images = data[:train_number]
        val_images = data[train_number:]
        # 训练数据集
        split_images(train_images, "train")
        # print("train images is ok")
        # 验证数据集
        split_images(val_images, "val")
    print("finished")
