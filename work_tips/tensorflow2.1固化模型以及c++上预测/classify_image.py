import glob
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import sys
import shutil


class Classify(object):
    # 指定存储的网络模型路径
    def __init__(self, s_modelpath):
        # 训练时独热编码所对应的类别 其中 污渍+ 代表的类别为 污渍or晶点
        self.label_dict = {0: "折痕", 1: "污渍+", 2: "破洞", 3: "错版"}
        self.base_model = load_model(s_modelpath)

    def judge(self, srcimg, orilabel):
        # 读取目录
        if os.path.isdir(srcimg):
            test_accuracy = self.all_image_accuracy(srcimg, orilabel)
            return test_accuracy
        # 读取图片
        else:
            try:
                image1 = cv2.imdecode(np.fromfile(srcimg, dtype=np.uint8), 1)
            except Exception as e:
                os.remove(srcimg)
                print("can not decode this image")
                sys.exit()
            result = self.judge_single_image(srcimg, orilabel)
            return result

    # 单张图片分类
    def judge_single_image(self, image1, orilabel):
        image = cv2.resize(image1, (128, 128))
        # 预测类别
        class_index = self.base_model.predict_classes(image.reshape(-1, 128, 128, 3) / 255.0)
        class_index = np.argmax(class_index, axis=1)
        # 数字标签判断
        if isinstance(orilabel, int):
            if orilabel == class_index:
                # print("Classified correctly")
                return True
            else:
                # print("wrong")
                return False
        # 字符类别判断
        elif isinstance(orilabel, str):
            if self.label_dict[class_index] == orilabel:
                # print("Classified correctly")
                return True
            else:
                # print("wrong")
                return False
        else:
            print("输入标签类型错误")
            sys.exit()

    # 用来测试某一个同类别目录下的准确度
    def all_image_accuracy(self, one_dir, orilabel):
        # 这里是 bmp 图片
        image_list = glob.glob(os.path.join(one_dir, "*.bmp"))
        # 实际测试的图片数量(有部分图片不能读取，防止对准确度进行干扰)  todo:是否需要加入时间模块
        test_image_number = 0
        # 分类准确的图片数量
        class_correct_number = 0
        for item in image_list:
            try:
                image1 = cv2.imdecode(np.fromfile(item, dtype=np.uint8), 1)
            except Exception as e:
                os.remove(item)
                continue
            result = self.judge_single_image(image1, orilabel)
            test_image_number += 1
            if result:
                class_correct_number += 1
        test_accuracy = class_correct_number / test_image_number
        print("the test accuracy amost %.2f" % test_accuracy)
        return test_accuracy

    # 识别图片
    def recognized(self, srcimg):
        # 文件夹
        if os.path.isdir(srcimg):
            # 这里是 bmp 图片
            image_list = glob.glob(os.path.join(srcimg, "*.bmp"))
            image_class_dict = dict()
            for item in image_list:
                try:
                    image1 = cv2.imdecode(np.fromfile(item, dtype=np.uint8), 1)
                except Exception as e:
                    os.remove(item)
                    continue
                image = cv2.resize(image1, (128, 128))
                # 预测类别
                class_index = self.base_model.predict_classes(image.reshape(-1, 128, 128, 3) / 255.0)
                class_index = np.argmax(class_index, axis=1)
                image_class_dict[item] = self.label_dict[class_index]
            # 返回字典 {图片：标签}  todo:是否需要将不同标签的图片放入特定文件夹
            return image_class_dict

        # 单张图片处理方式
        else:
            try:
                image1 = cv2.imdecode(np.fromfile(srcimg, dtype=np.uint8), 1)
            except Exception as e:
                print("can not decode this image")
                sys.exit()
            # resize图片至训练时大小, demo中图片为 128 * 128
            image = cv2.resize(image1, (128, 128))
            # 预测类别
            class_index = self.base_model.predict_classes(image.reshape(-1, 128, 128, 3) / 255.0)
            class_index = np.argmax(class_index, axis=1)
            # 返回类别名
            return self.label_dict[class_index]

    def continue_train(self):
        # 基于现有demo继续训练
        pass

