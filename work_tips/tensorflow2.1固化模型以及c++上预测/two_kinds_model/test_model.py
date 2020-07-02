# encoding: utf-8
from tensorflow.keras.models import load_model
import os
import cv2
import scipy.io
import numpy as np
from glob import glob
import time


# 双分类模型2准确度：污渍：65 晶点：84
class Deep_learning(object):

    def __init__(self):
        # 读取已经训练好的深度学习模型 注意:这句代码执行时间比较长建议该模型放在开始实例化
        # self.base_model = load_model('D:\\jupyter\\model\\model.h5')
        self.base_model = load_model("D:\\classification_demo\\two_kinds_model\\cnn_callback\\two_kinds_model.h5")
        self.label_dict = {0: u"晶点", 1: u"污渍"}
#       self.base_model = load_model.load_weights("D:\\jupyter\\model\\my_model.h5")


start_time = time.time()
s = Deep_learning()

# In[51]:
# image1 = cv2.imdecode(np.fromfile("D:\\data_1\\污渍\\" + "卷号0_缺陷数据包29.bmp", dtype=np.uint8), 1)
# image = cv2.resize(image1, (168, 168))
# class_indxe = s.base_model.predict(image.reshape(-1, 168, 168, 3) / 255.0)
# class_indxe = np.argmax(class_indxe, axis=1)

# In[ ]:
# class_indxe
# In[56]:
# 测试与人工分类的标签相比准确度
data_path = os.path.join("../../data/2", "*.bmp")
Formal = glob(data_path)
# 测试图片总数
total_number = len(Formal)
counter = 0
index = 0
for item in Formal:
    # linux下
    # current_label = data_path.split("/")[-2]
    # current_label = (data_path.split("/")[-1]).split("\\")[0]
    # 类别对应 {1：错版(index= 4), 2: 污渍(index= 2), 3: 折皱(index= 0), 4: 破洞(index= 3), 5: 晶点(index= 1)}
    # right_index = index_dict[current_label]
    # print(right_index)
    # index = 0
    try:
        image1 = cv2.imdecode(np.fromfile(item, dtype=np.uint8), 1)
    except Exception as e:
        print("----------------")
        continue
    # 将图片resize成训练时大小
    image = cv2.resize(image1, (128, 128))
    class_index = s.base_model.predict(image.reshape(-1, 128, 128, 3) / 255.0)
    class_index = np.argmax(class_index, axis=1)
    # 要看是哪种类别
    if class_index[0] == 1:
        counter += 1
    else:
        pass

    if class_index[0] == 0:
        path = "D:\\data\\分类_晶点\\" + "%d.bmp" % index
        cv2.imencode('.bmp', image1)[1].tofile(path)

    elif class_index[0] == 1:
        path = "D:\\data\\分类_污渍\\" + "%d.bmp" % index
        cv2.imencode('.bmp', image1)[1].tofile(path)
    index += 1
# elif class_index[0] == 1:
#     path = "D:\\resnet-in-tensorflow\\data\\data_1\\污渍\\" + "%d.bmp" % index
#     cv2.imencode('.bmp', image1)[1].tofile(path)
# elif class_index[0] == 2:
#     path = "D:\\resnet-in-tensorflow\\data\\data_1\\破洞\\" + "%d.bmp" % index
#     cv2.imencode('.bmp', image1)[1].tofile(path)
# elif class_index[0] == 3:
#     path = "D:\\resnet-in-tensorflow\\data\\data_1\\错版\\" + "%d.bmp" % index
#     cv2.imencode('.bmp', image1)[1].tofile(path)

predict_acc = counter / total_number
print("---------准确率---------", predict_acc)
end_time = time.time()
print("use time：", end_time - start_time)


# test acc:
# 1000张：
# 缺版准确度:
# resnet50_150s_168 : acc: 0.911  use: 42.43
# resnet50_100s_168 : acc: 0.8317  use: 41.69
# resnet50_100s_128 : acc: 0.825  use: 40.35
# resnet50_80s_168 : acc: 0.779  use: 42.60
# resnet50_80s_128 : acc:0.798  use:40.24

# resnet50V2_100s_128 : acc：0.822 use: 40.81
# InceptionResNetV2_100s_128 : acc: 0.852 use: 93.38
# InceptionV3_100s_model_128: acc: 0.820 use: 52.89

# 折皱准确度: 分得很差
# resnet50_150s_168 : acc: 0.138  use: 42.43

# resnet50V2_100s_128 : acc：use:
# InceptionResNetV2_100s_128 : acc:  use:
# InceptionV3_100s_model_128: acc:  use:

# 晶点准确度:
# resnet50_150s_168 : acc: 0.911  use: 42.43

# resnet50V2_100s_128 : acc：0.822 use: 40.81
# InceptionResNetV2_100s_128 : acc: 0.852 use: 93.38
# InceptionV3_100s_model_128: acc: 0.820 use: 52.89
