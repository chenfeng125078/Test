import os
import cv2
import scipy.io
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import regularizers
import tensorflow as tf
import os
import time


Stain = glob(os.path.join("./data/2", "*.bmp"))

Crystal_point = glob(os.path.join("./data/5", "*.bmp"))

image_list = []
label_list = []
i = 0
for a in Stain:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image, (128, 128))
    image_list.append(image)
    label_list.append(["污渍"])
    i += 1
    if i == 1650:
        # print(i)
        i = 0
        break
for a in Crystal_point:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image, (128, 128))
    image_list.append(image)
    label_list.append(["晶点"])
    i += 1
    if i == 1650:
        # print(i)
        i = 0
        break
# 对数据标签进行热编码转化成向量形式
label = pd.get_dummies(pd.DataFrame(label_list))
class_label = label.columns

label = np.array(label)
print(class_label)
print(label.shape)

image_list = np.array(image_list)
# 划分测试集和训练集
xtrain, xtest, ytrain, ytest = train_test_split(image_list / 255.0, label, test_size=0.30, random_state=0)
model = load_model("D:\\model\\two_kinds_model_170.h5")
hist = model.fit(xtrain, ytrain, epochs=30, batch_size=16, validation_data=(xtest, ytest))
# print(hist.history)

model.save("D:\\model\\two_kinds_model_200.h5")
