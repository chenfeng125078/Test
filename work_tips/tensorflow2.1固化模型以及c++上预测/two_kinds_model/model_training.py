#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import scipy.io
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import regularizers
import tensorflow as tf
import os
import time

# 指定数据路径

Stain = glob(os.path.join("./data/2", "*.bmp"))  # 污渍数据集

Crystal_point = glob(os.path.join("./data/5", "*.bmp"))  # 晶点数据集

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
    image = cv2.resize(image,(128,128))
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
# [晶点， 污渍]
print(label.shape)

image_list = np.array(image_list)
# 划分测试集和训练集
xtrain, xtest, ytrain, ytest = train_test_split(image_list / 255.0, label, test_size=0.30, random_state=0)
# print(ytrain)

# In[42]:
class_index = np.argmax(label, axis=1)
# print(class_indxe)

base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, input_tensor=None,
                                                     input_shape=None, pooling="avg")
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

# x = Dense(2048, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(2048, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(2048, activation='relu')(x)
# x = Dropout(0.5)(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

prediction = Dense(2, activation='softmax')(x)
# 构造完新的FC层，加入custom层
model = Model(inputs=base_model.input, outputs=prediction)

model.compile(loss='categorical_crossentropy',  # 多分类
              optimizer="adam",  # optimizer='Adadelta',
              metrics=['accuracy'])

log_dir = os.path.join("cnn_callback")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
output_file = os.path.join(log_dir, "two_kinds_model.h5")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
hist = model.fit(xtrain, ytrain, epochs=100, batch_size=16, validation_data=(xtest, ytest), callbacks=[tensorboard_callback])
# print(hist.history)

# 模型_步数_尺寸大小_model.h5  尺寸大小默认128
model.save(output_file)
