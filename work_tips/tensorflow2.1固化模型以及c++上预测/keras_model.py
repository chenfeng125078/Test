#!/usr/bin/env python
# coding: utf-8

# In[39]:


import os
import cv2
import scipy.io
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import regularizers
import tensorflow as tf
import os
import time


#指定数据路径
Wrong_edition = glob(os.path.join("./data/1", "*.bmp"))

Stain = glob(os.path.join("./data/2", "*.bmp"))

cockle = glob(os.path.join("./data/3", "*.bmp"))

hole = glob(os.path.join("./data/4", "*.bmp"))

# Crystal_point = glob(os.path.join("./labelme_data/5", "*.bmp"))



# In[40]:
print(len(Wrong_edition), len(Stain), len(cockle), len(hole))

image_list=[]
label_list=[]
i=0
for a in Wrong_edition:
    #将数据读取进来并将大小改变成（168，168）
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(128,128))
    #将数据和标签保存起来
    image_list.append(image)
    label_list.append(["错版"])
    i+=1
    if i == 3000:
        # print(i)
        i = 0
        break

for a in Stain:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(128,128))
    image_list.append(image)
    label_list.append(["污渍+"])
    i+=1
    if i==3000:
        # print(i)
        i=0
        break

for a in cockle:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(128,128))
    image_list.append(image)
    label_list.append(["折皱"])
    i += 1
    if i == 3000:
        # print(i)
        i = 0
        break

for a in hole:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(128,128))
    image_list.append(image)
    image_list.append(image)
    label_list.append(["破洞"])
    label_list.append(["破洞"])
    i += 1
    # print(i)
    if i == 1000:
        i = 0
        break

# for a in Crystal_point:
#     image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
#     image = cv2.resize(image,(128,128))
#     image_list.append(image)
#     label_list.append(["晶点"])
#     i+=1
#     if i==1000:
#         # print(i)
#         i=0
#         break


#对数据标签进行热编码转化成向量形式
label = pd.get_dummies(pd.DataFrame(label_list))
class_label = label.columns



label = np.array(label)
print(class_label)
print(label.shape)


# In[41]:


image_list = np.array(image_list)
#划分测试集和训练集
xtrain,xtest,ytrain,ytest = train_test_split(image_list/255.0, label, test_size=0.30 ,random_state=0)
# print(ytrain)

# In[42]:
class_index = np.argmax(label, axis=1)
# print(class_indxe)

# In[43]:

base_model=tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling="avg")

# In[44]:

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
prediction = Dense(4, activation='softmax')(x)
# 构造完新的FC层，加入custom层
model = Model(inputs=base_model.input, outputs=prediction)

model.compile(loss='categorical_crossentropy', #多分类
              optimizer="adam" ,  # optimizer='Adadelta',
              metrics=['accuracy'])


# In[46]:


hist = model.fit(xtrain,ytrain,epochs=100 ,batch_size=128,validation_data=(xtest,ytest))
print(hist.history)

#
# # In[47]:
#
# # 模型_步数_尺寸大小_model.h5  尺寸大小默认168 ，256 跑不起来
model.save("D:\\resnet-in-tensorflow\\model\\four_kinds_model.h5")
# import json
# with open("./1.json", "w") as fp:
#     json.dump(hist.history, fp)

# resnet50_80s : 0.780
# resnet50_100s : 0.820
# resnet50_80s_128 : 0.8000
# resnet50_100s_128 : 0.873
# resnet50V2_100s_128 : 0.8530
# InceptionResNetV2_100s_128 : 0.945
# InceptionV3_100s_128 : 0.8347
# MobileNetV2_100s_128 : 0.2160
# In[48]:


# from tensorflow.keras.models import load_model


# In[49]:



#

# In[ ]:




