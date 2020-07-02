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
Wrong_edition = glob(os.path.join("./data/data_2/1", "*.bmp"))

Stain = glob(os.path.join("./data/data_2/2", "*.bmp"))

cockle = glob(os.path.join("./data/data_2/3", "*.bmp"))

hole = glob(os.path.join("./data/data_2/data_stronger", "*.bmp"))

Crystal_point = glob(os.path.join("./data/data_2/5", "*.bmp"))



# In[40]:
# print(len(Wrong_edition))

image_list=[]
label_list=[]
i=0
for a in Wrong_edition:
    #将数据读取进来并将大小改变成（256，256）
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(168,168))
    #将数据和标签保存起来
    image_list.append(image)
    label_list.append(["错版"])
    i+=1
    if i==1000:
        # print(i)
        i=0
        break

for a in Stain:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(168,168))
    image_list.append(image)
    label_list.append(["污渍"])
    i+=1
    if i==1000:
        # print(i)
        i=0
        break

for a in cockle:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(168,168))
    image_list.append(image)
    label_list.append(["折皱"])
    i+=1
    if i==1000:
        # print(i)
        i=0
        break

for a in hole:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(168,168))
    image_list.append(image)
    label_list.append(["破洞"])
    i+=1
    # print(i)
    if i==1000:
        i=0
        break

for a in Crystal_point:
    image = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    image = cv2.resize(image,(168,168))
    image_list.append(image)
    label_list.append(["晶点"])
    i+=1
    if i==1000:
        # print(i)
        i=0
        break


#对数据标签进行热编码转化成向量形式
label = pd.get_dummies(pd.DataFrame(label_list))
class_label = label.columns



label = np.array(label)
print(class_label)
print(label.shape)


# In[41]:


image_list=np.array(image_list)
#划分测试集和训练集
xtrain,xtest,ytrain,ytest=train_test_split(image_list/255.0, label, test_size=0.30 ,random_state=0)


# In[42]:


class_indxe = np.argmax(label, axis=1)
print(class_indxe)


# In[43]:


base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling="avg")


# In[44]:


x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(5, activation='softmax')(x)
# 构造完新的FC层，加入custom层
model = Model(inputs=base_model.input, outputs=prediction)

model.compile(loss='categorical_crossentropy', #多分类
              optimizer='Adadelta',
              
              metrics=['accuracy'])


# In[46]:


model.fit(xtrain,ytrain,epochs=150 ,batch_size=100,validation_data=(xtest,ytest))


# In[47]:


model.save("D:\\resnet-in-tensorflow\\model\\my_model.h5")


# In[48]:


# from tensorflow.keras.models import load_model


# In[49]:


from tensorflow.keras.models import load_model
class Deep_learning:

    def __init__(self):
        # 读取已经训练好的深度学习模型 注意:这句代码执行时间比较长建议该模型放在开始实例化
#         self.base_model = load_model('D:\\jupyter\\model\\model.h5')
        self.base_model=load_model("D:\\resnet-in-tensorflow\\model\\my_model.h5")

#         self.base_model = load_model.load_weights("D:\\jupyter\\model\\my_model.h5")

    def predict(self, x_data):
        """初步的测试封装"""
        self.base_model.predict_classes(x_data)

s = Deep_learning()


# In[51]:


# image1 = cv2.imdecode(np.fromfile("D:\\data_1\\污渍\\" + "卷号0_缺陷数据包29.bmp", dtype=np.uint8), 1)
# image = cv2.resize(image1, (168, 168))
# class_indxe = s.base_model.predict(image.reshape(-1, 168, 168, 3) / 255.0)
# class_indxe = np.argmax(class_indxe, axis=1)


# In[ ]:


# class_indxe


# In[56]:



Formal = glob("D:\\data\\2\\" + "*.bmp")
index = 0
for a in Formal:
    try:
        image1 = cv2.imdecode(np.fromfile(a, dtype=np.uint8), 1)
    except Exception as e:
        os.remove(a)
    finally:
        os.remove(a)
        pass
    if index % 1000 == 0:
        print(index)

    if index==1000:
        break
    image = cv2.resize(image1, (256, 256))
    class_indxe = s.base_model.predict(image.reshape(-1, 256, 256, 3) / 255.0)
    class_indxe = np.argmax(class_indxe, axis=1)

    if class_indxe[0] == 0:
        path = "D:\\resnet-in-tensorflow\\data_1\\折皱\\" + "%d.bmp" % index
        cv2.imencode('.bmp', image1)[1].tofile(path)
    elif class_indxe[0] == 1 :
        path = "D:\\resnet-in-tensorflow\\data_1\\晶点\\" + "%d.bmp" % index
        cv2.imencode('.bmp', image1)[1].tofile(path)
    elif class_indxe[0] == 2 :
        path = "D:\\resnet-in-tensorflow\\data_1\\污渍\\" + "%d.bmp" % index
        cv2.imencode('.bmp', image1)[1].tofile(path)
    elif class_indxe[0] == 3 :
        path = "D:\\resnet-in-tensorflow\\data_1\\破洞\\" + "%d.bmp" % index
        cv2.imencode('.bmp', image1)[1].tofile(path)
    elif class_indxe[0] == 4 :
        path = "D:\\resnet-in-tensorflow\\data_1\\错版\\" + "%d.bmp" % index
        cv2.imencode('.bmp', image1)[1].tofile(path)
    index += 1
#

# In[ ]:




