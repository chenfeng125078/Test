import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import glob
import os
import numpy as np


def tf_rotate(input_image, min_angle = -np.pi/2, max_angle = np.pi/2):
  '''
  TensorFlow对图像进行随机旋转
  :param input_image: 图像输入
  :param min_angle: 最小旋转角度
  :param max_angle: 最大旋转角度
  :return: 旋转后的图像
  '''
  distorted_image = tf.expand_dims(input_image, 0)
  random_angles = tf.random.uniform(shape=(tf.shape(distorted_image)[0],), minval = min_angle , maxval = max_angle)
  distorted_image = tf.image.transpose(
    distorted_image,
    tf.image.transpose(
      random_angles, tf.cast(tf.shape(distorted_image)[1], tf.float32), tf.cast(tf.shape(distorted_image)[2], tf.float32)
    ))
  rotate_image = tf.squeeze(distorted_image, [0])
  return rotate_image


if __name__ == '__main__':
    base_dir = os.path.join("./data", "data_2/4")
    images_list = glob.glob(os.path.join(base_dir, "*.bmp"))
    target_path = os.path.join("./data", "data_stronger")
    i = 0
    times = 34
    for item in images_list:
        img = cv2.imread(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # global_init = tf.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            for number in range(times):
                # init = tf.initialize_local_variables()
                # sess.run([init, global_init])
                # coord = tf.train.Coordinator()
                # threads = tf.train.start_queue_runners(coord=coord)
                # 随机设置图片的亮度
                signal = i * times + number
                random_brightness = tf.image.random_brightness(img, max_delta=30)
                # 随机设置图片的对比度
                random_contrast = tf.image.random_contrast(img,lower=0,upper=1.8)
                # 随机设置图片的色度
                random_hue = tf.image.random_hue(img,max_delta=0.5)
                # 随机设置图片的饱和度
                random_satu = tf.image.random_saturation(img,lower=0,upper=1.8)
                # img = tf.image.random_flip_left_right(img)
                # # # 将图片随机进行垂直翻
                # img = tf.image.random_flip_up_down(img)
                # img = np.array(img)
                cv2.imwrite(os.path.join(target_path, "%d.bmp" % signal), img)
        i += 1
                # plt.imsave("./data/data_stronger/%d.bmp" % signal, img)
