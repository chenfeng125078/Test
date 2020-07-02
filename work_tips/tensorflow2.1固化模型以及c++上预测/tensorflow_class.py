import tensorflow as tf
import pandas
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)


if __name__ == '__main__':
    resize_width = 256
    resize_height = 256
    data_path_1 = os.path.join("./data","data_1" )
    data_path_2 = os.path.join("./data","data_2" )
    image_list_1 = glob.glob(os.path.join(data_path_1, "5/*.bmp"))
    image_list_2 = glob.glob(os.path.join(data_path_2, "5/*.bmp"))
    image_list = image_list_1 + image_list_2
    target_path = os.path.join("./data", "resize_data/5")
    if os.path.exists(target_path):
        pass
    else:
        os.mkdir(target_path)
        print("create dir")
    # print(len(image_list))  # 1299
    i = 0
    for item in image_list:
        # print(item)
        image = cv2.imread(item)

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        resize_image = tf.image.resize(image, (resize_width, resize_height))
        resize_image = np.array(resize_image)
        # print(resize_image)
        # cv2.imshow("image", resize_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(target_path, "%d.bmp" % i), resize_image * 255)
        i += 1
        # print(resize_image)
        # plt.imshow(resize_image)
        # plt.show()
        # print(image.shape)




