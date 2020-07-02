import numpy as np
import json
import os
import sys
import glob
import cv2


def clip_image(current_image, x1, x2, y1, y2, image_number):
    img = cv2.imread(current_image)
    # print(img.shape)
    cut_img = img[x1:x2, y1:y2, :]
    try:
        cv2.imwrite("%s.bmp" % image_number, cut_img)
    except:
        print("can not write")
    # cv2.imshow("image", img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    json_dir = "1_json"
    base_dir = os.path.join("./", json_dir)
    images = glob.glob(os.path.join(base_dir, "*.json"))
    # print(images)
    for item in images[:]:
        # print(item)
        # 对应的bmp文件
        image_number = item.split("\\")[-1].split(".")[0]
        if len(image_number) >= 2:
            image_number_source = image_number[:-1]
        # 对应的图像文件夹
        if os.path.exists(os.path.join("./1", "%s.bmp" % image_number)):
            current_image = os.path.join("./1", "%s.bmp" % image_number)
        elif os.path.exists(os.path.join("./1", "%s.bmp" % image_number_source)):
            current_image = os.path.join("./1", "%s.bmp" % image_number_source)
        else:
            continue
        # print(current_image)
        with open(item, "r") as f:
            data = json.load(f)
            # print(data)
            point_1 = data["shapes"][0]["points"][0]
            point_2 = data["shapes"][0]["points"][1]
            x1, y1 = point_1[0], point_1[1]
            x2, y2 = point_2[0], point_2[1]
            if x1 < x2:
                pass
            else:
                x1, x2 = x2, x1
            if y1 < y2:
                pass
            else:
                y1, y2 = y2, y1
            x1, y1 = int(np.ceil(x1)), int(np.ceil(y1))
            x2, y2 = int(np.floor(x2)), int(np.floor(y2))
            print(x1, y1, x2, y2)
            clip_image(current_image, x1, x2, y1, y2, image_number)
            # print(point_1, point_2)
