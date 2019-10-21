import sys
import os
import numpy
import cv2
import getopt


def handle_image(image_name):
    image_info = cv2.imread(image_name, 0)
    height, width = image_info.shape
    for i in range(height):
        for j in range(width):
            if image_info[i][j] <= 205:
                image_info[i][j] = 0
            else:
                image_info[i][j] = 255
    image_info[:10, :] = 0
    image_info[-16:, :] = 0
    image_info[:, :10] = 0
    image_info[:, -18:] = 0
    # print(hight, width)
    # print("--------", image_info)
    # cv2.imshow("image_info", image_info)
    # cv2.waitKey(0)
    return image_info


if __name__ == "__main__":
    img_path = os.path.realpath(".")
    argv = sys.argv
    short_args = "h"
    long_args = ["image=", "help"]
    opts, args = getopt.getopt(argv[1:], short_args, long_args)
    opts = dict(opts)
    image = "map.png"
    img_name = os.path.join(img_path, image)
    if "--image" in opts:
        image = opts["--image"]
        img_name = os.path.join(img_path, image)
    if "-h" in opts or "--help" in opts:
        print('输出形式为 python obstacle.py --image image_name')
        sys.exit()
    # print(img_name)
    image_info = handle_image(img_name)

