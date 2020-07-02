# -*- coding: utf-8 -*-
import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt


# case的情况是为了防止图像过小
def split_image(srcimg, center_point, split_mode, case=1):
    if split_mode == 1:
        # 横向框
        if case == 1:
            # print("-------------", srcimg.shape)
            start_coordinate = int(center_point - 300)
            end_coordinate = int(center_point + 300)
            return srcimg[start_coordinate:end_coordinate, :]
    else:
        # 纵向框
        if case == 1:
            # print("=============", srcimg.shape)
            start_coordinate = int(center_point - 300)
            end_coordinate = int(center_point + 300)
            return srcimg[:, start_coordinate:end_coordinate]


def compute_mat(cutimg_info, split_mode, threshold, combin_value):
    if split_mode == 1:
        # 计算竖线
        number_counter = 0
        one_list = np.sum(cutimg_info, axis=0)//600
        new_list = list()
        for i in range(len(one_list) // combin_value):
            cut_list = one_list[i * combin_value:(i + 1) * combin_value]
            value = np.sum(cut_list) // combin_value
            new_list.append(value)
        print(len(new_list))
        for i in range(len(new_list) - 1):
            if int(new_list[i + 1]) - int(new_list[i]) > threshold:
                number_counter += 1
        # 查看图像像素值分布
        # value_dict = Counter(one_list)
        x = list(range(len(new_list)))
        y = new_list
        plt.plot(x, y)
        plt.show()
        return number_counter

    else:
        # 计算横线
        number_counter = 0
        one_list = np.sum(cutimg_info, axis=1)//600
        new_list = list()
        print(len(one_list))
        for i in range(len(one_list) // combin_value):
            cut_list = one_list[i * combin_value:(i + 1) * combin_value]
            value = np.sum(cut_list) // combin_value
            new_list.append(value)
        for i in range(len(new_list) - 1):
            if int(new_list[i + 1]) - int(new_list[i]) > threshold:
                number_counter += 1
        # x = list(range(len(new_list)))
        # y = new_list
        # # y = value_dict.values()
        # plt.plot(x, y)
        # plt.show()
        return number_counter


if __name__ == '__main__':
    image_path = "./transform.bmp"
    image_info = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image_info, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    erode = cv2.erode(gray_img, kernel)
    dst1 = cv2.subtract(gray_img, erode)
    cv2.imwrite("./image/inner.bmp", dst1)
    height, width = dst1.shape
    # print(dst1[height-1][width-1])
    print(height, width)
    center_x, center_y = height // 4, width // 4
    split_case = 1
    # 根据不同情况截取部分图像矩阵
    row_mat = split_image(dst1, center_x, 1, split_case)
    # 横向阈值
    row_threshold = 30
    # 横向像素组合值
    combin_value_1 = 10
    cols_number = compute_mat(row_mat, 1, row_threshold, combin_value_1)
    col_mat = split_image(dst1, center_y, 0, split_case)
    # 竖向阈值
    col_threshold = 12
    # 竖向像素组合值
    combin_value_2 = 5
    # 减去两条边框
    rows_number = compute_mat(col_mat, 0, col_threshold, combin_value_2)
    print(cols_number, rows_number)
