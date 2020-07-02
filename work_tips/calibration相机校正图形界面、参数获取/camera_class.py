# -*- coding: utf-8 -*-
from imutils.perspective import four_point_transform
# from numpy._distributor_init import NUMPY_MKL  # requires numpy+mkl
import scipy
import cv2
import numpy as np
import glob
import os
import time


class CameraCorrect:
    def __init__(self, Img, width_cell_number=11, height_cell_number=8, cell_width=50):
        # 初始化图片路径
        # self.image_path = chess_board_path
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        # 宽度、长度方向上的角点个数
        self.width_number = width_cell_number
        self.height_number = height_cell_number
        self.objp = np.zeros((self.height_number * self.width_number, 3), np.float32)
        # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        self.objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        self.obj_points = []  # 存储3D点
        self.img_points = []  # 存储2D点
        if len(Img.shape) > 2:
            self.gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = Img
        self.size = self.gray.shape[::-1]
        start_time = time.time()
        # 均衡化图像高度 cv2.CALIB_CB_NORMALIZE_IMAGE  cv2.CALIB_CB_FILTER_QUADS
        self.ret, self.corners = cv2.findChessboardCorners(self.gray,
                                                           (self.width_number, self.height_number),
                                                           flags=cv2.CALIB_CB_FILTER_QUADS)
        end_time = time.time()
        print("--------", end_time - start_time)
        self.corners2 = ""
        self.center_point = tuple(["", ""])
        self.x_resolution, self.y_resolution = "", ""
        self.sharpness_score, self.brightness_score = "", ""
        self.x_angle, self.y_angle, self.z_angle = "", "", ""
        if self.ret:
            self.obj_points.append(self.objp)
            # 在原角点的基础上寻找亚像素角点坐标
            self.corners2 = cv2.cornerSubPix(self.gray, self.corners, (5, 5), (-1, -1), self.criteria)
            # 求棋盘的中心点
            self.center_point = self.chessboard_center_point()
            self.cell_width = cell_width
            # x方向 以及 y方向上的分辨率
            self.x_resolution, self.y_resolution = self.chess_resolution()
            # 透视变换后图像清晰度(四点透视变化)
            self.sharpness_score = self.transform_image_sharpness()
            # 灰度图像下求图像亮度
            self.brightness_score = self._image_brightness()
            if [self.corners2]:
                self.img_points.append(self.corners2)
            # cv2.drawChessboardCorners(img, (11, 8), self.corners, self.ret)  # OpenCV的绘制函数一般无返回值
            # 标定
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, self.size, None, None)
            self.x_angle = np.round(180 / np.pi * self.rvecs[0][0][0], 2)
            self.y_angle = np.round(180 / np.pi * self.rvecs[0][1][0], 2)
            self.z_angle = np.round(180 / np.pi * self.rvecs[0][2][0], 2)

    def chessboard_center_point(self):
        # 左上角点
        left_top_point = self.corners2[0]
        # 右下角点
        right_button_point = self.corners2[-1]
        x_top = left_top_point[0][0]
        y_top = left_top_point[0][1]
        x_button = right_button_point[0][0]
        y_button = right_button_point[0][1]
        # 棋盘格中心点
        x_center = int(np.ceil((x_top + x_button) / 2))
        y_center = int(np.ceil((y_top + y_button) / 2))
        result_point = tuple([x_center, y_center])
        return result_point

    def chess_resolution(self):
        min_x_resolution = None
        width_length = self.cell_width * (self.width_number - 1)
        height_length = self.cell_width * (self.height_number - 1)
        # x分辨率
        for i in range(self.height_number):
            start_point = self.corners2[self.width_number * i]
            end_point = self.corners2[self.width_number * (i + 1) - 1]
            start_point_x = start_point[0][0]
            start_point_y = start_point[0][1]
            end_point_x = end_point[0][0]
            end_point_y = end_point[0][1]
            pix_number = np.sqrt((end_point_x - start_point_x) ** 2 + (end_point_y - start_point_y) ** 2)
            current_resolution = width_length / pix_number
            if not min_x_resolution:
                min_x_resolution = current_resolution
            else:
                if current_resolution < min_x_resolution:
                    min_x_resolution = current_resolution
        min_x_resolution = np.round(min_x_resolution, 2)
        # y分辨率
        min_y_resolution = None
        for i in range(self.width_number):
            start_point = self.corners2[i]
            end_point = self.corners[i-8]
            start_point_x = start_point[0][0]
            start_point_y = start_point[0][1]
            end_point_x = end_point[0][0]
            end_point_y = end_point[0][1]
            pix_number = np.sqrt((end_point_x - start_point_x) ** 2 + (end_point_y - start_point_y) ** 2)
            current_resolution = height_length / pix_number
            if not min_y_resolution:
                min_y_resolution = current_resolution
            else:
                if current_resolution < min_y_resolution:
                    min_y_resolution = current_resolution
        min_y_resolution = np.round(min_y_resolution, 2)
        return min_x_resolution, min_y_resolution

    def _image_brightness(self):
        mean_value = np.round(cv2.mean(self.gray)[0], 2)
        # print(mean_value)
        return mean_value

    def transform_image_sharpness(self):
        # 转换为整型点
        # 左上点
        point_1 = (int(self.corners2[0][0][0]), int(self.corners2[0][0][1]))
        # 右上点
        point_2 = (int(self.corners2[self.width_number - 1][0][0]), int(self.corners2[self.width_number - 1][0][1]))
        # 左下点
        point_3 = (int(self.corners2[-self.width_number][0][0]), int(self.corners2[-self.width_number][0][1]))
        # 右下点
        point_4 = (int(self.corners2[-1][0][0]), int(self.corners2[-1][0][1]))
        docCnt = np.array([[point_1], [point_2], [point_3], [point_4]])
        paper = four_point_transform(self.gray, docCnt.reshape(4, 2))
        # brenner梯度求清晰度
        # brenner_score = self.brenner(paper)

        # sobel算子求清晰度
        sobel_score = float(self.sobel_compute(paper))

        # 拉普拉斯算子求清晰度
        # score = cv2.Laplacian(paper, cv2.CV_64F).var()
        # score = np.round(score, 3)
        return sobel_score

    def sobel_compute(self, img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        x = np.sum(np.abs(x))
        y = np.sum(np.abs(y))
        sobel_score = x + y
        return sobel_score

    def brenner(self, img):
        '''
        :param img:narray 二维灰度图像
        :return: float 图像约清晰越大
        '''
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 2):
            for y in range(0, shape[1]):
                out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
        return out


if __name__ == '__main__':
    example_1 = CameraCorrect("../image/1.bmp")
    example_2 = CameraCorrect("../image/5.bmp")
    example_3 = CameraCorrect("../image/6.bmp")
    example_4 = CameraCorrect("../image/8.bmp")
    example_5 = CameraCorrect("../image/9.bmp")
    print(float(example_1.sharpness_score)/float(example_2.sharpness_score))
    # print(example.center_point, example.sharpness_score)
