import cv2
import numpy as np
import glob
import os
from imutils.perspective import four_point_transform
import math


# 棋盘中心点坐标
def chessboard_center_point(corners2):
    # 左上角点
    left_top_point = corners2[0]
    # 右下角点
    right_button_point = corners2[-1]
    x_top = left_top_point[0][0]
    y_top = left_top_point[0][1]
    x_button = right_button_point[0][0]
    y_button = right_button_point[0][1]
    # 棋盘格中心点
    x_center = int(np.ceil((x_top + x_button) / 2))
    y_center = int(np.ceil((y_top + y_button) / 2))
    result_point = tuple([x_center, y_center])
    return result_point


# 棋盘分辨率：求最小分辨率（变形导致分辨率增大） 注意：此时长度单位为mm (**mm/pix)
def chess_resolution(corners2, width_number, height_number, cell_width):
    min_x_resolution = None
    width_length = cell_width * (width_number - 1)
    height_length = cell_width * (height_number - 1)
    # x分辨率
    for i in range(height_number):
        start_point = corners2[width_number * i]
        end_point = corners2[width_number * (i + 1) - 1]
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
    for i in range(width_number):
        start_point = corners2[i]
        end_point = corners[i-8]
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


# 图像清晰度
def image_sharpness(grayimg):
    score = cv2.Laplacian(grayimg, cv2.CV_64F).var()
    score = np.round(score)
    return score


# 透视变换后图像清晰度(四点透视变化)
def transform_image_sharpness(grayimg, corners2, width_number):
    # 转换为整型点
    point_1 = (int(corners2[0][0][0]), int(corners2[0][0][1]))
    point_2 = (int(corners2[width_number-1][0][0]), int(corners2[width_number-1][0][1]))
    point_3 = (int(corners2[-width_number][0][0]), int(corners2[-width_number][0][1]))
    point_4 = (int(corners2[-1][0][0]), int(corners2[-1][0][1]))
    docCnt = np.array([[point_1], [point_2], [point_3], [point_4]])
    paper = four_point_transform(grayimg, docCnt.reshape(4, 2))
    cv2.imwrite("transform_img1.jpg", paper)
    transform_score = image_sharpness(paper)
    return transform_score


if __name__ == '__main__':
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((8 * 11, 3), np.float32)
    # test_obj = np.mgrid[0:3, 0:2]
    # print('-----------', test_obj)

    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    # print(objp[:, :2])
    # print(objp.shape)  # (48, 3)

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    image_path = os.path.join("./image", "*.bmp")
    images = glob.glob(image_path)
    i = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 求图像清晰度评分
        current_score = image_sharpness(gray)  # 未透视变换 178
        print("----------清晰度----------\n", current_score)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)  # 寻找角点
        # print("---------corners-----------")
        # print(corners)
        if ret:
            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点坐标
            # print("--------------\n", corners2)  # （48, 1, 2）
            # 求得棋盘中心点坐标（图像上）
            center_point = chessboard_center_point(corners2)
            print("----------中心点----------\n", center_point)
            # width_cell_number 代表宽度方向上角点数量
            width_cell_number = 11
            height_cell_number = 8
            cell_width = 50  # 单位为mm 注意单位转换
            # 求分辨率
            x_resolution, y_resolution = chess_resolution(corners2, width_cell_number, height_cell_number, cell_width)
            # 透视变换后的清晰度  基于四点透视变换
            transform_score = transform_image_sharpness(gray, corners2, width_cell_number)
            print("----------分辨率----------\n", x_resolution, y_resolution)
            # print("-------------角点--------------")
            # print(corners2)
            if [corners2]:
                img_points.append(corners2)
            # else:
            #     img_points.append(corners)
            cv2.drawChessboardCorners(img, (11, 8), corners, ret)  # OpenCV的绘制函数一般无返回值
            i += 1
            cv2.imwrite('conimg' + str(i) + '.jpg', img)
            cv2.waitKey(1500)
    # print(len(img_points))
    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    # print("ret:", ret)
    # print("mtx:\n", mtx)  # 内参数矩阵
    # print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # print("rvecs:\n", rvecs)  # 旋转向量  # 外参数

    # x,y,z 轴的旋转角度分别如下
    x_angle = np.round(180 / np.pi * rvecs[0][0][0], 2)
    y_angle = np.round(180 / np.pi * rvecs[0][1][0], 2)
    z_angle = np.round(180 / np.pi * rvecs[0][2][0], 2)
    print("旋转角度分别为：", x_angle, y_angle, z_angle)
    print("tvecs:\n", tvecs)  # 平移向量  # 外参数

    # print("-----------------------------------------------------")
    # img = cv2.imread(images[0])
    # h, w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）
    # print(newcameramtx)
    # print("------------------使用undistort函数-------------------")
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # cv2.imshow("校正后", dst)
    # cv2.imwrite('undistort_calibration.png', dst)
    # x, y, w, h = roi
    # dst1 = dst[y:y + h, x:x + w]
    # cv2.imwrite('calibresult1.jpg', dst1)
    # print("方法一:dst的大小为:", dst1.shape)
    # print("-------------------使用重映射的方式-----------------------")
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)  # 获取映射方程
    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)      # 重映射
    # # dst = cv2.remap(img,mapx,mapy,cv2.INTER_CUBIC)
    # # 重映射后，图像变小了
    # x, y, w, h = roi
    # dst2 = dst[y:y+h, x:x+w]
    # cv2.imwrite('calibresult11_2.jpg', dst)
    # print("方法二:dst的大小为:", dst2.shape)

