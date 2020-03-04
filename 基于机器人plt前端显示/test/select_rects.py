import numpy
import os

# 机器人表示 {deviceCode: [左上角点x坐标, 左上角点y坐标, 安全半径]}
rects = {"Robot_1": [365.5, 365.5, 9], "Robot_2": [395.5, 395.5, 10]}
x_data = float(input("输入第一次按钮x坐标："))
y_data = float(input("输入第一次按钮y坐标："))
for item in rects:
    select_case = False
    # 当前楼层某一个机器人
    current_rect = rects[item]
    x_min = current_rect[0]
    y_min = current_rect[1]
    safe_distance = current_rect[2]
    x_max = x_min + 2 * safe_distance
    y_max = y_min + 2 * safe_distance
    # 判断是否选中某一个机器人
    if (x_data > x_min) and (x_data < x_max):
        print("-------------")
