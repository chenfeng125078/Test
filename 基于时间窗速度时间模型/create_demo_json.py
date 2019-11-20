import json
import os
import sys


def create_points_dict(number):
    points = {}
    string_choice = input("选择默认点还是自己输入点坐标,yes或y：默认, no或n:自己输入点坐标")
    if string_choice.lower().strip() == "yes" or string_choice.lower().strip() == "y" or not string_choice:
        points = {"point_1": (0, 4), "point_2": (0, 2), "point_3": (0, 0), "point_4": (2, 18.5), "point_5": (2, 2),
                  "point_6": (2, 0)}
    else:
        for i in range(number):
            key = "point_%d" % i
            point_coordinates = input("输入坐标 x,y 逗号分隔,例: 2,3")  # generate points
            value_x = float(point_coordinates.split(",")[0])
            value_y = float(point_coordinates.split(",")[1])
            value = (value_x, value_y)
            points[key] = value
    return points


if __name__ == '__main__':
    base_dir = os.path.realpath(".")
    string = input("输入点的数量：")
    if string:
        try:
            points_number = int(string)
        except:
            print("类型错误，请输入整数")
            sys.exit()
    else:
        print("string is None")
        points_number = 11
    data = create_points_dict(points_number)
    with open("points_2.json", "w") as fp:
        json.dump(data, fp)
