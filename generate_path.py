import json
import sys
import os
import time
import numpy as np
import cv2


def read_doors(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)
        doors_data = data["doors"]
        # print(len(doors_data))
        door_list = []
        for item in doors_data:
            door_now = eval(item["polygon"])
            door_list.append(list(door_now))
    # print(len(door_list))

    return door_list


def compute_door_direction(door_list, min_width):
    left_right_door = []
    up_down_door = []
    # door is four points list
    # cv2.imshow("yes", door_image)
    # cv2.waitKey(0)
    for item in door_list:
        point_1 = item[0]
        point_2 = item[1]
        point_3 = item[2]
        x_delta = abs(point_2[0] - point_1[0])
        y_delta = abs(point_2[1] - point_1[1])
        if x_delta > y_delta:
            x_value = x_delta
            y_value = abs(point_3[1] - point_2[1])
        else:
            y_value = y_delta
            x_value = abs(point_3[0] - point_2[0])
        # item内有五个点。每个点有(x， y)  item = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
        item.append((int(np.ceil(x_value / 2)), int(np.ceil(y_value / 2))))
        if x_value > y_value:
            # 注意该处 append 的是一个列表，其中item 是上面的列表
            # print(x_value)
            if x_value > min_width:
                up_down_door.append([item, x_value, y_value])
        else:
            # print(y_value)
            if y_value > min_width:
                left_right_door.append([item, x_value, y_value])
    # print(left_right_door)
    # print(len(up_down_door), "\n", len(left_right_door))
    return up_down_door, left_right_door


def compute_start_points(up_down_door, left_right_door, door_image, base_row_map, base_col_map, hight, width,
                         safe_interval):
    all_points_list = []
    col_points_list = []
    for door in up_down_door:
        x_min = door[0][0][0]
        y_min = door[0][0][1]
        # x_value = door[1]
        y_value = door[2]
        for point in door[0][1:4]:
            point_x = point[0]
            point_y = point[1]
            if point_x < x_min:
                x_min = point_x
            if point_y < y_min:
                y_min = point_y
        start_point = (hight - y_min - door[0][-1][1], x_min + door[0][-1][0])
        # 向上延伸的起点
        up_start_point = (hight - y_min - y_value, x_min + door[0][-1][0])
        # 向下延伸的起点
        down_start_point = (hight - y_min, x_min + door[0][-1][0])

        points_list = generate_up_room_point(start_point, up_start_point, down_start_point, door_image, hight,
                                             base_row_map, safe_interval)
        col_points_list.append(points_list)
        all_points_list.append(points_list)
    row_points_list = []
    for door in left_right_door:
        x_min = door[0][0][0]
        y_min = door[0][0][1]
        x_value = door[1]
        # y_value = door[2]
        for point in door[0][1:4]:
            point_x = point[0]
            point_y = point[1]
            if point_x < x_min:
                x_min = point_x
            if point_y < y_min:
                y_min = point_y
        start_point = (hight - y_min - door[0][-1][1], x_min + door[0][-1][0])
        # 向左延伸的起点
        left_start_point = (hight - y_min - door[0][-1][1], x_min)
        # 向右延伸的起点
        right_start_point = (hight - y_min - door[0][-1][1], x_min + x_value)
        points_list = generate_left_room_point(start_point, left_start_point, right_start_point, door_image, width,
                                               base_col_map, safe_interval)
        row_points_list.append(points_list)
        all_points_list.append(points_list)

    return all_points_list, col_points_list, row_points_list
    # cv2.imshow("test", door_image)
    # cv2.waitKey(0)


def generate_up_room_point(start_point, up_start_point, down_start_point, door_image, hight, base_row_map,
                           safe_interval):
    up_start_point_x = up_start_point[0]
    up_start_point_y = up_start_point[1]
    points_list = []
    for x_delta in range(1, up_start_point_x + 1):
        case = True
        now_x = up_start_point_x - x_delta
        for i in range(1, safe_interval + 1):
            if door_image[now_x][up_start_point_y] == 0 or door_image[now_x][up_start_point_y + i] == 0 or \
                    door_image[now_x][up_start_point_y - i] == 0:
                for red_delta in range(x_delta):
                    if not (base_row_map[up_start_point_x - red_delta][up_start_point_y] - [255, 0, 0]).any():
                        points_list.append((up_start_point_x - red_delta, up_start_point_y))
                        case = False
                        break
                if case:
                    points_list.append((up_start_point_x - int(np.ceil(x_delta / 2)), up_start_point_y))
                    case = False
                    break
                else:
                    break
        if case:
            continue
        else:
            break
    # points_list.append(start_point)
    down_start_point_x = down_start_point[0]
    down_start_point_y = down_start_point[1]
    for low_x in range(1, hight - down_start_point_x + 1):
        case = True
        now_x = down_start_point_x + low_x
        for i in range(1, safe_interval + 1):
            if door_image[now_x][down_start_point_y] == 0 or door_image[now_x][down_start_point_y + i] == 0 or \
                    door_image[now_x][down_start_point_y - i] == 0:
                for add_number in range(low_x):
                    if not (base_row_map[down_start_point_x + add_number][down_start_point_y] - [255, 0, 0]).any():
                        points_list.append((down_start_point_x + add_number, down_start_point_y))
                        case = False
                        # break
                if case:
                    points_list.append((down_start_point_x + int(np.floor(low_x / 2)), down_start_point_y))
                    case = False
                    break
                else:
                    break
        if case:
            continue
        else:
            break
    # 列表内的点都是从上到下（从左到右）
    # print("------------points-------------", points_list)
    return points_list


def generate_left_room_point(start_point, left_start_point, right_start_point, door_image, width, base_col_map,
                             safe_interval):
    left_start_point_x = left_start_point[0]
    left_start_point_y = left_start_point[1]
    points_list = []
    for y_delta in range(1, left_start_point_y + 1):
        case = True
        now_y = left_start_point_y - y_delta
        for i in range(1, safe_interval + 1):
            if door_image[left_start_point_x][now_y] == 0 or door_image[left_start_point_x + i][now_y] == 0 or \
                    door_image[left_start_point_x - i][now_y] == 0:
                for red_delta in range(y_delta):
                    if not (base_col_map[left_start_point_x][left_start_point_y - red_delta] - [255, 0, 0]).any():
                        points_list.append((left_start_point_x, left_start_point_y - red_delta))
                        case = False
                        # break
                if case:
                    points_list.append((left_start_point_x, left_start_point_y - int(np.ceil(y_delta / 2))))
                    case = False
                    break
                else:
                    break
        if case:
            continue
        else:
            break
    # points_list.append(start_point)
    right_start_point_x = right_start_point[0]
    right_start_point_y = right_start_point[1]
    for right_y in range(1, width - right_start_point_y + 1):
        case = True
        now_y = right_start_point_y + right_y
        for i in range(1, safe_interval + 1):
            if door_image[right_start_point_x][now_y] == 0 or door_image[right_start_point_x + i][now_y] == 0 or \
                    door_image[right_start_point_x - i][now_y] == 0:
                for add_number in range(right_y):
                    if not (base_col_map[right_start_point_x][right_start_point_y + add_number] - [255, 0, 0]).any():
                        points_list.append((right_start_point_x, right_start_point_y + add_number))
                        case = False
                        # break
                if case:
                    points_list.append((right_start_point_x, right_start_point_y + int(np.floor(right_y / 2))))
                    case = False
                    break
                else:
                    break
        if case:
            continue
        else:
            break
    return points_list


def show_on_map(all_points_list, add_points, door_image, width, hight):
    show_point_set = set()
    for one_point_list in all_points_list:
        for one_point in one_point_list:
            show_point_set.add(one_point)
    for point in add_points:
        show_point_set.add(point)
    points_list = list(show_point_set)
    show_list = []
    for first in range(len(points_list)):
        # 左.右.上.下
        tmp_link = [[], [], [], []]
        min_left_distance = width
        min_right_distance = width
        min_up_distance = hight
        min_down_distance = hight
        for second in range(first + 1, len(points_list)):
            case = True
            first_point = points_list[first]
            second_point = points_list[second]
            if first_point[0] == second_point[0]:
                if first_point[1] > second_point[1]:
                    for left_delta in range(1, first_point[1] - second_point[1]):
                        if door_image[first_point[0]][first_point[1] - left_delta] == 0:
                            case = False
                            break
                    if case:
                        left_distance = first_point[1] - second_point[1]
                        if left_distance < min_left_distance:
                            min_left_distance = left_distance
                            tmp_link[0] = [first_point, second_point]
                else:
                    for right_delta in range(1, second_point[1] - first_point[1]):
                        if door_image[first_point[0]][first_point[1] + right_delta] == 0:
                            case = False
                            break
                    if case:
                        right_distance = second_point[1] - first_point[1]
                        if right_distance < min_right_distance:
                            min_right_distance = right_distance
                            tmp_link[1] = [first_point, second_point]
            if first_point[1] == second_point[1]:
                if first_point[0] > second_point[0]:
                    for up_delta in range(1, first_point[0] - second_point[0]):
                        if door_image[first_point[0] - up_delta][first_point[1]] == 0:
                            case = False
                            break
                    if case:
                        up_distance = first_point[0] - second_point[0]
                        if up_distance < min_up_distance:
                            min_up_distance = up_distance
                            tmp_link[2] = [first_point, second_point]
                else:
                    for down_delta in range(1, second_point[0] - first_point[0]):
                        if door_image[first_point[0] + down_delta][first_point[1]] == 0:
                            case = False
                            break
                    if case:
                        down_distance = second_point[0] - first_point[0]
                        if down_distance < min_down_distance:
                            min_down_distance = down_distance
                            tmp_link[3] = [first_point, second_point]
        # print(tmp_link)
        show_list.append(tmp_link)
    for item in show_list:
        if item[0]:
            for i in range(item[0][0][1] - item[0][1][1]):
                door_image[item[0][0][0]][item[0][1][1] + i] = 0
        if item[1]:
            for i in range(item[1][1][1] - item[1][0][1]):
                door_image[item[1][0][0]][item[1][0][1] + i] = 0
        if item[2]:
            for i in range(item[2][0][0] - item[2][1][0]):
                door_image[item[2][1][0] + i][item[2][0][1]] = 0
        if item[3]:
            for i in range(item[3][1][0] - item[3][0][0]):
                door_image[item[3][0][0] + i][item[3][0][1]] = 0
    cv2.imshow("test", door_image)
    cv2.waitKey(0)


def link_two_points(all_points_list, add_points, door_image, width, hight):
    point_set = set()
    for one_point_list in all_points_list:
        for point in one_point_list:
            point_set.add(point)
    # print(len(point_set), point_set)
    for add_point in add_points:
        point_set.add(add_point)
    point_list = list(point_set)
    point_name_list = []
    for i in range(len(point_set)):
        point_name = ("Point-%04d" % (i + 1))
        point_name_list.append(point_name)
    point_dict = dict(zip(point_name_list, point_list))
    path_number = 1
    path_dict = {}
    for first in range(len(point_list)):
        tmp_link = [[], [], [], []]
        min_left_distance = width
        min_right_distance = width
        min_up_distance = hight
        min_down_distance = hight
        for second in range(len(point_list)):
            if first == second:
                continue
            case = True
            first_point = point_list[first]
            second_point = point_list[second]
            if first_point[0] == second_point[0]:
                if first_point[1] > second_point[1]:
                    for left_delta in range(1, (first_point[1] - second_point[1])):
                        if door_image[first_point[0]][first_point[1] - left_delta] == 0:
                            case = False
                            break
                    if case:
                        left_distance = first_point[1] - second_point[1]
                        if left_distance < min_left_distance:
                            min_left_distance = left_distance
                            tmp_link[0] = [point_name_list[first], point_name_list[second]]
                else:
                    for right_delta in range(1, (second_point[1] - first_point[1])):
                        if door_image[first_point[0]][first_point[1] + right_delta] == 0:
                            case = False
                            break
                    if case:
                        right_distance = second_point[1] - first_point[1]
                        if right_distance < min_right_distance:
                            min_right_distance = right_distance
                            tmp_link[1] = [point_name_list[first], point_name_list[second]]
            if first_point[1] == second_point[1]:
                if first_point[0] > second_point[0]:
                    for up_delta in range(1, (first_point[0] - second_point[0])):
                        if door_image[first_point[0] - up_delta][first_point[1]] == 0:
                            case = False
                            break
                    if case:
                        up_distance = first_point[0] - second_point[0]
                        if up_distance < min_up_distance:
                            min_up_distance = up_distance
                            tmp_link[2] = [point_name_list[first], point_name_list[second]]
                else:
                    for down_delta in range(1, (second_point[0] - first_point[0])):
                        if door_image[first_point[0] + down_delta][first_point[1]] == 0:
                            case = False
                            break
                    if case:
                        down_distance = second_point[0] - first_point[0]
                        if down_distance < min_down_distance:
                            min_down_distance = down_distance
                            tmp_link[3] = [point_name_list[first], point_name_list[second]]
        for item in tmp_link:
            tmp_dict = {}
            if item:
                tmp_dict["sourcePoint"] = item[0]
                tmp_dict["destinationPoint"] = item[1]
                path_name = ("connect%d" % path_number)
                path_dict[path_name] = tmp_dict
                path_number += 1
    # print("------------", path_number)
    with open("path.json", "w") as fp:
        json.dump(path_dict, fp)
    # print("---------finished---------")
    return point_dict


def write_point_json(point_dict, hight):
    for key in point_dict:
        value = point_dict[key]
        point_x = value[0]
        point_y = value[1]
        tmp_dict = {}
        tmp_dict["y"] = (-point_x) * 20
        # print(type(point_x))
        tmp_dict["x"] = point_y * 20
        tmp_dict["z"] = 9410
        point_dict[key] = tmp_dict
    with open("point.json", "w") as fp:
        json.dump(point_dict, fp)
    return point_dict


def compute_one_room(col_points_list, row_points_list, door_image, safe_interval):
    start_time = time.time()
    add_points = []
    two_points_list = []
    # 取起终点当作一条直线
    for point_list in col_points_list:
        tmp_col = []
        tmp_col.append(point_list[0])
        tmp_col.append(point_list[-1])
        two_points_list.append(tmp_col)
    for point_list in row_points_list:
        tmp_row = []
        tmp_row.append(point_list[0])
        tmp_row.append(point_list[-1])
        two_points_list.append(tmp_row)
    for i in range(len(two_points_list)):
        for j in range(i + 1, len(two_points_list)):
            # 两个点表示一条直线
            first_line = two_points_list[i]
            # print("-----------------first_line:", first_line)
            second_line = two_points_list[j]
            # print("----------second_line---:", second_line)
            for first_point in first_line:
                for second_point in second_line:
                    # print(first_point, second_point)
                    start_point_x = first_point[0]
                    start_point_y = first_point[1]
                    end_point_x = second_point[0]
                    end_point_y = second_point[1]
                    # 该距离主要防止路径相隔很近撞车
                    if abs(start_point_x - end_point_x) < safe_interval + 1 or \
                            abs(start_point_y - end_point_y) < safe_interval + 1:
                        continue
                    if start_point_x == end_point_x or start_point_y == end_point_y:
                        break
                    if (start_point_x < end_point_x and start_point_y < end_point_y) or (
                            start_point_x > end_point_x and start_point_y > end_point_y):
                        # 判断哪个点在左边
                        if start_point_x < end_point_x:
                            left_point = first_point
                            right_point = second_point
                        else:
                            left_point = second_point
                            right_point = first_point
                        # 第一个点为左下点， 第二个点为右上
                        need_judge_points = [(right_point[0], left_point[1]), (left_point[0], right_point[1])]
                        for judge_point in need_judge_points:
                            if judge_point == need_judge_points[0]:
                                # 左下角点
                                case = True
                                for row in range(left_point[0], right_point[0] + 1):
                                    for number_1 in range(1, safe_interval + 1):
                                        if door_image[row][left_point[1]] == 0 or \
                                                door_image[row][left_point[1] + number_1] == 0 or \
                                                door_image[row][left_point[1] - number_1] == 0:
                                            case = False
                                            break
                                    if case:
                                        continue
                                    else:
                                        break
                                if case:
                                    tmp_case = True
                                    for col in range(left_point[1], right_point[1] + 1):
                                        for number_2 in range(1, safe_interval + 1):
                                            if door_image[right_point[0]][col] == 0 or \
                                                    door_image[right_point[0] + number_2][col] == 0 or \
                                                    door_image[right_point[0] - number_2][col] == 0:
                                                tmp_case = False
                                                break
                                        if tmp_case:
                                            continue
                                        else:
                                            break
                                    if tmp_case:
                                        add_points.append((right_point[0], left_point[1]))
                            else:
                                # 右上角点
                                case = True
                                for col in range(left_point[1], right_point[1] + 1):
                                    for number_1 in range(1, safe_interval + 1):
                                        if door_image[left_point[0]][col] == 0 or\
                                                door_image[left_point[0] + number_1][col] == 0 or \
                                                door_image[left_point[0] - number_1][col] == 0:
                                            case = False
                                            break
                                    if case:
                                        continue
                                    else:
                                        break
                                if case:
                                    tmp_case = True
                                    for row in range(left_point[0], right_point[0] + 1):
                                        for number_2 in range(1, safe_interval + 1):
                                            if door_image[row][right_point[1]] == 0 or \
                                                    door_image[row][right_point[1] + number_2] == 0 or \
                                                    door_image[row][right_point[1] - number_2] == 0:
                                                tmp_case = False
                                                break
                                        if tmp_case:
                                            continue
                                        else:
                                            break
                                    if tmp_case:
                                        add_points.append((left_point[0], right_point[1]))
                    else:
                        # 判断哪个点在左边
                        if start_point_y < end_point_y:
                            left_point = first_point
                            right_point = second_point
                        else:
                            left_point = second_point
                            right_point = first_point
                        # 第一个点为右下点， 第二个点为左上
                        need_judge_points = [(left_point[0], right_point[1]), (right_point[0], left_point[1])]
                        for judge_point in need_judge_points:
                            if judge_point == need_judge_points[0]:
                                # 右下角点
                                case = True
                                for col in range(left_point[1], right_point[1] + 1):
                                    for number_1 in range(1, safe_interval + 1):
                                        if door_image[left_point[0]][col] == 0 or \
                                                door_image[left_point[0] + number_1][col] == 0 or \
                                                door_image[left_point[0] - number_1][col] == 0:
                                            case = False
                                            break
                                    if case:
                                        continue
                                    else:
                                        break
                                if case:
                                    tmp_case = True
                                    for row in range(right_point[0], left_point[0] + 1):
                                        for number_2 in range(1, safe_interval + 1):
                                            if door_image[row][right_point[1]] == 0 or \
                                                    door_image[row][right_point[1] + number_2] == 0 or \
                                                    door_image[row][right_point[1] - number_2] == 0:
                                                tmp_case = False
                                                break
                                        if tmp_case:
                                            continue
                                        else:
                                            break
                                    if tmp_case:
                                        add_points.append((left_point[0], right_point[1]))
                            else:
                                # 左上角点
                                case = True
                                for row in range(right_point[0], left_point[0] + 1):
                                    for number_1 in range(1, safe_interval + 1):
                                        if door_image[row][left_point[1]] == 0 or \
                                                door_image[row][left_point[1] + number_1] == 0 or \
                                                door_image[row][left_point[1] - number_1] == 0:
                                            case = False
                                            break
                                    if case:
                                        continue
                                    else:
                                        break
                                if case:
                                    tmp_case = True
                                    for col in range(left_point[1], right_point[1] + 1):
                                        for number_2 in range(1, safe_interval + 1):
                                            if door_image[right_point[0]][col] == 0 or \
                                                    door_image[right_point[0] + number_2][col] == 0 or \
                                                    door_image[right_point[0] - number_2][col] == 0:
                                                tmp_case = False
                                                break
                                        if tmp_case:
                                            continue
                                        else:
                                            break
                                    if tmp_case:
                                        add_points.append((right_point[0], left_point[1]))
    end_time = time.time()
    use_time = end_time - start_time
    print("----------", use_time)
    return add_points
