import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import getopt
"""
在该例中，考虑的是理想模型，即不考虑直角转弯的时间
也就是相当于把一条路径拉直进行计算, 第一个版本安全距离未考虑（后续更新）
在程序中只考虑 1.出发点第一个冲突区域的距离（初始段）
              2.冲突区域到冲突区域的距离（多个中间段）（当只有一个冲突区域时不存在）
              3.最后一个冲突区域到终点的距离（最后段）
"""


# 判断路径长度是否足够加速
def judge_length(distance, v_one, a_one, v_start, state):
    length_enough = True
    min_distance = (0.5 * v_one ** 2 - v_start ** 2) / a_one
    if distance < min_distance:
        if state == 1:
            print("小车1的路段不够以该初速度加速到最大速度")
        if state == 2:
            print("小车2的路段不够以该初速度加速到最大速度")
        length_enough = False

    return length_enough


# 判断两冲突段是否可进行任意时间规划
def judge_middle_length(conflict_distance, v_one, a_one, state):
    length_enough = True
    min_distance = (v_one ** 2) / a_one
    if conflict_distance < min_distance:
        if state == 1:
            print("中间段长度不够小车1规划过长时间")
        if state == 2:
            print("中间段长度不够小车2规划过长时间")
        length_enough = False

    return length_enough


# 计算理论第一段两车最快各自耗费时间以及最后速度
def compute_first_time(distance_1, distance_2, v, a, v_node_1, v_node_2):
    t1 = 0
    t2 = 0
    length_one_enough = judge_length(distance_1, v[0], a[0], v_node_1, state=1)
    length_two_enough = judge_length(distance_2, v[1], a[1], v_node_2, state=2)
    if length_one_enough:
        t1 += v[0] / a[0]
        t1 += (distance_1 - v[0] ** 2 / (2 * a[0])) / v[0]
        v_end_1 = v[0]
    else:
        t1 += (2 * distance_1 / a[0]) ** 0.5
        v_end_1 = a[0] * t1
    if length_two_enough:
        t2 += v[1] / a[1]
        t2 += (distance_2 - v[1] ** 2 / (2 * a[1])) / v[1]
        v_end_2 = v[1]
    else:
        t2 += (2 * distance_2 / a[1]) ** 0.5
        v_end_2 = a[1] * t2

    return t1, t2, v_end_1, v_end_2


# 不冲突起始段两车速度时间关系
def add_first_points_1(distance_1, distance_2, v, a, points_list_1, points_list_2, start_time_1, start_time_2, v_node_1, v_node_2):
    t1 = 0
    t2 = 0
    length_one_enough = judge_length(distance_1, v[0], a[0], v_node_1, state=1)
    length_two_enough = judge_length(distance_2, v[1], a[1], v_node_2, state=2)
    if length_one_enough:
        t1 += v[0] / a[0]
        points_list_1.append((start_time_1 + t1, v[0]))
        t1 += (distance_1 - v[0] ** 2 / (2 * a[0])) / v[0]
        points_list_1.append((start_time_1 + t1, v[0]))
    else:
        t1 += (2 * distance_1 / a[0]) ** 0.5
        v_end_1 = a[0] * t1
        points_list_1.append((start_time_1 + t1, v_end_1))
    if length_two_enough:
        t2 += v[1] / a[1]
        points_list_2.append((start_time_2 + t2, v[1]))
        t2 += (distance_2 - v[1] ** 2 / (2 * a[1])) / v[1]
        points_list_2.append((start_time_2 + t2, v[1]))
    else:
        t2 += (2 * distance_2 / a[1]) ** 0.5
        v_end_2 = a[1] * t2
        points_list_2.append((start_time_1 + t2, v_end_2))


# 计算初始段以规划速度最大可规划时间
def max_plan_time(end_time, start_time, distance, a, v_plan, v):
    need_time = end_time - start_time
    plan_distance = distance - 0.5 * v ** 2 / a
    max_time = v / a + plan_distance / v_plan
    if max_time >= need_time:
        return True
    else:
        return False


# 计算中间段以规划速度最大可规划时间
def max_plan_middle_time(end_time, start_time, distance, a, v_plan, v):
    need_time = end_time - start_time
    plan_distance = distance - v ** 2 / a
    max_time = (v - v_plan) / a * 2 + plan_distance / v_plan
    if max_time >= need_time:
        return True
    else:
        return False


# 按固定下阶段速度规划起始段的速度时间关系
def add_first_plan_points(distance_1, distance_2, v, v_plan, a, points_list_1, points_list_2, start_time_1, start_time_2, end_time, state):
    t1 = 0
    t2 = 0
    if state == 1:
        t1 += v[0] / a[0]
        points_list_1.append((start_time_1 + t1, v[0]))
        t1 += (distance_1 - v[0] ** 2 / (2 * a[0])) / v[0]
        points_list_1.append((start_time_1 + t1, v[0]))
        t2 += v_plan / a[1]
        points_list_2.append((start_time_2 + t2, v_plan))
        all_time = end_time - start_time_2 - v[1] / a[1]
        all_distance = distance_2 - 0.5 * v[1] ** 2 / a[1]
        t2 += (v[1] * all_time - all_distance) / (v[1] - v_plan)
        points_list_2.append((start_time_2 + t2, v_plan))
        t2 += (v[1] - v_plan) / a[1]
        points_list_2.append((start_time_2 + t2, v[1]))
        points_list_2.append((end_time, v[1]))
    if state == 2:
        t2 += v[1] / a[1]
        points_list_2.append((start_time_2 + t2, v[1]))
        t2 += (distance_2 - v[1] ** 2 / (2 * a[1])) / v[1]
        points_list_2.append((start_time_2 + t2, v[1]))
        t1 += v_plan / a[0]
        points_list_1.append((start_time_1 + t1, v_plan))
        all_time = end_time - start_time_1 - v[0] / a[0]
        all_distance = distance_1 - 0.5 * v[0] ** 2 / a[0]
        t1 += (v[0] * all_time - all_distance) / (v[0] - v_plan)
        points_list_1.append((start_time_1 + t1, v_plan))
        t1 += (v[0] - v_plan) / a[0]
        points_list_1.append((start_time_1 + t1, v[0]))
        points_list_1.append((end_time, v[0]))


# 计算冲突路段下阶段速度
def compute_off_vel(end_time, start_time, distance, v_end, a):
    need_time = end_time - start_time
    # 这里认为路径足够长
    a_time = v_end / a
    plan_time = need_time - a_time
    plan_distance = distance - (v_end ** 2 / (2 * a))
    v_off = v_end - (0.5 * plan_time * a) + 0.5 * (a * (plan_time ** 2 * a - 4 * v_end * plan_time + 4 * plan_distance)) ** 0.5

    return v_off


# 冲突起始段两车速度时间关系
def add_first_points_2(distance_1, distance_2, v, v_off, a, points_list_1, points_list_2, start_time_1, start_time_2, end_time, state):
    t1 = 0
    t2 = 0
    if state == 1:
        length_one_enough = judge_length(distance_1, v[0], a[0], 0, state=1)
        # car 1
        if length_one_enough:
            t1 += v[0] / a[0]
            points_list_1.append((start_time_1 + t1, v[0]))
            t1 += (distance_1 - v[0] ** 2 / (2 * a[0])) / v[0]
            points_list_1.append((start_time_1 + t1, v[0]))
        else:
            t1 += (2 * distance_1 / a[0]) ** 0.5
            v_end_1 = a[0] * t1
            points_list_1.append((start_time_1 + t1, v_end_1))
        # car 2
        t2 += v[1] / a[1]
        points_list_2.append((start_time_2 + t2, v[1]))
        t2 += (v[1] - v_off) / a[1]
        points_list_2.append((start_time_2 + t2, v_off))
        v_off_time = end_time - ((v[1] - v_off) / a[1])
        points_list_2.append((v_off_time, v_off))
        points_list_2.append((end_time, v[1]))

    if state == 2:
        length_two_enough = judge_length(distance_2, v[1], a[1], 0, state=2)
        if length_two_enough:
            t2 += v[1] / a[1]
            points_list_2.append((start_time_2 + t2, v[1]))
            t2 += (distance_2 - v[1] ** 2 / (2 * a[1])) / v[1]
            points_list_2.append((start_time_2 + t2, v[1]))
        else:
            t2 += (2 * distance_2 / a[1]) ** 0.5
            v_end_2 = a[1] * t2
            points_list_2.append((start_time_2 + t2, v_end_2))
        t1 += v[0] / a[0]
        points_list_1.append((start_time_1 + t1, v[0]))
        t1 += (v[0] - v_off) / a[0]
        points_list_1.append((start_time_1 + t1, v_off))
        v_off_time = end_time - ((v[0] - v_off) / a[0])
        points_list_1.append((v_off_time, v_off))
        points_list_1.append((end_time, v[0]))


# 通过时间关系判断两车是否在路径重叠部分有冲突 (state=1表示小车1先到，state=2表示小车2先到）
def judge_conflict(time_1, time_2, v_end, conflict_distance, v, a, state):
    conflict_state = True
    if state == 1:
        if v_end == v[0]:
            conflict_cost_time = conflict_distance / v_end
        else:
            x = (v[0] ** 2 - v_end ** 2) / (2 * a[0])
            conflict_cost_time = (conflict_distance - x) / v[0] + (v[0] - v_end) / a[0]
        end_time = time_1 + conflict_cost_time
        if time_1 + conflict_cost_time <= time_2:
            conflict_state = False
    if state == 2:
        if v_end == v[1]:
            conflict_cost_time = conflict_distance / v_end
        else:
            x = (v[1] ** 2 - v_end ** 2) / (2 * a[1])
            conflict_cost_time = (conflict_distance - x) / v[1] + (v[1] - v_end) / a[1]
        end_time = time_2 + conflict_cost_time
        if time_2 + conflict_cost_time <= time_1:
            conflict_state = False

    return conflict_state, end_time


# 计算中间部分两车以最大速度理论用时
def compute_middle_time(distance_1, distance_2, v):
    t1 = distance_1 / v[0]
    t2 = distance_2 / v[1]

    return t1, t2


# 中间段两车不冲突速度时间关系
def add_middle_points_1(distance_1, distance_2, v, points_list_1, points_list_2, start_time_1, start_time_2):
    t1 = 0
    t2 = 0
    t1 += distance_1 / v[0]
    points_list_1.append((start_time_1 + t1, v[0]))
    t2 += distance_2 / v[1]
    points_list_2.append((start_time_2 + t2, v[1]))


# 按照规划速度规划中间段速度时间关系
def add_middle_plan_points(distance_1, distance_2, v, v_plan, a, points_list_1, points_list_2, start_time_1, start_time_2, end_time, state):
    t1 = 0
    t2 = 0
    if state == 1:
        t1 += distance_1 / v[0]
        points_list_1.append((start_time_1 + t1, v[0]))
        plan_distance = distance_2 - (v[1] ** 2 - v_plan ** 2) / a[1]
        plan_time = end_time - start_time_2 - 2 * (v[1] - v_plan) / a[1]
        t2 += (plan_distance - v_plan * plan_time) / (v[1] - v_plan)
        points_list_2.append((start_time_2 + t2, v[1]))
        t2 += (v[1] - v_plan) / a[1]
        points_list_2.append((start_time_2 + t2, v_plan))
        points_list_2.append((end_time - (v[1] - v_plan) / a[1], v_plan))
        points_list_2.append((end_time, v[1]))
    if state == 2:
        t2 += distance_2 / v[1]
        points_list_2.append((start_time_2 + t2, v[1]))
        plan_distance = distance_1 - (v[0] ** 2 - v_plan ** 2) / a[0]
        plan_time = end_time - start_time_1 - 2 * (v[0] - v_plan) / a[0]
        t1 += (plan_distance - v_plan * plan_time) / (v[0] - v_plan)
        points_list_1.append((start_time_1 + t1, v[0]))
        t1 += (v[0] - v_plan) / a[0]
        points_list_1.append((start_time_1 + t1, v_plan))
        points_list_1.append((end_time - (v[0] - v_plan) / a[0], v_plan))
        points_list_1.append((end_time, v[0]))


# 中间段两车冲突速度时间关系
def add_middle_points_2(distance_1, distance_2, v, v_off, a, points_list_1, points_list_2, start_time_1, start_time_2, end_time, state=1):
    t1 = 0
    t2 = 0
    if state == 1:
        t1 += distance_1 / v[0]
        points_list_1.append((start_time_1 + t1, v[0]))
        t2 += (v[1] - v_off) / a[1]
        points_list_2.append((start_time_2 + t2, v_off))
        v_off_time = end_time - ((v[1] - v_off) / a[1])
        points_list_2.append((v_off_time, v_off))
        points_list_2.append((end_time, v[1]))
    if state == 2:
        t2 += distance_2 / v[1]
        points_list_2.append((start_time_2 + t2, v[1]))
        t1 += (v[0] - v_off) / a[0]
        points_list_1.append((start_time_1 + t1, v_off))
        v_off_time = end_time - ((v[0] - v_off) / a[0])
        points_list_1.append((v_off_time, v_off))
        points_list_1.append((end_time, v[0]))


# 计算最后段速度时间关系
def compute_end_path(x1, x2, points_list_1, points_list_2, v, a, conflict, start_time_1, start_time_2):
    t1 = 0
    t2 = 0
    distance_1 = sum(x1[conflict[-1][0][0]:])
    distance_2 = sum(x2[conflict[-1][1][0]:])
    t1 += (distance_1 - v[0] ** 2 / (2 * a[0])) / v[0]
    points_list_1.append((start_time_1 + t1, v[0]))
    t1 += v[0] / a[0]
    points_list_1.append((start_time_1 + t1, 0))
    t2 += (distance_2 - v[1] ** 2 / (2 * a[1])) / v[1]
    points_list_2.append((start_time_2 + t2, v[1]))
    t2 += v[1] / a[1]
    points_list_2.append((start_time_2 + t2, 0))


# 将两车的速度时间关系以图像表示出来
def show_picture(points_list_1, points_list_2):
    x, y, a, b = [], [], [], []
    for item in points_list_1:
        x.append(item[0])
        y.append(item[1])
    for item in points_list_2:
        a.append(item[0])
        b.append(item[1])
    plt.plot(x, y, label="car_1")
    plt.plot(a, b, label="car_2")
    plt.title("V-T")
    plt.xlabel("t")
    plt.ylabel("V")
    plt.legend()
    plt.show()


# 生成两车整体速度时间图像
def compute_profiles(x1, x2, v, v_plan, a, conflict, start_time):
    points_list_1 = []
    points_list_2 = []
    start_time_1 = start_time[0]
    points_list_1.append((start_time_1, 0))
    start_time_2 = start_time[1]
    points_list_2.append((start_time_2, 0))
    v_node_1, v_node_2 = 0, 0
    for i in range(len(conflict)):
        if i == 0:
            distance_1 = sum(x1[:conflict[i][0][0]])
            distance_2 = sum(x2[:conflict[i][1][0]])
            use_time_1, use_time_2, v_end_1, v_end_2 = compute_first_time(distance_1, distance_2, v, a, v_node_1, v_node_2)
            time_1 = use_time_1 + start_time_1
            time_2 = use_time_2 + start_time_2
            # 冲突路段长度
            conflict_distance = sum(x1[conflict[i][0][0]:conflict[i][0][1]])
            # 通过时间判断两车是否冲突
            if use_time_1 + start_time_1 <= use_time_2 + start_time_2:
                # state=1 表示小车1先到冲突区域 ，end_time表示先行车出冲突区域时间
                conflict_state, end_time = judge_conflict(time_1, time_2, v_end_1, conflict_distance, v, a, state=1)
                # 若不冲突
                if not conflict_state:
                    add_first_points_1(distance_1, distance_2, v, a, points_list_1, points_list_2, start_time_1, start_time_2, v_node_1, v_node_2)
                    start_time_1 += use_time_1
                    start_time_2 += use_time_2
                # 冲突
                else:
                    # 先判断路段长是否足够通过指定下阶段速度v = 1m/s进行速度规划
                    if max_plan_time(end_time, start_time_2, distance_2, a[1], v_plan[1], v[1]):
                        # print("is_True")
                        add_first_plan_points(distance_1, distance_2, v, v_plan[1], a, points_list_1, points_list_2, start_time_1, start_time_2, end_time, state=1)
                    else:
                        # 计算下阶段速度进行避让
                        v_off = compute_off_vel(end_time, start_time_2, distance_2, v_end_2, a[1])
                        add_first_points_2(distance_1, distance_2, v, v_off, a, points_list_1, points_list_2,
                                           start_time_1, start_time_2, end_time, state=1)
                    start_time_1 += use_time_1
                    start_time_2 = end_time
            else:
                # state=2 表示小车2先到冲突区域
                conflict_state, end_time = judge_conflict(time_1, time_2, v_end_2, conflict_distance, v, a, state=2)
                if not conflict_state:
                    add_first_points_1(distance_1, distance_2, v, a, points_list_1, points_list_2, start_time_1, start_time_2, v_node_1, v_node_2)
                    start_time_1 += use_time_1
                    start_time_2 += use_time_2
                else:
                    if max_plan_time(end_time, start_time_1, distance_1, a[0], v_plan[0], v[0]):
                        add_first_plan_points(distance_1, distance_2, v, v_plan[0], a, points_list_1, points_list_2,
                                              start_time_1, start_time_2, end_time, state=2)
                    else:
                        v_off = compute_off_vel(end_time, start_time_1, distance_1, v_end_1, a[0])
                        add_first_points_2(distance_1, distance_2, v, v_off, a, points_list_1, points_list_2,
                                           start_time_1, start_time_2, end_time, state=2)
                    start_time_1 = end_time
                    start_time_2 += use_time_2
            # print(points_list_1, "\n", points_list_2)
        else:
            distance_1 = sum(x1[conflict[i - 1][0][0]:conflict[i][0][0]])
            distance_2 = sum(x2[conflict[i - 1][1][0]:conflict[i][1][0]])
            use_time_1, use_time_2 = compute_middle_time(distance_1, distance_2, v)
            time_1 = use_time_1 + start_time_1
            time_2 = use_time_2 + start_time_2
            conflict_distance = sum(x1[conflict[i][0][0]:conflict[i][0][1]])
            # 先到先得
            if time_1 <= time_2:
                conflict_state, end_time = judge_conflict(time_1, time_2, v[0], conflict_distance, v, a, state=1)
                if not conflict_state:
                    add_middle_points_1(distance_1, distance_2, v, points_list_1, points_list_2, start_time_1, start_time_2)
                    start_time_1 += use_time_1
                    start_time_2 += use_time_2
                else:
                    if max_plan_middle_time(end_time, start_time_2, distance_2, a[1], v_plan[1], v[1]):
                        add_middle_plan_points(distance_1, distance_2, v, v_plan[1], a, points_list_1, points_list_2,
                                               start_time_1, start_time_2, end_time, state=1)
                    else:
                        v_off = compute_off_vel(end_time, start_time_2, distance_2, v[1], a[1])
                        add_middle_points_2(distance_1, distance_2, v, v_off, a, points_list_1, points_list_2, start_time_1, start_time_2, end_time, state=1)
                    start_time_1 += use_time_1
                    start_time_2 = end_time
            else:
                conflict_state, end_time = judge_conflict(time_1, time_2, v[1], conflict_distance, v, a, state=2)
                if not conflict_state:
                    add_middle_points_1(distance_1, distance_2, v, points_list_1, points_list_2, start_time_1, start_time_2)
                    start_time_1 += use_time_1
                    start_time_2 += use_time_2
                else:
                    if max_plan_middle_time(end_time, start_time_1, distance_1, a[0], v_plan[0], v[0]):
                        add_middle_plan_points(distance_1, distance_2, v, v_plan[0], a, points_list_1, points_list_2,
                                               start_time_1, start_time_2, end_time, state=2)
                    else:
                        v_off = compute_off_vel(end_time, start_time_1, distance_1, v[0], a[0])
                        add_middle_points_2(distance_1, distance_2, v, v_off, a, points_list_1, points_list_2, start_time_1, start_time_2, end_time, state=2)
                    start_time_1 = end_time
                    start_time_2 += use_time_2
    compute_end_path(x1, x2, points_list_1, points_list_2, v, a, conflict, start_time_1, start_time_2)
    print(points_list_1)
    print(points_list_2)
    show_picture(points_list_1, points_list_2)


if __name__ == '__main__':
    argv = sys.argv
    short_args = "h"
    """
    x1，x2表示小车 1、2 的路径中点到点的距离列表[x1,x2,.....]
    v表示小车 1、2 的速度[v1, v2]
    a表示小车 1、2 的加速度[a1, a2]
    conflict表示小车 1、2 冲突的路段部分 例小车 x1[2]与x2[4]冲突，则表示为 conflict=[2, 4]
    start_time代表两车的发车时间 [time1, time2]
    """
    long_args = ["x1=", "x2=", "v=", "a=", "conflict=", "start_time=", "help"]
    opts, args = getopt.getopt(sys.argv[1:], short_args, long_args)
    opts = dict(opts)
    # path代表点
    path_1 = [1, 2, 3, 4, 5, 6]
    # x代表两点之间距离
    x1 = [6, 2, 2, 2, 2, 6, 2]
    path_2 = [1, 2, 3, 4, 5]
    x2 = [3, 2, 2, 2, 2, 6, 2]
    v = [2, 2]
    a = [2, 2]
    start_time = [0, 0]
    v_plan = [v[0] / 2, v[1] / 2]
    # conflict代表冲突分别再两条path上的点位
    conflict = [[[2, 4], [2, 4]], [[5, 6], [5, 6]]]
    if "-h" in opts or "--help" in opts:
        print('--x1，x2表示小车 1、2 的路径中点到点的距离列表[x1,x2,.....]'
              '--v表示小车 1、2 的速度[v1, v2]'
              '--a表示小车 1、2 的加速度[a1, a2]'
              '--conflict表示小车 1、2 冲突的路段部分 例小车 x1[2：4]与x2[1：3]冲突，则表示为 conflict=[ [ [2, 4], [1, 3] ] ]'
              '--start_time代表两车的发车时间[time1, time2]'
              )
        sys.exit()
    if "--x1" in opts:
        x1 = opts["--x1"]
    if "--x2" in opts:
        x2 = opts["--x2"]
    if "--v" in opts:
        v = opts["--v"]
    if "--a" in opts:
        a = opts["--a"]
    if "--conflict" in opts:
        conflict = opts["--conflict"]
    if "--start_time" in opts:
        start_time = opts["--start_time"]
    compute_profiles(x1, x2, v, v_plan, a, conflict, start_time)
