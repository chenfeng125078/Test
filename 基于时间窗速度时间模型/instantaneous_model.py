import sys
import numpy as np
import time
import getopt
import json
import matplotlib.pyplot as plt
"""
该模型考虑单段路段的平均速度
并考虑安全距离以及直角转弯时间
该模型依然是分三段考虑：初始段，中间段，最末段
后续优化可考虑相向冲突以及同向冲突
重叠区域不一定冲突，根据时间关系需判断重叠是否冲突
"""


# 读取路网信息
def read_json(file):
    with open(file, "r") as fp:
        data = json.load(fp)

    return data


# 计算路径重叠分别在两车路径的位置
def compute_conflict_1(path1, path2):
    relation = []
    conflict_1 = []
    conflict_2 = []
    conflict = []
    for i in range(len(path1)):
        for j in range(len(path2)):
            if path1[i] == path2[j]:
                relation.append((i, j))
                conflict_1.append(i)
                conflict_2.append(j)
    count = 0
    cut_area = [count]
    while count < len(conflict_1):
        num = conflict_1[count]
        if num + 1 in conflict_1:
            count += 1
        else:
            cut_area.append(count)
            count += 1
            if count < len(conflict_1):
                cut_area.append(count)
    for i in range(0, len(cut_area), 2):
        start_point_1 = conflict_1[cut_area[i]]
        end_point_1 = conflict_1[cut_area[i + 1]]
        # print(start_point_1, end_point_1)
        start_point_2 = 0
        end_point_2 = 0
        for j in range(len(path2)):
            if path2[j] == path1[start_point_1]:
                start_point_2 = j
            if path2[j] == path1[end_point_1]:
                end_point_2 = j
        if start_point_2 > end_point_2:
            start_point_2, end_point_2 = end_point_2, start_point_2
        conflict.append([[start_point_1, end_point_1], [start_point_2, end_point_2]])
    print(conflict)

    return conflict


# 计算两点之间的欧式距离
def compute_distance(point_1, point_2):
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    return np.sqrt(np.sum((point_1 - point_2) ** 2))


# 计算路段长度
def compute_segment_length(points_list):
    length_list = []
    for i in range(len(points_list) - 1):
        point_1 = points_list[i]
        point_2 = points_list[i + 1]
        distance = compute_distance(point_1, point_2)
        length_list.append(distance)

    return length_list


# 计算分段起点到冲突的时间
def compute_use(segment_length, start, end, points_list, velocity, turn_use):
    total_distance = np.sum(segment_length[start:end])
    use_time = (total_distance - safe_distance) / velocity
    turn_count = 0
    for i in range(start + 1, end):
        x1 = np.array(points_list[i]) - np.array(points_list[i - 1])
        x2 = np.array(points_list[i + 1]) - np.array(points_list[i])
        if np.dot(x1, x2) == 0:
            turn_count += 1
    use_time += turn_count * turn_use
    return use_time


# 计算先到小车所占用冲突区域的时间(包括冲突区域直角转弯时间)
def compute_conflict_time(segment_length, start_point, end_point, points_list, velocity, turn_use, safe_distance):
    conflict_distance = np.sum(segment_length[start_point:end_point])
    use_time_more = (conflict_distance + 2 * safe_distance) / velocity
    use_time_less = (conflict_distance + safe_distance) / velocity
    turn_count = 0
    for i in range(start_point, end_point + 1):
        x1 = np.array(points_list[i - 1]) - np.array(points_list[i])
        x2 = np.array(points_list[i]) - np.array(points_list[i + 1])
        if np.dot(x1, x2) == 0:
            turn_count += 1
    use_time_more += turn_count * turn_use
    use_time_less += turn_count * turn_use

    return use_time_less, use_time_more


# 判断第一个重叠区域是否冲突
def whether_conflict(first_time, second_time, less_time, more_time, length, velocity):
    if (first_time + less_time) <= second_time and (first_time + more_time) <= (second_time + length / velocity):
        return False
    else:
        return True


# 通过向量点乘判断是否为转弯点
def whether_turn(points_list, current_point):
    x1 = np.array(points_list[current_point + 1]) - np.array(points_list[current_point])
    x2 = np.array(points_list[current_point + 2]) - np.array(points_list[current_point + 1])
    if np.dot(x1, x2) == 0:
        return True
    else:
        return False


# 最终段速度时间规划
def end_velocity_plan(segment_length, start_point, end_point, points_list, velocity, turn_use, current_time,
                      velocity_list):
    for i in range(start_point, end_point - 1):
        current_distance = segment_length[i]
        use_time = current_distance / velocity
        current_time += use_time
        velocity_list.append((current_time, velocity))
        if i == end_point - 2:
            return current_time
        else:
            if whether_turn(points_list, i):
                velocity_list.append((current_time, 0))
                current_time += turn_use
                velocity_list.append((current_time, 0))
                velocity_list.append((current_time, velocity))
    return current_time


# 先行车的速度规划(无冲突速度规划)
def no_conflict_velocity_plan(segment_length, start_point, end_point, points_list, velocity, turn_use, current_time,
                              velocity_list):
    # print("no conflict velocity planning")
    for i in range(start_point, end_point):
        current_distance = segment_length[i]
        use_time = current_distance / velocity
        current_time += use_time
        velocity_list.append((current_time, velocity))
        if i == end_point - 1:
            if whether_turn(points_list, i):
                velocity_list.append((current_time, 0))
                current_time += turn_use
                velocity_list.append((current_time, 0))
        else:
            if whether_turn(points_list, i):
                velocity_list.append((current_time, 0))
                current_time += turn_use
                velocity_list.append((current_time, 0))
                velocity_list.append((current_time, velocity))
    return current_time


# 后行车 冲突情况1 速度规划
def second_velocity_plan_1(segment_length, start_point, end_point, points_list, turn_use, current_time,
                           velocity_list, end_time, distance, velocity_self, velocity_max):
    # 先要计算平均速度
    print("second velocity 1 planning")
    total_distance = np.sum(segment_length[start_point:end_point])
    plan_distance = total_distance - distance
    count_turn = 0
    for i in range(start_point, end_point - 1):
        if whether_turn(points_list, i):
            count_turn += 1
    plan_time = end_time - current_time - turn_use * count_turn
    plan_velocity = plan_distance / plan_time
    velocity_list.append((current_time, plan_velocity))
    for i in range(start_point, end_point - 1):
        current_distance = segment_length[i]
        use_time = current_distance / plan_velocity
        current_time += use_time
        velocity_list.append((current_time, plan_velocity))
        if whether_turn(points_list, i):
            velocity_list.append((current_time, 0))
            current_time += turn_use
            velocity_list.append((current_time, 0))
            velocity_list.append((current_time, plan_velocity))
    end_distance = segment_length[end_point - 1] - distance
    time_1 = end_distance / plan_velocity
    current_time += time_1
    velocity_list.append((end_time, plan_velocity))
    current_time = end_time
    if velocity_self >= velocity_max:
        t2 = distance / velocity_max
        current_time += t2
        velocity_list.append((end_time, velocity_max))
        velocity_list.append((current_time, velocity_max))
    else:
        t2 = distance / velocity_self
        current_time += t2
        velocity_list.append((end_time, velocity_self))
        velocity_list.append((current_time, velocity_self))
    if whether_turn(points_list, end_point - 1):
        velocity_list.append((current_time, 0))
        current_time += turn_use
        velocity_list.append((current_time, 0))
    return current_time


# 后行车 冲突情况2 速度规划
def second_velocity_plan_2(segment_length, start_point, end_point, points_list, turn_use, current_time,
                           velocity_list, end_time):
    print("second velocity 2 planning")
    plan_distance = np.sum(segment_length[start_point:end_point])
    count_turn = 0
    for i in range(start_point, end_point - 1):
        if whether_turn(points_list, i):
            count_turn += 1
    plan_time = end_time - current_time - turn_use * count_turn
    plan_velocity = plan_distance / plan_time
    velocity_list.append((current_time, plan_velocity))
    for i in range(start_point, end_point - 1):
        current_distance = segment_length[i]
        use_time = current_distance / plan_velocity
        current_time += use_time
        velocity_list.append((current_time, plan_velocity))
        if whether_turn(points_list, i):
            velocity_list.append((current_time, 0))
            current_time += turn_use
            velocity_list.append((current_time, 0))
            velocity_list.append((current_time, plan_velocity))
    velocity_list.append((end_time, plan_velocity))
    current_time = end_time
    if whether_turn(points_list, end_point - 1):
        velocity_list.append((current_time, 0))
        current_time += turn_use
        velocity_list.append((current_time, 0))
    return current_time


# 同向冲突解决方式(根据速度判断是否有追尾)处理 最优解为：先行车驶出冲突区域的一瞬间，后面的车与它相隔安全距离
def same_direction_way():
    print("同向冲突")


# 相向冲突解决方式 最优解为：1.先行车驶出（冲突区域 + 安全距离）的一瞬间 另外小车刚进冲突点
# 2.先行车刚出冲突区域 小车 在 冲突区域 + 安全距离 位置 取两者 速度小的一个 作为 平均速度
def opposite_direction_way():
    print("相向冲突")


# 点冲突解决方式 最优解：只需考虑安全距离
def point_conflict_way():
    print("十字路口式点冲突")


# 生成速度时间图像
def show_velocity_segment(velocity_list_1, velocity_list_2):
    t1, v1, t2, v2 = [], [], [], []
    for item in velocity_list_1:
        t1.append(item[0])
        v1.append(item[1])
    for item in velocity_list_2:
        t2.append(item[0])
        v2.append(item[1])
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t1, v1, label="car_1")
    ax[0].legend()
    ax[1].plot(t2, v2, color='red', label="car_2")
    ax[1].legend()
    plt.show()


# 生成速度剖面
def velocity_profile(path_1, path_2, points_info, conflict, v, start_time, safe_distance, turn_time):
    velocity_1 = v[0]
    velocity_2 = v[1]
    start_time_1 = start_time[0]
    start_time_2 = start_time[1]
    # 点列表
    points_list_1 = []
    points_list_2 = []
    # 速度变化列表
    velocity_list_1 = []
    velocity_list_2 = []
    # 速度时间图像起点
    velocity_list_1.append((start_time_1, 0))
    velocity_list_2.append((start_time_2, 0))
    for item in path_1:
        point_name = "point_%d" % item
        point = points_info[point_name]
        points_list_1.append(point)
    # 计算 1车的路段长度
    segment_length_1 = compute_segment_length(points_list_1)
    for item in path_2:
        point_name = "point_%d" % item
        point = points_info[point_name]
        points_list_2.append(point)
    # 计算 2车的路段长度
    segment_length_2 = compute_segment_length(points_list_2)
    # 冲突路段
    for i in range(len(conflict)):
        # 两车起点到第一段冲突段的时间
        if i == 0:
            print("----起点到第1段重叠段规划----")
            use_time_1 = compute_use(segment_length_1, 0, conflict[i][0][0], points_list_1, velocity_1, turn_time[0])
            use_time_2 = compute_use(segment_length_2, 0, conflict[i][1][0], points_list_2, velocity_2, turn_time[1])
            time_1 = start_time_1 + use_time_1
            time_2 = start_time_2 + use_time_2
            # 以下冲突默认为相向冲突(最好通过路径上重叠点进行判断分类处理)
            # 1车先行
            if start_time_1 + use_time_1 <= start_time_2 + use_time_2:
                print("1车先行")
                velocity_list_1.append((start_time_1, velocity_1))
                use_time_less, use_time_more = compute_conflict_time(segment_length_1, conflict[i][0][0],
                                                                     conflict[i][0][1], points_list_1, velocity_1,
                                                                     turn_time[0], safe_distance)
                # 判断是否冲突
                if whether_conflict(time_1, time_2, use_time_less, use_time_more, safe_distance, velocity_2):
                    start_time_1 = no_conflict_velocity_plan(segment_length_1, 0, conflict[i][0][0], points_list_1,
                                                             velocity_1, turn_time[0], start_time_1, velocity_list_1)
                    # print(start_time_1)
                    if (time_1 + use_time_less) > time_2:
                        # print("-----第一种情况规划-----")
                        start_time_2 = second_velocity_plan_1(segment_length_2, 0, conflict[i][1][0], points_list_2,
                                                              turn_time[1], start_time_2, velocity_list_2,
                                                              time_1 + use_time_less, safe_distance, velocity_2,
                                                              velocity_1)
                        # print(velocity_list_2)
                    else:
                        # print("-----第二种情况规划-----")
                        start_time_2 = second_velocity_plan_2(segment_length_2, 0, conflict[i][1][0], points_list_2,
                                                              turn_time[1], start_time_2, velocity_list_2,
                                                              time_1 + use_time_more)
                        # print(start_time_2)
                else:
                    print("-----%d path is not conflict---------" % i)
                    velocity_list_2.append((start_time_2, velocity_2))
                    start_time_1 = no_conflict_velocity_plan(segment_length_1, 0, conflict[i][0][0], points_list_1,
                                                             velocity_1, turn_time[0], start_time_1, velocity_list_1)
                    start_time_2 = no_conflict_velocity_plan(segment_length_2, 0, conflict[i][1][0], points_list_2,
                                                             velocity_2, turn_time[1], start_time_2, velocity_list_2)
            # 2车先行
            else:
                print("2车先行")
                velocity_list_2.append((start_time_2, velocity_2))
                use_time_less, use_time_more = compute_conflict_time(segment_length_2, conflict[i][1][0],
                                                                     conflict[i][1][1], points_list_2, velocity_2,
                                                                     turn_time[1], safe_distance)
                if whether_conflict(time_2, time_1, use_time_less, use_time_more, safe_distance, velocity_1):
                    start_time_2 = no_conflict_velocity_plan(segment_length_2, 0, conflict[i][1][0], points_list_2,
                                                             velocity_2, turn_time[1], start_time_2, velocity_list_2)
                    if (time_2 + use_time_less) > time_1:
                        print("-----第一种情况规划-----")
                        start_time_1 = second_velocity_plan_1(segment_length_1, 0, conflict[i][0][0], points_list_1,
                                                              turn_time[0], start_time_1, velocity_list_1,
                                                              time_2 + use_time_less, safe_distance, velocity_1,
                                                              velocity_2)
                        # print(velocity_list_2)
                    else:
                        # print("-----第二种情况规划-----")
                        start_time_1 = second_velocity_plan_2(segment_length_1, 0, conflict[i][0][0], points_list_1,
                                                              turn_time[0], start_time_1, velocity_list_1,
                                                              time_2 + use_time_more)
                        # print(start_time_2)
                else:
                    # print("-----one is not conflict---------")
                    velocity_list_1.append((start_time_1, velocity_1))
                    start_time_1 = no_conflict_velocity_plan(segment_length_1, 0, conflict[i][0][0], points_list_1,
                                                             velocity_1, turn_time[0], start_time_1, velocity_list_1)
                    start_time_2 = no_conflict_velocity_plan(segment_length_2, 0, conflict[i][1][0], points_list_2,
                                                             velocity_2, turn_time[1], start_time_2, velocity_list_2)
        # 有两段甚至更多重叠路径
        else:
            print("----第%d段重叠段规划----" % (i + 1))
            # print(start_time_1, start_time_2)
            use_time_1 = compute_use(segment_length_1, conflict[i-1][0][0], conflict[i][0][0], points_list_1,
                                     velocity_1, turn_time[0])
            use_time_2 = compute_use(segment_length_2, conflict[i-1][1][0], conflict[i][1][0], points_list_2,
                                     velocity_2, turn_time[1])
            time_1 = start_time_1 + use_time_1
            time_2 = start_time_2 + use_time_2
            # print(use_time_1, use_time_2)
            if start_time_1 + use_time_1 <= start_time_2 + use_time_2:
                print("1车先行")
                velocity_list_1.append((start_time_1, velocity_1))
                use_time_less, use_time_more = compute_conflict_time(segment_length_1, conflict[i][0][0],
                                                                     conflict[i][0][1], points_list_1, velocity_1,
                                                                     turn_time[0], safe_distance)
                # 判断是否冲突
                if whether_conflict(time_1, time_2, use_time_less, use_time_more, safe_distance, velocity_2):
                    start_time_1 = no_conflict_velocity_plan(segment_length_1, conflict[i-1][0][0], conflict[i][0][0],
                                                             points_list_1, velocity_1, turn_time[0], start_time_1,
                                                             velocity_list_1)
                    # print(start_time_1)
                    if (time_1 + use_time_less) > time_2:
                        # print("-----第一种情况规划-----")
                        start_time_2 = second_velocity_plan_1(segment_length_2, conflict[i-1][1][0], conflict[i][1][0],
                                                              points_list_2, turn_time[1], start_time_2,
                                                              velocity_list_2, time_1 + use_time_less, safe_distance,
                                                              velocity_2, velocity_1)
                        # print(velocity_list_2)
                    else:
                        # print("-----第二种情况规划-----")
                        start_time_2 = second_velocity_plan_2(segment_length_2, conflict[i-1][1][0], conflict[i][1][0],
                                                              points_list_2, turn_time[1], start_time_2,
                                                              velocity_list_2, time_1 + use_time_more)
                        # print(start_time_2)
                else:
                    # print("-----one is not conflict---------")
                    velocity_list_2.append((start_time_2, velocity_2))
                    start_time_1 = no_conflict_velocity_plan(segment_length_1, conflict[i-1][0][0], conflict[i][0][0],
                                                             points_list_1, velocity_1, turn_time[0], start_time_1,
                                                             velocity_list_1)
                    start_time_2 = no_conflict_velocity_plan(segment_length_2, conflict[i-1][1][0], conflict[i][1][0],
                                                             points_list_2, velocity_2, turn_time[1], start_time_2,
                                                             velocity_list_2)
            # 2车先行
            else:
                print("2车先行")
                velocity_list_2.append((start_time_2, velocity_2))
                use_time_less, use_time_more = compute_conflict_time(segment_length_2, conflict[i][1][0],
                                                                     conflict[i][1][1], points_list_2, velocity_2,
                                                                     turn_time[1], safe_distance)
                if whether_conflict(time_2, time_1, use_time_less, use_time_more, safe_distance, velocity_1):
                    start_time_2 = no_conflict_velocity_plan(segment_length_2, conflict[i-1][1][0], conflict[i][1][0],
                                                             points_list_2, velocity_2, turn_time[1], start_time_2,
                                                             velocity_list_2)
                    if (time_2 + use_time_less) > time_1:
                        print("-----第一种情况规划-----")
                        start_time_1 = second_velocity_plan_1(segment_length_1, conflict[i-1][0][0], conflict[i][0][0],
                                                              points_list_1, turn_time[0], start_time_1,
                                                              velocity_list_1, time_2 + use_time_less, safe_distance,
                                                              velocity_1, velocity_2)
                        # print(velocity_list_2)
                    else:
                        print("-----第二种情况规划-----")
                        start_time_1 = second_velocity_plan_2(segment_length_1, conflict[i-1][0][0], conflict[i][0][0],
                                                              points_list_1, turn_time[0], start_time_1,
                                                              velocity_list_1, time_2 + use_time_more)
                        # print(start_time_2)
                else:
                    print("-----this path is not conflict---------")
                    velocity_list_1.append((start_time_1, velocity_1))
                    start_time_1 = no_conflict_velocity_plan(segment_length_1, conflict[i-1][0][0], conflict[i][0][0],
                                                             points_list_1, velocity_1, turn_time[0], start_time_1,
                                                             velocity_list_1)
                    start_time_2 = no_conflict_velocity_plan(segment_length_2, conflict[i-1][1][0], conflict[i][1][0],
                                                             points_list_2, velocity_2, turn_time[1], start_time_2,
                                                             velocity_list_2)
    print(velocity_list_1)
    print("----最后段规划----")
    velocity_list_1.append((start_time_1, velocity_1))
    velocity_list_2.append((start_time_2, velocity_2))
    start_time_1 = end_velocity_plan(segment_length_1, conflict[-1][0][0], len(points_list_1), points_list_1,
                                     velocity_1, turn_time[0], start_time_1, velocity_list_1)
    start_time_2 = end_velocity_plan(segment_length_2, conflict[-1][1][0], len(points_list_2), points_list_2,
                                     velocity_2, turn_time[1], start_time_2, velocity_list_2)
    velocity_list_1.append((start_time_1, 0))
    velocity_list_2.append((start_time_2, 0))
    # print(velocity_list_1)
    # print(velocity_list_2)
    show_velocity_segment(velocity_list_1, velocity_list_2)


if __name__ == '__main__':
    argv = sys.argv
    short_args = 'h'
    long_args = ["file_name=", "path_1=", "path_2=", "v=", "start_time=", "safe_distance=", "turn_time=", "help"]
    opts, args = getopt.getopt(sys.argv[1:], short_args, long_args)
    opts = dict(opts)
    file_name = "./points.json"
    # 路网地图中点坐标
    points_info = read_json(file_name)
    # 两车路径经过的点
    path_1 = [1, 2, 3, 6, 5, 8, 9, 11]
    path_2 = [4, 3, 6, 7, 9, 8, 10]
    # path_1 = [1, 2, 5, 6]
    # path_2 = [4, 5, 2, 3]
    # 冲突段分别在两车的位置
    conflict = compute_conflict_1(path_1, path_2)
    # 两车最大速度
    v = [2, 2]
    # 两车出发时间
    start_time = [0, 0]
    # 安全距离
    safe_distance = 0.5
    # 转弯时间
    turn_time = [1, 1]
    if "-h" in opts or "--help" in opts:
        print("需要提示 ---------- need help")
    if "--file_name" in opts:
        file_name = opts["--file_name"]
        points_info = read_json(file_name)
    if "--path_1" in opts:
        path_1 = opts["--path_1"]
    if "--path_2" in opts:
        path_2 = opts["--path_2"]
    if "--v" in opts:
        v = opts["--v"]
    if "--start_time" in opts:
        start_time = opts["--start_time"]
    if "--safe_distance" in opts:
        safe_distance = opts["--safe_distance"]
    if "--turn_time" in opts:
        turn_time = opts["--turn_time"]
    velocity_profile(path_1, path_2, points_info, conflict, v, start_time, safe_distance, turn_time)
