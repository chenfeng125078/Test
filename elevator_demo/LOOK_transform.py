import numpy as np
import sys


# 下面是先到先得电梯算法,只能接收一个任务（队列为一）
def FCFS(orderList, order_time, startFloor=1, no_conflict_time=1, conflict_time=5, one_floor_time=1):
    total_time = 0
    total_var = 0
    finished_order_history = []
    start_floor = startFloor
    for i in range(len(orderList)):
        current_order = orderList[i]
        order_floor = current_order[0]
        end_floor = current_order[1]
        direction = current_order[2]
        floor_count = np.abs((order_floor - start_floor)) + np.abs((end_floor - order_floor))
        use_time = floor_count * one_floor_time + 2 * no_conflict_time
        total_time += use_time
        finished_order_history.append(total_time)
        # 此时电梯处于目标楼层
        start_floor = end_floor
    for i in range(len(orderList)):
        total_var += ((finished_order_history[i] - order_time[i]) / len(orderList)) ** 2
    total_var = np.sqrt(total_var) / len(orderList)
    # print(total_time)
    return total_time, total_var


# 下面是最短路径算法，接收一个任务（队列为一），有可能出现“饿死”现象
def SSTF(orderList, order_time, startFloor=1, no_conflict_time=1, conflict_time=5, one_floor_time=1):
    # 根据当前时间寻找最近订单
    for i in range(len(orderList)):
        orderList[i].append(order_time[i])
    current_time = 0
    start_floor = startFloor
    finished_order_history = []
    while orderList:
        arrived_order = []
        for i in range(len(orderList)):
            if orderList[i][2] <= current_time:
                arrived_order.append((orderList[i]))
        distance_list = []
        for i in range(len(arrived_order)):
            distance = np.abs(arrived_order[i][0] - start_floor)
            distance_list.append(distance)
        number = np.argsort(distance_list)[0]
        current_order = arrived_order[number]
        floor_count = np.abs(current_order[0] - start_floor) + np.abs(current_order[1] - current_order[0])
        use_time = floor_count * one_floor_time + 2 * no_conflict_time
        current_time += use_time
        finished_order_history.append((current_order[3], current_time))
        start_floor = current_order[1]
        orderList.remove(current_order)
    # print(finished_order_history, current_time)
    total_var = 0
    for item in finished_order_history:
        total_var += ((item[1] - item[0]) / len(finished_order_history)) ** 2
    # print(current_time)
    return current_time, total_var


# 一笼两机look变种算法：只有当待进入的机器人目标楼层低于当前执行机器人目标楼层才会让其进入
def LOOK_transorform(orderList, startFloor=1, no_conflict_time=60, conflict_time=100, one_floor_time=5):
    recieved_order = []
    execute_order = []
    current_floor = startFloor
    current_time = 0
    # 电梯状态
    elevator_state = False
    # 电梯运行方向
    elevator_direction = True
    while True:
        arrived_order = []
        for item in orderList:
            if current_time >= item[3]:
                arrived_order.append(item)
        for item in arrived_order:
            orderList.remove(item)
        # print("current_time is %d" % current_time)
        # 根据当前时间判断是否有订单 这里用input来实时输入订单
        # arrived_order = input("输入出发楼层以及目标楼层：以,隔开 format：2,5")
        new_order = []
        try:
            if arrived_order == "":
                pass
            else:
                for item in arrived_order:
                    new_order.append(item)
        except:
            # print("输入格式有误")
            sys.exit()
        # 电梯从静止到开始向第一个新订单运动
        if new_order:
            if not elevator_state:
                # print("----静止到运动----------------")
                for item in new_order:
                    recieved_order.append(item)
                elevator_state = True
                if new_order[0][0] > current_floor:
                    # 电梯运行方向
                    elevator_direction = True
                elif new_order[0][0] == current_floor and new_order[0][1] > new_order[0][0]:
                    elevator_direction = True
                elif new_order[0][0] == current_floor and new_order[0][1] < new_order[0][0]:
                    elevator_direction = False
                else:
                    elevator_direction = False
            # 电梯在运行过程中来了新订单
            else:
                for item in new_order:
                    # print("-------运动来新订单-------")
                    recieved_order.append(item)
        # 如果执行中的订单到达目标楼层
        if execute_order:
            remove_list = []
            for item in execute_order:
                if item[1] == current_floor:
                    # print("----------下机器人---------------")
                    remove_list.append(item)
                    current_time += no_conflict_time
            for item in remove_list[:]:
                execute_order.remove(item)
        # 电梯内机器人数量小于2时
        if len(execute_order) < 2:
            if len(execute_order) == 0:
                for item in recieved_order:
                    if item[0] == current_floor and item[2] == elevator_direction:
                        # 实际增加时间不是 2 * no_conflict_time
                        # print("------上机器人----0--------")
                        execute_order.append(item)
                        current_time += no_conflict_time
                    if len(execute_order) == 2:
                        break
                for item in execute_order:
                    recieved_order.remove(item)
            else:
                for item in recieved_order:
                    if item[0] == current_floor and item[2] == elevator_direction and item[1] <= execute_order[0][1]:
                        # print("------上机器人----1-------")
                        execute_order.append(item)
                        recieved_order.remove(item)
                        current_time += no_conflict_time
                        break
        # 电梯内已无执行订单 并且当前方向没有订单 则转向 转向时要判断当前楼层是否有反向订单，有，需要接单
        if not execute_order:
            # 上行时 若高层无订单
            if elevator_direction:
                change_state = True
                for item in recieved_order:
                    if item[0] > current_floor:
                        change_state = False
                if change_state:
                    elevator_direction = False
                    for item in recieved_order:
                        if item[0] == current_floor and item[2] == elevator_direction:
                            # print("----------0--------")
                            execute_order.append(item)
                            current_time += no_conflict_time
                        if len(execute_order) == 2:
                            break
                    for item in execute_order:
                        recieved_order.remove(item)
            else:
                change_state = True
                for item in recieved_order:
                    if item[0] < current_floor:
                        change_state = False
                if change_state:
                    elevator_direction = True
                    for item in recieved_order:
                        if item[0] == current_floor and item[2] == elevator_direction:
                            # print("----------0--------")
                            execute_order.append(item)
                            current_time += no_conflict_time
                        if len(execute_order) == 2:
                            break
                    for item in execute_order:
                        recieved_order.remove(item)
        # print("execute_order", execute_order)
        # print("received_order", recieved_order)
        # 所有任务执行完 电梯停止
        if (not execute_order) and (not recieved_order):
            elevator_state = False
        # 更新当前所在楼层
        if elevator_state:
            if elevator_direction:
                # print("-------上行-------")
                add_floor = 1
            else:
                # print("-------下行-------")
                add_floor = -1
            current_floor += add_floor * 1
        current_time += 1 * one_floor_time
        # print(current_floor, current_time)
        if not elevator_state:
            # print(current_floor, current_time)
            return current_time


# ordList = [[2, 4, True, 0], [2, 6, True, 0], [6, 7, True, 0], [6, 8, True, 0], [4, 8, True, 0]]
# LOOK_transorform(ordList)
def create_orderList(order_list, order_time):
    orderList = []
    for i in range(len(order_list)):
        # print('=====')
        order_change = []
        current_order = order_list[i]
        order_change.append(current_order[0])
        order_change.append(current_order[1])
        if current_order[0] < current_order[1]:
            order_change.append(True)
        else:
            order_change.append(False)
        recieve_time = order_time[i]
        order_change.append(recieve_time)
        orderList.append(order_change)
    return orderList


# 楼层高设为9
building_floor = 30
order_num = 100
startFloor = 1
one_floor_time = 5
no_conflict_time = 60
conflict_time = 100
for i in range(10):
    np.random.seed(i)
    start_list = list(np.random.randint(1, building_floor + 1, order_num))
    end_list = list(np.random.randint(1, building_floor + 1, order_num))
    same_number = []
    for i in range(len(start_list)):
        if start_list[i] == end_list[i]:
            if start_list[i] == building_floor:
                end_list[i] = end_list[i] - 1
            else:
                end_list[i] = end_list[i] + 1
    length = len(start_list)
    order_time = list(np.zeros(order_num))
    order_list = []
    for i in range(len(start_list)):
        tmp_list = [start_list[i], end_list[i]]
        if end_list[i] > start_list[i]:
            tmp_list.append(True)
        else:
            tmp_list.append(False)
        order_list.append(tmp_list)
        order_time[i] = i
    # print(order_list)
    # print(order_time)
    time_fcfs, var_fcfs = FCFS(order_list[:], order_time[:], startFloor=1, no_conflict_time=no_conflict_time,
                               conflict_time=conflict_time, one_floor_time=one_floor_time)
    time_sstf, var_sstf = SSTF(order_list[:], order_time[:], startFloor=1, no_conflict_time=no_conflict_time,
                               conflict_time=conflict_time, one_floor_time=one_floor_time)
    orderList = create_orderList(order_list[:], order_time[:])
    # print(orderList)
    time_look = LOOK_transorform(orderList, startFloor=1, no_conflict_time=no_conflict_time,
                                 conflict_time=conflict_time, one_floor_time=one_floor_time)
    print(time_fcfs, time_sstf, time_look)

