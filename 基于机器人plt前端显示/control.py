from matplotlib.gridspec import GridSpec
from package_image import *
import os
import time
import json
import numpy as np
from matplotlib.widgets import Button
import requests
import sys
from watchdog.observers import Observer
from watchdog.events import *
import uuid
import threading

"""该测试前端显示界面，步骤如下
1. 运行 deviceinfo.py 通过 mqtt 实时更新机器人位置
2. 运行 get_vehicle_path.py 通过 mqtt 实时监控路径下发
3. 运行control.py 进行路径以及机器人位置实时显示 """
# matplotlib.use('Qt5Agg')


# 筛选最新机器人位置信息
def select_recent_info(building_now, floor_now, position_dict, expire=60, dir='data'):
    # 筛选出最近的信息
    # print("-------------")
    position_dict.clear()
    files = os.listdir(dir)
    for file in files:
        file_path = dir + "/" + file
        last_time = os.stat(file_path).st_mtime  # 获取文件的时间戳
        now_time = time.time()
        # print(file)
        if last_time + expire > now_time:
            # 满足条件的机器人
            # print(file)
            # 心跳正常
            # print("----------", file)
            with open(file_path, 'r') as f:
                data = f.read()
            data = json.loads(data)
            # 提取坐标
            if 'Location-x' in data and 'Location-y' in data:
                # print("have right format")
                if 'FloorNum' in data:
                    floor_num = data['FloorNum']
                else:
                    floor_num = floor_now
                # if 'BuildingNum' in data:
                #     building_num = data['BuildingNum']
                # else:
                building_num = building_now
                # print(file)
                # print(building_num, floor_num)
                if (building_num, floor_num) not in position_dict:
                    position_dict[(building_num, floor_num)] = [data]
                else:
                    position_dict[(building_num, floor_num)].append(data)
            # break
            # else:
            #     # 判断数据格式是否有问题
            #     print("no build no floor")
    return position_dict


# 画机器人
def plot_robot(axis, a, b, r, name):
    # 将机器人在画布上表示为一个正方形
    x1 = a - r
    x2 = a + r
    y1 = b - r
    y2 = b + r
    # print(a, x1)
    x = np.array([x1, x1, x2, x2, x1])
    y = np.array([y1, y2, y2, y1, y1])
    f1 = axis.plot(x, y)
    f2 = axis.text(a, b + r, name, family='monospace', fontsize=10, horizontalalignment='center')
    return f1, f2


# 回到图像主界面快捷键
def key_press(event):
    if event.key == 'h':
        figuration.fig.canvas.toolbar.home()
        figuration.img_origin.reverse('y_axis')


# 判断是否选中机器人
def judge_select_robot(rects, x_data, y_data):
    select_robot = None
    select_case = False
    for item in rects:
        # 当前楼层某一个机器人
        current_rect = rects[item]
        x_min = current_rect[0]
        y_min = current_rect[1]
        safe_distance = current_rect[2]
        x_max = x_min + 2 * safe_distance
        y_max = y_min + 2 * safe_distance
        if (x_max > x_data > x_min) and (y_max > y_data > y_min):
            select_case = True
            select_robot = item
            print("当前选中的机器人为%s" % item)
            break
    if select_case:
        pass
    else:
        print("未选中机器人")
    return select_robot, select_case


# 创建订单函数
def create_order(x, y, select_robot):
    # 这个地方orderId 需要及时修改 后续需完善
    # print(select_robot, type(select_robot))
    headers = {'Content-Type': 'application/json'}
    # todo:orderid需自动生成
    order_id = ""
    i = np.random.randint(4, 14)
    for item in range(i):
        j = np.random.randint(0, 10)
        order_id += str(j)
    # print(order_id)
    datas = json.dumps({"destination": {"buildingName": None,
                                        "buildingNum": None,
                                        "deviceCode": None,
                                        "floorNum": "7",
                                        "state": None,
                                        "typeCode": None,
                                        "x": x,
                                        "y": y,
                                        "yaw": None,
                                        "z": None}, "deviceCode": select_robot,
                        "orderId": order_id,
                        "route": None, "startFloorNum": "7"})
    r = requests.post("http://10.8.202.166:65200/v1/tcs/request/createTransportOrder", data=datas, headers=headers)
    print(r.text)


def lines_plot(x, y):
    f1 = figuration.ax_map.plot(x, y)
    return f1


# 监测下发路径的线程
class PathFileMonitorHandler(FileSystemEventHandler):
    def __init__(self, **kwargs):
        super(PathFileMonitorHandler, self).__init__(**kwargs)
        # 监控目录 目录下面以device_id为目录存放各自的图片
        self._watch_path = "./path"
        self.case = False

    # 重写文件改变函数，文件改变都会触发文件夹变化
    def on_modified(self, event):
        if not event.is_directory:
            file_path = event.src_path
            # print("文件改变: %s " % file_path)
            self.case = True
            x_list, y_list = [], []
            with open(event.src_path, "r") as fp:
                robot_data = json.load(fp)
                route_points = robot_data["route"]["RouteData"]["RoutePts"]
                # print(len(route_points))
                for item in route_points:
                    x = item["X"] / 20 - x_delta
                    x_list.append(x)
                    y = item["Y"] / 20 - y_delta
                    y_list.append(y)
            x_point = np.array(x_list)
            y_point = np.array(y_list)
            f1 = lines_plot(x_point, y_point)
            if "path" not in plot_dict:
                plot_dict["path"] = f1[0]
            else:
                figuration.ax_map.lines.remove(plot_dict["path"])
                plot_dict["path"] = f1[0]
                plt.draw()
            plt.draw()

    def on_created(self, event):
        print('创建了文件夹', event.src_path)

    # def on_moved(self, event):
    #     print("移动了文件", event.src_path)
    #
    # def on_deleted(self, event):
    #     print("删除了文件", event.src_path)


def monitor_file():
    observer.schedule(event_handler, path="./path", recursive=True)  # recursive递归的
    observer.start()
    observer.join()


# 监控机器人位置信息
class PositionFileMonitorHandler(FileSystemEventHandler):
    def __init__(self, **kwargs):
        super(PositionFileMonitorHandler, self).__init__(**kwargs)
        # 监控目录 目录下面以device_id为目录存放各自的图片
        self._watch_path = "./data"
        self.case = False

    # 重写文件改变函数，文件改变都会触发文件夹变化
    def on_modified(self, event):
        if not event.is_directory:
            file_path = event.src_path
            # print("文件改变: %s " % file_path)
            self.case = True
            with open(event.src_path, "r") as fp:
                data = json.load(fp)
                device_code = data["DeviceCode"]
                x = data["X"] / 20 - x_delta
                y = data["Y"] / 20 - y_delta
                if 'Radius' in data:
                    r = data['Radius'] / 20
                else:
                    r = 9
                f1, f2 = plot_robot(figuration.ax_map, x, y, r, device_code)
                if device_code not in plot_dict:
                    plot_dict[device_code] = {'plot': f1[0], 'text': f2}
                else:
                    figuration.ax_map.lines.remove(plot_dict[device_code]['plot'])
                    figuration.ax_map.texts.remove(plot_dict[device_code]['text'])
                    plot_dict[device_code] = {'plot': f1[0], 'text': f2}
                    plt.draw()
                with open(os.path.join("./path", device_code + ".json")) as path_fp:
                    data_path = json.load(path_fp)
                    target_x = data_path["route"]["RouteData"]["RoutePts"][-1]["X"]
                    target_y = data_path["route"]["RouteData"]["RoutePts"][-1]["Y"]
                    if target_x == data["X"] and target_y == data["Y"]:
                        try:
                            figuration.ax_map.lines.remove(plot_dict["path"])
                        except Exception as e:
                            pass
                        plt.draw()

    def on_created(self, event):
        print('创建了文件夹', event.src_path)


# 监控机器人位置线程
def monitor_position():
    position_observer.schedule(position_handler, path="./data", recursive=True)  # recursive递归的
    position_observer.start()
    position_observer.join()
    print("---------------")
    # time.sleep(1)


# 创建订单按钮事件触发函数
def orderclick(event, robot):
    if not event.xdata or not event.ydata:
        print("end point can not arrive")
        sys.exit()
    # data_x, data_y
    if event.xdata > 1 and event.ydata > 1:
        data_x = (np.round(event.xdata) + x_delta) * 20
        data_y = (np.round(event.ydata) + y_delta) * 20
        # 此时需要判断是否要切换楼层 -----可能z值在上报过程中也要上报
        # -----注意：目标楼层要从plt中获取，但是在同层下单移动中从robot_info中获取
        # print(data_x, data_y)
        # 确定订单机器人
        the_robot = robot
        # 根据data_x, data_y 生成订单上报(json数据格式)
        create_order(data_x, data_y, the_robot)
        ordercallback.Stop(event)
    else:
        print("please select the correct point")
        sys.exit()


# 选择机器人按钮事件函数
def onclick(event):
    # 判断点是否无效
    if not event.xdata or not event.ydata:
        print("please select correct robot")
    elif event.xdata > 1 and event.ydata > 1:
        confirm = " "
        if confirm == "n" or confirm == "no":
            print("please input correct end point")
        else:
            # 判断是否选中某一个机器人
            # print(event.xdata, event.ydata)
            select_robot, select_case = judge_select_robot(rects, event.xdata, event.ydata)
            # print(select_robot, select_case)
            if select_case:
                print("选中机器人")
                # 调试时注释，方便调试
                # choice = input("选择输入坐标还是鼠标点击事件,YES或Y代表输入坐标")
                # if choice.upper() == "YES" or choice.upper() == "Y":
                #     x = float(input("输入目标点x坐标: "))
                #     y = float(input("输入目标点y坐标: "))
                #     callback.Stop(event)
                #     create_order(x, y, select_robot)
                #     data_case = monitor_file()
                #     if data_case:
                #         redraw_image(select_robot)
                #     else:
                #         print("订单上报异常.重新派发订单")
                #         sys.exit()
                # else:
                ordercallback.Start(event, select_robot)
                # 在图像上生成可视化路径
                # todo:监听数据到来
                callback.Stop(event)
            else:
                print("当前未选中机器人，请重新选择需要派发订单机器人")


# 画布布局调整
def show():
    # 根据楼层号设置
    with open('map_setting.json', 'r') as f:
        setting = json.loads(f.read())
    figurepath = 'map/' + setting[building_now][str(floor_now)]
    # print(figurepath)
    img_origin = Figure.build(cv2.imread(figurepath))
    img_origin.reverse('y_axis')
    figuration.fig.clf()
    gs = GridSpec(7, 7)
    figuration.ax_head = plt.subplot(gs[0, :])
    figuration.ax_map = plt.subplot(gs[1:, 1:-1])
    # figuration.ax_left = plt.subplot(gs[1:, 0])
    # figuration.ax_right = plt.subplot(gs[1:, -1])
    figuration.img_origin = img_origin
    figuration.ax_map.imshow(img_origin)
    figuration.ax_map.invert_yaxis()


class Figuration:

    def __init__(self):
        fig = plt.figure()
        self.fig = fig
        self.ax_head = None
        self.ax_map = None
        self.ax_left = None
        self.ax_right = None
        self.img_origin = None


# 在地图上显示出机器人所在位置
def show_robot(building_now, floor_now, position_dict):
    # 更新数据
    position_dict = select_recent_info(building_now, floor_now, position_dict, expire=100000)
    rects = {}
    # print(position_dict)
    # 根据当前需展示的楼栋和楼层，画出当前机器人的位置
    if (building_now, floor_now) in position_dict:
        # print("--------")
        # print(building_now, floor_now)
        for data in position_dict[(building_now, floor_now)]:
            # 这里将坐标偏量返回
            x = data['Location-x'] / 20 - x_delta
            y = data['Location-y'] / 20 - y_delta
            device_code = data['DeviceCode']
            if 'Radius' in data:
                r = data['Radius'] / 20
            else:
                r = 9
            # 画机器人:机器人表示 {deviceCode: [左上角点x坐标, 左上角点y坐标, 安全半径]}
            rects[device_code] = [x - r, y - r, r]
            f1, f2 = plot_robot(figuration.ax_map, x, y, r, device_code)
            if device_code not in plot_dict:
                plot_dict[device_code] = {'plot': f1[0], 'text': f2}
            else:
                figuration.ax_map.lines.remove(plot_dict[device_code]['plot'])
                figuration.ax_map.texts.remove(plot_dict[device_code]['text'])
                plt.draw()
                plot_dict[device_code] = {'plot': f1[0], 'text': f2}
    # print(rects)
    return rects


# 创建订单按钮
class OrderHandler:
    def __init__(self):
        self.flag = False
        self.cid = "button_press_event"

    # 线程函数，用来更新数据并重新绘制图形
    def threadStart(self):
        if self.flag:
            # 更新数据
            self.judge_case(robot=None)
            # 重新绘制图形
            self.flag = False

    def Start(self, event, robot=None):
        self.flag = True
        self.judge_case(robot)

    def Stop(self, event, robot=None):
        self.flag = False
        self.judge_case(robot)

    def judge_case(self, robot):
        if self.flag:
            # print("-----------")
            self.cid = figuration.fig.canvas.mpl_connect("button_press_event", lambda event: orderclick(event, robot))
        else:
            figuration.fig.canvas.mpl_disconnect(self.cid)


# 鼠标点击事件：选择机器人按钮
class ButtonHandler:
    def __init__(self):
        self.flag = False
        self.cid = "button_press_event"

    # 线程函数，用来更新数据并重新绘制图形
    def threadStart(self):
        if self.flag:
            # 更新数据
            self.judge_case()
            # 重新绘制图形
            self.flag = False

    def Start(self, event):
        self.flag = True
        self.judge_case()

    def Stop(self, event):
        self.flag = False
        self.judge_case()

    def judge_case(self):
        if self.flag:
            # print("now self.flag is %s" % self.flag)
            self.cid = figuration.fig.canvas.mpl_connect("button_press_event", lambda event: onclick(event))
        else:
            # print("now self.flag is %s" % self.flag)
            figuration.fig.canvas.mpl_disconnect(self.cid)


if __name__ == '__main__':
    # 楼栋号以及楼层号
    building_now = '4#'
    floor_now = 7
    # 调整图像
    # plt.subplots_adjust(bottom=0.2)
    # 每栋楼(每层楼)有坐标偏量
    x_delta = -500 / 20
    y_delta = -np.round(6656 / 20 - 172)
    figuration = Figuration()
    # 这里需要根据选择的目标楼层去进行展示 后续需开发
    show()
    # plt.show()
    position_dict = {}
    plot_dict = {}
    # 注册函数:'h'返回原图像
    figuration.fig.canvas.mpl_connect('key_press_event', key_press)
    time_now = time_old = time.time()
    # 创建路径文件监听函数
    event_handler = PathFileMonitorHandler()
    observer = Observer()
    # 创建机器人位置信息文件监听函数
    position_handler = PositionFileMonitorHandler()
    position_observer = Observer()
    # 返回机器人的位置、大小信息
    rects = show_robot(building_now, floor_now, position_dict)
    # 创建并启动新线程监听路径数据下发
    threading.Thread(target=monitor_file, args=()).start()
    # 创建并启动新线程监听机器人位置数据下发
    threading.Thread(target=monitor_position, args=()).start()
    # 创建按钮并设置单击事件处理函数
    callback = ButtonHandler()
    ordercallback = OrderHandler()
    axprev = plt.axes([0.81, 0.05, 0.1, 0.075])
    bprev = Button(axprev, 'Cancel')
    bprev.on_clicked(callback.Stop)
    bprev.on_clicked(ordercallback.Stop)
    axnext = plt.axes([0.7, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Create')
    bnext.on_clicked(callback.Start)
    plt.show()


