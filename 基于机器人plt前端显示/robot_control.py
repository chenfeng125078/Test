import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from package_geometry2 import *
from package_image import *
import os
import time
import json
import numpy as np
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle


def select_recent_info(building_now, floor_now, expire=60, dir='data'):
    # 筛选出最近的信息
    # print("-------------")
    position_dict.clear()
    files = os.listdir(dir)
    for file in files:
        file_path = dir + "/" + file
        last_time = os.stat(file_path).st_mtime  # 获取文件的时间戳
        now_time = time.time()
        if last_time + expire > now_time:
            # 心跳正常
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
                if 'BuildingNum' in data:
                    building_num = data['BuildingNum']
                else:
                    building_num = building_now
                # print(file)
                # print(building_num, floor_num)
                if (building_num, floor_num) not in position_dict:
                    position_dict[(building_num, floor_num)] = [data]
                else:
                    position_dict[(building_num, floor_num)].append(data)


class Figuration:

    def __init__(self):
        fig = plt.figure()
        self.fig = fig
        self.ax_head = None
        self.ax_map = None
        self.ax_left = None
        self.ax_right = None
        self.img_origin = None


def show():
    # 根据楼层号设置
    with open('map_setting.json', 'r') as f:
        setting = json.loads(f.read())
    figurepath = 'map/' + setting[building_now][str(floor_now)]
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


def key_press(event):
    time_now = time.time()
    # print(time_now)
    if event.key == 'h':
        figuration.fig.canvas.toolbar.home()
        figuration.img_origin.reverse('y_axis')


def plot_robot(axis, a, b, r, name):
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    f1 = axis.plot(x, y, picker=True)
    f2 = axis.text(a, b + r, name, family='monospace', fontsize=10, horizontalalignment='center')
    return f1, f2


def show_robot(building_now, floor_now):
    # 更新数据
    select_recent_info(building_now, floor_now, expire=100000)
    # 根据当前楼栋和楼层，画出当前机器人的位置
    if (building_now, floor_now) in position_dict:
        for data in position_dict[(building_now, floor_now)]:
            x = data['Location-x'] / 20
            y = data['Location-y'] / 20
            device_code = data['DeviceCode']
            if 'Radius' in data:
                r = data['Radius'] / 20
            else:
                r = 9
            # 画机器人
            f1, f2 = plot_robot(figuration.ax_map, x, y, r, device_code)
            if device_code not in plot_dict:
                plot_dict[device_code] = {'plot': f1[0], 'text': f2}
            else:
                figuration.ax_map.lines.remove(plot_dict[device_code]['plot'])
                figuration.ax_map.texts.remove(plot_dict[device_code]['text'])
                plt.draw()
                plot_dict[device_code] = {'plot': f1[0], 'text': f2}


def onclick(event):
    if not event.xdata or not event.ydata:
        print("end point can not arrive")
    elif event.xdata > 1 and event.ydata > 1:
        confirm = " "
        if confirm == "n" or confirm == "no":
            print("please input correct end point")
        else:
            print(event.xdata, event.ydata)
            end_point = (event.xdata, event.ydata)
            # 将终点数据通过mqtt发送给OpenTCS


class ButtonHandler:
    def __init__(self):
        self.flag = False
        self.cid = "button_press_event"
        # print("----------", self.flag)

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
        self.flag =False
        self.judge_case()

    def judge_case(self):
        if self.flag:
            # print("now self.flag is %s" % self.flag)
            self.cid = figuration.fig.canvas.mpl_connect("button_press_event", onclick)
        else:
            # print("now self.flag is %s" % self.flag)
            figuration.fig.canvas.mpl_disconnect(self.cid)


def onpick(event):
    this_robot = event.artist
    xdata = this_robot.get_xdata()
    ydata = this_robot.get_ydata()
    ind = event.ind
    print('onpick points:', (xdata[ind], ydata[ind]))


if __name__ == '__main__':
    building_now = '6#'
    floor_now = -1
    # 画布定义
    figuration = Figuration()
    # 画布数据、显示
    show()
    position_dict = {}
    plot_dict = {}
    # 注册函数
    figuration.fig.canvas.mpl_connect('key_press_event', key_press)
    show_robot(building_now, floor_now)
    callback = ButtonHandler()
    axprev = plt.axes([0.81, 0.05, 0.1, 0.075])
    bprev = Button(axprev, 'Cancel')
    bprev.on_clicked(callback.Stop)
    axnext = plt.axes([0.7, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Create')
    bnext.on_clicked(callback.Start)
    figuration.fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
