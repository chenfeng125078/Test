# -*- coding: utf-8 -*-
# 画卷积神经网络的户型图语义分割样本
# 这是第二版，用多边形的方法来表示元素。
# 可以对标注多边形进行调整，所以需要有点的选中(ok)
# 可以对标注多边形进行删除
# 围成一个多边形就停止绘图模式(ok)
# Enter键直接强行连接成多边形(ok)
# ESC键直接强行清除当前已画的图形
# 多边形显示颜色(ok)
# 以列表形式存储已有的标注(ok)
# alt切换类型(ok)
# alt+s进行保存
# control进行绘图(ok)


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import PIL
import numpy as np
from package_geometry2 import *
from functools import reduce
import time
import glob
from enum import Enum
import os
import json
import PIL.Image


markersize = 5  # 标记大小
alpha = 0.2  # 透明度
point_accute = 2  # 选点误差
digit = 0  # 精确位数
x_error = y_error = 0.5  # x坐标和y坐标的偏离


class element(Enum):
    # 线段类型
    wall = 0  # 墙体
    corner = 1  # 墙体拐角
    nb_wall = 2  # 非承重墙
    door = 3  # 单开门
    double_door = 4  # 双开门
    sliding_door = 5  # 推拉门
    unequal_door = 6  # 子母门
    window = 7  # 单开窗
    bay_window = 8  # 飘窗
    balcony = 9  # 阳台
    elevator = 10  # 电梯
    stair = 11  # 楼梯
    pineline = 12  # 管道间
    nan = 13  # 无意义区域


def bulid_figure(figurepath):
    # 建立图像
    ax1 = figuration.ax1
    ax_now = figuration.axis
    img_origin = PIL.Image.open(figurepath).convert('RGB')
    ax_now.imshow(img_origin)
    show_title()
    ax1.set_title(container.element_type.name, color=[1, 0, 0], fontsize=30)
    plt.draw()


def load_data(figurepath):
    # 读取图像对应的数据
    ax_now = figuration.axis
    txtpath = figurepath.replace('floorplan_image', 'floorplan_segmantation').replace('jpg', 'json')
    figuration.figurepath = figurepath
    figuration.txtpath = txtpath
    # if flag_recognition:
    #     if not os.path.exists(txtpath):
    #         # 直接执行模型，然后保存为json
    #         img_origin = np.array(Image.open(figurepath).convert('RGB'))
    #         factor = 100
    #         high_house = 3000
    #         house, element_list, status, msg = HLD_main(img_origin, model, factor, high_house, fig_reverse=False)
    #         room_list, segment_list = element_list
    #         data_dict = {}
    #         poly_list = []
    #         i_poly = 0
    #         for _, poly in enumerate(room_list):
    #             for i, _ in enumerate(poly):
    #                 p_now = poly[i - 1]
    #                 p_next = poly[i]
    #                 s = Segment.build([p_now, p_next])
    #                 poly_s = s.transform('s_poly', width=5)
    #                 poly_list.append(poly_s)
    #                 data_dict[i_poly] = {'type': 'wall', 'polygen': poly_s}
    #                 i_poly += 1
    #         for d_list, d_type in zip(segment_list, ['window', 'door']):
    #             for _, s in enumerate(d_list):
    #                 p_now = s.p1
    #                 p_next = s.p2
    #                 s = Segment.build([p_now, p_next])
    #                 poly_s = s.transform('s_poly', width=5)
    #                 poly_list.append(poly_s)
    #                 data_dict[i_poly] = {'type': d_type, 'polygen': poly_s}
    #                 i_poly += 1
    #         with open(txtpath, 'w', encoding='gbk') as file_object:
    #             json.dump(data_dict, file_object, indent=2)
    if os.path.exists(txtpath):
        with open(txtpath, 'r') as f:
            data_dict = json.load(f)
        key_list = sorted(list(data_dict.keys()))
        for key in key_list:
            poly_type = data_dict[key]['type']
            if poly_type in dict(element.__members__):
                poly_now = data_dict[key]['polygen']
                color, alpha_now = get_color(poly_type)
                container.build_queue.append([[]])
                # 画边
                point_dict = {}  # 点字典
                for idx, p in enumerate(poly_now):
                    p = tuple(p)
                    p_last = tuple(poly_now[idx - 1])
                    x1, y1 = p
                    x2, y2 = p_last
                    x1 = int(x1) + x_error
                    x2 = int(x2) + x_error
                    y1 = int(y1) + y_error
                    y2 = int(y2) + y_error
                    p = (x1, y1)
                    p_last = (x2, y2)
                    # 判断点是否使用过
                    if p_last not in point_dict:
                        p1 = Point(p_last)
                        point_dict[p_last] = p1
                    else:
                        p1 = point_dict[p_last]
                    if p not in point_dict:
                        p2 = Point(p)
                        point_dict[p] = p2
                    else:
                        p2 = point_dict[p]
                    # 建立线段
                    s = Segment.build([p1, p2])
                    s.type = poly_type
                    line_n = ax_now.plot([x1, x2], [y1, y2], color=color, marker='o', markersize=markersize)
                    container.build_queue[-1][0].append([[p1, p2], s, line_n[0]])
                # 填充
                p_list = [term[0][0] for term in container.build_queue[-1][0]]
                x_list = [p[0] for p in p_list]
                y_list = [p[1] for p in p_list]
                patch_n = ax_now.fill(x_list, y_list, color=color, alpha=alpha_now)
                container.build_queue[-1].append(patch_n[0])
        plt.draw()


def load_figuration():
    # 读取配置信息
    if not os.path.exists(root + 'figuration.json'):
        # 如果不存在，创建一个
        figuration = {'checkpoint': 0}
        with open(root + 'figuration.json', 'w', encoding='gbk') as file_object:
            json.dump(figuration, file_object, indent=2)
    with open(root + 'figuration.json', 'r', encoding='gbk') as file_object:
        info = json.load(file_object)
    figuration = Figuration()
    figuration.checkpoint = info['checkpoint']
    return figuration


def logo(ax, p_start, idx):
    # 用来画标志
    x, y = p_start
    e = element(idx)
    string = e.name
    color, alpha_now = get_color(string)
    rect = plt.Rectangle((x, y), 1, 1, color=color, alpha=0.3)  # 左下起点，长，宽，颜色，α
    ax.text(x + 1.1, y, string, family='monospace', fontsize=10)
    ax.add_patch(rect)


def show(figurepath):
    # 显示图像和读取数据
    container.__init__()
    # 清理图像，重新建立一个轴
    fig = figuration.fig
    fig.clf()
    # ax2 = fig.add_subplot(111)

    gs = GridSpec(7, 7)
    ax1 = plt.subplot(gs[:, 0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax1.set_xlim((-5, 0))
    ax1.set_ylim((-len(element) * 2, 0))
    for i in range(len(element)):
        logo(ax1, (-5, -2 * i - 1), i)
    ax2 = plt.subplot(gs[:, 1:])

    figuration.ax1 = ax1
    figuration.axis = ax2
    bulid_figure(figurepath)
    load_data(figurepath)


def clear_point_move():
    # 清理当前移动点
    if container.point_move_info is not None:
        ax_now = figuration.axis
        _, _, _, h = container.point_move_info
        ax_now.lines.remove(h)  # 010
        container.point_move_info = None


def get_color(element_name):
    # 获取当前应该的颜色。由当前绘图的元素种类决定颜色
    color = [1, 0, 0]  # 墙体：纯红
    alpha_now = alpha
    if element_name == 'corner':
        # 非承重墙：军校蓝
        color = [95/255, 158/255, 160/255]
        alpha_now = 0.9  # 透明度
    if element_name == 'nb_wall':
        # 非承重墙：纯蓝
        color = [0/255, 0/255, 255/255]
    if element_name == 'door':
        # 单开门：橙色
        color = [255/255, 165/255, 0/255]
    if element_name == 'double_door':
        # 双开门：青色
        color = [0/255, 139/255, 139/255]
    if element_name == 'sliding_door':
        # 推拉门：紫色
        color = [128/255, 0/255, 128/255]
    if element_name == 'unequal_door':
        # 子母门：巧克力
        color = [210/255, 105/255, 30/255]
    if element_name == 'window':
        # 单开窗：洋红
        color = [255/255, 0/255, 255/255]
    if element_name == 'bay_window':
        # 飘窗：绿色
        color = [30/255, 255/255, 144/255]
    if element_name == 'balcony':
        # 阳台：秋麒麟
        color = [218 / 255, 165 / 255, 32 / 255]
    if element_name == 'elevator':
        # 电梯：草坪绿
        color = [255/255, 99/255, 71/255]
    if element_name == 'stair':
        # 楼梯：蓝色
        color = [30/255, 144/255, 255/255]
    if element_name == 'pineline':
        # 管道间：钢蓝
        color = [70/255, 130/255, 180/255]
        alpha_now = 0.8
    if element_name == 'nan':
        # 无意义区域：灰色
        color = [128/255, 128/255, 128/255]
        alpha_now = 0.8
    return color, alpha_now


def point_floor_calc():
    # 计算点的层信息，以字典形式表示
    p_dict = {}  # 点对应的层
    for i_poly, info_polygen in enumerate(container.build_queue):
        poly = info_polygen[0]
        for idx, info in enumerate(poly):
            # 计算点属于第几个点
            p1, p2 = info[0]
            s_type = info[1].type
            if tuple(p1) not in p_dict:
                p_dict[tuple(p1)] = {'point': p1, 'polygen': i_poly, 'floor': [], 'type': ''}
            if tuple(p2) not in p_dict:
                p_dict[tuple(p2)] = {'point': p2, 'polygen': i_poly, 'floor': [], 'type': ''}
            # 每个点只接受一个多边形的统计信息，按照历史次序来，这样当点重叠的时候不会选中多个点
            if p_dict[tuple(p1)]['polygen'] == i_poly:
                p_dict[tuple(p1)]['floor'].append(idx)
                p_dict[tuple(p1)]['type'] = s_type
            if p_dict[tuple(p2)]['polygen'] == i_poly:
                p_dict[tuple(p2)]['floor'].append(idx)
                p_dict[tuple(p2)]['type'] = s_type
    return p_dict


def show_title():
    # 显示线段类型
    plt.title('filepath: ' + imagePaths[figuration.checkpoint])
    plt.draw()


def button_press(event):
    fig = figuration.fig
    ax_now = figuration.axis
    if 'figure' in event.inaxes.__dir__():
        x_event, y_event = event.xdata, event.ydata
        if x_event < 0 and y_event < 0:
            # 选择画笔类型
            if -5 < x_event < 0:
                place = math.floor(y_event)
                if place % 2 == 1:
                    container.element_type = element((-place - 1) // 2)
                    show_title()
                    ax1 = figuration.ax1
                    ax1.set_title(container.element_type.name, color=[1, 0, 0], fontsize=30)
        if container.control:
            if container.draw_mode is None or container.draw_mode == 'draw':
                # 控制的情况下才能新增点
                container.draw_mode = 'draw'  # 进入画图模式
                x_n, y_n = x_error + round(event.xdata, digit), y_error + round(event.ydata, digit)
                # 先产生一个新的点
                p = Point([x_n, y_n])
                if container.point_now is not None:
                    # 建立元件并且绘图
                    x_last, y_last = p_last = container.point_now
                    if x_n != x_last or y_n != y_last:
                        # 只有选中不同的点才能建立元件
                        s = Segment.build([p, p_last])
                        s.type = container.element_type.name
                        color, alpha_now = get_color(s.type)
                        line_n = ax_now.plot([x_last, x_n], [y_last, y_n], color=color, marker='o', markersize=markersize)
                        container.build_queue[-1][0].append([[p_last, p], s, line_n[0]])
                else:
                    # 这是第一个点，可以新建一个空的列表来进行储存
                    container.build_queue.append([[]])

                # 更新当前点
                container.point_now = p
                # 对上一个当前点描绘进行清理
                if container.temp_point is not None:
                    ax_now.lines.remove(container.temp_point)
                    container.temp_point = None
                # 当前点描绘
                point_n = ax_now.plot([x_n], [y_n], color=[0, 0, 1], marker='o', markersize=markersize)
                container.temp_point = point_n[0]
                plt.draw()
        elif container.shift:
            # 可以直接画矩形
            if container.draw_mode is None or container.draw_mode == 'rect':
                container.draw_mode = 'rect'  # 进入画图模式
                x_n, y_n = x_error + round(event.xdata, digit), y_error + round(event.ydata, digit)
                if container.point_now is not None:
                    # 建立元件并且绘图
                    x_last, y_last = container.point_now
                    if x_n != x_last or y_n != y_last:
                        p1 = Point([x_last, y_last])
                        p2 = Point([x_n, y_last])
                        p3 = Point([x_n, y_n])
                        p4 = Point([x_last, y_n])
                        for p_n1, p_n2 in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
                            x_n1, y_n1 = p_n1
                            x_n2, y_n2 = p_n2
                            s = Segment.build([p_n1, p_n2])
                            s.type = container.element_type.name
                            color, alpha_now = get_color(s.type)
                            line_n = ax_now.plot([x_n1, x_n2], [y_n1, y_n2], color=color, marker='o',
                                                 markersize=markersize)
                            container.build_queue[-1][0].append([[p_n1, p_n2], s, line_n[0]])
                        # 进行填充
                        p_list = [term[0][0] for term in container.build_queue[-1][0]]
                        x_list = [p[0] for p in p_list]
                        y_list = [p[1] for p in p_list]
                        patch_n = ax_now.fill(x_list, y_list, color=color, alpha=alpha_now)
                        container.build_queue[-1].append(patch_n[0])
                        # 删除引导线
                        if container.temp_rect_line is not None:
                            for line in container.temp_rect_line:
                                ax_now.lines.remove(line)
                            container.temp_rect_line = None
                        # 退出绘图模式
                        container.draw_mode = None
                        container.point_now = None
                        # 清理当前点的句柄
                        ax_now.lines.remove(container.temp_point)
                        container.temp_point = None
                else:
                    # 这是第一个点，可以新建一个空的列表来进行储存
                    container.build_queue.append([[]])
                    # 更新当前点
                    container.point_now = [x_n, y_n]
                    # 对上一个当前点描绘进行清理
                    if container.temp_point is not None:
                        ax_now.lines.remove(container.temp_point)
                        container.temp_point = None
                    # 当前点描绘
                    point_n = ax_now.plot([x_n], [y_n], color=[0, 0, 1], marker='o', markersize=markersize)
                    container.temp_point = point_n[0]
                plt.draw()
        else:
            # 在没有控制的情况下可以直接移动点
            build_queue = container.build_queue
            if len(build_queue) >= 1:
                x_n = event.xdata
                y_n = event.ydata
                p_dict = point_floor_calc()
                p_list = list(p_dict.keys())
                dist_list = [Point.distance(p, [x_n, y_n]) for p in p_list]
                if min(dist_list) < point_accute:
                    # 如果范围足够小，选中该点
                    container.draw_mode = 'move'  # 进入移动模式
                    # 清理当前选中点
                    clear_point_move()
                    # 新建选中点
                    p_min = p_list[dist_list.index(min(dist_list))]
                    # 选中点的信息
                    i_polygen = p_dict[p_min]['polygen']
                    i_floor = p_dict[p_min]['floor']
                    s_type = p_dict[p_min]['type']
                    point_n = ax_now.plot([x_n], [y_n], color=[0, 0, 1], marker='o', markersize=markersize)
                    container.point_move_info = [p_dict[p_min]['point'], i_polygen, i_floor, point_n[0]]
                    ax_now.set_xlabel('selected element type: ' + s_type, fontsize=20)
                    plt.draw()


def button_release(event):
    if container.draw_mode == 'move':
        # 移动模式下
        # 退出移动模式
        container.draw_mode = None


def key_press(event):
    print(event.key)
    ax1 = figuration.ax1
    ax_now = figuration.axis
    # fig = event.inaxes.figure
    # ax_now = fig.gca()
    if event.key == 'control':
        container.control = True
    if event.key == 'ctrl+z':
        # 在非绘图模式下，才允许撤回
        if container.draw_mode is None:
            # 删除最后一层
            if len(container.build_queue) > 0:
                # 删除所有的线段
                poly_now = container.build_queue[-1]
                for floor in poly_now[0]:
                    _, _, handle_w = floor
                    ax_now.lines.remove(handle_w)
                # 删除填充
                patch_n = poly_now[1]
                ax_now.patches.remove(patch_n)
                del container.build_queue[-1]
                # 删除选中点图层
                clear_point_move()
                plt.draw()
    if event.key == 'alt':
        container.element_type = element((container.element_type.value + 1) % len(element))
        show_title()
        ax1.set_title(container.element_type.name, color=[1, 0, 0], fontsize=30)
    if event.key in ['ctrl+s', 'S']:
        # 保存数据
        txtpath = figuration.txtpath
        dirname = os.path.dirname(txtpath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        data_dict = {}
        for i_poly, info_poly in enumerate(container.build_queue):
            poly = info_poly[0]
            if len(poly) > 0:
                p_list = []
                poly_type = poly[0][1].type
                for floor in poly:
                    p_segment, _, _ = floor
                    p_list.append(p_segment[0])
                data_dict[i_poly] = {'type': poly_type, 'polygen': p_list}
        with open(txtpath, 'w', encoding='gbk') as file_object:
            json.dump(data_dict, file_object, indent=2)
        plt.title('Save successfully!', Fontsize=30)
        plt.draw()
    if event.key == 'o':
        fig.canvas.toolbar.zoom()
    if event.key == 'h':
        fig.canvas.toolbar.home()
    if event.key == 'backspace':
        fig.canvas.toolbar.back()
    if event.key == 'p' or event.key == ' ':
        fig.canvas.toolbar.pan()
        fig.canvas.toolbar._set_cursor(event)
    if event.key == 'shift':
        container.shift = True
    if event.key == '.':
        # 前进一个图片
        checkpoint = figuration.checkpoint
        figuration.checkpoint = (checkpoint + 1) % len(imagePaths)
        figurepath = container.imagePaths[figuration.checkpoint]
        show(figurepath)
    if event.key == ',':
        # 后退一个图片
        checkpoint = figuration.checkpoint
        figuration.checkpoint = (checkpoint - 1) % len(imagePaths)
        figurepath = container.imagePaths[figuration.checkpoint]
        show(figurepath)
    if event.key == '>':
        # 前进10个图片
        checkpoint = figuration.checkpoint
        figuration.checkpoint = (checkpoint + 10) % len(imagePaths)
        figurepath = container.imagePaths[figuration.checkpoint]
        show(figurepath)
    if event.key == '<':
        # 后退10个图片
        checkpoint = figuration.checkpoint
        figuration.checkpoint = (checkpoint - 10) % len(imagePaths)
        figurepath = container.imagePaths[figuration.checkpoint]
        show(figurepath)
    if event.key == 'delete':
        # 删除与选中点相关的层
        if container.point_move_info is not None:
            p, i_polygen, _, handle_p = container.point_move_info
            # 删除所有的线段
            poly_now = container.build_queue[i_polygen]
            for floor in poly_now[0]:
                _, _, handle_w = floor
                ax_now.lines.remove(handle_w)
            del container.build_queue[i_polygen]
            # 删除填充
            patch_n = poly_now[1]
            ax_now.patches.remove(patch_n)
            # 删除选中点图层
            ax_now.lines.remove(handle_p)
            container.point_move_info = None
            plt.draw()
    if event.key in ['escape', 'u']:
        # 退出绘图模式
        if container.draw_mode == 'draw':
            container.draw_mode = None
            container.point_now = None
            # 删除所有的线段
            poly_now = container.build_queue[-1]
            for floor in poly_now[0]:
                _, _, handle_w = floor
                ax_now.lines.remove(handle_w)
            del container.build_queue[-1]
            # 清理当前点的句柄
            if container.temp_point is not None:
                ax_now.lines.remove(container.temp_point)
                container.temp_point = None
            # 清理临时线
            if container.temp_line is not None:
                ax_now.lines.remove(container.temp_line)
                container.temp_line = None
            plt.draw()
        elif container.draw_mode == 'rect':
            del container.build_queue[-1]
            # 删除引导线
            if container.temp_rect_line is not None:
                for line in container.temp_rect_line:
                    ax_now.lines.remove(line)
                container.temp_rect_line = None
            # 退出绘图模式
            container.draw_mode = None
            container.point_now = None
            # 清理当前点的句柄
            ax_now.lines.remove(container.temp_point)
            container.temp_point = None
            plt.draw()
        else:
            # 删除选中点图层
            clear_point_move()
            plt.draw()
    if event.key == 'enter' or event.key == ' ':
        if container.draw_mode == 'draw':
            # 在绘图模式下
            queue = container.build_queue[-1][0]
            if len(queue) >= 2:
                # 长度足够，进行强行封闭
                x_start, y_start = p_start = queue[0][0][0]
                x_end, y_end = p_end = queue[-1][0][1]
                s = Segment.build([p_end, p_start])
                s.type = container.element_type.name
                color, alpha_now = get_color(s.type)
                line_n = ax_now.plot([x_start, x_end], [y_start, y_end], color=color, marker='o', markersize=markersize)
                container.build_queue[-1][0].append([[p_end, p_start], s, line_n[0]])
                # 进行填充
                p_list = [term[0][0] for term in queue]
                x_list = [p[0] for p in p_list]
                y_list = [p[1] for p in p_list]
                patch_n = ax_now.fill(x_list, y_list, color=color, alpha=alpha_now)
                container.build_queue[-1].append(patch_n[0])
                # 清理临时线
                if container.temp_line is not None:
                    ax_now.lines.remove(container.temp_line)
                    container.temp_line = None
                # 清理临时点
                if container.temp_point is not None:
                    ax_now.lines.remove(container.temp_point)
                    container.temp_point = None
                # 退出绘图模式
                container.draw_mode = None
                container.point_now = None
                plt.draw()
    if event.key in ['up', 'down', 'left', 'right']:
        # 移动模式下，对点进行移动
        if container.point_move_info is not None:
            build_queue = container.build_queue
            p, i_polygen, relation_floor, handle_p = container.point_move_info
            # 重新设置点的数值
            x_old, y_old = p
            x_n = x_old
            y_n = y_old
            if event.key == 'up':
                y_n -= 1
            if event.key == 'down':
                y_n += 1
            if event.key == 'left':
                x_n -= 1
            if event.key == 'right':
                x_n += 1
            p.__init__([x_n, y_n])
            # 找到相应的墙体句柄，更新清理
            walls = [build_queue[i_polygen][0][floor][1] for floor in relation_floor]
            handles = [build_queue[i_polygen][0][floor][2] for floor in relation_floor]
            color, alpha_now = get_color(walls[0].type)
            for wall, h, floor in zip(walls, handles, relation_floor):
                ax_now.lines.remove(h)  # 015
                x1, y1 = wall.p1
                x2, y2 = wall.p2
                line_n = ax_now.plot([x1, x2], [y1, y2], color=color, marker='o', markersize=markersize)
                build_queue[i_polygen][0][floor][2] = line_n[0]
            # 更新点的句柄
            ax_now.lines.remove(handle_p)
            point_n = ax_now.plot([x_n], [y_n], color=[0, 0, 1], marker='o', markersize=markersize)
            container.point_move_info[3] = point_n[0]
            # 更新填充
            handle_patch = container.build_queue[i_polygen][1]
            ax_now.patches.remove(handle_patch)
            p_list = [term[0][0] for term in build_queue[i_polygen][0]]
            x_list = [p[0] for p in p_list]
            y_list = [p[1] for p in p_list]
            patch_n = ax_now.fill(x_list, y_list, color=color, alpha=alpha_now)
            container.build_queue[i_polygen][1] = patch_n[0]
            plt.draw()


def key_release(event):
    if event.key == 'control':
        container.control = False
    if event.key == 'shift':
        container.shift = False


def mouse_motion(event):
    figuration.action_count += 1
    if 'figure' in event.inaxes.__dir__():
        ax_now = figuration.axis
        x_e = event.xdata
        y_e = event.ydata
        if container.draw_mode == 'draw':
            # 绘图模式下，画引导线
            print(figuration.action_count)
            if container.point_now is not None:
                x_n, y_n = container.point_now
                if container.temp_line is not None:
                    ax_now.lines.remove(container.temp_line)
                    container.temp_line = None
                color, alpha_now = get_color(container.element_type.name)
                line_n = ax_now.plot([x_n, x_e], [y_n, y_e], color=color)
                container.temp_line = line_n[0]
                plt.draw()
        if container.draw_mode == 'rect':
            # 矩形绘图模式下，画引导线
            if container.point_now is not None:
                x_n, y_n = container.point_now
                if container.temp_rect_line is not None:
                    for line in container.temp_rect_line:
                        ax_now.lines.remove(line)
                    container.temp_rect_line = None
                color, alpha_now = get_color(container.element_type.name)
                temp_rect_line = []
                p1 = (x_n, y_n)
                p2 = (x_e, y_n)
                p3 = (x_e, y_e)
                p4 = (x_n, y_e)
                for p_n1, p_n2 in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
                    x_n1, y_n1 = p_n1
                    x_n2, y_n2 = p_n2
                    line_n = ax_now.plot([x_n1, x_n2], [y_n1, y_n2], color=color)
                    temp_rect_line.append(line_n[0])
                container.temp_rect_line = temp_rect_line
                plt.draw()
        if container.draw_mode == 'move':
            # 移动模式下，对点进行移动
            if container.point_move_info is not None:
                build_queue = container.build_queue
                p, i_polygen, relation_floor, handle_p = container.point_move_info
                # 重新设置点的数值
                x_n, y_n = x_error + round(event.xdata, digit), y_error + round(event.ydata, digit)
                p.__init__([x_n, y_n])
                # 找到相应的墙体句柄，更新清理
                walls = [build_queue[i_polygen][0][floor][1] for floor in relation_floor]
                handles = [build_queue[i_polygen][0][floor][2] for floor in relation_floor]
                color, alpha_now = get_color(walls[0].type)
                for wall, h, floor in zip(walls, handles, relation_floor):
                    ax_now.lines.remove(h)  # 015
                    x1, y1 = wall.p1
                    x2, y2 = wall.p2
                    line_n = ax_now.plot([x1, x2], [y1, y2], color=color, marker='o', markersize=markersize)
                    build_queue[i_polygen][0][floor][2] = line_n[0]
                # 更新点的句柄
                ax_now.lines.remove(handle_p)
                point_n = ax_now.plot([x_n], [y_n], color=[0, 0, 1], marker='o', markersize=markersize)
                container.point_move_info[3] = point_n[0]
                # 更新填充
                handle_patch = container.build_queue[i_polygen][1]
                ax_now.patches.remove(handle_patch)
                p_list = [term[0][0] for term in build_queue[i_polygen][0]]
                x_list = [p[0] for p in p_list]
                y_list = [p[1] for p in p_list]
                patch_n = ax_now.fill(x_list, y_list, color=color, alpha=alpha_now)
                container.build_queue[i_polygen][1] = patch_n[0]
                plt.draw()


def on_close(event):
    # 保存配置
    info = {'checkpoint': figuration.checkpoint}
    with open(root + 'figuration.json', 'w', encoding='gbk') as file_object:
        json.dump(info, file_object, indent=2)


class Container:

    def __init__(self):
        self.control = False  # 控制键。按着控制键可以新建一条线段
        self.shift = False  # shift键，按着shift键，可以新建一个矩形
        self.point_now = None  # 当前点，即当前绘图点
        self.temp_point = None  # 当前点的句柄，用来显示当前点的位置，是一个句柄
        self.point_move_info = None  # 被移动的点信息（点+所在建立列表polygen位置+所在建立列表floor位置+点句柄）
        self.temp_line = None  # draw绘图模式下临时线。临时线的作用是在鼠标移动的时候，可以看到形成的墙体
        self.temp_rect_line = None  # rect绘图模式下临时线。临时线的作用是在鼠标移动的时候，可以看到形成的墙体
        self.draw_mode = None  # 画图模式。在非画图模式下才能进行撤销
        self.build_queue = []  # 元件建立过程（一层包括点列表+墙体+句柄）
        self.element_idx = 0  # 当前绘图类型默认是0
        self.element_type = element(0)  # 当前绘图类型默认是0


class Figuration:

    def __init__(self):
        self.checkpoint = None  # 检查点
        self.fig = None  # 当前图像
        self.ax1 = None  # 左边的axis
        self.axis = None    # 当前绘图的axis
        self.figurepath = None  # 当前图像的地址
        self.txtpath = None  # 当前图像的标记
        self.action_count = 0  # 动作计数


# 主程序
# 读取路径
root = './data_wujwu/'  # 数据根目录
imagePaths = glob.glob(root + 'floorplan_image/kjl/*')  # 图片目录
figuration = load_figuration()
# 初始化
container = Container()
container.imagePaths = imagePaths
if figuration.checkpoint > len(imagePaths):
    figuration.checkpoint = 0
figurepath = imagePaths[figuration.checkpoint]
# 建立图像
fig = plt.figure()
fig.canvas.mpl_disconnect(3)
figuration.fig = fig
# 显示图像和读取数据
show(figurepath)
# 注册函数
fig.canvas.mpl_connect('button_press_event', button_press)
fig.canvas.mpl_connect('button_release_event', button_release)
fig.canvas.mpl_connect('key_press_event', key_press)
fig.canvas.mpl_connect('key_release_event', key_release)
fig.canvas.mpl_connect('motion_notify_event', mouse_motion)
fig.canvas.mpl_connect('close_event', on_close)
