# 几何库2
from base_tool import *
import numpy as np
import math
import matplotlib.pyplot as plt
from functools import lru_cache  # 缓存函数或者类的结果 加快运行速度
import cv2
# from shapely.geometry import Polygon as SPolygon


# 参数
digit = 6  # 精度
dis_tol = 1e-6  # 距离误差（程序的保留位数是3位）
angle_tol = 1e-4  # 角度误差
angle_big_tol = 5/180*math.pi  # 大角度误差，用于人类视角

# 线段array索引
S_P1 = 0
S_P2 = 1
S_LENGTH = 2
S_DIRECTION = 3
S_LOCATE = 4
S_P_MIN = 5
S_P_MAX = 6
S_IND = 7  # 线段索引


# 角度
class Angle:

    @staticmethod
    def atan(x, y):
        # 计算反正切（考虑特殊情况，范围是0~2pi）
        if x == 0:
            if y > 0:
                return math.pi / 2
            if y < 0:
                return math.pi * 3 / 2
        elif x > 0:
            if y >= 0:
                return math.atan(y / x)
            if y < 0:
                return 2 * math.pi + math.atan(y / x)
        elif x < 0:
            if y >= 0:
                return math.pi + math.atan(y / x)
            if y < 0:
                return math.pi + math.atan(y / x)

    @staticmethod
    def confirm(v_sin_list, v_cos_list):
        # 角度类函数，角度范围是[0, 2pi)
        # 根据sin和cos值计算准确弧度
        direction_list = []
        for v_sin, v_cos in zip(v_sin_list, v_cos_list):
            direction = math.asin(v_sin)
            if direction < 0:  # 将小于[-pi, 0]部分转移到[pi, 2pi]
                direction += 2 * math.pi
            if v_cos < 0:
                if v_sin >= 0:
                    direction = math.pi - direction
                if v_sin < 0:
                    direction = math.pi + (2 * math.pi - direction)
            direction_list.append(direction)
        return direction_list

    @staticmethod
    def diff(angle1, angle2, angle_range_max=2 * math.pi):
        # 计算角度之间的角度距离
        return cycle_difference(angle1, angle2, angle_range_max)

    @staticmethod
    def diff_real(angle1, angle2, angle_range_max=2 * math.pi):
        # 计算角度之间的最短距离
        return cycle_difference_real(angle1, angle2, angle_range_max)

    @staticmethod
    def normal_calc(direction):
        # 计算角度的法线角度(主要针对直线，在0~180度范围内计算）
        if direction >= math.pi / 2:
            return direction - math.pi / 2
        else:
            return direction + math.pi / 2

    @staticmethod
    def tri_value(p_list, p_center):
        '''
        计算点序列相对于中心点的三角函数值。注意，点序列当中没有中心点，否则报错
        :param p_list: 点列表
        :param p_center:  中心点坐标 (x, y)
        :return: angle_sin 正弦值列表
                 angle_cos 余弦值列表
        '''
        xc, yc = p_center
        xy = np.array(p_list)
        x = xy[:, 0]
        y = xy[:, 1]
        if (np.sqrt(pow(x - xc, 2) + pow(y - yc, 2)) == 0).any():
            raise Exception
        angle_cos = (x - xc) / np.sqrt(pow(x - xc, 2) + pow(y - yc, 2))
        angle_sin = (y - yc) / np.sqrt(pow(x - xc, 2) + pow(y - yc, 2))
        return angle_sin, angle_cos

    @staticmethod
    def to_angle(value):
        # 返回角度对应的弧度值
        return value*180/math.pi

    @staticmethod
    def to_value(angle):
        # 返回角度对应的弧度值
        return angle/180*math.pi


# 角度集合
class Angles:

    @staticmethod
    def kmeans(info_list, diff_threshold=0.1, angle_num=2):
        '''
        使用k_means算法分成两个类，找到角度中心
        :param info_list: info_list_list有两列：角度和权重
        :param diff_threshold: 角度阈值
        :param angle_num: 提取的角度数量
        :return:
        '''
        category = []
        mean_list = []
        weight_list = []
        for i_term, term in enumerate(info_list):
            angle, weight = term
            if len(category) == 0:
                # 没有分类，直接创建分类
                mean_list.append(angle)
                weight_list.append(weight)
                category.append([angle])
            else:
                diff_list = []
                for value in mean_list:
                    diff_list.append(Angle.diff_real(angle, value, math.pi))
                if min(diff_list) <= diff_threshold:
                    i_find = diff_list.index(min(diff_list))
                    category[i_find].append(angle)
                    mean_list[i_find] = Angles.mean(category[i_find], math.pi)
                    weight_list[i_find] += weight
                else:
                    # 直接创建一个分类
                    mean_list.append(angle)
                    weight_list.append(weight)
                    category.append([angle])
        # 找出2个主方向
        weight_index = np.argsort(weight_list)[::-1]
        mean_rst_list = []
        for i in range(angle_num):
            try:
                mean_rst_list.append(mean_list[weight_index[i]])
            except:
                break
        return mean_rst_list

    @staticmethod
    def mean(angle_list, angle_range_max=2 * math.pi):
        # 计算角度均值
        angle_min = min(angle_list)
        angle_max = max(angle_list)
        if angle_max - angle_min <= angle_range_max / 2:
            # 在正常的半区
            return np.mean(angle_list)
        else:
            # 跨越了零轴，则将小于angle_max对称角度的所有角度加上整个范围
            angle_sum = 0
            for angle in angle_list:
                if angle < angle_max - angle_range_max / 2:
                    angle_sum += angle + angle_range_max
                else:
                    angle_sum += angle
            rst = angle_sum / len(angle_list)
            return rst % angle_range_max


# 点
class Point(list):

    def __init__(self, seq=()):
        assert len(seq) == 2  # 二维坐标
        super().__init__(seq)
        self.ind = 0  # 索引
        self.type = ''  # 点的类型

    def float_ban(self, digit=digit):
        # 由于python的精度计算问题，需要对点的位数进行截取
        x, y = self
        self[0] = round(x, digit)
        self[1] = round(y, digit)

    def move(self, p_delta):
        # 对点进行平移，移动幅度为p_delta
        x, y = self
        x_delta, y_delta = p_delta
        super().__init__([x + x_delta, y + y_delta])

    def replace_min_distance(self, p_list, threshold=10):
        # 将点替换为p_list当中最接近的点，只要这个点的距离小于阈值
        dist_list = [Point.distance(self, p_cmp) for p_cmp in p_list]
        if min(dist_list) < threshold:
            p_rst = p_list[np.argmin(dist_list)]
            return p_rst, True
        else:
            return self, False

    def transfer(self, p_new):
        # 点转移p_new
        x_delta, y_delta = Point.delta(self, p_new)
        self.move((x_delta, y_delta))

    @staticmethod
    def angle_original(p):
        '''
        计算点p与原点所形成的直线与水平方向形成的夹角
        :param p: 点p
        :return:
        '''
        x, y = p
        v_sin = y / np.sqrt(pow(x, 2) + pow(y, 2))
        v_cos = x / np.sqrt(pow(x, 2) + pow(y, 2))
        a = Angle.confirm([v_sin], [v_cos])[0]
        return a

    @staticmethod
    def build(p, type=''):
        # 直接给定类型
        p_n = Point(p)
        p_n.type = type
        return p_n

    @staticmethod
    def combination(p_start, p_end, c):
        # 两个点进行线性组合，得到中间点
        x1, y1 = p_start
        x2, y2 = p_end
        return Point.float_cut((x1 * c + x2 * (1 - c), y1 * c + y2 * (1 - c)))

    @staticmethod
    def distance(p1, p2, method='direct'):
        x1, y1 = p1
        x2, y2 = p2
        if method == 'direct':
            # 计算直线距离
            return np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
        if method in ['manhattan', 'chessblock']:
            # 计算曼哈顿距离
            return np.abs(x1 - x2) + np.abs(y1 - y2)
        if method in ['chebyshev', 'maxside']:
            # 计算切比雪夫距离
            return max([np.abs(x1 - x2), np.abs(y1 - y2)])

    @staticmethod
    def delta(p_start, p_end):
        # 计算以p_end为终点，p_start移动到p_end的x_delta和y_delta
        x_s, y_s = p_start
        x_e, y_e = p_end
        return x_e - x_s, y_e - y_s

    @staticmethod
    def extend_direction(p_start, direction, d_extend):
        # 点借助方向进行延伸
        x1, y1 = p_start
        return [x1 + d_extend * math.cos(direction), y1 + d_extend * math.sin(direction)]

    @staticmethod
    def float_cut(p, digit=digit):
        # 由于python的精度计算问题，需要对点的位数进行截取
        x, y = p
        return [round(x, digit), round(y, digit)]

    @staticmethod
    @lru_cache(None)
    def gen(p):
        # 点生成器，利用lru_cache进行点缓存
        x, y = p
        p = Point([x, y])
        return p

    @staticmethod
    def in_convex(p_now, border):
        '''
        判断点是否在凸多边形当中，带误差计算
        :param p_now:
        :param border:
        :return: 1:点在多边形中（内部或边界） 0:点不在多边形中
        '''
        vectors = Polygen.vetorize(border)
        flag_in = 1
        for vector in vectors:
            if Point.is_RtInSide(vector, p_now) > 0:
                flag_in = 0
        return flag_in

    @staticmethod
    def in_polygen(point, border, method='radial', output='general'):
        '''
        判断当前点是否在多边形内部，可以处理任意多边形（注意，精确位数是程序的精确位数）
        :param point: 当前点
        :param border: 多边形
        :param method: radial 射线法（精确到程序的最小位数）  round 环绕法（多边形边数在一定范围内完全精确）
        :param output:
            radial 射线法
            detail:明细输出，是明确计算点在多边形的什么位置
                         0  在多边形外
                         1  在多边形内部（不包含边界和顶点）
                         2  在多边形边界上（不包含顶点）
                         3  在多边形的顶点上
            general:大概输出  0  在多边形外
                             1  在多边形内部（包含边界和顶点）
            round 环绕法，精确计算（一般不建议使用）
            0  在多边形外
            1  在多边形内部（包含边界和顶点）
        :return:
        '''
        xp, yp = point
        vectors = Polygen.vetorize(border)
        rst = 0  # 假设点在外面
        if point in border:
            # 点在顶点上
            rst = 3
        else:
            if method == 'radial':
                for v in vectors:
                    if Point.in_segment(v, point):
                        # 点在边界上
                        rst = 2
                if rst != 2:
                    cross_count = 0  # 交点数量
                    l_radial = [(xp, yp), (xp + 10000, yp)]  # 水平射线（先这么设置）
                    for v in vectors:
                        p1, p2 = v
                        flag1 = Point.in_segment(l_radial, p1)
                        flag2 = Point.in_segment(l_radial, p2)
                        if flag1 == 1 and flag2 == 1:
                            # 两个端点都在射线上，则没有交点
                            pass
                        elif flag1 == 1 and flag2 == 0:
                            if p2[1] > yp:
                                # 两个点都在扫描行上或者上方，不算相交
                                pass
                            else:
                                # 一个点在扫描行上，一个点在扫描行下，算相交
                                cross_count += 1
                        elif flag1 == 0 and flag2 == 1:
                            if p1[1] > yp:
                                # 两个点都在扫描行上或者上方，不算相交
                                pass
                            else:
                                # 一个点在扫描行上，一个点在扫描行下，算相交
                                cross_count += 1
                        else:
                            # 两个端点都不在射线上，则按照正常算法
                            inter_point = Segment.interpoint(l_radial, v)
                            if inter_point != -1:
                                # 产生交点
                                cross_count += 1
                    if cross_count % 2 == 1:
                        # 交点数为奇数个，即在多边形内部
                        rst = 1
                    else:
                        # 交点数为偶数个，即在多边形外部
                        rst = 0
            elif method == 'round':
                for v in vectors:
                    if Point.in_segment(v, point):
                        # 点在边界上
                        rst = 2
                if rst != 2:
                    acc_seta = 0
                    for i, p in enumerate(border):
                        xn, yn = p
                        xb, yb = border[i - 1]
                        va = ((xp, yp), (xb, yb))
                        vb = ((xp, yp), (xn, yn))
                        if Segment.length(va) != 0 and Segment.length(vb) != 0:
                            seta = Vector.interangle(va, vb)
                            if Point.is_RtInSide(va, p, 'acc') == 0:
                                seta = 0
                            elif Point.is_RtInSide(va, p, 'acc') < 0:
                                # 相对原来的角度逆时针为正，顺时针为负
                                seta *= -1
                            acc_seta += seta
                    if math.fabs(acc_seta) < math.pi / 180:  # 这个数可以视情况放大
                        # 点在多边形外部
                        rst = 0
                    else:
                        # 点在多边形内部
                        rst = 1
        if output == 'detail':
            return rst
        elif output == 'general':
            if rst == 0:
                return 0
            else:
                return 1

    @staticmethod
    def in_rectangle(p, rect, tol=dis_tol):
        # 判断一个点是否在[x_min, x_max, y_min, y_max]组成的区域内
        x, y = p
        x_min, x_max, y_min, y_max = rect
        if x_min - tol <= x <= x_max + tol and y_min - tol <= y <= y_max + tol:
            return 1
        else:
            return 0

    @staticmethod
    def in_segment(s, p_now):
        '''
        判断点是否在线段（segment）上，带误差的计算
        :param s: 线段
        :param p_now: 点
        :return:
        '''
        x1, y1 = s[0]
        x2, y2 = s[1]
        xn, yn = p_now
        x_min, x_max = min([x1, x2]), max([x1, x2])
        y_min, y_max = min([y1, y2]), max([y1, y2])
        p1 = (x_min, y_min)
        p2 = (x_min, y_max)
        p3 = (x_max, y_max)
        p4 = (x_max, y_min)
        rst = 0  # 假设不在线段上
        if Point.is_RtInSide(s, p_now) == 0:
            # 判断点是否在直线上
            if Point.is_RtInSide(s, p_now) == 0:
                # 判断点是否在直线上。为什么要先判断X方向和Y方向，主要是有可能线段就是水平的和竖直的。
                if x_min == x_max:
                    # 如果在x方向上
                    if y_min - dis_tol < yn < y_max + dis_tol:  # 考虑精度影响
                        rst = 1
                elif y_min == y_max:
                    # 如果在y方向上
                    if x_min - dis_tol < xn < x_max + dis_tol:  # 考虑精度影响
                        rst = 1
                elif Point.in_convex(p_now, [p1, p2, p3, p4]):
                    # 判断点是否在盒子范围内（盒子是凸多边形）
                    rst = 1
        return rst

    @staticmethod
    def int(p):
        # 将点整数化
        x, y = p
        return [int(round(x)), int(round(y))]

    @staticmethod
    def is_RtInSide(vector, point, method='general'):
        # 计算是否在顺时针边界的内侧，小于0是内侧，大于0是外侧，等于0是边界
        # method: general：近似  acc:需要准确数字
        spx, spy = vector[0]
        epx, epy = vector[1]
        nx, ny = point
        if spx == nx and spy == ny:
            # 与起点相同，在边界上
            return 0
        else:
            # 将向量标准化
            v1_len = math.sqrt(pow((epx - spx), 2) + pow((epy - spy), 2))
            v2_len = math.sqrt(pow((nx - spx), 2) + pow((ny - spy), 2))
            value = (epx - spx) / v1_len * (ny - spy) / v2_len - (epy - spy) / v1_len * (nx - spx) / v2_len
            # 以距离容忍值作为基础计算角度容忍值
            # 点p到向量vector的距离小于dis_tol时则表示点在向量上
            sinseta_max = dis_tol / v2_len
            if method == 'general':
                if -sinseta_max < value < sinseta_max:
                    return 0
                else:
                    return value
            elif method == 'acc':
                return value

    @staticmethod
    def neighbor(p, method='8-dire'):
        # 输出点p的1阶邻居
        if method == '8-dire':
            iter_list = Points([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
            iter_list.move(p)
        elif method == 'square':
            iter_list = Points([(1, 0), (0, 1), (-1, 0), (0, -1)])
            iter_list.move(p)
        elif method == 'diag':
            iter_list = Points([(1, 1), (-1, 1), (-1, -1), (1, -1)])
            iter_list.move(p)
        return iter_list

    @staticmethod
    def plot(p, color=(1, 0, 0), fig=[], text='', markersize=10, scaled=False):
        if fig == []:
            fig = plt.figure()
        x, y = p
        kwargs = {'color': color, 'markersize': markersize}
        kwargs['marker'] = "o"
        plt.plot(x, y, **kwargs)
        if len(text) > 0:
            plt.text(x, y, text, fontsize=20)
        if scaled:
            plt.axis('scaled')

    @staticmethod
    def position(p, direction):
        # 计算点p关于法线方向direction的position值，具体用法最主要是计算p_min和p_max
        l_abc = Line.transform(([0, 0], direction), 'pseta_ABC')
        bound = Line.point_distance(p, l_abc)
        if bound != 0:
            pd = Point.extend_direction([0, 0], direction, 1)
            if Point.is_RtInSide([pd, [0, 0]], p) > 0:
                # 在外侧为负，在内侧为正数
                bound *= -1
        return bound

    @staticmethod
    def rotate_axis(p, seta):
        '''
        对点做坐标轴旋转，顺时针旋转seta角度。
        :param seta: 顺时针旋转的角度
        :return:
        '''
        x_p, y_p = p
        x_n = x_p * math.cos(seta) + y_p * math.sin(seta)
        y_n = y_p * math.cos(seta) - x_p * math.sin(seta)
        return x_n, y_n

    @staticmethod
    def rotate_center(p, p_center, seta):
        # 绕p_center顺时针旋转seta角度
        x_p, y_p = p
        x_center, y_center = p_center
        x_n = (x_p - x_center) * math.cos(seta) + (y_p - y_center) * math.sin(seta) + x_center
        y_n = (y_p - y_center) * math.cos(seta) - (x_p - x_center) * math.sin(seta) + y_center
        return x_n, y_n


# 点集合函数
class Points(list):

    def __init__(self, p_list, hold=False):
        super().__init__([])
        for p in p_list:
            if hold:
                # 不新建点，保留点的信息
                self.append(p)
            else:
                # 直接新建点
                self.append(Point(p))

    def close_origin(self):
        # 将多边形尽可能往原点靠，同时保持所有点都在第一象限上。
        x_min, x_max, y_min, y_max = self.range()
        poly = copy.deepcopy(self)
        x_delta = -x_min
        y_delta = -y_min
        poly.move((-x_min, -y_min))
        return poly, x_delta, y_delta

    def coco_bbox(self):
        # 计算coco数据集格式的bbox
        x_min, x_max, y_min, y_max = self.range()
        x_delta = x_max - x_min
        y_delta = y_max - y_min
        return x_min, y_min, x_delta, y_delta

    def coco_poly(self):
        # 计算coco数据格式的多边形数据
        xy_list = []
        for p in self:
            xy_list.extend(p)
        return xy_list

    def convex(self):
        # 计算点的凸包
        convex_hull = cv2.convexHull(np.array(self).astype('int'))
        convex = [tuple(d[0].tolist()) for d in convex_hull]
        return convex

    def extract_mseg(self, epsilon=1):
        # 将点集转化为多段线
        segment_list = []
        line_array = np.array(self, 'float32')
        key_points = cv2.approxPolyDP(line_array, epsilon, False)
        key_points = Points.transform(key_points, 'cv_point')
        for j in range(len(key_points) - 1):
            p = key_points[j]
            p_next = key_points[j + 1]
            s = Segment((p, p_next))
            # 记录原始端点之间的距离
            s.mid_length = Point.distance(s.p1, s.p2, 'direct')
            s.weight = s.mid_length
            segment_list.append(s)
        return segment_list

    def gravity(self):
        # 计算多边形重心
        x_list = []
        y_list = []
        for p in self:
            x, y = p
            x_list.append(x)
            y_list.append(y)
        return sum(x_list) / len(x_list), sum(y_list) / len(y_list)

    def merge_close_point(self, threshold=10):
        '''
        合并相近的点，提供点收集字典
        :param threshold: 距离阈值
        :return:
        '''

        # 合并相近的关键点
        merge_group = Grouper()
        for p1 in self:
            for p2 in self:
                if p1 != p2:
                    if Point.distance(p1, p2) < threshold:
                        merge_group.add({tuple(p1), tuple(p2)})
        delete_list = []
        append_list = []
        collect_dict = {}  # 点收集字典
        for group in merge_group.data:
            delete_list.extend(group)
            p_now = tuple(Point.float_cut(Points.gravity(group)))
            collect_dict[p_now] = group
        return collect_dict

    def min_area_rect(self):
        # 计算多边形的最小外接矩形，以及中心线
        rect = cv2.minAreaRect(np.array(self))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        center = rect[0]
        width, high = rect[1]
        width_direction = (rect[2] + 90)/180*math.pi  # 宽度方向direction
        high_direction = Angle.normal_calc(width_direction)  # 高度方向direction
        bone_list = []
        if width >= high:
            s = Segment.build_from_direction(width_direction)
            p_min, p_max = -width/2, width/2
            s.feature_fix((p_min, p_max), 'minmax')
            s.move(Point.delta(s.p_mid, center))
            bone_list.append(s)
        if width <= high:
            s = Segment.build_from_direction(high_direction)
            p_min, p_max = -high/2, high/2
            s.feature_fix((p_min, p_max), 'minmax')
            s.move(Point.delta(s.p_mid, center))
            bone_list.append(s)
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点
        box = np.round(box).tolist()
        return box, bone_list

    def move(self, p_delta):
        # 点列表进行移动
        for p in self:
            p.move(p_delta)

    def range(self):
        # 计算这条线段的范围
        x_list = []
        y_list = []
        for p in self:
            x, y = p
            x_list.append(x)
            y_list.append(y)
        x_list = sorted(x_list)
        y_list = sorted(y_list)
        x_min = x_list[0]
        x_max = x_list[-1]
        y_min = y_list[0]
        y_max = y_list[-1]
        return x_min, x_max, y_min, y_max

    def rotate_center(self, seta, p_center=(0, 0), replace=True):
        # 绕p_center顺时针旋转seta角度
        p_list_new = []
        for i, p in enumerate(self):
            x_p, y_p = p
            x_center, y_center = p_center
            x_n = (x_p - x_center) * math.cos(seta) + (y_p - y_center) * math.sin(seta) + x_center
            y_n = (y_p - y_center) * math.cos(seta) - (x_p - x_center) * math.sin(seta) + y_center
            if replace:
                self[i] = (x_n, y_n)
            else:
                p_list_new.append((x_n, y_n))
        return p_list_new

    def plot(self, color=(1, 0, 0), fig=[], texted=False, markersize=10):
        if fig == []:
            fig = plt.figure()
        for i, p in enumerate(self):
            if texted:
                text = r'w' + str(p.ind)
            else:
                text = ''
            kwargs = {'color': color, 'fig': fig, 'text': text, 'markersize': markersize}
            Point.plot(p, **kwargs)
        plt.axis('scaled')

    def transform(self, method):
        '''
        点集形式转化，可以将opencv的格式和point的格式互换
        :param d_list:
        :param method: cv_point  将opencv的格式转换为point格式
        :return:
        '''
        if method == 'cv_point':
            rst = []
            for d in self:
                rst.append(d[0].tolist())
            return rst

    @staticmethod
    def reset(p_list):
        '''
        重置点列表
        :param p_list: 点列表
        :return:
        '''

        for i, p in enumerate(p_list):
            p.ind = i

    @staticmethod
    def rotate_axis(p_list, seta):
        '''
        对点做坐标轴旋转，顺时针旋转seta角度。
        :param seta: 顺时针旋转的角度
        :return:
        '''
        p_list_new = []
        for p in p_list:
            x_p, y_p = p
            x_n = x_p * math.cos(seta) + y_p * math.sin(seta)
            y_n = y_p * math.cos(seta) - x_p * math.sin(seta)
            p_list_new.append((x_n, y_n))
        return p_list_new

    @staticmethod
    def sorted(p_list, method):
        '''

        :param p_list: 点列表
        :param method: line: 对在一条单调线上的点列表，按照与某一个端点的距离进行排序
                       mean: 对圆圈状的点列表进行顺时针排序
        :return:
        '''
        if method == 'line':
            # 对在一条线上的点进行排序
            # 随意以一个点作为基准点，找出最远点。再以该最远点作为基准点，按照距离进行排序
            p_base = p_list[0]
            dis_list = [Point.distance(p, p_base, 'direct') for p in p_list]
            max_ind = dis_list.index(max(dis_list))
            p_max = p_list[max_ind]
            dis_list2 = [[p, Point.distance(p, p_max, 'direct')] for p in p_list]
            dis_list2 = sorted(dis_list2, key=itemgetter(1))
            xy_new = [term[0] for term in dis_list2]
        if method == 'mean':
            # 围绕均值将点进行顺时针排序（在0~2pi范围内从角度最大值排到角度最小值）
            xy = np.array(p_list)
            center_x, center_y = np.mean(xy[:, 0]), np.mean(xy[:, 1])
            angle_sin, angle_cos = Angle.tri_value(xy, [center_x, center_y])
            angle = Angle.confirm(angle_sin, angle_cos)
            xy_angle = np.column_stack((xy, angle))
            xy_angle = np.array(sorted(xy_angle, key=itemgetter(2), reverse=True))
            xy_new = xy_angle[:, :-1].tolist()
        return xy_new


# 直线
class Line(list):

    def __init__(self, seq=()):
        assert len(seq) == 3  # 直线标准方程的A、B、C系数
        super().__init__(seq)
        rho, seta = Line.transform(seq, 'ABC_polar')
        self.polar = [rho, seta]  # 极坐标系

    def get_direction(self):
        # 计算直线的方向，返回一个较小的方向。实际上一条直线是有两个方向的。
        A, B, C = self
        if B != 0:
            direction1 = math.atan(-A/B)
            direction2 = (direction1 + np.pi) % (2 * np.pi)
        else:
            direction1 = np.pi/2
            direction2 = np.pi/2 * 3
        return min([direction1, direction2])

    @staticmethod
    def interpoint(l1_abc, l2_abc):
        '''
        计算两条直线的交点
        :param l1_abc: 直线1（ABC形式）
        :param l1_abc: 直线2（ABC形式）
        :return: 交点
        '''
        A1, B1, C1 = l1_abc
        A2, B2, C2 = l2_abc
        if A1 * B2 - A2 * B1 != 0:
            x = -(C1 * B2 - C2 * B1) / (A1 * B2 - A2 * B1)
            y = -(C1 * A2 - C2 * A1) / (B1 * A2 - B2 * A1)
            return [x, y]
        else:
            return -1

    @staticmethod
    def interpoint_within_error(l1_abc, l2_abc, angle_threshold=angle_tol):
        '''
        计算两条直线的交点
        :param l1_abc: 直线1（ABC形式）
        :param l2_abc: 直线2（ABC形式）
        :return: 交点
        '''
        direction1 = Line.get_direction(l1_abc)
        direction2 = Line.get_direction(l2_abc)
        if Angle.diff_real(direction1, direction2, np.pi) > angle_threshold:
            # 角度相差较大
            A1, B1, C1 = l1_abc
            A2, B2, C2 = l2_abc
            if A1 * B2 - A2 * B1 != 0:
                x = -(C1 * B2 - C2 * B1) / (A1 * B2 - A2 * B1)
                y = -(C1 * A2 - C2 * A1) / (B1 * A2 - B2 * A1)
                return [x, y]
            else:
                return -1
        else:
            return -1

    @staticmethod
    def point_distance(p, l):
        '''
        计算点到直线的距离
        :param p: 点
        :param l: 直线（ABC形式）
        :return:
        '''
        A, B, C = l
        x0, y0 = p
        return np.abs((A * x0 + B * y0 + C) / np.sqrt(A * A + B * B))

    @staticmethod
    def transform(l_now, method='2p_ABC'):
        '''
        直线形式转换
        :param l_now: 直线数据
        :param method: 转换方法
        :return:
        '''
        if method == '2p_slope':
            # 已知直线两点坐标，输出点斜式直线
            p1, p2 = l_now
            x1, y1 = p1
            x2, y2 = p2
            k = (y2 - y1) / (x2 - x1)
            b = y1 - (y2 - y1) / (x2 - x1) * x1
            return k, b
        if method == '2p_ABC':
            # 已知直线两点坐标，输出ABC形式直线
            p1, p2 = l_now
            x1, y1 = p1
            x2, y2 = p2
            if x1 == x2:
                A = 1
                B = 0
                C = -x1
            else:
                k = (y2 - y1) / (x2 - x1)
                b = y1 - (y2 - y1) / (x2 - x1) * x1
                A = -k
                B = 1
                C = -b
            return A, B, C
        if method == 'pseta_ABC':
            # 已知点和夹角，求直线（ABC形式）
            p, seta = l_now
            x, y = p
            x_next = x + 1 * np.cos(seta)
            y_next = y + 1 * np.sin(seta)
            l1_abc = Line.transform([(x, y), (x_next, y_next)], '2p_ABC')
            return l1_abc
        if method == 'pk_ABC':
            # 已知点和斜率求直线
            p, k = l_now
            x_m, y_m = p
            return (k, -1, -k * x_m + y_m)
        if method == 'polar_ABC':
            # 极坐标转ABC形式
            rho, seta = l_now
            if seta == 0:
                A = np.cos(seta)
                B = 0
                C = -rho
            else:
                A = np.cos(seta) / np.sin(seta)
                B = 1
                C = -rho / np.sin(seta)
            return A, B, C
        if method == 'ABC_polar':
            # ABC形式转极坐标形式
            A, B, C = l_now
            locate = Line.point_distance([0, 0], l_now)
            # 以极坐标方程来对号入座
            if B == 0:
                # 此时为垂直于极轴的一条直线
                v_sin = 0
                v_cos = A
            elif C == 0:
                # 穿过零点
                if A > 0:
                    v_cos = math.sqrt(A * A / (1 + A * A))
                    v_sin = v_cos / A
                elif A < 0:
                    v_cos = -math.sqrt(A * A / (1 + A * A))
                    v_sin = v_cos / A
                else:
                    # 此时为水平与极轴的直线（A == 0）
                    v_cos = 0
                    v_sin = 1
            else:
                v_sin = -locate / C
                v_cos = A * v_sin
            direction = Angle.confirm([v_sin], [v_cos])[0]
            return locate, direction

    @staticmethod
    def verticalpoint(p, l, method='ABC'):
        '''
        计算垂足的位置
        :param p: 直线外的一点
        :param l: 直线数据
        :param method: ABC 直线数据为ABC形式
                       polar 直线数据为极坐标形式
        :return:
        '''
        # 转换直线
        if method == 'ABC':
            l_polar = Line.transform(l, 'ABC_polar')
        elif method == 'polar':
            l_polar = l
        # 计算垂足
        rho, seta = l_polar
        l0_abc = Line.transform(l_polar, 'polar_ABC')
        l1_abc = Line.transform((p, seta), 'pseta_ABC')
        return Line.interpoint(l0_abc, l1_abc)

    @staticmethod
    def x_coor(l_abc, x):
        # 计算x对应的y坐标
        A, B, C = l_abc
        return (-A * x - C) / B

    @staticmethod
    def y_coor(l_abc, y):
        # 计算y对应的x坐标
        A, B, C = l_abc
        return (-B * y - C) / A


# 向量
class Vector(list):
    def __init__(self, seq):
        assert len(seq) == 2  # 平面向量
        p1, p2 = seq
        super().__init__([])
        self.extend([p1, p2])

    @staticmethod
    def angle(v):
        # 计算向量角度
        x1, y1 = v[0]
        x2, y2 = v[1]
        x, y = (x2 - x1, y2 - y1)
        return Angle.atan(x, y)

    @staticmethod
    def interangle(v1, v2):
        # 向量夹角
        angle1 = Vector.angle(v1)
        angle2 = Vector.angle(v2)
        return Angle.diff_real(angle1, angle2)

    @staticmethod
    def roundangle(v1, v2, method='zeronormal'):
        # 计算v1顺时针旋转多少角度到达v2（顺时针，可以不需要同起点）
        # method: zeromax:如果角度是0，设为2pi  zeronormal:如果角度为0，就是为0
        angle1 = Vector.angle(v1)
        angle2 = Vector.angle(v2)
        angle = Angle.diff(angle2, angle1)  # 按照角度正方向来摆动是顺时针，所以要对调一下
        if method == 'zeromax':
            if angle == 0:
                angle = 2 * math.pi
        return angle


# 线段
class Segment(list):
    # 线段类

    def __init__(self, seq=(), hold=False):
        '''
        一个总的原则是，一旦用点计算好了特征。当你修改某个特征的时候，其他特征可以不变就不变，
        不要进行重复计算。重复计算必然会带来误差。
        进入这个初始化函数的序列是单纯不带任何信息的。如果带信息的要使用build函数。
        :param l_now: 输入线段两个端点的坐标， 形式是[(x1, y1), (x2, y2)]。
        '''
        p1_old, p2_old = seq
        p1, p2, length, locate, direction, p_min, p_max = Segment.feature_calc(p1_old, p2_old)
        super().__init__([])
        if hold:
            # 不新建点，保留点的信息
            if tuple(p1_old) == tuple(p1):
                p1 = p1_old
                p2 = p2_old
            else:
                p1 = p2_old
                p2 = p1_old
            self.p1_raw = p1_old  # 原始端点1
            self.p2_raw = p2_old  # 原始端点2
        else:
            # 直接新建点
            p_dict = Storer()
            p1 = p_dict.gen(Point(p1))
            p2 = p_dict.gen(Point(p2))
            self.p1_raw = p_dict.gen(Point(p1_old))  # 原始端点1
            self.p2_raw = p_dict.gen(Point(p2_old))  # 原始端点2
        self.extend([p1, p2])  # 数据保留
        self.p1 = p1  # 线段端点1（精确数值）
        self.p2 = p2  # 线段端点2（精确数值）
        self.length = length  # 线段长度
        self.direction = direction  # 极坐标的seta
        self.locate = locate  # 极坐标的rho
        self.p_min = p_min  # 在线段方向上的最小值（一维坐标, position最小值）
        self.p_max = p_max  # 在线段方向上的最大值（一维坐标，position最大值）
        self.p_mid = Point.combination(p1, p2, 0.5)  # 中心点，旋转有用
        self.ind = 0  # 线段列表索引
        self.type = ''  # 线段的类型
        self.weight = 0  # 线段的权重，用来做比较、排序使用
        # 其他属性
        self.interpoint_list = []  # 线段上的交点列表
        self.intersegment_list = []  # 与线段本身相交的线段列表

    def angle_raw(self):
        # 返回线段的原始夹角
        p1_raw = self.p1_raw
        p2_raw = self.p2_raw
        return Vector.angle([p1_raw, p2_raw])

    def another_coord(self, x=None, y=None):
        '''
        已知线段和x坐标（或者y坐标），计算另一个坐标
        :param x: 已知的x坐标
        :param y: 已知的y坐标
        :return: 返回另一个需要计算的坐标
        '''
        x1, y1 = self.p1
        x2, y2 = self.p2
        if y is not None:
            x = (x2 - x1) / (y2 - y1) * (y - y1) + x1
            return x
        elif x is not None:
            y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
            return y
        else:
            raise NotImplementedError

    def cut(self, p_cut, method):
        '''
        线段切割函数, 切割点是p_cut
        :param p_cut: 切割点（注意这个切割点是一维的position，范围是p_min~p_max）
        :param method: 切割方法： max: 切走max方向 min：切走min方向
        :return: l_out:  切割出来的线段
        '''
        l_out = copy.deepcopy(self)
        if method == 'max':
            self.feature_fix(p_cut, 'max')
            l_out.feature_fix(p_cut, 'min')
        elif method == 'min':
            self.feature_fix(p_cut, 'min')
            l_out.feature_fix(p_cut, 'max')
        return l_out

    def direction_adsorb(self, direction_list):
        '''
        对线段强制吸附至最近的方向，
        :param direction_list: 方向列表
        '''
        p1 = self.p1_raw
        p2 = self.p2_raw
        angle = Vector.angle([p1, p2])  # 计算在0~180度半区的对应角度
        diff_list = []
        for direction in direction_list:
            diff_list.append(Angle.diff_real(angle, direction))
        # 计算新角度，以及旋转角度后的新端点
        direction_new = direction_list[diff_list.index(min(diff_list))]
        self.rotate_center(self.p_mid, -(direction_new - angle))

    def direction_adjust(self, direction_new):
        '''
        对线段进行绕中心旋转至新角度
        :param direction_new: 最新方向
        :return:
        '''
        self.p1.__init__(Point.extend_direction(self.p_mid, direction_new, self.length / 2))
        self.p2.__init__(Point.extend_direction(self.p_mid, direction_new, -self.length / 2))
        self.update()

    def fix(self, p_value, method):
        '''
        单项修改直线的最大值、最小值和高度，可以相应将直线的其他特征进行计算。
        注意在这个过程中是不需要修改方向的。
        :param p_value:
        :param method: min: 修改min_position值
                       max: 修改max_position值
                       min_point: 修改最小点
                       max_point: 修改最大点
                       point_extend：修改点来延长线段。如果不能延长则不延长
                       注意min_point、max_point、point_extend方法要求修改点必须要在直线上。所以一般这类点是交点。
                       p1_extend:以p1为起点进行延长
                       p2_extend:以p2为起点进行延长
                       locate:修改高度
        :return:
        '''
        x1, y1 = self.p1
        x2, y2 = self.p2
        l = self.length
        p_min = self.p_min
        p_max = self.p_max
        if method == 'max' or method == 'min':
            ratio = (p_value - p_min) / (p_max - p_min)
            x_new = x1 + ratio * (x2 - x1)
            y_new = y1 + ratio * (y2 - y1)
            if method == 'max':
                if p_value < p_min:
                    raise NotImplementedError
                self.p_max = p_value
                self.p2.__init__([x_new, y_new])
            if method == 'min':
                if p_value > p_max:
                    raise NotImplementedError
                self.p_min = p_value
                self.p1.__init__([x_new, y_new])
            # 还需要更新的信息
            self.length = Point.distance(self.p1, self.p2)  # 线段长度
            self.p_mid = Point.combination(self.p1, self.p2, 0.5)  # 中心点，旋转有用
        elif method in ['min_point', 'max_point']:
            p = p_value
            position_now = Point.position(p, self.direction)
            if method == 'min_point':
                if position_now >= p_max:
                    raise NotImplementedError
                self.p1.transfer(p)
                self.p_min = position_now
            elif method == 'max_point':
                if position_now <= p_min:
                    raise NotImplementedError
                self.p2.transfer(p)
                self.p_max = position_now
            # 还需要更新的信息
            self.length = Point.distance(self.p1, self.p2)  # 线段长度
            self.p_mid = Point.combination(self.p1, self.p2, 0.5)  # 中心点，旋转有用
        elif method == 'point_extend':
            # 用点来延长线段
            p_list = p_value
            info_list = []
            for p in p_list:
                info_list.append([p, Point.position(p, self.direction)])
            info_list.sort(key=lambda info: info[1])
            point_min, posi_min = info_list[0]
            point_max, posi_max = info_list[-1]
            if posi_max > p_max:
                self.p2.transfer(point_max)
                self.p_max = posi_max
            if posi_min < p_min:
                self.p1.transfer(point_min)
                self.p_min = posi_min
            # 还需要更新的信息
            self.length = Point.distance(self.p1, self.p2)  # 线段长度
            self.p_mid = Point.combination(self.p1, self.p2, 0.5)  # 中心点，旋转有用
        elif method == 'p1_extend' or method == 'p2_extend':
            # 修改水平宽度
            d_extend = p_value
            if d_extend <= -l:
                # 整条线段都修改没了
                raise NotImplementedError
            if method == 'p1_extend':
                x_s, y_s = (x1, y1)
                x_e, y_e = (x2, y2)
            else:
                x_s, y_s = (x2, y2)
                x_e, y_e = (x1, y1)
            xn = x_s + (x_e - x_s) * (l + d_extend) / l
            yn = y_s + (y_e - y_s) * (l + d_extend) / l
            pn = [xn, yn]
            if method == 'p1_extend':
                self.p2.__init__(pn)
            else:
                self.p1.__init__(pn)
            # 还需要更新的信息
            _, _, length, locate, direction, p_min, p_max = Segment.feature_calc(self.p1, self.p2, self.direction, self.locate)
            self.length = length  # 线段长度
            self.p_min = p_min  # 在线段方向上的最小值（一维坐标, position最小值）
            self.p_max = p_max  # 在线段方向上的最大值（一维坐标，position最大值）
            self.p_mid = Point.combination(self.p1, self.p2, 0.5)  # 中心点，旋转有用
        elif method == 'locate':
            # 修改高度
            locate = p_value
            l_abc = Line.transform((locate, self.direction), 'polar_ABC')
            p1_new = Line.verticalpoint(self.p1, l_abc)
            p2_new = Line.verticalpoint(self.p2, l_abc)
            self.p1.__init__(p1_new)
            self.p2.__init__(p2_new)
            self.locate = locate  # 极坐标的rho
            self.p_mid = Point.combination(self.p1, self.p2, 0.5)  # 中心点，旋转有用

    def feature_fix(self, p_value, method):
        # 修改直线的最大值、最小值和高度，可以相应将直线的其他特征进行吸怪
        if method in ['max', 'min', 'max_point', 'min_point', 'locate', 'p1_extend', 'p2_extend', 'point_extend']:
            self.fix(p_value, method)
        elif method == 'minmax':
            # 同时修改position的最大值和最小值
            self.fix(max([p_value[1] + 1, self.p_max]), 'max')  # 先变长一点，这样不会产生误差
            self.fix(p_value[0], 'min')
            self.fix(p_value[1], 'max')
        elif method == 'extend':
            # 两边同时延长
            self.fix(p_value, 'p1_extend')
            self.fix(p_value, 'p2_extend')
        elif method == 'position_extend':
            # 希望添加一堆新的position，然后再重新计算线段的范围
            position_list = [self.p_min, self.p_max] + p_value
            p_min_new = min(position_list)
            p_max_new = max(position_list)
            self.feature_fix([p_min_new, p_max_new], 'minmax')

    def move(self, p_delta):
        # 对线段进行整体平移（方向不变，长度不变，高度变）
        p1 = self.p1
        p2 = self.p2
        direction = self.direction
        locate = self.locate
        p1.move(p_delta)
        p2.move(p_delta)
        # 更新相应的信息
        super().__init__([])
        self.extend([self.p1, self.p2])
        _, _, _, locate, direction, p_min, p_max = Segment.feature_calc(p1, p2, direction, locate)
        self.locate = locate  # 极坐标的rho
        self.p_min = p_min  # 在线段方向上的最小值（一维坐标, position最小值）
        self.p_max = p_max  # 在线段方向上的最大值（一维坐标，position最大值）
        self.p_mid = Point.combination(p1, p2, 0.5)  # 中心点，旋转有用

    def plot(self, color=[1, 0, 0], fig=[], text='', linewidth=1, scaled=False, marked=False,
             markersize=5):
        if fig == []:
            fig = plt.figure()
        x1, y1 = self.p1
        x2, y2 = self.p2
        kwargs = {'color': color, 'linewidth': linewidth}
        if marked:
            kwargs['marker'] = "o"
            kwargs['markersize'] = markersize
        plt.plot([x1, x2], [y1, y2], **kwargs)
        if len(text) > 0:
            x_c, y_c = (x1 + x2)/2, (y1 + y2)/2
            plt.text(x_c, y_c, text, fontsize=20)
        if scaled:
            plt.axis('scaled')

    def point_distance(self, p, method='vert'):
        '''
        计算点到线段的距离
        :param p: 点
        :param method:
        :return:
        '''
        if method == 'vert':
            # 计算点p到线段s的垂足的距离
            p_vert = Segment.point_vertical(self, p)
            return Point.distance(p, p_vert)
        elif method == 'point':
            # 计算点p到线段两个端点的最短距离
            return min([Point.distance(p, p_now) for p_now in self])
        elif method == 'clever':
            # 如果点p在线段的position范围内，就用vert方法
            # 如果在position范围外，就用point方法/vert方法中的最大值
            direction = self.direction
            p_min = self.p_min
            p_max = self.p_max
            posi = Point.position(p, direction)
            if p_min <= posi <= p_max:
                return self.point_distance(p, 'vert')
            else:
                return max([self.point_distance(p, 'vert'), self.point_distance(p, 'point')])

    def point_vertical(self, p):
        # 计算点p到线段s的垂点
        l = Line.transform(self)
        p_vert = Line.verticalpoint(p, l)
        return p_vert

    def position_to_point(self, position):
        # 计算特定position下的点位置
        x1, y1 = self.p1
        x2, y2 = self.p2
        p_min = self.p_min
        p_max = self.p_max
        ratio = (position - p_min) / (p_max - p_min)
        x_new = x1 + ratio * (x2 - x1)
        y_new = y1 + ratio * (y2 - y1)
        return [x_new, y_new]

    def range(self):
        # 计算这条线段的范围
        x1, y1 = self[0]
        x2, y2 = self[1]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        return x_min, x_max, y_min, y_max

    def rotate_axis(self, seta):
        '''
        对直线做坐标轴旋转，逆时针旋转seta角度。
        :param seta: 顺时针旋转的角度
        :return:
        '''
        self.p1.__init__(Point.rotate_axis(self.p1, seta))
        self.p2.__init__(Point.rotate_axis(self.p2, seta))
        self.update()

    def rotate_center(self, p_center, seta):
        '''
        对直线做中心点旋转，逆时针旋转seta角度。
        :param seta: 顺时针旋转的角度
        :param p_center: 中心点坐标
        :return:
        '''
        self.p1.__init__(Point.rotate_center(self.p1, p_center, seta))
        self.p2.__init__(Point.rotate_center(self.p2, p_center, seta))
        self.update()

    def simplify(self):
        # 将线段拆散为(x1, y1, x2, y2)形式
        return tuple(self.p1 + self.p2)

    def to_pixel_locate(self):
        # 将线段转化为单像素宽的点，返回像素坐标列表
        p1, p2 = self
        x1, y1 = p1
        x2, y2 = p2
        x_min, x_max = min([x1, x2]), max([x1, x2])
        y_min, y_max = min([y1, y2]), max([y1, y2])
        x_delta = x_max - x_min
        y_delta = y_max - y_min
        p_list = []
        if x_delta >= y_delta:
            for x in np.arange(int(x_min), int(x_max) + 1):
                y = self.another_coord(x=x)
                p_list.append((x, int(y)))
        else:
            for y in np.arange(int(y_min), int(y_max) + 1):
                x = self.another_coord(y=y)
                p_list.append((int(x), y))
        return p_list

    def transform(self, method, **kwargs):
        # 线段形式转换
        if method == 's_pp':
            # 将线段转换为平行线段
            width = kwargs['width']
            locate = self.locate
            s1 = copy.deepcopy(self)
            s2 = copy.deepcopy(self)
            s1.fix(locate + width, 'locate')
            s2.fix(locate - width, 'locate')
            pp = Parallel([s1, s2])
            return pp
        elif method == 's_poly':
            width = kwargs['width']
            # 将线段转换为平行线段的多边形形式
            pp = self.transform('s_pp', width=width)
            poly = pp.point_extract()
            return poly

    def tuplify(self):
        # 将线段元组化
        return tuple(self.p1), tuple(self.p2)

    def update(self):
        '''
        更新线段特征信息，其他附带的信息不进行修改。一般是用在当线段点更新以后，对特征进行更新。
        '''
        p1 = self.p1
        p2 = self.p2
        p1, p2, length, locate, direction, p_min, p_max = Segment.feature_calc(p1, p2)
        # 重新设置线段点的顺序
        self[0] = p1
        self[1] = p2
        self.p1 = p1  # 线段端点1（精确数值）
        self.p2 = p2  # 线段端点2（精确数值）
        self.length = length  # 线段长度
        self.direction = direction  # 极坐标的seta
        self.locate = locate  # 极坐标的rho
        self.p_min = p_min  # 在线段方向上的最小值（一维坐标, position最小值）
        self.p_max = p_max  # 在线段方向上的最大值（一维坐标，position最大值）
        self.p_mid = Point.combination(p1, p2, 0.5)  # 中心点，旋转有用

    def weight_init(self, method='length'):
        # 权值初始化方法
        if method == 'length':
            self.weight = self.length

    @staticmethod
    def build(seq, type=''):
        # 线生成器，保存p1和p2点的相关属性
        p1, p2 = seq
        s = Segment([p1, p2], hold=True)
        s.type = type
        return s

    @staticmethod
    def build_from_direction(direction):
        # 通过线段的法线方向建立，长度为2个单位（端点在单位圆上）
        angle1 = direction + math.pi/2
        angle2 = direction - math.pi/2
        # x_delta, y_delta = np.cos(direction), np.sin(direction)
        # x1, y1 = np.cos(angle1) + x_delta, np.sin(angle1) + y_delta
        # x2, y2 = np.cos(angle2) + x_delta, np.sin(angle2) + y_delta
        x1, y1 = np.cos(angle1), np.sin(angle1)
        x2, y2 = np.cos(angle2), np.sin(angle2)
        s = Segment([(x1, y1), (x2, y2)])
        return s

    @staticmethod
    def build_from_point_direction(p, direction):
        # 通过线段的法线方向建立，长度为2个单位（端点在单位圆上）
        x, y = p
        xn, yn = x + np.cos(direction), y + np.sin(direction)
        s = Segment([(x, y), (xn, yn)])
        return s

    @staticmethod
    def build_from_point_list(p_list):
        # 由点列表中抽取出最远的两个点，形成线段。注意，这些点都是要在一条直线上的
        # 情景可以是直线、线段形成的交点
        p_list = Points.sorted(p_list, 'line')
        p1 = p_list[0]
        p2 = p_list[-1]
        return Segment([p1, p2])

    @staticmethod
    def cut_2d(s, v_min, v_max, method='position'):
        '''
        线段切割函数，将线段切割走p_min到p_max的部分
        :param v_min: 下切割点
        :param v_max: 上切割点
        :param method: position: 按照位置的方法进行切分
                       point: 按照点直接进行切分
        :return: 剩下的线段部分。切割出的线段按照position进行排序
        '''
        s_now1 = copy.deepcopy(s)
        s_now2 = copy.deepcopy(s)
        p_min = s.p_min
        p_max = s.p_max
        direction = s.direction
        s_left = []
        if method == 'position':
            cut_pmin = cut_min = v_min
            cut_pmax = cut_max = v_max
            low_method = 'min'
            high_method = 'max'
        elif method == 'point':
            cut1 = Point.position(v_min, direction)
            cut2 = Point.position(v_max, direction)
            info = [[cut1, v_min], [cut2, v_max]]
            info = sorted(info, key=itemgetter(0))
            cut_min = info[0][1]
            cut_max = info[1][1]
            cut_pmin = info[0][0]
            cut_pmax = info[1][0]
            low_method = 'min_point'
            high_method = 'max_point'
        else:
            raise NotImplementedError
        if cut_pmin <= p_min and p_max <= cut_pmax:
            # 全切割走了
            pass
        elif cut_pmin <= p_min < cut_pmax:
            s_now1.feature_fix(cut_max, low_method)
            s_left.append(s_now1)
        elif p_min < cut_pmin and cut_pmax < p_max:
            s_now1.feature_fix(cut_min, high_method)
            s_now2.feature_fix(cut_max, low_method)
            s_left.extend([s_now1, s_now2])
        elif cut_pmin < p_max <= cut_pmax:
            s_now1.feature_fix(cut_min, high_method)
            s_left.append(s_now1)
        else:
            # 没有进行切割
            s_left.append(s)
        return s_left

    @staticmethod
    def feature_calc(p1, p2, direction=None, locate=None):
        # 通过线段的两个端点按照position值来对点的顺序进行排序，同时计算线段的其他特征
        length = Point.distance(p1, p2, 'direct')
        if direction is None or locate is None:
            # 如果没有方向和locate，则用p1和p2重新计算
            # 方向为[0, 360)这个范围
            l_now_abc = Line.transform([p1, p2], '2p_ABC')
            locate, direction = Line.transform(l_now_abc, 'ABC_polar')
        else:
            # 如果没有方向和locate，则不用计算，p1和p2的信息为辅
            pass
        # 计算特征
        p1, p2, direction, p_min, p_max = Segment.feature_calc_shadow(p1, p2, direction)
        return p1, p2, length, locate, direction, p_min, p_max

    @staticmethod
    def feature_calc_shadow(p1, p2, direction):
        # 计算p1, p2投影到法线方向为direction的线段上的线段特征
        position1 = Point.position(p1, direction)
        position2 = Point.position(p2, direction)
        ind_p_min = np.argmin([position1, position2])
        if ind_p_min == 0:
            p_min, p_max = position1, position2
        else:
            p_min, p_max = position2, position1
            p2, p1 = p1, p2
        return p1, p2, direction, p_min, p_max

    @staticmethod
    def length(s):
        p1, p2 = s
        return Point.distance(p1, p2, 'direct')

    @staticmethod
    def interpoint(s1, s2, flag_cut=True):
        '''
        计算线段是否存在交点。先考虑是否存在共同点，然后再考虑是否不平行，最后计算交点
        :param s1: segment1
        :param s2: segment2
        :return:
        '''
        # s1_now = eval(str(s1))
        # s2_now = eval(str(s2))
        p11, p12 = s1
        p21, p22 = s2
        tuple_list = list_tuplize([p11, p12, p21, p22])
        tuple_unique_list = list_unique(tuple_list)
        if len(set(tuple_list)) < 4:
            # 存在共同点吗，则将共同点作为交点
            count_info = [[tuple_list.count(p), p] for p in tuple_unique_list]
            count_info.sort(key=lambda info: info[0], reverse=True)
            _, p_common = count_info[0]
            return p_common
        else:
            l1 = Line.transform(s1, '2p_ABC')
            l2 = Line.transform(s2, '2p_ABC')
            direction1 = Line.get_direction(l1)
            direction2 = Line.get_direction(l2)
            if Angle.diff(direction1, direction2) < angle_tol:
                # 角度差距太小，不存在直线交点
                return -1
            else:
                # flag1 = Point.in_segment(s1_now, p_inter)
                # flag2 = Point.in_segment(s2_now, p_inter)
                p_inter = Line.interpoint(l1, l2)
                flag1 = Point.in_rectangle(p_inter, Segment.range(s1))
                flag2 = Point.in_rectangle(p_inter, Segment.range(s2))
                if flag1 == 1 and flag2 == 1:
                    # 存在交点
                    if flag_cut:
                        return Point.float_cut(p_inter)
                    else:
                        return p_inter
                else:
                    # 不存在交点
                    return -1

    @staticmethod
    def interpoint_within_error(s1, s2, method, angle_threshold=angle_tol):
        '''
        在角度误差内计算线段是否存在交点
        :param s1: segment1
        :param s2: segment2
        :param method: l_l: 计算两条线段所在的直线是否产生交点
                       s_s: 计算两条线段是否产生交点
                       l_s: 就散第一条线段是直线，第二条是线段下是否产生交点
        :return:
        '''
        direction1 = s1.direction
        direction2 = s2.direction
        l1 = Line.transform(s1, '2p_ABC')
        l2 = Line.transform(s2, '2p_ABC')
        p_inter = Line.interpoint(l1, l2)
        if Angle.diff_real(direction1, direction2, np.pi) > angle_threshold:
            # 如果角度误差大于阈值
            if p_inter == -1:
                # 不存在直线交点
                return -1
            else:
                if method == 'l_l':
                    return p_inter
                elif method == 's_s':
                    flag1 = Point.in_rectangle(p_inter, s1.range())
                    flag2 = Point.in_rectangle(p_inter, s2.range())
                    if flag1 == 1 and flag2 == 1:
                        # 存在交点
                        return p_inter
                    else:
                        # 不存在交点
                        return -1
                elif method == 'l_s':
                    flag2 = Point.in_rectangle(p_inter, s2.range())
                    if flag2 == 1:
                        # 存在交点
                        return p_inter
                    else:
                        # 不存在交点
                        return -1
        else:
            # 交点过远，则直接设为不存在交点
            return -1

    @staticmethod
    def shadow(s, direction, locate):
        # 计算线段s在法线方向为direction，高度为locate的线上的投影
        p1 = s.p1
        p2 = s.p2
        _, _, direction, p_min, p_max = Segment.feature_calc_shadow(p1, p2, direction)
        s_new = Segment.build_from_direction(direction)
        Segment.fix(s_new, locate, 'locate')
        Segment.feature_fix(s_new, [p_min, p_max], 'minmax')
        return s_new

    @list_join
    @list_strize
    def show(self):
        return [self.p1, self.p2, self.length, self.direction, self.locate,
                self.p_min, self.p_max, self.ind]


# 线段集合函数
class Segments(list):
    # 线段集合模块
    def __init__(self, s_list, hold=False):
        super().__init__([])
        for s in s_list:
            # 不新建线段，保留线段的信息
            self.append(Segment(s, hold))
        self.reindex()

    def calc_ori_dict(self):
        # 计算原始线段对应当前哪些线段
        ori_dict = {}
        for i, s in enumerate(self):
            ori_ind = s.ori_ind
            dict_add(ori_dict, ori_ind, s.ind, 'append')
        return ori_dict

    def calc_ss_relation(self):
        # 计算线段相连（通过点）的关系
        # 在线段中添加变量
        # linked_list：该骨架跟哪些索引的骨架相连
        # linked_dict：该骨架跟特定索引的骨架相连的原本位置
        Segments.reset(self)
        ps_dict = {}
        for s in self:
            p1 = tuple(s.p1)
            p2 = tuple(s.p2)
            dict_add(ps_dict, p1, s.ind, 'append')
            dict_add(ps_dict, p2, s.ind, 'append')
        for s in self:
            p1 = tuple(s.p1)
            p2 = tuple(s.p2)
            s.linked_list = list_unique(ps_dict[p1] + ps_dict[p2])  # 线段与哪些线段相连
            s.linked_dict = {}  # 线段与线段之间的原本交点
            for i in ps_dict[p1]:
                s.linked_dict[i] = p1
            for i in ps_dict[p2]:
                s.linked_dict[i] = p2

    def close_origin(self):
        # 将多边形尽可能往原点靠，同时保持所有点都在第一象限上。
        x_min, x_max, y_min, y_max = self.range()
        s_list = copy.deepcopy(self)
        x_delta = -x_min
        y_delta = -y_min
        s_list.move((x_delta, y_delta))
        return s_list, x_delta, y_delta

    def convex(self):
        # 计算线段列表的凸包
        p_list = []
        for s in self:
            p_list.extend([s.p1, s.p2])
        p_list = Points(p_list)
        return p_list.convex()

    def is_interpoint_normal(self, method='end'):
        # 判断两条向量的交点是否正常。交点在向量的前方
        s1, s2 = self
        p_start1 = s1.p1_raw
        p_end1 = s1.p2_raw
        p_start2 = s2.p1_raw
        p_end2 = s2.p2_raw
        direction1 = s1.direction
        posi_end1 = Point.position(p_start1, direction1)
        posi_extend1 = Point.position(p_end1, direction1)
        if posi_extend1 < posi_end1:
            # 数值往小的方向走
            flag_neg1 = True
        else:
            flag_neg1 = False
        direction2 = s2.direction
        posi_end2 = Point.position(p_start2, direction2)
        posi_extend2 = Point.position(p_end2, direction2)
        if posi_extend2 < posi_end2:
            # 数值往小的方向走
            flag_neg2 = True
        else:
            flag_neg2 = False
        l1 = Line.transform(s1, '2p_ABC')
        l2 = Line.transform(s2, '2p_ABC')
        p_inter = Line.interpoint(l1, l2)
        status = False
        if p_inter != -1:
            posi1 = Point.position(p_inter, direction1)
            posi2 = Point.position(p_inter, direction2)
            # 计算是否相交在正常的地方
            if method == 'front':
                if flag_neg1:
                    if posi1 < posi_extend1:
                        # 表示相交在正常的地方
                        flag_normal1 = True
                    else:
                        flag_normal1 = False
                else:
                    if posi1 > posi_extend1:
                        # 表示相交在正常的地方
                        flag_normal1 = True
                    else:
                        flag_normal1 = False
                if flag_neg2:
                    if posi2 < posi_extend2:
                        # 表示相交在正常的地方
                        flag_normal2 = True
                    else:
                        flag_normal2 = False
                else:
                    if posi2 > posi_extend2:
                        # 表示相交在正常的地方
                        flag_normal2 = True
                    else:
                        flag_normal2 = False
            else:
                if flag_neg1:
                    if posi1 < posi_end1:
                        # 表示相交在正常的地方
                        flag_normal1 = True
                    else:
                        flag_normal1 = False
                else:
                    if posi1 > posi_end1:
                        # 表示相交在正常的地方
                        flag_normal1 = True
                    else:
                        flag_normal1 = False
                if flag_neg2:
                    if posi2 < posi_end2:
                        # 表示相交在正常的地方
                        flag_normal2 = True
                    else:
                        flag_normal2 = False
                else:
                    if posi2 > posi_end2:
                        # 表示相交在正常的地方
                        flag_normal2 = True
                    else:
                        flag_normal2 = False
            if flag_normal1 and flag_normal2:
                status = True
        return p_inter, status

    def move(self, p_delta):
        # 对线段列表当中的所有线段进行平移
        for s in self:
            s.move(p_delta)

    def range(self):
        # 计算线段列表的范围
        x_list = []
        y_list = []
        for s in self:
            for p in [s.p1, s.p2]:
                x, y = p
                x_list.append(x)
                y_list.append(y)
        x_list = sorted(x_list)
        y_list = sorted(y_list)
        x_min = x_list[0]
        x_max = x_list[-1]
        y_min = y_list[0]
        y_max = y_list[-1]
        return x_min, x_max, y_min, y_max

    def reindex(self):
        '''
        重置直线列表索引
        :return:
        '''

        for i, l in enumerate(self):
            l.ind = i

    def sort_for_weight(self, reverse=False):

        self.sort(key=lambda x: x.weight, reverse=reverse)
        Segments.reset(self)

    def to_pixel_locate(self):
        # 将线段列表变成像素坐标列表
        pixel_locate_list = []
        for s in self:
            pixel_locate_list.extend(s.to_pixel_locate())
        pixel_locate_list = list_unique(pixel_locate_list)
        return pixel_locate_list

    def weight_filter(self, fs):
        # 按照权重来筛选
        low, high, low_type, high_type = fs_process(fs)
        # 进行筛选
        if low_type == 0 and high_type == 0:
            list_new = [s for s in self if low < s.weight < high]
        elif low_type == 1 and high_type == 0:
            list_new = [s for s in self if low <= s.weight < high]
        elif low_type == 0 and high_type == 1:
            list_new = [s for s in self if low < s.weight <= high]
        elif low_type == 1 and high_type == 1:
            list_new = [s for s in self if low <= s.weight <= high]
        else:
            raise NotImplementedError
        self.__init__(list_new)

    def weight_init(self, method):
        # 权值初始化
        for s in self:
            s.weight_init(method=method)

    @staticmethod
    def array(segment_list):
        '''
        将线段集合array化
        :param segment_list: 线段列表
        :return:
        '''
        for i, l in enumerate(segment_list):
            eval(str(l.show()))

        line_array = np.array([eval(str(l.show())) for l in segment_list])
        return line_array

    @staticmethod
    @time_logger('线段方向强制吸附至最近的主方向')
    def direction_adsorb(segment_list, direction_list):
        '''
        线段方向强制吸附至最近的方向，线段绕中心旋转
        :param segment_list: 线段列表
        :param direction_list: 方向列表
        '''
        for s in segment_list:
            s.direction_adsorb(direction_list)
        Segments.reset(segment_list)

    @staticmethod
    def distance(s1, s2):
        # 计算两条线段之间的距离
        d1 = Segment.point_distance(s2, s1.p_mid)
        d2 = Segment.point_distance(s1, s2.p_mid)
        return (d1 + d2)/2

    @staticmethod
    def filter_length(segment_list, threshold):
        # 按照长度来筛选线段
        segment_list_new = [s for s in segment_list if s.length > threshold]
        Segments.reset(segment_list_new)
        return segment_list_new

    @staticmethod
    def fix_feature(segment_list, p_value, method):
        for s in segment_list:
            s.feature_fix(p_value, method)
        Segments.reset(segment_list)

    @staticmethod
    def generate(segment_raw_list):
        '''
        生成线段列表
        :param segment_raw_list: 线段原始列表
        :return:
        '''
        segment_list = []
        for i, l in enumerate(segment_raw_list):
            l_obj = Segment(l)
            segment_list.append(l_obj)
        Segments.reset(segment_list)
        return segment_list

    @staticmethod
    def group_parallel(segment_list, maxwidth=2, simthre=0.75):
        '''
        将线段划分为相近且平行的组别。最开始的应用是在语义识别转换。
        :param segment_list: 线段列表
        :param maxwidth: 划分为平行的最大距离
        :param simthre: 相似度阈值。大于这个相似度的两条线段才能看成为一组
        :return: parallel_group: 索引进行分类
        '''

        parallel_group = Grouper()
        # 临时添加变量
        for s in segment_list:
            s.cleared = 0  # 在计算过程中被清理的标志
        segment_array = Segments.array(segment_list)
        for i, s in enumerate(segment_list):
            if s.cleared == 0:  # 并没有被清理
                direction = s.direction
                locate = s.locate
                p_min = s.p_min
                p_max = s.p_max
                # 降低搜索范围
                slice = segment_array
                slice = slice[np.abs(locate - slice[:, S_LOCATE]) < maxwidth]  # 位置相近
                slice = slice[slice[:, S_DIRECTION] == direction]  # 同方向
                slice = slice[slice[:, S_P_MAX] > p_min]  # 重叠
                temp = slice[slice[:, S_P_MIN] < p_max]
                ind_set = set()
                for array_cmp in temp:
                    i_cmp = array_cmp[S_IND]
                    s_cmp = segment_list[i_cmp]
                    # 重叠程度足够高的情况下才能算是连续的语义
                    if Segments.calc_overlap(s, s_cmp)['iou'] > simthre:
                        ind_set.add(i_cmp)
                        s_cmp.clear = 1
                # 组别分组
                if len(ind_set) > 0:
                    parallel_group.add(ind_set)
        return parallel_group

    @staticmethod
    @time_logger('计算线段交点')
    def interpoint(segment_list, AngleRange=0.05):
        '''
        计算线段交点
        :param segment_list: 线段列表
        :param AngleRange: 角度误差范围
        :return:
        '''
        segment_array = Segments.array(segment_list)
        for s in segment_list:
            s.cleared = 0
            # 在计算前清零
            s.interpoint_list = []
            s.intersegment_list = []
        for segment in segment_list:
            direction = segment.direction
            slice = segment_array
            temp = slice[(slice[:, S_DIRECTION] < direction - AngleRange) |
                          (slice[:, S_DIRECTION] > direction + AngleRange)]  # 不同方向
            for s_array in temp:
                s_cmp = segment_list[s_array[S_IND]]
                if s_cmp.cleared == 0:
                    interpoint = Segment.interpoint(segment, s_cmp)
                    if interpoint != -1:
                        # 添加交点
                        segment.interpoint_list.append(tuple(interpoint))
                        s_cmp.interpoint_list.append(tuple(interpoint))
                        # 添加与自己相交的线段
                        segment.intersegment_list.append(s_cmp)
            segment.cleared = 1
        # 使得交点唯一
        for s in segment_list:
            s.interpoint_list = list_unique(s.interpoint_list)

    @staticmethod
    def interpoint_single(segment_list, s_n):
        '''
        计算线段与线段列表产生的交点
        :param segment_list: 线段列表
        :param s_n: 线段
        :return interpoint_list: 交点列表
        '''
        interpoint_list = []
        for s in segment_list:
            p1_t, p2_t = s
            p_inter = Segment.interpoint(s_n, s, False)
            if p_inter != -1:
                interpoint_list.append(p_inter)
        return interpoint_list

    @staticmethod
    # @time_logger('生成交点结构字典')
    def interpoint_struct(skeleton_list, method='interpoint', flag_cut=True, threshold=dis_tol):
        '''
        在线段集合中寻找封闭的多边形，利用交点来进行寻找
        :param skeleton_list: 骨架集合
        :param method: interpoint: 只是提取交点之间的骨架
                       all: 提取所有骨架
        :return:
        struct_dict  交点结构
        skeleton_list_new  新的骨架结构
        point_segment_dict  点-线段索引字典
        '''
        # 将长度为0的线段筛走
        delete_index = []
        for i, s in enumerate(skeleton_list):
            if s.length == 0:
                delete_index.insert(0, i)
        for i in delete_index:
            del skeleton_list[i]
        Segments.reset(skeleton_list)
        # 边的交点集合
        interpoint_dict = {}
        # 收集交点
        for i in range(len(skeleton_list) - 1):
            for j in range(i + 1, len(skeleton_list)):
                p11, p12 = s1 = skeleton_list[i]
                p21, p22 = s2 = skeleton_list[j]
                tuple_list = list_tuplize([p11, p12, p21, p22])
                flag_common = False
                for p_now in tuple_list:
                    if tuple_list.count(p_now) == 2:
                        # 有共同点
                        flag_common = True
                        dict_add(interpoint_dict, i, p_now, 'append')
                        dict_add(interpoint_dict, j, p_now, 'append')
                        break
                if flag_common is False:
                    # 没有共同点
                    interpoint = Segment.interpoint(s1, s2, flag_cut)
                    if interpoint != -1:
                        # 计算交点是否为线段的点之一
                        info_list = [[Point.distance(interpoint, p_tuple), p_tuple] for p_tuple in tuple_list]
                        info_list.sort(key=lambda x: x[0])
                        diff, p_now = info_list[0]
                        if diff < threshold:
                            # 如果差距小于误差，则判断为该点
                            dict_add(interpoint_dict, i, p_now, 'append')
                            dict_add(interpoint_dict, j, p_now, 'append')
                        else:
                            # 如果差距大于误差，则直接使用交点
                            dict_add(interpoint_dict, i, interpoint, 'append')
                            dict_add(interpoint_dict, j, interpoint, 'append')
        # 交点排序，创造交点结构字典，也就是计算每个交点的相邻交点
        skeleton_list_new = []
        struct_dict = {}
        point_segment_dict = {}  # 点-线段索引字典
        p_dict = Storer()
        for i_s, s in enumerate(skeleton_list):
            if i_s in interpoint_dict:
                sorted_list = interpoint_dict[i_s]
            else:
                sorted_list = []
            if method == 'all':
                # 每一条线都存在
                p1, p2 = s
                sorted_list.extend([p1, p2])
            if len(sorted_list) > 0:
                point_list = Points.sorted(sorted_list, 'line')  # 按照线的方法进行排序
                point_list = list_tuplize(point_list)
                # 按顺序放入struct_dict
                for ind_p, p in enumerate(point_list):
                    if ind_p > 0:
                        p_b = point_list[ind_p - 1]
                        if Point.distance(p, p_b) > 0:
                            # 防止有些点相等而产生长度为0的线段
                            dict_add(struct_dict, p, p_b, 'append')
                            dict_add(struct_dict, p_b, p, 'append')
                            p1 = p_dict.gen(Point(p))
                            p2 = p_dict.gen(Point(p_b))
                            s_new = Segment.build([p1, p2])
                            s_new.refer = s  # 记录这条线从属于哪条线
                            s_new.ori_ind = s.ind
                            skeleton_list_new.append(s_new)
                            # 保存点-线段字典
                            dict_add(point_segment_dict, p, s_new, 'append')
                            dict_add(point_segment_dict, p_b, s_new, 'append')
        Segments.reset(skeleton_list_new)
        return struct_dict, skeleton_list_new, point_segment_dict

    @staticmethod
    def calc_overlap(s1, s2):
        # 计算两条平行线段的重叠率
        min1 = s1.p_min
        max1 = s1.p_max
        min2 = s2.p_min
        max2 = s2.p_max
        if min1 < max2 and min2 < max1:
            # 重叠
            min_list = sorted([min1, min2])
            max_list = sorted([max1, max2])
            overlap = (max_list[0] - min_list[1])
            max_iou = overlap/max([max1 - min1, max2 - min2])
            rst = {'iou': overlap/(max_list[1] - min_list[0]),  # 重叠率
                   'p_min': min_list[1],  # 重叠部分的p_min
                   'p_max': max_list[0],  # 重叠部分的p_max
                   'max_iou': max_iou,    # 重叠部分占最长线段的比例
                   }
            return rst
        else:
            return {'iou': 0,
                    'p_min': 0,
                    'p_max': float('inf'),
                    'max_iou': 0}

    @staticmethod
    # @time_logger('计算线段集合主方向')
    def main_direction(segment_list):
        '''
        计算线段集合的主方向
        :param segment_list: 线段列表
        :return:
        '''
        diffthreshold = 0.15  # 角度最大误差
        info_list = []
        for l in segment_list:
            p1 = l.p1
            p2 = l.p2
            v = (p1, p2)
            info_list.append([Vector.roundangle(v, ((0, 0), (1, 0))) % math.pi, Segment.length(v)])
        return Angles.kmeans(info_list, diffthreshold, 1)[0]

    @staticmethod
    def __merge(segment_list, MinLength, MaxWidth, AngleRange, locate_method, SimThre=0):
        '''
        合并相近的线段
        :param segment_list: 线段列表
        :param MinLength: 所处理线段的最小长度
        :param MaxWidth: 所合并线段的最大距离
        :param AngleRange: 允许的最大角度误差
        :param: locate_method  计算合并线高度的方法
        :param SimThre: 相似度阈值
        :return: line_stayed  合并后的直线
        '''
        # 临时添加变量
        for l in segment_list:
            l.cleared = 0  # 在计算过程中被清理的标志
            l.stayed = 0  # 在计算过程中被保留的标志
        segment_array = Segments.array(segment_list)
        for i, term in enumerate(zip(segment_list, segment_array)):
            l, l_array = term
            if l.cleared == 0:  # 并没有被清理
                length = l.length
                direction = l.direction
                locate = l.locate
                p_min = l.p_min
                p_max = l.p_max
                if length >= MinLength:
                    # 降低搜索范围
                    slice = segment_array
                    slice = slice[np.abs(locate - slice[:, S_LOCATE]) < MaxWidth]  # 位置相近
                    slice = slice[slice[:, S_DIRECTION] <= direction + AngleRange]  # 同方向
                    slice = slice[slice[:, S_DIRECTION] >= direction - AngleRange]
                    slice = slice[slice[:, S_P_MAX] > p_min]  # 重叠
                    temp = slice[slice[:, S_P_MIN] < p_max]
                    # 找出相似度高的出来进行合并
                    cmp_list = []
                    for s_cmp in temp:
                        p_min_c = s_cmp[S_P_MIN]
                        p_max_c = s_cmp[S_P_MAX]
                        length_c = s_cmp[S_LENGTH]
                        d = min([p_max_c - p_min, p_max - p_min_c])
                        sim = max([d / length, d / length_c])
                        if sim > SimThre:
                            cmp_list.append(s_cmp)
                    if len(cmp_list) >= 2:
                        # 只有可以合并的线大于两条，才能进行合并。合并以后剩下自己
                        # 计算范围
                        min_new = min([s_n[S_P_MIN] for s_n in cmp_list])
                        max_new = max([s_n[S_P_MAX] for s_n in cmp_list])
                        l.feature_fix([min_new, max_new], 'minmax')  # 调整position
                        if locate_method == 'weighted':
                            len_sum = sum([s_n[S_LENGTH] for s_n in cmp_list])
                            locate_m_len_sum = sum([s_n[S_LENGTH] * s_n[S_LOCATE] for s_n in cmp_list])
                            locate_new = locate_m_len_sum / len_sum  # 加权高度
                            l.feature_fix(locate_new, 'locate')  # 调整locate
                        elif locate_method == 'max_length':
                            len_list = [s_n[S_LENGTH] for s_n in cmp_list]
                            locate_new = cmp_list[len_list.index(max(len_list))][S_LOCATE]
                            l.feature_fix(locate_new, 'locate')  # 调整locate
                        # 线段清理
                        for s_n in cmp_list:
                            segment_list[s_n[S_IND]].cleared = 1  # 被清理的线段下一次不需要再进行计算
                            segment_list[s_n[S_IND]].cleanor = i
                        l.stayed = 1  # 这条线段被保留下来
                    else:
                        # 没有可以合并的，直接保留
                        l.stayed = 1
        # 找到留下来的直线
        line_stayed = [l for l in segment_list if l.stayed == 1]
        Segments.reset(line_stayed)
        return line_stayed

    @staticmethod
    def __merge2(segment_list, MinLength, MaxWidth, AngleRange, locate_choice='max_length', iou_thre=0, weight_method=None):
        '''
        合并相近的线段
        :param segment_list: 线段列表
        :param MinLength: 所处理线段的最小长度
        :param MaxWidth: 所合并线段的最大距离
        :param AngleRange: 允许的最大角度误差
        :param: locate_method  计算合并线高度的方法
        :param SimThre: 相似度阈值
        :return: line_stayed  合并后的直线
        '''
        # 临时添加变量
        if locate_choice in ['max_length', 'mid']:
            segment_info = [[segment.length, segment] for segment in segment_list]
        elif locate_choice == 'max_weight':
            segment_info = [[segment.weight, segment] for segment in segment_list]
        else:
            raise NotImplementedError
        segment_info = sorted(segment_info, key=lambda x: x[0], reverse=True)
        segment_list_now = [term[1] for term in segment_info]
        Segments.reset(segment_list_now)
        for l in segment_list_now:
            l.cleared = 0  # 在计算过程中被合并清理的标志
            l.stayed = 0  # 在计算过程中被保留的标志
        for i, s in enumerate(segment_list_now):
            if s.cleared == 0:  # 并没有被清理
                length = s.length
                direction = s.direction
                if length >= MinLength:
                    # 降低搜索范围
                    temp = [s_cmp for s_cmp in segment_list_now
                            if Angle.diff_real(s_cmp.direction, direction) <= AngleRange and
                            Segments.distance(s_cmp, s) <= MaxWidth and
                            s_cmp.cleared == 0]
                    cmp_list = []
                    for s_cmp in temp:
                        # 进行合并
                        s1, s2 = Segments.pairize(s, s_cmp, method='first', flag_copy=False)
                        if Segments.calc_overlap(s1, s2)['iou'] > iou_thre:
                            cmp_list.append(s2)
                    if len(cmp_list) >= 2:
                        # 只有可以合并的线大于两条，才能进行合并。合并以后剩下自己
                        # 计算范围
                        min_new = min([s_n.p_min for s_n in cmp_list])
                        max_new = max([s_n.p_max for s_n in cmp_list])
                        s.feature_fix([min_new, max_new], 'minmax')  # 调整position
                        # 以最长的线的高度作为基准线
                        if locate_choice == 'max_length':
                            sort_list = [s_n.length for s_n in cmp_list]
                            locate_new = cmp_list[sort_list.index(max(sort_list))].locate
                        elif locate_choice == 'max_weight':
                            sort_list = [s_n.weight for s_n in cmp_list]
                            locate_new = cmp_list[sort_list.index(max(sort_list))].locate
                        elif locate_choice == 'mid':
                            locate_new = np.mean([s_n.locate for s_n in cmp_list])
                        else:
                            raise NotImplementedError
                        s.feature_fix(locate_new, 'locate')  # 调整locate
                        # 线段清理
                        for s_n in cmp_list:
                            s_n.cleared = 1  # 被清理的线段下一次不需要再进行计算
                            s_n.cleanor = i
                        if weight_method == 'sum':
                            # 权重进行累加
                            s.weight = sum([s_n.weight for s_n in cmp_list])
                        s.stayed = 1  # 这条线段被保留下来
                    else:
                        # 没有可以合并的，直接保留
                        s.stayed = 1
        # 找到留下来的直线
        line_stayed = [l for l in segment_list if l.stayed == 1]
        Segments.reset(line_stayed)
        return line_stayed

    @staticmethod
    def merge(segment_list, MinLength, MaxWidth, AngleRange, locate_method='max_length', SimThre=0, Times=1):
        '''
        合并相近的线段
        :param segment_list: 线段列表
        :param MinLength: 所处理线段的最小长度
        :param MaxWidth: 所合并线段的最大距离
        :param AngleRange: 允许的最大角度误差
        :param Times: 重复次数
        :return:
        '''
        segment_list_new = segment_list
        for i in range(Times):
            segment_list_new = Segments.__merge(segment_list_new, MinLength, MaxWidth, AngleRange, locate_method, SimThre)
        return segment_list_new

    @staticmethod
    def merge_extend(segment_list, extend=10):
        '''
        # 将靠近的线进行合并在一起，方法是先延长线，然后再进行合并
        :param segment_list: 待合并直线
        :return: segment_list: 完成合并直线
        '''

        # 延长线段
        Segments.fix_feature(segment_list, extend, 'extend')
        # 合并线段，靠近的小线段合并
        segment_list = Segments.merge_rep(segment_list, MinLength=1, MaxWidth=10,
                                          AngleRange=10 / 180 * math.pi, Times=3)
        # 缩短线段
        Segments.fix_feature(segment_list, -extend, 'extend')
        return segment_list

    @staticmethod
    def merge_rep(segment_list, MinLength, MaxWidth, AngleRange, locate_choice='max_length', iou_thre=0, weight_method=None, Times=1):
        '''
        合并相近的线段
        :param segment_list: 线段列表
        :param MinLength: 所处理线段的最小长度
        :param MaxWidth: 所合并线段的最大距离
        :param AngleRange: 允许的最大角度误差
        :param Times: 重复次数
        :return:
        '''
        segment_list_new = segment_list
        for i in range(Times):
            segment_list_new = Segments.__merge2(segment_list_new, MinLength, MaxWidth, AngleRange, locate_choice,
                                                 iou_thre, weight_method)
        return segment_list_new

    @staticmethod
    def overlap_extract(segment1_old, segment2_old):
        '''
        计算两条平行线段的重叠部分，同时切出非重叠部分
        :param segment1_old: 线段1
        :param segment2_old: 线段2
        :return: d: 两条线段间的高度差
        :return: lm: 两条线段的position重叠部分
        :return: lr: 两条线断各自被切割出的部分
        :return: iou: 重叠部分占占据部分的比例
        '''
        segment1 = copy.deepcopy(segment1_old)
        segment2 = copy.deepcopy(segment2_old)
        min1 = segment1.p_min
        max1 = segment1.p_max
        min2 = segment2.p_min
        max2 = segment2.p_max
        locate1 = segment1.locate
        locate2 = segment2.locate
        p_min = min([min1, min2])
        p_max = max([max1, max2])
        d = np.abs(locate1 - locate2)
        if min1 < max2 and min2 < max1:
            # 重叠的情况
            l1_r = []
            l2_r = []
            if max1 > max2:
                lr = segment1.cut(max2, 'max')
                l1_r.append(lr)
            elif max1 < max2:
                lr = segment2.cut(max1, 'max')
                l2_r.append(lr)
            if min1 > min2:
                lr = segment2.cut(min1, 'min')
                l2_r.append(lr)
            elif min1 < min2:
                lr = segment1.cut(min2, 'min')
                l1_r.append(lr)
            lm = [segment1, segment2]
            lr = [l1_r, l2_r]
            iou = segment1.length/(p_max - p_min)
        else:
            lm = []
            lr = []
            iou = 0
        return d, lm, lr, iou

    @staticmethod
    @time_logger('匹配平行线段')
    def parallelize(segment_list, MinLength, MinWidth, MaxWidth, AngleRange, method='overlap'):
        '''
        将线段匹配成平行线段
        :param segment_list: 线段列表
        :param MinLength: 所处理线段的最小长度
        :param MinWidth: 所平行线段的最小距离
        :param MaxWidth: 所平行线段的最大距离
        :param AngleRange: 允许的最大角度误差
        :return:
        '''
        segment_info = [[segment.locate, segment] for segment in segment_list]
        segment_info = sorted(segment_info, key=lambda x: x[0])
        segment_list_now = [term[1] for term in segment_info]
        Segments.reset(segment_list_now)
        segment_array_now = Segments.array(segment_list_now)
        for s in segment_list_now:
            s.cleared = 0
        pp_list = []
        for i, s in enumerate(segment_list_now):
            if s.cleared == 0:
                length = s.length
                direction = s.direction
                locate = s.locate
                p_min = s.p_min
                p_max = s.p_max
                if length >= MinLength:
                    # 降低搜索范围
                    slice = segment_array_now
                    slice = slice[np.abs(locate - slice[:, S_LOCATE]) <= MaxWidth]  # 位置相近
                    slice = slice[np.abs(locate - slice[:, S_LOCATE]) >= MinWidth]
                    slice = slice[slice[:, S_DIRECTION] <= direction + AngleRange]  # 同方向
                    slice = slice[slice[:, S_DIRECTION] >= direction - AngleRange]
                    slice = slice[slice[:, S_P_MAX] > p_min]  # 重叠
                    temp = slice[slice[:, S_P_MIN] < p_max]
                    # temp: 可以进行匹配的线段
                    for s_array in temp:
                        s_cmp = segment_list_now[s_array[S_IND]]
                        if s_cmp.cleared == 0:
                            # 如果此前这条线没有被匹配过
                            s1, s2 = Segments.pairize(s, s_cmp)
                            if method == 'overlap':
                                # 提取重叠部分
                                _, lm, _, _ = Segments.overlap_extract(s1, s2)
                            elif method == 'union':
                                # 提取联合部分
                                lm = Segments.union_extract(s1, s2)
                            else:
                                raise NotImplementedError
                            if len(lm) > 0:
                                pp_list.append(Parallel(lm))
                # 清掉这条线段，以后不需要再处理
                s.cleared = 1
        return pp_list

    @staticmethod
    @time_logger('匹配平行线段')
    def parallelize2(segment_list, MinLength, MaxWidth, AngleRange, method='overlap'):
        # 计算平行线
        '''
        将线段匹配成平行线段
        :param segment_list: 线段列表
        :param MinLength: 所处理线段的最小长度
        :param MaxWidth: 所平行线段的最大距离
        :param AngleRange: 允许的最大角度误差
        :return:
        '''
        segment_info = [[segment.locate, segment] for segment in segment_list]
        segment_info = sorted(segment_info, key=lambda x: x[0])
        segment_list_now = [term[1] for term in segment_info]
        Segments.reset(segment_list_now)
        for s in segment_list_now:
            s.cleared = 0
        pp_list = []
        for i, s in enumerate(segment_list_now):
            if s.cleared == 0:
                length = s.length
                direction = s.direction
                if length >= MinLength:
                    # 降低搜索范围
                    temp = [s_cmp for s_cmp in segment_list_now
                            if Angle.diff_real(s_cmp.direction, direction) <= AngleRange and
                            Segments.distance(s_cmp, s) <= MaxWidth and
                            s_cmp.cleared == 0 and s_cmp != s]
                    # temp: 可以进行匹配的线段
                    for s_cmp in temp:
                        # 如果此前这条线没有被匹配过
                        s1, s2 = Segments.pairize(s, s_cmp)
                        if method == 'overlap':
                            # 提取重叠部分
                            _, lm, _, _ = Segments.overlap_extract(s1, s2)
                        elif method == 'union':
                            # 提取联合部分
                            lm = Segments.union_extract(s1, s2)
                        else:
                            raise NotImplementedError
                        if len(lm) > 0:
                            pp_list.append(Parallel(lm))
                # 清掉这条线段，以后不需要再处理
                s.cleared = 1
        return pp_list

    @staticmethod
    def pairize(s1_old, s2_old, method='max', flag_copy=True):
        '''
        将两条线段做成平行线
        :param s1_old: 线段1
        :param s2_old: 线段2
        :param method: max: 以最长的线段作为基础
                       first: 以第一条线段为基础
        :return:
        '''

        if flag_copy:
            s1 = copy.deepcopy(s1_old)
            s2 = copy.deepcopy(s2_old)
        else:
            s1 = s1_old
            s2 = s2_old
        if method == 'max':
            if s1.length >= s2.length:
                s_base = s1
                s_other = s2
            else:
                s_base = s2
                s_other = s1
        elif method == 'first':
            s_base = s1
            s_other = s2
        else:
            raise NotImplementedError
        direction = Angle.normal_calc(s_base.direction)
        s_other.direction_adsorb([direction])
        return s1, s2

    @staticmethod
    def plot(segment_list, fig=[], texted=False, y_reverse=False, **kwargs):
        if fig == []:
            fig = plt.figure()
        for i, l in enumerate(segment_list):
            if texted:
                text = r's' + str(l.ind)
            else:
                text = ''
            kwargs['text'] = text
            kwargs['fig'] = fig
            l.plot(**kwargs)
        plt.axis('scaled')
        if y_reverse:
            fig.gca().invert_yaxis()

    @staticmethod
    @time_logger('线段退行')
    def regress(skeleton_list, RegreeeLength):
        '''
        骨架退行，退行到交点
        :param skeleton_list: 骨架列表
        :param RegreeeLength:  退行距离
        :return:
        '''
        for s in skeleton_list:
            s.cleared = 0
            s.p_grow = []
        for s in skeleton_list:
            direction = s.direction
            p_min = s.p_min + RegreeeLength
            p_max = s.p_max - RegreeeLength
            if p_min < p_max:
                s.feature_fix([p_min, p_max], 'minmax')
                p_list = s.interpoint_list[:]
                position_p1 = Point.position(s.p1, direction)
                position_p2 = Point.position(s.p2, direction)
                position_left = [Point.position(p, direction) for p in p_list]
                position_list = [position_p1, position_p2] + position_left
                p_min_new = min(position_list)
                p_max_new = max(position_list)
                # 去除成为端点的交点。
                index_remove = []
                if p_min_new in position_left:
                    index_remove.append(position_left.index(p_min_new))
                if p_max_new in position_left:
                    index_remove.append(position_left.index(p_max_new))
                index_remove = sorted(index_remove, reverse=True)
                for ind in index_remove:
                    del s.interpoint_list[ind]
                s.feature_fix([p_min_new, p_max_new], 'minmax')
                if p_min_new in [position_p1, position_p2]:
                    s.p_grow.append(s.p1)
                if p_max_new in [position_p1, position_p2]:
                    s.p_grow.append(s.p2)
            else:
                s.cleared = 1
        skeleton_list_new = [s for s in skeleton_list if s.cleared == 0]
        Segments.reset(skeleton_list_new)
        return skeleton_list_new

    @staticmethod
    def reset(segment_list):
        '''
        重置直线列表
        :param segment_list: 线段列表
        :return:
        '''

        for i, l in enumerate(segment_list):
            l.ind = i

    @staticmethod
    def rotate_axis(segment_list, seta):
        '''
        线段集合进行坐标轴旋转，顺时针旋转seta角度
        :param segment_list: 线段列表
        :param seta: 逆时针旋转的角度
        :return:
        '''
        for l in segment_list:
            l.rotate_axis(seta)
        Segments.reset(segment_list)

    @staticmethod
    def rotate_center(segment_list, p_center, seta):
        '''
        线段集合进行中心点旋转，顺时针旋转seta角度
        :param segment_list: 线段列表
        :param p_center: 中心点坐标
        :param seta: 逆时针旋转的角度
        :return:
        '''
        for l in segment_list:
            l.rotate_center(p_center, seta)
        Segments.reset(segment_list)

    @staticmethod
    def sort_for_length(segment_list):
        '''
        按照长度对线段列表进行排序
        :param segment_list: 线段列表
        :return:
        '''
        segment_list_new = sorted(segment_list, key=lambda x: x.length, reverse=True)
        Segments.reset(segment_list_new)
        return segment_list_new

    @staticmethod
    def union_extract(segment1_old, segment2_old):
        '''
        计算两条线段的position的联合
        :param segment1_old: 线段1
        :param segment2_old: 线段2
        :return: lm: 返回两条线段联合后的结果
        '''
        segment1 = copy.deepcopy(segment1_old)
        segment2 = copy.deepcopy(segment2_old)
        min1 = segment1.p_min
        max1 = segment1.p_max
        min2 = segment2.p_min
        max2 = segment2.p_max
        if min1 < max2 and min2 < max1:
            max_new = np.max([max1, max2])
            min_new = np.min([min1, min2])
            segment1.feature_fix([min_new, max_new], 'minmax')
            segment2.feature_fix([min_new, max_new], 'minmax')
            lm = [segment1, segment2]
        else:
            lm = []
        return lm


# 骨架
class Skeleton(Segment):
    # 骨架是一种更加规范的线段。在骨架中所有的线段都共享端点
    # 命名为骨架是因为骨架这个词体现了结构的含义。
    pass


# 骨架列表
class Skeletons(Segments):

    def __init__(self, s_list=(), hold=True, collect=True):
        '''
        骨架初始化
        :param s_list: 骨架列表原始数据。其中每一条骨架中的点只要坐标相同，id是一致的
        :param hold: True，保留原来的线段信息。False，不保留。默认为True，能够保留线段
        集合原来的信息。
        :param collect: True: 统一生成点 False: 分开生成点
        '''
        super().__init__([])
        p_dict = Storer()
        for s in s_list:
            if hold:
                self.append(s)
            else:
                if collect:
                    p1 = p_dict.gen(Point(s.p1))
                    p2 = p_dict.gen(Point(s.p2))
                else:
                    p1 = Point(s.p1)
                    p2 = Point(s.p2)
                self.append(Segment.build([p1, p2]))
        self.update_data()
        self.reindex()

    def adjust_point_from_direction(self, direction_list):
        # 通过调整点来调整骨架至规定的方向
        # 依次调整骨架方向
        p_dict = self.p_dict
        pp_dict = self.pp_dict
        p_list = []
        for p in p_dict:
            p_list.append(p_dict[p])
        stable_set = set()
        while True:
            # 选择一个位置在最下方的点进行计算
            if len(p_list) > 0:
                p_list.sort(key=lambda x: (x[1], x[0]))
                p_now = p_list.pop(0)
            else:
                # 已经没有可计算的点
                break
            if len(stable_set) == 0:
                # 如果一个点都没有，则直接将这个点固定下来
                stable_set.add(id(p_now))
            else:
                p_next_list = pp_dict[tuple(p_now)]
                id_set = set()
                for p in p_next_list:
                    id_set.add(id(p_dict[p]))
                linked_id_set = stable_set & id_set
                if len(linked_id_set) == 0:
                    # 如果与stable_list中的点没有任何联系，则直接固定下来
                    stable_set.add(id(p_now))
                else:
                    # 如果与stable_list中的点有联系，那么则与最近的点进行方向纠正
                    linked_list = []
                    for i_p in linked_id_set:
                        linked_list.append(id_to_object(i_p))
                    dist_list = [Point.distance(p_now, p_next) for p_next in linked_list]
                    p_near = p_dict[tuple(linked_list[np.argmin(dist_list)])]
                    # 计算线段方向
                    angle = Vector.angle([p_near, p_now])
                    # 计算线段方向与哪个主方向靠近
                    diff_list = [Angle.diff_real(angle, direction, 2 * math.pi) for direction in direction_list]
                    # 计算新角度，以及旋转角度后的新端点
                    direction_new = direction_list[diff_list.index(min(diff_list))]
                    # 计算p_now新的位置
                    l = Line.transform((p_near, direction_new), 'pseta_ABC')
                    new_site = Line.verticalpoint(p_now, l)
                    # p_now移动到新的位置
                    p_now.transfer(new_site)
                    # p_now进入稳定列表中
                    stable_set.add(id(p_now))
                    # 更新数据
                    self.update_data()
                    p_dict = self.p_dict
                    pp_dict = self.pp_dict

    def adjust_fork_skeleton_direction(self, direction_list):
        # 调整经过分叉点的骨架的方向至direction_list方向序列
        # 其中，从分叉点引出来的只有一条线的不进行合并

        def walked_record(walked_set, p, p_now):
            # 记录行走路线，同时将反过来的路线标记为已行走
            # 从p点出发，到p_now结束
            walked_set.add((p_now, p))
            walked_set.add((p, p_now))

        def search(p_now, walked_set, s_list):
            # 根据点p_now进行搜索
            while True:
                if len(pp_dict[p_now]) >= 3:
                    next_list2 = pp_dict[p_now]
                    info_list = []
                    for p_next2 in next_list2:
                        if (p_now, p_next2) not in walked_set:
                            # 没有走过，并且是分叉点
                            direction_next = direction_dict[(p_now, p_next2)]
                            diff_next = Angle.diff_real(direction_next, direction_now, np.pi)
                            info_list.append([p_next2, diff_next])
                    if len(info_list) > 0:
                        info_list.sort(key=lambda x: x[1])
                        p_raw, diff_raw = info_list[0]
                        if diff_raw == 0:
                            # 找到一个方向
                            walked_record(walked_set, p_now, p_raw)
                            s_list.append(pps_dict[(p_raw, p_now)])
                            p_list.append(p_raw)
                            p_now = p_raw
                        else:
                            # 找不到可以前进的点
                            break
                    else:
                        # 找不到可以前进的点
                        break
                else:
                    break

        # 计算交点字典（键为交点，值为线段索引）
        Segments.reset(self)
        self.update_data()
        ps_dict = self.ps_dict
        inter_dict = {}
        for p in ps_dict:
            inter_dict[p] = [id(s) for s in ps_dict[p]]

        pp_dict = self.pp_dict
        pps_dict = self.pps_dict
        direction_dict = {}  # 线段方向列表
        for s in self:
            p1, p2 = s
            p1 = tuple(p1)
            p2 = tuple(p2)
            direction = s.direction
            direction_now = direction_list[np.argmin([Angle.diff_real(dire, direction, 2 * np.pi) for dire in direction_list])]
            direction_dict[(p1, p2)] = direction_now
            direction_dict[(p2, p1)] = direction_now

        delete_index_list = []
        walked_set = set()
        skeleton_big_list = []
        collasp_dict = {}  # 新线id对应哪些旧线id
        for p in pp_dict:
            if len(pp_dict[p]) >= 3:
                # print(p)
                # 分叉点
                next_list = pp_dict[p]
                for p_next in next_list:
                    if (p, p_next) not in walked_set:
                        s_list = []  # 所寻找的线段
                        p_list = []  # 所经过的点
                        direction_now = direction_dict[(p, p_next)]
                        walked_record(walked_set, p, p_next)
                        s_list.append(pps_dict[(p, p_next)])
                        p_list.extend([p, p_next])
                        # 向前找
                        p_now = p_next
                        search(p_now, walked_set, s_list)
                        # 向后找
                        info_list = []
                        for p_next2 in next_list:
                            if (p_next2, p) not in walked_set:
                                # 没有走过，并且是分叉点
                                direction_next = direction_dict[(p_next2, p)]
                                diff_next = Angle.diff_real(direction_next, direction_now, np.pi)
                                info_list.append([p_next2, diff_next])
                        if len(info_list) > 0:
                            info_list.sort(key=lambda x: x[1])
                            p_now, diff_raw = info_list[0]
                            if diff_raw == 0:
                                walked_record(walked_set, p_now, p)
                                s_list.append(pps_dict[(p, p_now)])
                                p_list.append(p_now)
                                search(p_now, walked_set, s_list)
                        if len(s_list) >= 1:
                            # 计算新的骨架
                            # s_list.sort(key=lambda x: x.length, reverse=True)
                            # locate_new = s_list[0].locate
                            # min_new = min([s.p_min for s in s_list])
                            # max_new = max([s.p_max for s in s_list])
                            # 计算中心点
                            p_list = Points.sorted(p_list, 'line')
                            p_mid = p_list[int((len(p_list) - 1)/2)]
                            position_list = [Point.position(p, direction_now) for p in p_list]
                            min_new = min(position_list)
                            max_new = max(position_list)
                            # 构造线段
                            s = Segment.build_from_direction(direction_now)
                            # 线段经过中心点
                            s.move(p_mid)
                            # s.feature_fix(locate_new, 'locate')
                            s.feature_fix([min_new, max_new], 'minmax')
                            skeleton_big_list.append(s)
                            # 记录索引
                            index_list = [s.ind for s in s_list]
                            delete_index_list.extend(index_list)
                            collasp_dict[id(s)] = [id(s) for s in s_list]

        # 合并为新的骨架序列
        skeleton_left_list = []
        for i, s in enumerate(self):
            if i not in delete_index_list:
                skeleton_left_list.append(s)
        skeleton_new_list = skeleton_big_list + skeleton_left_list
        Segments.reset(skeleton_new_list)
        # 替换inter_dict
        collasp_swap_dict = dict_swap_key_value(collasp_dict)  # 旧线id对应新线id
        key_list = list(inter_dict.keys())
        for p in key_list:
            id_list = inter_dict[p]
            id_list_new = []
            for id_now in id_list:
                if id_now in collasp_swap_dict:
                    # print('产生替换', id_now, '->', collasp_swap_dict[id_now])
                    id_list_new.append(id_to_object(collasp_swap_dict[id_now][0]).ind)
                else:
                    id_list_new.append(id_to_object(id_now).ind)
            inter_dict[p] = list_unique(id_list_new)
        # Segments.plot(skeleton_new_list, marked=True)
        return skeleton_new_list, inter_dict

    def count_point_linked_each_side(self, p, main_direction):
        '''
        计算一个点的非主干边在同一直线上延伸的最大骨架数量
        :param p: 出发点
        :param main_direction: 主干方向
        :return: found_slist_all: 搜索到的线段
                 found_plist_all: 搜索到的线段对应的点
                 linked_num: 可延伸数量
        '''
        pp_dict = self.pp_dict
        linked_num = 0
        found_slist_all = {}
        found_plist_all = {}
        for p_next in pp_dict[tuple(p)]:
            s_next = self.extract((p, p_next), 'pp')
            direction_now = s_next.direction
            if Angle.diff_real(direction_now, main_direction, math.pi) != 0:
                found_slist, found_plist = self.find_same_line(s_next)
                linked_num += len(found_slist)
                found_slist_all[direction_now] = found_slist
                found_plist_all[direction_now] = []
                for p_now in found_plist:
                    if tuple(p_now) != tuple(p):
                        found_plist_all[direction_now].append(p_now)
        return found_slist_all, found_plist_all, linked_num

    def clear_abnormal_stretch(self, bottom_threshold=Angle.to_value(75), top_threshold=Angle.to_value(105)):
        '''
        去除不正常胡须
        :param bottom_threshold: 胡须正常角度下限
        :param top_threshold: 胡须正常角度上限
        :return:
        '''

        skeleton_list = copy.deepcopy(self)
        pp_dict = skeleton_list.pp_dict
        pps_dict = skeleton_list.pps_dict
        while True:
            delete_dict = {}
            for p in pp_dict:
                if len(pp_dict[p]) == 1:
                    p_next = pp_dict[p][0]
                    s_now = pps_dict[(p, p_next)]
                    # 计算最小夹角
                    flag_clear = False
                    angle_min = None
                    if p_next in pp_dict:
                        next_list2 = pp_dict[p_next]
                        angle_list = []
                        for p_next2 in next_list2:
                            if p_next2 != p:
                                v1 = [p_next, p]
                                v2 = [p_next, p_next2]
                                angle = Vector.interangle(v1, v2)
                                angle_list.append(angle)
                        if len(angle_list) > 0:
                            angle_list.sort()
                            angle_min = angle_list[0]
                            if angle_min < bottom_threshold or angle_min > top_threshold:
                                flag_clear = True
                    if flag_clear:
                        # 将这条线去掉
                        ind = s_now.ind
                        dict_add(delete_dict, p_next, [ind, angle_min, s_now.length], 'append')
            if len(delete_dict) == 0:
                break
            else:
                delete_index = []
                for p_now in delete_dict:
                    info = delete_dict[p_now]
                    info.sort(key=lambda x: (x[1], x[2]))
                    delete_index.append(info[0][0])
                delete_index.sort(reverse=True)
                for i_delete in delete_index:
                    del skeleton_list[i_delete]
            skeleton_list_new = Skeletons(skeleton_list)
            skeleton_list = skeleton_list_new
            pp_dict = skeleton_list.pp_dict
            pps_dict = skeleton_list.pps_dict
        # 数据更新
        self.__init__(skeleton_list)

    def clear_short_stretch(self, pro_threshold=0.2):
        '''
        去除短的胡须
        :param pro_threshold: 胡须正常比例下限
        :return:
        '''

        skeleton_list = Skeletons(self, hold=False, collect=False)
        # 两边进行延长
        for s in skeleton_list:
            s.feature_fix(5, 'extend')
        pp_dict = self.pp_dict
        pps_dict = self.pps_dict
        delete_index = []
        for p in pp_dict:
            if len(pp_dict[p]) == 1:
                p_next = pp_dict[p][0]
                s_now = pps_dict[(p_next, p)]
                # 计算最接近的点的距离
                direction_now = Vector.angle([p_next, p])
                p_inter_list = []
                for s in skeleton_list:
                    p_inter = Segment.interpoint_within_error(s_now, s, 'l_s')
                    if p_inter != -1:
                        if tuple(p_inter) != tuple(p):
                            direction_inter = Vector.angle([p, p_inter])
                            if Angle.diff_real(direction_now, direction_inter) < angle_tol:
                                p_inter_list.append(p_inter)
                # 计算比例
                dist_list = []
                for p_inter in p_inter_list:
                    dist_list.append(Point.distance(p, p_inter))
                if len(dist_list) > 0:
                    dist_max = min(dist_list)
                    dist_now = Point.distance(p, p_next)
                    pro = dist_now / dist_max
                    if pro < pro_threshold:
                        # 将这条线去掉
                        delete_index.append(s_now.ind)
        # 删除线段
        delete_index = list_unique(delete_index)  # 对于单条线段，这条线段会出现两次
        delete_index.sort(reverse=True)
        for i_delete in delete_index:
            del self[i_delete]
        # 更新数据
        self.update_data()
        self.reindex()

    def clear_small_space(self, factor, areathre=0.5, method='interpoint'):
        '''
        通过删除线段来破坏小空间
        清理原理如下：
        将面积小于阈值的房间称为小房间。将小房间的邻接房间按照面积从大到小排序，判断清理边以后是否产生边数下降。
        如果产生边数下降，则清理邻接边。否则，如果它是边界房间，则把边界全部去掉。如果不是边界房间，则选择一个
        边数上升最小的邻接房间，删除邻接边。
        输入可以是任意线段集合
        :param factor:
        :param areathre:
        :return:
        '''
        # 将小房间的最短从属边删线段，达到破坏小房间的目的
        # 寻找房间
        _, skeleton_list, ps_dict = Segments.interpoint_struct(self, method)
        while True:
            Segments.reset(skeleton_list)
            room_list, _ = Skeletons.find_polygen(skeleton_list)  # 利用骨架寻找空间多边形
            # 筛选小空间
            room_delete_list = []
            room_ind_delete_list = []
            for i, room in enumerate(room_list):
                # 如果面积小于0.5平方，那么把这个房间去掉
                if Polygen.area(room) * factor * factor < 1000000 * areathre:
                    room_delete_list.append(room)
                    room_ind_delete_list.append(room.ind)
            if len(room_delete_list) <= 0:
                break
            wall_room_dict = Polygens.room_edge_classify(room_list)
            # 在每一个小房间中找到从属最短的骨架筛选出来准备删除
            skeleton_delete_list = []
            for room in room_delete_list:
                i_room = room.ind
                s_list = room.s_list
                room_next_list = []
                for s in s_list:
                    room_next_list.append(set(wall_room_dict[s.tuplify()]) - {i_room})
                next_all_list = []
                flag_board = False  # 房间是否接触边界区域
                for num_set in room_next_list:
                    next_all_list += list(num_set)
                    if len(num_set) == 0:
                        flag_board = True
                next_all_list = list_unique(next_all_list)
                # 按照房间大小进行排序
                next_all_info = []
                for i in next_all_list:
                    if i not in room_ind_delete_list:
                        # 只有不被删除的房间才需要计算是否边数下降
                        next_all_info.append([i, Polygen.area(room_list[i_room])])
                next_all_info.sort(key=lambda x: x[1], reverse=True)
                next_all_list = [info[0] for info in next_all_info]
                # 循环所有邻接房间，将相连线段尝试删除，看看能否被删除
                drop_flag = False  # 是否产生了边数下降
                s_delete_info = []
                for room_num in next_all_list:
                    # 计算边数
                    room_origin = room_list[room_num]
                    # 将多边形拆散为骨架
                    # 将同一方向且相连的骨架进行合并
                    room_skeleton = Skeletons(Polygens.skeletonize([room_origin]))
                    room_skeleton.merge_linked_same_direction()
                    num_origin = len(room_skeleton)
                    # 找到相连线段
                    s_delete_list = []
                    for i, num_set in enumerate(room_next_list):
                        if num_set == {room_num}:
                            s_delete_list.append(s_list[i].tuplify())
                    # 找到邻接房间和本房间的所有线段
                    room_stay_list = []
                    for s in wall_room_dict:
                        if room_num in wall_room_dict[s] or i_room in wall_room_dict[s]:
                            room_stay_list.append(s)
                    # 进行删除
                    skeleton_list_new2 = []
                    for s in skeleton_list:
                        if s.tuplify() not in s_delete_list and s.tuplify() in room_stay_list:
                            skeleton_list_new2.append(s)
                    room_now_list, _ = Skeletons.find_polygen(skeleton_list_new2)  # 利用骨架寻找空间多边形
                    # 计算当前边数
                    room_skeleton = Skeletons(Polygens.skeletonize(room_now_list))
                    room_skeleton.merge_linked_same_direction()
                    num_now = len(room_skeleton)
                    s_delete_info.append([num_now - num_origin, s_delete_list])
                    # 如果产生了边数下降，则可以进行边删除
                    if num_now < num_origin:
                        drop_flag = True
                        break
                if drop_flag is False:
                    # 如果有边界，则将边界线全都删除，否则不做处理
                    if flag_board:
                        # 删除所有临近边界区域的线段
                        s_delete_list = [s_list[i] for i, num_set in enumerate(room_next_list) if num_set == set()]
                    else:
                        if len(s_delete_info) > 0:
                            s_delete_info.sort(key=lambda x: x[0])
                            s_delete_list = s_delete_info[0][1]
                        else:
                            s_delete_list = []
                # 开始进行切割
                for s in s_delete_list:
                    p1, p2 = s
                    p1 = tuple(p1)
                    p2 = tuple(p2)
                    s_delete = id_to_object(
                        list(set(list_id(ps_dict[p1])) & set(list_id(ps_dict[p2])))[0])
                    # 记录被切割的部分
                    # if 'cut_set' in s_delete.refer.__dict__:
                    #     s_delete.refer.cut_set.add(s_delete.tuplify())
                    # else:
                    #     s_delete.refer.cut_set = {s_delete.tuplify()}
                    skeleton_delete_list.append(tuple(s_delete))
            # 进行删除
            skeleton_list_new2 = []
            for s in skeleton_list:
                if tuple(s) not in skeleton_delete_list:
                    skeleton_list_new2.append(s)
            skeleton_list = skeleton_list_new2
        self.__init__(skeleton_list)

    def extend_to_interpoint(self, inter_dict):
        '''
        已知线段集合，以及各线段的相交信息，让线段延长到相交点
        :param inter_dict: 相交点信息。键是相交点，值是线段的索引号
        :return:
        '''
        p_list_dict = {}
        line_dict = {}
        for s in self:
            p_list_dict[s.ind] = [s[0], s[1]]
            line_dict[s.ind] = Line.transform(s, '2p_ABC')
        for p in inter_dict:
            i_list = inter_dict[p]
            for i in range(len(i_list) - 1):
                for j in range(i + 1, len(i_list) - 1):
                    i_l1 = i_list[i]
                    i_l2 = i_list[j]
                    l1 = line_dict[i_l1]
                    l2 = line_dict[i_l2]
                    p_inter = Line.interpoint(l1, l2)
                    if p_inter != -1:
                        p_list_dict[i_l1].append(p_inter)
                        p_list_dict[i_l2].append(p_inter)
        # 构造新的线
        skeleton_list_new = []
        for s in self:
            p_list = p_list_dict[s.ind]
            p_list = Points.sorted(p_list, 'line')
            s_new = Segment((p_list[0], p_list[-1]))
            skeleton_list_new.append(s_new)
        # 产生新的骨架集合
        skeleton_list_new = Skeletons(skeleton_list_new, hold=False, collect=True)
        return skeleton_list_new

    def find_same_line(self, s):
        '''
        在骨架列表中寻找在同一直线上的骨架
        :param s: 骨架列表中的其中一条骨架s
        :return: found_slist: 返回与骨架s在同一直线上且有不中断相邻关系的所有骨架
                 found_plist: found_slist对应的所有点
        '''
        pp_dict = self.pp_dict
        p1 = tuple(s.p1)
        p2 = tuple(s.p2)
        direction = s.direction
        unwalk_plist = [p1, p2]
        found_slist = [s]
        found_plist = [s.p1, s.p2]
        while len(unwalk_plist) > 0:
            p = unwalk_plist.pop()
            n_plist = pp_dict[p]
            for p_next in n_plist:
                s_next = self.extract((p, p_next), 'pp')
                if s_next.direction == direction and s_next not in found_slist:
                    # 找到同伙
                    unwalk_plist.append(p_next)
                    found_slist.append(s_next)
                    found_plist.append(self.extract(p_next, 'p'))
        return found_slist, found_plist

    def find_linked_points(self, p, main_direction):
        # 寻找点p的非main_direction方向的所有关联点
        p_dict = self.p_dict
        pp_dict = self.pp_dict
        found_list = [p]
        search_list = [p]
        while len(search_list) > 0:
            p_now = search_list.pop()
            for p_next in pp_dict[tuple(p_now)]:
                if p_dict[p_next] not in found_list:
                    s_next = self.extract((p_now, p_next), 'pp')
                    direction_now = s_next.direction
                    if Angle.diff_real(direction_now, main_direction, math.pi) >= 1e-4:
                        # 方向不同
                        search_list.append(p_dict[p_next])
                        found_list.append(p_dict[p_next])
        return found_list

    def find_linked_points2(self, p, p_target):
        # 寻找点p移动到p_target时，所有必须移动的点
        p_dict = self.p_dict
        pp_dict = self.pp_dict
        move_dict = {tuple(p): Point.delta(p, p_target)}  # 点移动列表
        stable_list = [tuple(p), tuple(p_target)]  # 不需要移动的点
        search_list = [tuple(p)]
        while len(search_list) > 0:
            p_now = search_list.pop()
            for p_next in pp_dict[p_now]:
                if p_next not in stable_list:
                    # 找到p_next的所有线，然后计算一条最近的交点
                    p_now2 = copy.deepcopy(p_dict[p_now])
                    p_now2.move(move_dict[tuple(p_now2)])
                    # 计算p_next新点产生的
                    s1 = Segment.build((p_now, p_next))
                    direction1 = s1.direction
                    l1 = Line.transform((p_now2, Angle.normal_calc(direction1 % np.pi)), 'pseta_ABC')
                    intersect_list = []
                    for p_next2 in pp_dict[p_next]:
                        if p_next2 != p_now:
                            s2 = Segment.build((p_next, p_next2))
                            l2 = Line.transform(s2)
                            # 设得阈值大一些
                            interpoint = Line.interpoint_within_error(l1, l2, Angle.to_value(15))
                            if interpoint != -1:
                                d = Point.distance(p_now2, interpoint)
                                intersect_list.append([interpoint, d, p_next2])
                    if len(intersect_list) > 0:
                        p_target_now, d, p_direction = sorted(intersect_list, key=lambda x: x[1])[0]
                        # 计算移动距离
                        d_move = Point.distance(p_next, p_target_now)
                        if d_move > dis_tol:
                            # 获得新的位置
                            move_dict[p_next] = Point.delta(p_next, p_target_now)
                            stable_list.append(p_next)
                            # 计算
                            flag_change = False
                            for p_next2 in pp_dict[p_next]:
                                if p_next2 not in stable_list:
                                    direction_origin = Segment.build((p_next, p_next2)).direction
                                    direction_new = Segment.build((p_target_now, p_next2)).direction
                                    if Angle.diff_real(direction_origin, direction_new, math.pi) >= 1e-4:
                                        # 方向不同，则这个点就要进行处理
                                        flag_change = True
                            if flag_change:
                                search_list.append(p_next)
                    else:
                        # 这里有两种情况：
                        # 一是p_next是悬空点
                        # 二是pp_dict[p_next]只有一个点a，而与这个点a与下一个邻居形成的线平行
                        # 计算这个点在新线上的投影
                        p_target_now = Line.verticalpoint(p_next, l1)
                        d_move = Point.distance(p_next, p_target_now)
                        if d_move > dis_tol:
                            move_dict[p_next] = Point.delta(p_next, p_target_now)
        return move_dict

    def find_polygen(self):
        '''
        在线段集合中寻找所有多边形（保留点的信息），以及外轮廓
        :param segment_list: 线段列表
        :return:
        polygen_list: 多边形列表
        contour_list: 外轮廓列表
        '''
        struct_dict = {}
        for s in self:
            # 这一层有两个点的情况下，才能进行层的计算
            p1 = s.p1
            p2 = s.p2
            id_p1 = id(p1)
            id_p2 = id(p2)
            dict_add(struct_dict, id_p1, id_p2, 'append')
            dict_add(struct_dict, id_p2, id_p1, 'append')
        polygen_list = []  # 封闭多边形列表
        # 开始寻找封闭多边形
        while len(struct_dict) > 0:
            # 寻找坐标最低点（最外轮廓的一点），找到一个外轮廓则将内部的所有点清除
            p_list = []
            for key in struct_dict.keys():
                p_n = id_to_object(key)
                p_list.append(p_n)
            xy = np.array(p_list)
            x_min = np.min(xy[:, 0])
            xy_sub = xy[xy[:, 0] == x_min]
            y_min = np.min(xy_sub[:, 1])
            # 最外轮廓点
            p_start = p_list[np.where((xy[:, 0] == x_min) & (xy[:, 1] == y_min))[0][0]]
            # 从坐标最低点出发，寻找最外层轮廓
            p_next = p_start
            path = [p_next]
            end_flag = 0
            p_direction = (p_next[0] + 1, p_next[1])
            count2 = 0
            while end_flag == 0:
                if count2 == 1000:
                    print('计算最大轮廓迭代次数过多')
                    raise Exception
                count2 += 1
                id_next_list = struct_dict[id(p_next)]
                # roundangle_info_list包含两个数值，第一个是经过余弦公式算出来的，第二个是直接算正弦且按照区域来划分数值
                # 在右半区，这个数值为正弦值，在左半区，这个数值为正弦值的负数，当第一个数值一样的情况下看第二个数值的大小，小的表示roundangle更小
                roundangle_list = []
                for i_p, id_p in enumerate(id_next_list):
                    p = id_to_object(id_p)
                    roundangle = Vector.roundangle([tuple(p_next), tuple(p_direction)], [tuple(p_next), tuple(p)],
                                                   'zeronormal')
                    roundangle_list.append(roundangle)
                # 选择旋转角度最小的作为下个起点
                id_next_new = id_next_list[roundangle_list.index(max(roundangle_list))]
                # 从结构字典中删掉
                struct_dict[id(p_next)].remove(id_next_new)
                p_next = id_to_object(id_next_new)
                # 产生新的方向
                p_direction = path[-1]
                if p_next == p_start:
                    end_flag = 1
                else:
                    path.append(p_next)
            polygen_list.append(Polygen(path, hold=True))
            # 清理被清空的点
            for key in list(struct_dict.keys()):
                if len(struct_dict[key]) == 0:
                    del struct_dict[key]
        # 将点进行分组
        p_group_list = []
        for i_poly, path in enumerate(polygen_list):
            path_tuple = [tuple(p) for p in path]
            # 寻找所在的组
            group_in = -1
            for p in path_tuple:
                for i_group, info in enumerate(p_group_list):
                    _, p_group = info
                    if p in p_group:
                        group_in = i_group
                        break
                if group_in != -1:
                    break
            # 组信息更新
            if group_in == -1:
                # 没有找到组，新建一个集合
                p_group_list.append([[i_poly], set(path_tuple)])
            else:
                # 找到组，添加到原集合
                p_group_list[group_in][0].append(i_poly)
                p_group_list[group_in][1] = set(list(p_group_list[group_in][1]) + path_tuple)
        # # 计算最外层轮廓，并进行删除
        i_contour_list = []
        for info in p_group_list:
            i_poly_list, _ = info
            poly_info_list = []
            for i_poly in i_poly_list:
                poly_info_list.append([i_poly, Polygen.area(polygen_list[i_poly]), len(polygen_list)])
            # 找出面积最大的情况下，点数量最多的多边形，也就是外轮廓
            poly_info_list = sorted(poly_info_list, key=itemgetter(1, 2), reverse=True)
            i_contour_list.append(poly_info_list[0][0])
        # 删除外轮廓
        i_contour_list = sorted(i_contour_list, reverse=True)
        contour_list = []
        for i_poly in i_contour_list:
            contour_list.append(polygen_list[i_poly])
            del polygen_list[i_poly]
        if len(polygen_list) > 0:
            # 删掉外轮廓（位置为1的多边形即外轮廓）
            # del polygen_list[1]
            Polygens.reset(polygen_list)
            return polygen_list, contour_list
        else:
            # 没有找到多边形，返回空值
            return [], []

    def extract(self, data, method):
        '''
        在骨架中提取信息
        :param data: 数据
        :param method: p: 进行点转换，提取实际的点数据
                       s: 进行线段转换，提取实际的线段数据
                       pp: 将两个端点在骨架列表中找到相应的骨架
        :return:
        '''
        if method == 'p':
            # 进行点转换
            p_dict = self.p_dict
            return p_dict[tuple(data)]
        elif method == 's':
            # 进行线转换
            ss_dict = self.ss_dict
            return ss_dict[tuple(data)]
        elif method == 'pp':
            # 利用两个端点在骨架列表中找到相应的骨架
            ss_dict = self.ss_dict
            p1, p2 = data
            return ss_dict[Segment([p1, p2]).tuplify()]

    def merge_close_point(self, thre_d=5):
        '''
        对相近的点进行合并，计算移动骨架的代价，选择代价最小的方案进行合并。永远只有两个点之间的合并。
        :param skeleton_list: 骨架列表
        :param thre_d: 合并距离阈值
        :return: 骨架列表
        '''

        flag_end = 0
        while flag_end == 0:
            # 搜索需要合并的点，
            merge_list = []
            pp_dict = self.pp_dict
            for p in pp_dict:
                n_list = pp_dict[p]
                for p_next in n_list:
                    if Point.distance(p, p_next) <= thre_d:
                        # 发现需要合并的点
                        p1 = self.extract(p, 'p')
                        p2 = self.extract(p_next, 'p')
                        merge_list.append([[p1, p2], Point.distance(p1, p2)])
            if len(merge_list) == 0:
                flag_end = 1
            else:
                # 选择最近的两个点进行合并
                merge_list = sorted(merge_list, key=itemgetter(1))
                p1, p2 = merge_list[0][0]
                self.merge_point(p1, p2)
                # 更新骨架中所有点的数据，然后将长度为0的线段干掉
                self.update_point()

    def merge_linked_same_direction(self, threshold=1e-2):
        '''
        将同一方向且相连的骨架进行合并
        :param s_list: 骨架列表
        :return:
        '''
        ps_dict = self.ps_dict
        ss_dict = self.ss_dict
        # 将非分叉点上的同方向骨架进行合并
        # 按照点组合两条线段，只要同一直线则视为同组
        pair_group = Grouper()
        for p in ps_dict:
            if len(ps_dict[p]) == 2:
                s1, s2 = ps_dict[p]
                if Angle.diff_real(s1.direction, s2.direction) < threshold:
                    s1 = s1.tuplify()
                    s2 = s2.tuplify()
                    # 两条线段同组
                    pair_group.add({s1, s2})
        # 按组进行合并
        for group in pair_group.data:
            p_list = []
            for s_tuple in group:
                s = ss_dict[s_tuple]
                p_list.extend([s.p1, s.p2])
            p_list = Points.sorted(p_list, 'line')
            p1 = p_list[0]
            p2 = p_list[-1]
            skeleton_new = Segment.build([p1, p2])
            ss_dict[skeleton_new.tuplify()] = skeleton_new
            for skeleton_tuple in group:
                del ss_dict[skeleton_tuple]
        # 取出骨架
        skeleton_list = []
        for skeleton_tuple in ss_dict:
            skeleton_list.append(ss_dict[skeleton_tuple])
        self.__init__(skeleton_list)

    def merge_linked_same_direction_fork(self, threshold=1e-2):
        '''
        将从分叉点出发，同一方向且相连的骨架进行合并
        :param s_list: 骨架列表
        :return:
        '''
        # 将分叉点上的同方向骨架进行合并
        ps_dict = self.ps_dict
        pp_dict = self.pp_dict
        pps_dict = self.pps_dict
        ss_dict = self.ss_dict
        # 按照点组合两条线段，只要同一直线则视为同组
        pair_group = Grouper()
        for p in pp_dict:
            if len(pp_dict[p]) >= 3:
                # 分叉点
                p_list = pp_dict[p]
                for i in range(len(p_list) - 1):
                    for j in range(i + 1, len(p_list)):
                        p1 = p_list[i]
                        p2 = p_list[j]
                        s1 = Segment.build((p, p1))
                        s2 = Segment.build((p, p2))
                        direction1 = s1.angle_raw()
                        direction2 = s2.angle_raw()
                        if Angle.diff_real(direction1, direction2) < threshold:
                            if Point.in_segment(s2, p1) == 1 or Point.in_segment(s1, p2) == 1:
                                s1 = s1.tuplify()
                                s2 = s2.tuplify()
                                # 两条线段同组
                                pair_group.add({s1, s2})
        # 按组进行合并
        for group in pair_group.data:
            p_list = []
            for s_tuple in group:
                s = ss_dict[s_tuple]
                p_list.extend([s.p1, s.p2])
            p_list = Points.sorted(p_list, 'line')
            p1 = p_list[0]
            p2 = p_list[-1]
            skeleton_new = Segment.build([p1, p2])
            for skeleton_tuple in group:
                del ss_dict[skeleton_tuple]
            ss_dict[skeleton_new.tuplify()] = skeleton_new
        # 取出骨架
        skeleton_list = []
        for skeleton_tuple in ss_dict:
            skeleton_list.append(ss_dict[skeleton_tuple])
        self.__init__(skeleton_list)

    def merge_point(self, p1, p2):
        # 合并点p1和点p2，结果是这两个点将重合
        # 计算需要移动p1所需要的点
        p1_movedict = self.find_linked_points2(p1, p2)
        p2_movedict = self.find_linked_points2(p2, p1)
        # fig = Figure.plot(wall_segmentation)
        # Points.plot(list(p2_movedict.keys()), fig=fig)
        if len(p1_movedict) <= len(p2_movedict):
            # 调整p1，p1端点往p2端点靠拢
            move_dict = p1_movedict
        else:
            move_dict = p2_movedict
        # 执行调整
        for p in move_dict:
            delta = move_dict[p]
            self.p_dict[p].move(delta)

    def merge_point_from_keypoint(self, keypoint_list, threshold=10):
        '''
        将关键点附近的点合并成一个点
        :param keypoint_list: 关键点列表
        :param threshold: 关键点范围阈值
        '''

        p_dict = Storer(self.p_dict)
        pp_dict = self.pp_dict
        p_list = list(pp_dict.keys())
        for keypoint in keypoint_list:
            # 将关键点附近的点合并成一个点
            p_near_list = [p for p in p_list if Point.distance(keypoint, p) < threshold]
            # 提取这些点的所有关系
            next_list = []
            for p_near in p_near_list:
                next_list.extend(pp_dict[p_near])
            # 在合并点以后在外面的关系
            outside_list = set(next_list) - set(p_near_list)
            # 将原来包含被合并点的所有线删掉
            delete_list = []
            for i, s in enumerate(self):
                p1, p2 = s
                if tuple(p1) in p_near_list or tuple(p2) in p_near_list:
                    delete_list.append(i)
            delete_list.sort(reverse=True)
            for i in delete_list:
                del self[i]
            # 新增一个点
            for p_out in outside_list:
                p1 = p_dict.gen(Point(keypoint))
                p2 = p_dict.gen(Point(p_out))
                self.append(Segment.build((p1, p2)))
            # 更新数据
            self.update_data()
            p_dict = Storer(self.p_dict)
            pp_dict = self.pp_dict
            p_list = list(pp_dict.keys())
        self.reindex()

    def merge_point_from_mergedict_hold(self, merge_dict):
        # 根据merge_dict来进行点合并，保留线段关系
        # merge_dict的格式：键为需要合并的点的组名，值为需要合并点的信息列表
        # 其中，元素的组合形式是[A, B]，A为B的元祖形式, B是类型为Point的点
        # 对merge_dict进行排序，按照点集所占据的面积大小
        info_list = []
        for key in merge_dict:
            p_list = [info[0] for info in merge_dict[key]]
            area = Polygen.area(Points.convex(p_list))
            info_list.append([area, key])
        info_list.sort(key=lambda x:x[0])
        key_list = [info[1] for info in info_list]
        count = 1
        for key in key_list:
            j = 0
            while len(merge_dict[key]) >= 2:
                # 可以进行合并
                # if j == 0:
                #     print(count, key)
                #     Segments.plot(self, marked=True)
                #     # raise Exception
                #     count += 1
                # j = 1
                merge_list = merge_dict[key]
                merge_pair_list = permutation_combination(merge_list)
                length_old = len(merge_list)
                for pair in merge_pair_list:
                    if len(pair) == 2:
                        pps_dict = self.pps_dict
                        p_dict = self.p_dict
                        p1 = pair[0][1]
                        p2 = pair[1][1]
                        if tuple(p1) in p_dict and tuple(p2) in p_dict and p1 != p2:
                            # 如果点存在
                            if (tuple(p1), tuple(p2)) in pps_dict:
                                # 如果边存在，才可以进行合并
                                # raise Exception
                                self.merge_point(p1, p2)
                                # print(pair)
                                # Segments.plot(self, marked=True, texted=True)
                                # 更新merge_dict的所有键值
                                for p_now in merge_dict:
                                    merge_list_now = merge_dict[p_now]
                                    for i, _ in enumerate(merge_list_now):
                                        p_tuple, p_real = merge_list_now[i]
                                        if tuple(p_real) != p_tuple:
                                            merge_list_now[i] = [tuple(p_real), p_real]
                                # 更新骨架的所有点，同时去掉长度为0的骨架
                                self.update_point()
                                # 让merge_dict认领新的点
                                p_dict = self.p_dict
                                for p_now in merge_dict:
                                    merge_list_now = merge_dict[p_now]
                                    for i, _ in enumerate(merge_list_now):
                                        p_tuple, _ = merge_list_now[i]
                                        merge_list_now[i] = [p_tuple, p_dict[p_tuple]]
                # 更新合并列表
                pair_dict = {}
                for info in merge_list:
                    p_tuple, p_real = info
                    pair_dict[p_tuple] = p_real
                merge_list_new = []
                for pair in pair_dict:
                    merge_list_new.append([pair, pair_dict[pair]])
                merge_dict[key] = merge_list_new
                length_now = len(merge_dict[key])
                if length_old == length_now:
                    # 已经不能产生合并了，终止循环
                    break

    def merge_point_from_mergedict_main(self, merge_dict):
        # 根据merge_dict来进行点合并，保留线段主要关系
        # merge_dict的格式：键为需要合并的点的组名，值为需要合并点的信息列表
        # # 其中，元素的组合形式是[A, B]， A为类型为Point的点，B为构成点的两条原始线段的长度
        # p_dict = Storer(self.p_dict)
        # pp_dict = self.pp_dict
        for key in merge_dict:
            if len(merge_dict[key]) >= 2:
                # 可以进行点合并
                merge_list = merge_dict[key]
                # 计算新增点坐标，为所属长度最长的点
                merge_list.sort(key=lambda x: x[1], reverse=True)
                keypoint = merge_list[0][0]
                for info in merge_list:
                    p_merge, _ = info
                    p_merge.transfer(keypoint)
        # 索引重排
        self.update_point()

    def singlize(self):
        # 将线段列表唯一化，防止有重叠的线段出现
        # 将所有点进行位数截断
        p_dict = Storer()
        for i, s in enumerate(self):
            p1, p2 = s
            p1 = p_dict.gen(Point.float_cut(p1))
            p2 = p_dict.gen(Point.float_cut(p2))
            self[i] = Segment.build([p1, p2])
        # 计算交点
        _, skeleton_list, _ = Segments.interpoint_struct(self, 'all')
        # 线段唯一化
        ss_dict = Skeletons.transform(skeleton_list, 'sl_ssdict')
        skeleton_list_new = []
        for s_tuple in ss_dict:
            skeleton_list_new.append(ss_dict[s_tuple])
        Segments.reset(skeleton_list_new)
        self.__init__(skeleton_list_new)

    def update_point(self):
        # 更新骨架当中所有的点，并且将长度为0的线段去掉
        p_dict = Storer()
        for i, s in enumerate(self):
            s.p1 = p_dict.gen(Point(tuple(s.p1)))
            s.p2 = p_dict.gen(Point(tuple(s.p2)))
            self[i] = Segment.build([s.p1, s.p2])
        # 更新数据，重新计算骨架特征
        s_list = copy.deepcopy(self)
        self.__init__(s_list)
        # 将长度为0的线段去掉
        self.weight_init('length')
        self.weight_filter('(0')

    def update_data(self):
        # 单独更新数据
        self.ss_dict = Skeletons.transform(self, 'sl_ssdict')
        self.ps_dict = Skeletons.transform(self, 'sl_psdict')
        self.pp_dict = Skeletons.transform(self, 'sl_ppdict')
        self.p_dict = Skeletons.transform(self, 'sl_pdict')
        self.pps_dict = Skeletons.transform(self, 'sl_ppsdict')

    @staticmethod
    def transform(data, method):
        '''
        骨架形式变换
        :param data: 数据
        :param method:  sl_psdict: 骨架列表转换为点-骨架关系字典
                        sl_ssdict: 骨架列表转换为骨架-骨架关系字典
                        sl_ppdict: 骨架列表转换为点-点关系字典
                        sl_pdict: 骨架列表转换为点-点索引字典
                        sl_ppsdict: 骨架列表转换为点点-骨架索引字典
                        ssdict_sl: 骨架-骨架关系字典转换为骨架列表
        :return:
        '''

        if method == 'sl_psdict':
            # 生成点-骨架字典
            ps_dict = {}
            for s in data:
                p1 = s.p1
                p2 = s.p2
                dict_add(ps_dict, tuple(p1), s, 'append')
                dict_add(ps_dict, tuple(p2), s, 'append')
            return ps_dict
        elif method == 'sl_ssdict':
            # 生成骨架-骨架字典
            ss_dict = {}
            for s in data:
                ss_dict[s.tuplify()] = s
            return ss_dict
        elif method == 'sl_ppdict':
            # 生成点-点关系字典
            pp_dict = {}
            for s in data:
                p1 = tuple(s.p1)
                p2 = tuple(s.p2)
                dict_add(pp_dict, p1, p2, 'append')
                dict_add(pp_dict, p2, p1, 'append')
            return pp_dict
        elif method == 'sl_pdict':
            # 生成点-点索引字典
            p_dict = {}
            for s in data:
                p1 = tuple(s.p1)
                p2 = tuple(s.p2)
                p_dict[p1] = s.p1
                p_dict[p2] = s.p2
            return p_dict
        elif method == 'sl_ppsdict':
            # 生成点点-骨架索引字典
            pps_dict = {}
            for s in data:
                p1 = tuple(s.p1)
                p2 = tuple(s.p2)
                pps_dict[(p1, p2)] = s
                pps_dict[(p2, p1)] = s
            return pps_dict
        elif method == 'ssdict_sl':
            # 从骨架-骨架字典生成骨架列表
            s_list = []
            for key in data:
                s_list.append(data[key])
            return s_list


# 多段线
class Curve(Points):

    def __init__(self, p_list=(), hold=False):
        super().__init__([])
        for p in p_list:
            if hold:
                # 不新建点，保留点的信息
                self.append(p)
            else:
                # 直接新建点
                self.append(Point(p))
        self.x_min = 0
        self.y_min = 0
        self.ind = 0
        self.weight = 0  # 多边形的权重，用来做比较、排序使用
        self.update_data()

    def collasp(self, angle_thre=1e-4, dist_thre=1e-4):
        # 去除同方向上的冗余点，以及重叠点
        # 按照点组合两条线段，只要同一直线则视为同组
        # 先清理距离太近的
        delete_index = []
        for i, p in enumerate(self):
            if i < len(self) - 2:
                p_next = self[i + 1]
                p_next2 = self[i + 2]
                # 距离太近（甚至点重合)，也可以直接删除
                if Point.distance(p, p_next) < dist_thre:
                    delete_index.insert(0, i + 1)
        for i in delete_index:
            del self[i]
        # 再清理角度相近的
        delete_index = []
        for i, p in enumerate(self):
            if i < len(self) - 2:
                p_next = self[i + 1]
                p_next2 = self[i + 2]
                # 角度相近
                direction1 = Vector.angle([p, p_next])
                direction2 = Vector.angle([p_next, p_next2])
                if Angle.diff_real(direction1, direction2) < angle_thre:
                    delete_index.insert(0, i + 1)
        for i in delete_index:
            del self[i]

    def corrcoef(self):
        # 计算相关系数
        x_list = []
        y_list = []
        for p in self:
            x_list.append(p[0])
            y_list.append(p[1])
        r_square = abs(np.corrcoef(x_list, y_list)[0][1])
        return r_square

    def line_distance(self, l_abc, method='normal'):
        # 计算曲线到直线的距离
        p_array = np.array(self)
        x = p_array[:, 0]
        y = p_array[:, 1]
        A, B, C = l_abc
        dist_array = np.abs(A * x + B * y + C) / np.sqrt(A * A + B * B)
        if method == 'square':
            dist_array = pow(dist_array, 2)
        return np.sum(dist_array)

    def line_distance_list(self, l_abc, method='normal'):
        # 计算曲线到直线的距离序列
        dist_list = []
        for p in self:
            dist = Line.point_distance(p, l_abc)
            if method == 'square':
                dist_list.append(dist * dist)
            else:
                dist_list.append(dist)
        return dist_list

    def line_regression_quarter(self):
        # 用四分位估计一条近似直线的系数
        length = len(self)
        curve1 = self[:int(length * 3 / 4)]
        curve2 = self[int(length / 4):]
        # 计算中心点
        p1 = np.mean(np.array(curve1), axis=0)
        p2 = np.mean(np.array(curve2), axis=0)
        l_abc = Line.transform((p1, p2), '2p_ABC')
        residual = Curve.line_distance(self, l_abc, 'square') / length
        return l_abc, residual

    def line_regression(self):
        # 计算拟合直线
        length = len(self)
        curve = np.array(self)
        x = curve[:, 0]
        y = curve[:, 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        xy_mean = np.mean(x * y)
        xx_mean = np.mean(x * x)
        yy_mean = np.mean(y * y)
        if x_mean * x_mean - xx_mean != 0:
            m = (x_mean * y_mean - xy_mean) / (x_mean * x_mean - xx_mean)
            b = y_mean - m * x_mean
            l_abc = (m, -1, b)
        else:
            l_abc = (1, 0, -x_mean)

        # residual = max(Curve.line_distance_list(self, l_abc))
        residual = Curve.line_distance(self, l_abc, 'square') / length
        # 计算拟合优度
        # f1 = m * x + b
        # sst1 = sum((y - y_mean) * (y - y_mean))
        # sse1 = sum((y - f1) * (y - f1))
        # ssr1 = sum((f1 - y_mean) * (f1 - y_mean))
        # f2 = (y - b)/m
        # sst2 = sum((x - x_mean) * (x - x_mean))
        # sse2 = sum((x - f2) * (x - f2))
        # ssr2 = sum((f2 - x_mean) * (f2 - x_mean))
        # sse = min([sse1, sse2])
        # ssr = min([ssr1, ssr2])
        # # 计算指标
        # residual = sse / length
        # r2 = ssr1 / sst1
        return l_abc, residual

    def lines_distance(self, l_list, method='square', rst_type='num'):
        # 计算线段到直线集合的最短距离
        dist_list = []
        p_array = np.array(self)
        x = p_array[:, 0]
        y = p_array[:, 1]
        for l in l_list:
            A, B, C = l
            dist_list.append(np.abs(A * x + B * y + C) / np.sqrt(A * A + B * B))
        dist_array = np.column_stack(dist_list)
        if method == 'square':
            if rst_type == 'list':
                dist_total = pow(np.min(dist_array, axis=1), 2)
            else:
                dist_total = np.sum(pow(np.min(dist_array, axis=1), 2))
        else:
            if rst_type == 'list':
                dist_total = np.min(dist_array, axis=1)
            else:
                dist_total = np.sum(np.min(dist_array, axis=1))

        # 原始代码
        # dist_total = 0
        # for p in self:
        #     dist_list = []
        #     for l in l_list:
        #         dist_list.append(Line.point_distance(p, l))
        #     dist = min(dist_list)
        #     if method == 'square':
        #         dist_total += dist * dist
        #     else:
        #         dist_total += dist
        return dist_total

    def plot(self, color=[1, 0, 0], fig=[], texted=False, linewidth=1, scaled=False, marked=False,
             markersize=5):
        if fig == []:
            fig = plt.figure()
        s_list, _ = Curve.segmentize(self)
        kwargs = {'color': color, 'linewidth': linewidth, 'marked': marked}
        for i, s in enumerate(s_list):
            if texted:
                kwargs['text'] = 'c' + str(self.ind)
            else:
                kwargs['text'] = ''
            kwargs['fig'] = fig
            s.plot(**kwargs)
        if scaled:
            plt.axis('scaled')

    def raw_point_relation(self, curve_raw):
        # 计算曲线上的原始点属于当前曲线的哪一条
        point_group = {}
        point_ind_group = {}
        # 计算点到所有直线的距离
        l_list = []
        for i in range(len(self) - 1):
            p1 = self[i]
            p2 = self[i + 1]
            l_list.append(Line.transform((p1, p2), '2p_ABC'))
        if len(l_list) == 1:
            # 都属于一条线
            point_group[0] = curve_raw
            point_ind_group[0] = list(range(len(curve_raw)))
        else:
            # 属于不同的线
            dist_list = []
            p_array = np.array(curve_raw)
            x = p_array[:, 0]
            y = p_array[:, 1]
            for l in l_list:
                A, B, C = l
                dist_list.append(np.abs(A * x + B * y + C) / np.sqrt(A * A + B * B))
            # 从头到尾开始进行比较
            ind_list = list(range(len(curve_raw)))
            i_now = 0
            if curve_raw[0] == curve_raw[-1]:
                # 封闭曲线，需要考虑首尾衔接
                for i in range(len(dist_list)):
                    array_now = dist_list[i][i_now:]
                    array_next = dist_list[(i + 1) % len(dist_list)][i_now:]
                    ind_minus = np.where((array_next - array_now) < 0)[0]
                    if len(ind_minus) > 0:
                        # 存在分割点
                        i_end = i_now + min(np.where((array_next - array_now) < 0)[0])
                        point_group[i] = curve_raw[i_now:i_end + 1]
                        point_ind_group[i] = ind_list[i_now:i_end + 1]
                        i_now = i_end
                        # 剩下的全部属于下一条线（其实就是第一条线，由于首尾连接，可能出现起点偏移）
                        i_next = (i + 1) % len(dist_list)
                        if i_next not in point_group:
                            point_group[i_next] = curve_raw[i_now:]
                            point_ind_group[i_next] = ind_list[i_now:]
                        else:
                            point_group[i_next].extend(curve_raw[i_now:])
                            point_ind_group[i_next].extend(ind_list[i_now:])
                    else:
                        # 不存在分割点
                        point_group[i] = curve_raw[i_now:]
                        point_ind_group[i] = ind_list[i_now:]
            else:
                # 不封闭曲线，到最后全部属于最后一条线
                for i in range(len(dist_list)):
                    if i < len(dist_list) - 1:
                        array_now = dist_list[i][i_now:]
                        array_next = dist_list[(i + 1) % len(dist_list)][i_now:]
                        ind_minus = np.where((array_next - array_now) < 0)[0]
                        i_end = i_now + min(np.where((array_next - array_now) < 0)[0])
                        point_group[i] = curve_raw[i_now:i_end + 1]
                        point_ind_group[i] = ind_list[i_now:i_end + 1]
                        i_now = i_end
                    else:
                        # 全部属于最后一条线
                        point_group[i] = curve_raw[i_now:]
                        point_ind_group[i] = ind_list[i_now:]

        # # 原始代码
        # ind = 0
        # point_group = {}
        # point_ind_group = {}
        # for i_p, p in enumerate(curve_raw):
        #     if ind < len(self) - 2:
        #         l_now = Line.transform([self[ind], self[ind + 1]], '2p_ABC')
        #         l_next = Line.transform([self[ind + 1], self[ind + 2]], '2p_ABC')
        #         dist_now = Line.point_distance(p, l_now)
        #         dist_next = Line.point_distance(p, l_next)
        #         if dist_now < dist_next:
        #             dict_add(point_group, ind, p, 'append')
        #             dict_add(point_ind_group, ind, i_p, 'append')
        #         else:
        #             dict_add(point_group, ind, p, 'append')
        #             dict_add(point_ind_group, ind, i_p, 'append')
        #             # 转换到下一条曲线
        #             ind += 1
        #             dict_add(point_group, ind, p, 'append')
        #             dict_add(point_ind_group, ind, i_p, 'append')
        #     else:
        #         # 都属于最后一条线
        #         dict_add(point_group, ind, p, 'append')
        #         dict_add(point_ind_group, ind, i_p, 'append')
        # 可视化
        # fig = plt.figure()
        # for i in point_group:
        #     Points.plot(point_group[i], color=randomcolor(), fig=fig)
        return point_group, point_ind_group

    def rdp(self, epsilon=5):
        '''
        进行rdp线段抽稀
        :param epsilon: 距离阈值
        :param replace: 是否替换原来的序列
        :return:
        '''

        if self[0] == self[-1]:
            closed = True
        else:
            closed = False
        line_array = np.array(self, 'float32')
        key_points = cv2.approxPolyDP(line_array, epsilon, closed)
        key_points = Points.transform(key_points, 'cv_point')
        i_start = 0
        if closed:
            # 如果是闭合算法，会产生抽稀点的顺序会更换，因此需要旋转原来的原始曲线
            # 寻找起点
            p_start = key_points[0]
            dist_list = [Point.distance(p_start, p_raw) for p_raw in self]
            i_start = np.argmin(dist_list)
            raw_new = self[i_start:] + self[:i_start]
            # 因为是闭合的，所以要多产生一个点
            self.__init__(raw_new + [raw_new[0]])
            key_points += [key_points[0]]
        return key_points, i_start

    def segmentize(self, hold=True, collect=True):
        # 返回多段线的线段形式
        s_list = []
        ps_dict = {}
        p_dict = Storer()
        for i, _ in enumerate(self):
            if i > 0:
                if hold:
                    p1 = self[i - 1]
                    p2 = self[i]
                else:
                    if collect:
                        p1 = p_dict.gen(Point(self[i - 1]))
                        p2 = p_dict.gen(Point(self[i]))
                    else:
                        p1 = Point(self[i - 1])
                        p2 = Point(self[i])
                s_new = Segment.build([p1, p2])
                s_list.append(s_new)
                dict_add(ps_dict, tuple(self[i - 1]), s_new, 'append')
                dict_add(ps_dict, tuple(self[i]), s_new, 'append')
        return s_list, ps_dict

    def update_data(self):
        # 更新数据
        s_list, ps_dict = self.segmentize(hold=True)
        self.s_list = s_list
        self.ps_dict = ps_dict


# 多段线集合
class Curves(list):

    def __init__(self, m_list, hold=False):
        super().__init__([])
        for m in m_list:
            # 不新建多段线，保留多段线的信息
            self.append(Curve(m, hold))
        self.reindex()

    def extract_segment(self, threshold=5):
        # 从曲线集合中提取线段

        def find_reverse(p_start, p_end, i_route):
            # 已知起点p_start和终点p_end以及当前第i_route条路线
            # 返回起点p_end和终点p_start以及对应第几条路线
            for i, curve in enumerate(curve_dict[(p_end, p_start)]):
                curve_now = curve[::-1]
                if curve_now == curve_dict[(p_start, p_end)][i_route]:
                    return i

        def walked_record(walked_set, p, p_now, i_now):
            # 记录行走路线，同时将反过来的路线标记为已行走
            # 从p点出发，到p_now结束，走第i_now条路线
            walked_set.add((p, p_now, i_now))
            # 计算从p点出发，到p_now结束的逆转
            i = find_reverse(p, p_now, i_now)
            walked_set.add((p_now, p, i))

        def search(p_now, walked_set, curve_all, threshold, flag):
            # 从点p_now出发，寻找直线
            # 进行搜索
            while True:
                next_list2 = relation_dict[p_now]
                # 计算哪一个分支是残差最小
                info_list2 = []
                for p_next2 in next_list2:
                    if flag == 'forward':
                        for i, curve in enumerate(curve_dict[(p_now, p_next2)]):
                            if (p_now, p_next2, i) not in walked_set:
                                curve_now = curve_all + curve
                                _, sse = Curve.line_regression_quarter(curve_now)
                                info_list2.append([p_next2, i, sse])
                    elif flag == 'backward':
                        for i, curve in enumerate(curve_dict[(p_next2, p_now)]):
                            if (p_next2, p_now, i) not in walked_set:
                                curve_now = curve + curve_all
                                _, sse = Curve.line_regression_quarter(curve_now)
                                info_list2.append([p_next2, i, sse])
                    else:
                        raise NotImplementedError
                if len(info_list2) > 0:
                    # 寻找残差最小的
                    info_list2.sort(key=lambda x: x[2])
                    p_raw, i_raw, sse_raw = info_list2[0]
                    if sse_raw < threshold:
                        walked_record(walked_set, p_now, p_raw, i_raw)
                        if flag == 'forward':
                            curve_all = curve_all + curve_dict[(p_now, p_raw)][0]
                        elif flag == 'backward':
                            curve_all = curve_dict[(p_now, p_raw)][0] + curve_all
                        else:
                            raise NotImplementedError
                        p_now = p_raw
                    else:
                        # 找不到符合要求的点
                        break
                else:
                    # 找不到下一个点
                    break
            return curve_all

        # 提取起点-终点索引路径的字典
        curve_dict = {}
        for curve in self:
            p_start = tuple(curve[0])
            p_end = tuple(curve[-1])
            dict_add(curve_dict, (p_start, p_end), curve, 'append')
        # 从骨架线中提取分叉点关系
        relation_dict = {}
        for key in curve_dict:
            p1, p2 = key
            dict_add(relation_dict, p1, p2, 'append')
            dict_add(relation_dict, p2, p1, 'append')
        # 状态记录
        # 提取直线
        segment_list = []
        walked_set = set()
        key_list = list(relation_dict.keys())
        for i, p in enumerate(key_list):
            next_list = relation_dict[p]
            # 挑一个最直的（残差最小的）方向
            info_list = []
            for p_next in next_list:
                for i, curve in enumerate(curve_dict[(p, p_next)]):
                    if (p, p_next, i) not in walked_set:
                        _, residual = Curve.line_regression_quarter(curve)
                        info_list.append([p_next, i, residual])
            if len(info_list) > 0:
                info_list.sort(key=lambda x: x[2])
                p_now, i_now, residual_now = info_list[0]
                if residual_now < threshold:
                    # 当前路径
                    curve_all = curve_dict[(p, p_now)][i_now]
                    # 记录行走路线，同时将反过来的路线标记为已行走
                    walked_record(walked_set, p, p_now, i_now)
                    # 往前进行搜索
                    curve_all = search(p_now, walked_set, curve_all, threshold, 'forward')
                    # 完成往前搜索以后，观察一下回到原点那个方向是可行的
                    info_list = []
                    for p_next in next_list:
                        for i, curve in enumerate(curve_dict[(p_next, p)]):
                            if (p_next, p, i) not in walked_set:
                                curve_now = curve + curve_all
                                _, residual = Curve.line_regression_quarter(curve_now)
                                info_list.append([p_next, i, residual])
                    if len(info_list) > 0:
                        info_list.sort(key=lambda x: x[2])
                        p_now, i_now, residual_now = info_list[0]
                        if residual_now < threshold:
                            # 往后搜索
                            # 记录行走路线，同时将反过来的路线标记为已行走
                            walked_record(walked_set, p_now, p, i_now)
                            curve_all = curve_dict[(p_now, p)][i_now] + curve_all
                            # 往后进行搜索
                            curve_all = search(p_now, walked_set, curve_all, threshold, 'backward')
                    # 保存直线
                    x1, y1 = curve_all[0]
                    x2, y2 = curve_all[-1]
                    l_abc, _ = Curve.line_regression_quarter(curve_all)
                    A, B, C = l_abc
                    if abs(A) > 1 or B == 0:
                        x1 = Line.y_coor(l_abc, y1)
                        x2 = Line.y_coor(l_abc, y2)
                    else:
                        y1 = Line.x_coor(l_abc, x1)
                        y2 = Line.x_coor(l_abc, x2)
                    s = Segment(((x1, y1), (x2, y2)))
                    segment_list.append(s)
        return segment_list

    def merge_point_for_voronoi(curve_group):
        '''
        在对泰森图进行曲线抽稀（point_wedding）后，合并属于同一泰森分叉点的点
        :return:
        '''
        # 构造分叉点对应的线段
        pass_point_dict = {}  # 经过点的线段集合
        curve_segment_list = []  # 曲线拆散后的线段集合
        threshold = 5  # 分叉点距离阈值
        for i_curve in range(len(curve_group)):
            # 计算关键点属于曲线当中的那一段
            curve = curve_group[i_curve]
            # 重新构造线段，方便独立延伸
            s_list, _ = curve.segmentize(hold=False, collect=False)
            # Segments.plot(s_list, marked=True)
            curve_segment_list.extend(s_list)
            Segments.reset(curve_segment_list)
            p_walked = curve.p_walked
            p_walked_add = curve.p_walked_add
            # 两端的点只有一条线段经过
            p_walked_1 = {p_walked[0], p_walked[-1]}
            # 中间的点相当于有两条线段经过
            p_walked_2 = set(p_walked[1:-1] + p_walked_add)
            walked_list = list(p_walked_1) + list(p_walked_2) * 2
            p_unique_list = p_walked_1 | p_walked_2
            # 进行匹配
            allo_dict = {}
            for p in p_unique_list:
                times = walked_list.count(p)
                # 跟所有的线段进行比较
                info_list = [[Segment.point_distance(s, p, 'clever'), s] for s in s_list]
                info_list.sort(key=lambda x: x[0])
                # 找到距离最近的线段
                ind1 = None
                for i in range(times):
                    if i < len(info_list):
                        diff, s = info_list[i]
                        if i == 0:
                            # 第一条一定能够分配成功的，因为必须匹配一条
                            ind1 = s.ind
                            dict_add(pass_point_dict, tuple(p), s, 'append')
                        else:
                            # 第二条，分配给可以接受的最近的
                            if diff < threshold:
                                ind2 = s.ind
                                if (ind1, ind2) not in allo_dict:
                                    # 可以分配
                                    allo_dict[(ind1, ind2)] = [diff, s, p]
                                    allo_dict[(ind2, ind1)] = [diff, s, p]
                                else:
                                    diff_old, s_old, _ = allo_dict[(ind1, ind2)]
                                    if diff < diff_old:
                                        # 如果距离更短，则可以保存
                                        allo_dict[(ind1, ind2)] = [diff, s, p]
                                        allo_dict[(ind2, ind1)] = [diff, s, p]
            # 补充第二条线的分配
            for key in allo_dict:
                _, s, p = allo_dict[key]
                dict_add(pass_point_dict, tuple(p), s, 'append')
        # 索引重置
        Segments.reset(curve_segment_list)
        # Segments.plot(curve_segment_list, texted=True, marked=True)
        # 让线段延长到交点
        angle_inter_threshold = Angle.to_value(15)
        connect_dict = {}  # 连接线段字典，即连接线段对应哪个分叉点
        s_inter_dict = {}  # 线段相交字典
        s_not_cross_dict = {}  # 线段不可能相交字典
        for i, s in enumerate(curve_segment_list):
            s.flag = 'main'  # 主要线段
            s_not_cross_dict[i] = [s]
        for p in pass_point_dict:
            s_list = pass_point_dict[p]
            length = len(s_list)
            for i in range(length - 1):
                for j in range(i + 1, length):
                    p11, p12 = s1 = s_list[i]
                    p21, p22 = s2 = s_list[j]
                    tuple_list = []
                    for p_now in [p11, p12, p21, p22]:
                        tuple_list.append(tuple(p_now))
                    # 计算公用一个端点
                    flag_common = False
                    for p_now in tuple_list:
                        if tuple_list.count(p_now) == 2:
                            flag_common = True
                            break
                    if flag_common is False:
                        l1 = Line.transform(s1, '2p_ABC')
                        l2 = Line.transform(s2, '2p_ABC')
                        p_inter = Line.interpoint(l1, l2)
                        flag_inter = False
                        if p_inter != -1:
                            # 防止平行的线计算交点
                            direction1 = s1.direction
                            direction2 = s2.direction
                            diff = Angle.diff_real(direction1, direction2)
                            if diff > angle_inter_threshold:
                                # 计算是否可行
                                dict_add(s_inter_dict, p, [s1, s2, p_inter], 'append')
                                flag_inter = True
                        # 判断是否相交在正常的地方
                        p1_info = [[Point.distance(p, p_now), p_now] for p_now in [p11, p12]]
                        p1_info.sort(key=lambda x: x[0])
                        p2_info = [[Point.distance(p, p_now), p_now] for p_now in [p21, p22]]
                        p2_info.sort(key=lambda x: x[0])
                        p1_close_info = p1_info[0]
                        p2_close_info = p2_info[0]
                        p1_close = p1_close_info[1]
                        p2_close = p2_close_info[1]
                        if flag_inter is False:
                            # 如果没有相交，则直接从距离最短的点出发向另一个点作垂线
                            close_info = [p1_close_info + [1]] + [p2_close_info + [2]]
                            close_info.sort(key=lambda x: x[0])
                            _, p_close, ind = close_info[0]
                            if ind == 1:
                                # 第一条线上的点向第二条线作垂线
                                p_next = Line.verticalpoint(p_close, l2)
                            else:
                                # 第二条线上的点向第一条线作垂线
                                p_next = Line.verticalpoint(p_close, l1)
                            # 建立一条新的线，将两条线的最靠近端点连起来
                            s_new = Segment([p_close, p_next])
                            s_new.flag = 'connect'  # 连接线段
                            curve_segment_list.append(s_new)
                            # 记录不可能相交字典
                            dict_add(s_not_cross_dict, s1.ind, s_new, 'append')
                            dict_add(s_not_cross_dict, s2.ind, s_new, 'append')
                            # 记录这条线段的分叉点
                            dict_add(pass_point_dict, tuple(p), s_new, 'append')
                            connect_dict[s_new.tuplify()] = tuple(p)
                        else:
                            # 如果相交，则直接建立一条从端点到交点的线
                            for p_close, s_now in zip([p1_close, p2_close], [s1, s2]):
                                # 都建立一条新的线，将两条线的最靠近端点连起来
                                ind_now = s_now.ind
                                if Point.in_segment(s_now, p_inter) != 1:
                                    # 如果交点不在线段上，可以建立线段
                                    s_new = Segment([p_close, p_inter])
                                    s_new.flag = 'sub'  # 辅助线段
                                    curve_segment_list.append(s_new)
                                    # 记录不可能相交字典
                                    dict_add(s_not_cross_dict, ind_now, s_new, 'append')
                                    # 记录这条线段的分叉点
                                    dict_add(pass_point_dict, tuple(p), s_new, 'append')
        # 索引重置
        Segments.reset(curve_segment_list)
        for i, s in enumerate(curve_segment_list):
            if i not in s_not_cross_dict:
                s_not_cross_dict[i] = [s]

        # 收集合理的相交点
        # 合理的相交点来源如下：
        # 1、原本已经存在的交点；
        # 2、延长后的线与其他线相交以后产生的点是合理相交点时，该交点有效
        p_fairset = set()
        # 计算已经存在的交点
        for i in range(len(curve_segment_list)):
            for j in range(i + 1, len(curve_segment_list)):
                s1 = curve_segment_list[i]
                s2 = curve_segment_list[j]
                interpoint = Segment.interpoint(s1, s2)
                if interpoint != -1:
                    p_fairset.add(tuple(interpoint))
        # 寻找更多的合理点
        for s in curve_segment_list:
            s.extend_point = []
        while True:
            temp_set = set()
            for p in s_inter_dict:
                info_list = s_inter_dict[p]
                for info in info_list:
                    s1, s2, p_inter = info
                    p11, p12 = s1
                    p21, p22 = s2
                    # 判断s1产生了哪些交点，并且在总交点集合中去除合理的交点
                    p_list1 = [p11, p12, p_inter]
                    s1_now = Segment.build_from_point_list(p_list1)
                    not_cross_sind = [s.ind for s in s_not_cross_dict[s1.ind]]
                    p_list1 = [tuple(Point.float_cut(p_now)) for p_now in p_list1]
                    sub_list = List.filter_sub(curve_segment_list, not_cross_sind)
                    inter_list1 = Segments.interpoint_single(sub_list, s1_now)
                    inter_list1 = [tuple(Point.float_cut(p_now)) for p_now in inter_list1 if
                                   tuple(Point.float_cut(p_now)) not in p_fairset]
                    # 判断s2产生了哪些交点，并且在总交点集合中去除合理的交点
                    p_list2 = [p21, p22, p_inter]
                    s2_now = Segment.build_from_point_list(p_list2)
                    not_cross_sind = [s.ind for s in s_not_cross_dict[s2.ind]]
                    p_list2 = [tuple(Point.float_cut(p_now)) for p_now in p_list2]
                    sub_list = List.filter_sub(curve_segment_list, not_cross_sind)
                    inter_list2 = Segments.interpoint_single(sub_list, s2_now)
                    inter_list2 = [tuple(Point.float_cut(p_now)) for p_now in inter_list2]
                    inter_list2 = [tuple(Point.float_cut(p_now)) for p_now in inter_list2 if
                                   tuple(Point.float_cut(p_now)) not in p_fairset]
                    if len(set(inter_list1) - set(p_list1)) == 0 and len(set(inter_list2) - set(p_list2)) == 0:
                        # 如果没有其他交点
                        # 记录交点
                        s1.extend_point.append(p_inter)
                        s2.extend_point.append(p_inter)
                        temp_set.add(tuple(Point.float_cut(p_inter)))
            if len(temp_set) > 0:
                num_old = len(p_fairset)
                p_fairset.update(temp_set)
                num_new = len(p_fairset)
                if num_new == num_old:
                    # 当数量没有增加的时候，结束搜索
                    break
            else:
                break
        # 延长线段
        for s in curve_segment_list:
            extend_point = s.extend_point
            if len(extend_point) > 0:
                s.feature_fix(extend_point, 'point_extend')
        # 清除辅助线
        delete_index = []
        for s in curve_segment_list:
            if s.flag == 'sub':
                delete_index.insert(0, s.ind)
        for i_delete in delete_index:
            del curve_segment_list[i_delete]
        Segments.reset(curve_segment_list)

        # 修改connect_dict，即以线段的索引作为键，分叉点作为值
        ss_dict = {}
        for s in curve_segment_list:
            ss_dict[s.tuplify()] = s
        connect_index_dict = {}
        for s_tuple in connect_dict:
            s = ss_dict[s_tuple]
            connect_index_dict[s.ind] = connect_dict[s_tuple]
        connect_index_set = set(connect_index_dict.keys())
        # Segments.plot(curve_segment_list, texted=True, marked=True)
        # 产生交点
        _, skeleton_list, point_segment_dict = Segments.interpoint_struct(curve_segment_list, 'all', False)
        # Segments.plot(skeleton_list, marked=True, texted=True)
        # 将连接线的交点存入pass_point_dict
        for p_inter in point_segment_dict:
            s_list = point_segment_dict[p_inter]
            ind_set = set([s.ori_ind for s in s_list])
            common_set = connect_index_set & ind_set
            if len(common_set) > 0:
                for i in common_set:
                    # 将交点加入到pass_point_dict中
                    p_raw = connect_index_dict[i]
                    dict_add(pass_point_dict, p_raw, curve_segment_list[i], 'append')
        # 计算合并字典
        skeleton_list = Skeletons(skeleton_list)
        p_dict = skeleton_list.p_dict
        # 计算inter_dict（点是由什么线交成的）
        inter_dict = {}
        for p in pass_point_dict:
            s_list = pass_point_dict[p]
            inter_dict[p] = [s.ind for s in s_list if s.flag in ['main', 'connect']]
        # 计算inter_dict_new（点是由什么线交成的）
        inter_dict_new = {}
        for p in point_segment_dict:
            s_inter_list = point_segment_dict[p]
            inter_dict_new[p] = list_unique([s.ori_ind for s in s_inter_list])
        # 计算两两线段索引相交原点
        index_point_dict = {}
        for p in inter_dict:
            linked_list = inter_dict[p]
            for i in range(len(linked_list)):
                for j in range(i + 1, (len(linked_list))):
                    dict_add(index_point_dict, (linked_list[i], linked_list[j]), p, 'append')
                    dict_add(index_point_dict, (linked_list[j], linked_list[i]), p, 'append')
        # 计算合并字典（键：原始点，值：当前相交出来的点、相交线的总长度）
        merge_dict = {}
        for p in inter_dict_new:
            key_list = permutation_combination(inter_dict_new[p])
            for key_now in key_list:
                # 遍历所有的可能性
                if len(key_now) == 2:
                    if tuple(key_now) in index_point_dict:
                        # 寻找两条线的相交以前是否存在
                        real_point = index_point_dict[tuple(key_now)]
                        for p_real in real_point:
                            dict_add(merge_dict, p_real, [p, p_dict[p]], 'append')
        # 合并字典进行合并
        skeleton_list.merge_point_from_mergedict_hold(merge_dict)
        # Segments.plot(skeleton_list, marked=True, texted=True)
        return skeleton_list

    def plot(self, color=[1, 0, 0], fig=[], texted=False, linewidth=1, marked=False,
             markersize=5):
        if fig == []:
            fig = plt.figure()
        kwargs = {'color': color, 'linewidth': linewidth, 'marked': marked, 'texted': texted}
        kwargs['fig'] = fig
        for curve in self:
            curve.plot(**kwargs)
        plt.axis('scaled')

    def reindex(self):
        '''
        重置直线列表索引
        :return:
        '''

        for i, m in enumerate(self):
            m.ind = i

    def segmentize(self, collect=True, rdp=False, epsilon=5):
        '''
        将多段线集合线段化
        :param collect: 是否集中生成点。
        如果True，同坐标是同一个点，生成集中性骨架。
        如果为False，则生成分散性骨架
        :param rdp: 是否进行线段抽稀
        :param epsilon: rdp算法的epsilon参数
        :return: 线段化的结果
        '''

        segment_list = []
        s_store = set()
        p_dict = Storer()
        for curve in self:
            if rdp:
                key_points, _ = Curve.rdp(curve, epsilon)
                # 固定第一个点和最后一个点
                key_points[0] = curve[0]
                key_points[-1] = curve[-1]
            else:
                # 对所有点进行几何
                key_points = []
                for p in curve:
                    key_points.append(p)
            if collect:
                # 集中生成点
                for i, p in enumerate(key_points):
                    key_points[i] = p_dict.gen(Point(p))
            # 线段化
            s_list, _ = Curve.segmentize(key_points, collect)
            for s in s_list:
                if s.tuplify() not in s_store:
                    segment_list.append(s)
                    s_store.add(s.tuplify())
        Segments.reset(segment_list)
        return segment_list

    def straightize(curve_group, direction_square, error_factor=0.75):
        '''
        对从曲线进行纠正，其中曲线是带原始点的。如果误差更低，直接摆正
        :param direction_square: 规范方向
        :param error_factor: 纠正道规范方向后的曲线误差的衰减因子。一般小于1，鼓励纠正道规范方向上
        '''

        def calc_straightize_min_error(p_now, p_next, raw_points, dire_best):
            '''
            计算纠正到最接近的规范方向的最低误差方式
            :param p_now: 点1
            :param p_next: 点2
            :param raw_points: 线段[p1, p2]苏属于的原始点
            :param dire_best: 当前最佳规范方向
            :return diff: 当前规范方向最佳直线的误差
            :return l_old: 线段当前直线
            :return l_now: 当前规范方向最佳直线
            :return p_mid: 当前规范方向最佳直线的中点
            :return len_raw: 当前规范方向最佳直线的原始点数量
            '''

            l_old = Line.transform((p_now, p_next), '2p_ABC')
            # 由于泰森图的特征，仅取出中间的部分作为计算
            i_1_8 = int(len(raw_points) / 8)
            i_1_4 = int(len(raw_points) / 4)
            i_1_2 = int(len(raw_points) / 2)
            i_3_4 = int(len(raw_points) * 3 / 4)
            i_7_8 = int(len(raw_points) * 7 / 8)
            i_1_1 = len(raw_points) - 1
            # 计算三种不同的中点
            mid_list = [[raw_points[i_1_2], dire_best, i_1_8, i_7_8, 1],
                        [np.mean(raw_points, axis=0), dire_best, i_1_8, i_7_8, 1],
                        [raw_points[i_1_4], dire_best, 0, i_1_2, 2],
                        [raw_points[i_3_4], dire_best, i_1_2, i_1_1, 2]]
            # 计算误差
            error_list = []
            for mid_info in mid_list:
                p_mid, dire_now, i_start, i_end, factor = mid_info
                l_now = Line.transform((p_mid, dire_now), 'pseta_ABC')
                old_list = list(Curve.lines_distance(raw_points, [l_old], rst_type='list'))
                now_list = list(Curve.lines_distance(raw_points, [l_now], rst_type='list'))
                # Points.plot(raw_points[i_start:i_end], color=[0,1,0], fig=fig)
                error_old = sum(old_list[i_start:i_end])
                error_now = sum(now_list[i_start:i_end]) * factor * error_factor
                diff = error_now - error_old
                error_list.append([diff, l_now, p_mid, len(raw_points)])
            error_list.sort(key=lambda info: info[0])
            diff, l_now, p_mid, len_raw = error_list[0]
            return diff, l_old, l_now, p_mid, len_raw

        for i_curve in range(len(curve_group)):
            curve_now = curve_group[i_curve]
            curve_raw = curve_now.raw  # 曲线的原始点
            s_list, _ = Curve.segmentize(curve_now)
            raw_list, _ = Curve.segmentize(curve_raw)
            # fig = Figure.plot(wall_segmentation)
            # Segments.plot(s_list, marked=True, color=[0, 0, 1], fig=fig)
            # fig = Segments.plot(raw_list, marked=True)
            # Segments.plot(s_list, marked=True, fig=fig, color=[0,0,1])
            if len(curve_now) == 2:
                # 直线绕中心点旋转
                raw_points = curve_raw
                p1, p2 = curve_now
                s = Segment((p1, p2))
                direction = s.direction
                i_dire = np.argmin([Angle.diff_real(dire, direction) for dire in direction_square])
                # 最佳方向
                dire_best = direction_square[i_dire]
                l_old = Line.transform((p1, p2), '2p_ABC')
                # 由于泰森图的特征，仅取出中间的部分作为计算
                i_1_8 = int(len(raw_points) / 8)
                i_1_4 = int(len(raw_points) / 4)
                i_1_2 = int(len(raw_points) / 2)
                i_3_4 = int(len(raw_points) * 3 / 4)
                i_7_8 = int(len(raw_points) * 7 / 8)
                i_1_1 = len(raw_points) - 1
                # 计算三种不同的中点
                mid_list = [[raw_points[i_1_2], dire_best, i_1_8, i_7_8],
                            [np.mean(raw_points, axis=0), dire_best, i_1_8, i_7_8],
                            [raw_points[0], dire_best, 0, i_1_2],
                            [raw_points[i_1_4], dire_best, 0, i_1_2]]
                # 计算误差
                error_list = []
                for mid_info in mid_list:
                    p_mid, dire_now, i_start, i_end = mid_info
                    s_new = Segment.build_from_direction(dire_best)
                    s_new.move(p_mid)
                    l_now = Line.transform(s_new, '2p_ABC')
                    old_list = list(Curve.lines_distance(raw_points, [l_old], rst_type='list'))
                    now_list = list(Curve.lines_distance(raw_points, [l_now], rst_type='list'))
                    error_old = sum(old_list[i_start:i_end])
                    error_now = sum(now_list[i_start:i_end]) * error_factor
                    diff = error_now - error_old
                    error_list.append([diff, l_now, p_mid, len(raw_points)])
                error_list.sort(key=lambda info: info[0])
                diff, l_now, p_mid, len_raw = error_list[0]
                if diff <= 0:
                    # 找到更优解
                    s_new = Segment.build_from_direction(dire_best)
                    s_new.move(p_mid)
                    direciton = s_new.direction
                    posi1 = Point.position(p1, direciton)
                    posi2 = Point.position(p2, direciton)
                    s = Segment([p1, p2])
                    if posi1 < posi2:
                        flag_posi = True
                        p_min, p_max = posi1, posi2
                    else:
                        flag_posi = False
                        p_min, p_max = posi2, posi1
                    s_new.feature_fix([p_min, p_max], 'minmax')
                    p1_new, p2_new = s_new
                    if flag_posi:
                        p1.transfer(p1_new)
                        p2.transfer(p2_new)
                    else:
                        p1.transfer(p2_new)
                        p2.transfer(p1_new)
            else:
                # 第一步：所有线段绕中心旋转，判断是否误差更小
                # 对曲线进行分组
                point_group, _ = Curve.raw_point_relation(curve_now, curve_raw)
                # 计算原本所有线段对应的直线
                l_list = []
                for i in range(len(curve_now) - 1):
                    p_now = curve_now[i]
                    p_next = curve_now[i + 1]
                    l_list.append(Line.transform((p_now, p_next), '2p_ABC'))
                # 计算摆正自己的误差是多少
                info_list = []
                for i in range(len(curve_now) - 1):
                    p_now = curve_now[i]
                    p_next = curve_now[i + 1]
                    raw_points = point_group[i]
                    s = Segment([p_now, p_next])
                    direction = s.angle_raw()
                    i_dire = np.argmin([Angle.diff_real(dire, direction) for dire in direction_square])
                    dire_best = direction_square[i_dire]
                    diff, l_old, l_now, p_mid, len_raw = calc_straightize_min_error(p_now, p_next, raw_points, dire_best)
                    if diff <= 0:
                        # print(i, '更好')
                        info_list.append([l_now, p_mid, dire_best, len_raw, True])
                    else:
                        # print(i, '不好')
                        info_list.append([l_old, p_mid, dire_best, len_raw, False])
                # 计算新的交点
                same_dire_ind_set = set()
                for i in range(len(curve_now) - 2):
                    l_now, p_mid_now, dire_now, _, status_now = info_list[i]
                    l_next, p_mid_next, dire_next, _, status_next = info_list[i + 1]
                    p_inter = Line.interpoint(l_now, l_next)
                    flag_normal = False
                    flag_changed = False  # 同方向线数量是否改变
                    if p_inter != -1:
                        # 判断是否交点是否合理
                        flag_normal_back = False
                        flag_normal_forward = False
                        # 向后判断
                        p_last = curve_now[i]
                        p_now = curve_now[i + 1]
                        s_last = Segment([p_last, p_now])
                        direction_last = s_last.direction
                        posi_now = Point.position(p_now, direction_last)
                        posi_last = Point.position(p_last, direction_last)
                        posi_inter = Point.position(p_inter, direction_last)
                        if posi_now > posi_last:
                            if posi_inter > posi_last:
                                flag_normal_back = True
                        if posi_now < posi_last:
                            if posi_inter < posi_last:
                                flag_normal_back = True
                        # 向前判断
                        p_now = curve_now[i + 1]
                        p_next = curve_now[i + 2]
                        s_next = Segment([p_next, p_now])
                        direction_next = s_next.direction
                        posi_now = Point.position(p_now, direction_next)
                        posi_next = Point.position(p_next, direction_next)
                        posi_inter = Point.position(p_inter, direction_next)
                        if posi_now > posi_next:
                            if posi_inter > posi_next:
                                flag_normal_forward = True
                        if posi_now < posi_next:
                            if posi_inter < posi_next:
                                flag_normal_forward = True
                        if flag_normal_forward and flag_normal_back:
                            flag_normal = True
                    if flag_normal:
                        # 交点合理
                        if i == 0:
                            # 修改起点，使得方向正常
                            p_now = curve_now[0]
                            p_now2 = Line.verticalpoint(p_now, l_now)
                            p_now.transfer(p_now2)
                        if i == len(curve_now) - 3:
                            # 修改终点
                            p_now = curve_now[-1]
                            p_now2 = Line.verticalpoint(p_now, l_next)
                            p_now.transfer(p_now2)
                        curve_now[i + 1].transfer(p_inter)
                    else:
                        # 没有交点或者交点不合理
                        if dire_now == dire_next and status_now and status_next:
                            # 如果方向相同，并且规范直线存在的情况下
                            n_old = len(same_dire_ind_set)
                            same_dire_ind_set.update({i, i + 1})
                            n_new = len(same_dire_ind_set)
                            if n_new > n_old:
                                flag_changed = True
                    # 判断条件
                    # 如果数量没有发生变动
                    flag1 = flag_changed is False
                    # 如果数量发生变动，且到了最后一个点
                    flag2 = flag_changed is True and i == len(curve_now) - 3
                    if flag1 or flag2:
                        if len(same_dire_ind_set) > 0:
                            ind_list = list(same_dire_ind_set)
                            j_min = min(ind_list)
                            j_max = max(ind_list)
                            info_list_now = info_list[j_min:j_max + 1]
                            # 选择raw点数量最多的直线
                            info_list_now.sort(key=lambda info: info[3], reverse=True)
                            l_now = info_list_now[0][0]
                            if flag2:
                                # 此时也要考虑最后一个点，因为结束了
                                ind_list.append(len(curve_now) - 1)
                            # 计算新的位置
                            for j in ind_list:
                                p_now = curve_now[j]
                                p_now2 = Line.verticalpoint(p_now, l_now)
                                curve_now[j].transfer(p_now2)
                            # 清空
                            same_dire_ind_set = set()
                # 如果是闭合曲线，那么计算第一条线和最后一条线的交点
                if curve_raw[0] == curve_raw[-1]:
                    p_s1 = curve_now[0]
                    p_s2 = curve_now[1]
                    p_e1 = curve_now[-1]
                    p_e2 = curve_now[-2]
                    l1 = Line.transform((p_s1, p_s2), '2p_ABC')
                    l2 = Line.transform((p_e1, p_e2), '2p_ABC')
                    p_inter = Line.interpoint(l1, l2)
                    # 如果没有交点，则是两条直线刚好平行
                    if p_inter != -1:
                        # 存在交点，因为在拐角处
                        curve_now[0].transfer(p_inter)
                        curve_now[-1].transfer(p_inter)

    @staticmethod
    def point_wedding_for_voronoi(curve_dict, epsilon=5, epsilon_make=5, addpoint_set=set()):
        '''
        将曲线直线化
        :param curve_dict: 曲线字典。以起点-终点为键保存一条曲线。双向字典
        :param epsilon: 点抽稀的参数
        :param epsilon_make: 构造线段的抽稀参数
        :param addpoint_set: 需要加入关键点的点集合
        :return: segment_list: 抽稀后的结果
        '''

        def find_reverse(p_start, p_end, i_route):
            # 已知起点p_start和终点p_end以及当前第i_route条路线
            # 返回起点p_end和终点p_start以及对应第几条路线
            for i, curve in enumerate(curve_dict[(p_end, p_start)]):
                curve_now = curve[::-1]
                if curve_now == curve_dict[(p_start, p_end)][i_route]:
                    return i

        def walked_record(walked_set, p, p_now, i_now):
            # 记录行走路线，同时将反过来的路线标记为已行走
            # 从p点出发，到p_now结束，走第i_now条路线
            walked_set.add((p, p_now, i_now))
            # 计算从p点出发，到p_now结束的逆转
            i = find_reverse(p, p_now, i_now)
            walked_set.add((p_now, p, i))

        def search(p_now, walked_set, p_walked, curve_all, curve_info, num_all, flag):
            j = 0
            while True:
                j += 1
                next_list2 = relation_dict[p_now]
                # 计算哪一个分支是rdp总和最小的
                info_list2 = []
                for p_next2 in next_list2:
                    if flag == 'forward':
                        for i, curve in enumerate(curve_dict[(p_now, p_next2)]):
                            if (p_now, p_next2, i) not in walked_set:
                                curve_now = curve_all + curve
                                curve_rdp, _ = Curve.rdp(curve_now, epsilon)
                                num_rdp = len(curve_rdp) - 1
                                num_second = rdpnum_dict[(p_now, p_next2)][i]
                                # 数值分别是合并后增加的线段量，合并后rdp线段量，合并前rdp线段总量
                                info_list2.append(
                                    [p_next2, i, num_rdp - (num_all + num_second), num_rdp, num_all + num_second])
                    elif flag == 'backward':
                        for i, curve in enumerate(curve_dict[(p_next2, p_now)]):
                            if (p_next2, p_now, i) not in walked_set:
                                curve_now = curve + curve_all
                                # s_temp = Curves.segmentize([curve_now], rdp=False)
                                # Segments.plot(s_temp, marked=True)
                                curve_rdp, _ = Curve.rdp(curve_now, epsilon)
                                num_rdp = len(curve_rdp) - 1
                                num_second = rdpnum_dict[(p_next2, p_now)][i]
                                # 数值分别是合并后增加的线段量，合并后rdp线段量，合并前rdp线段总量
                                info_list2.append(
                                    [p_next2, i, num_rdp - (num_all + num_second), num_rdp, num_all + num_second])
                    else:
                        raise NotImplementedError
                if len(info_list2) > 0:
                    info_list2.sort(key=lambda x: x[2])
                    p_raw, i_raw, delta, num_raw, _ = info_list2[0]
                    if delta < 0:
                        # 线段总量减少
                        if flag == 'forward':
                            curve_n = curve_dict[(p_now, p_raw)][i_raw]
                            curve_all = curve_all + curve_n[1:]
                            curve_info.append(curve_n)
                            walked_record(walked_set, p_now, p_raw, i_raw)
                            p_walked.append(p_raw)
                        elif flag == 'backward':
                            curve_n = curve_dict[(p_raw, p_now)][i_raw]
                            curve_all = curve_n[:-1] + curve_all
                            curve_info.insert(0, curve_n)
                            walked_record(walked_set, p_raw, p_now, i_raw)
                            p_walked.insert(0, p_raw)
                        else:
                            raise NotImplementedError
                        p_now = p_raw
                        num_all = num_raw
                    else:
                        # 线段总量没有减少
                        break
                else:
                    # 没有找到下一个点
                    break
            return curve_all, num_all

        # 计算每条曲线的rdp后线段数量
        rdpnum_dict = {}
        for key in curve_dict:
            curves = curve_dict[key]
            rdpnum_list = []
            for curve in curves:
                curve_rdp, _ = Curve.rdp(curve, epsilon)
                rdpnum_list.append(len(curve_rdp) - 1)
            rdpnum_dict[key] = rdpnum_list
        # 计算关系字典
        relation_dict = {}
        for key in curve_dict:
            p1, p2 = key
            dict_add(relation_dict, p1, p2, 'append')
            dict_add(relation_dict, p2, p1, 'append')
        # 从任意一个起点出发
        walked_set = set()  # 走过的地方
        key_list = list(relation_dict.keys())
        curve_group = []  # 抽稀曲线集合
        for j, p in enumerate(key_list):
            while True:
                next_list = relation_dict[p]
                # 选一个没有走过的点
                info_list = []
                for p_next in next_list:
                    for i, curve in enumerate(curve_dict[(p, p_next)]):
                        if (p, p_next, i) not in walked_set:
                            info_list.append([p_next, i])
                if len(info_list) == 0:
                    break
                else:
                    p_walked = [p]  # 当前走过的点
                    p_now, i_now = info_list[0]
                    # 提取路径
                    curve_all = curve_dict[(p, p_now)][i_now]
                    curve_info = [curve_all]  # 记录曲线附属的关键点
                    num_all = rdpnum_dict[(p, p_now)][i_now]
                    # 记录行走路线，同时将反过来的路线标记为已行走
                    walked_record(walked_set, p, p_now, i_now)
                    p_walked.append(p_now)
                    # 向前搜索
                    curve_all, num_all = search(p_now, walked_set, p_walked, curve_all, curve_info, num_all, 'forward')
                    # s_temp = Curves.segmentize([curve_all], rdp=False)
                    # Segments.plot(s_temp, marked=True)
                    # 完成往前搜索以后，观察一下回到原点那个方向是可行的
                    info_list = []
                    for p_next in next_list:
                        for i, curve in enumerate(curve_dict[(p_next, p)]):
                            if (p_next, p, i) not in walked_set:
                                curve_now = curve + curve_all
                                curve_rdp = Curve.rdp(curve_now, epsilon)
                                num_rdp = len(curve_rdp) - 1
                                num_second = rdpnum_dict[(p_next, p)][i]
                                # 数值分别是合并后增加的线段量，合并后rdp线段量，合并前rdp线段总量
                                info_list.append(
                                    [p_next, i, num_rdp - (num_all + num_second), num_rdp, num_all + num_second])
                    if len(info_list) > 0:
                        info_list.sort(key=lambda x: x[2])
                        p_now, i_now, delta, num_now, _ = info_list[0]
                        if delta < 0:
                            # 往后搜索
                            # 记录行走路线，同时将反过来的路线标记为已行走
                            walked_record(walked_set, p_now, p, i_now)
                            p_walked.insert(0, p_now)
                            curve_n = curve_dict[(p_now, p)][i_now]
                            curve_all = curve_n[:-1] + curve_all
                            curve_info.insert(0, curve_n)
                            # 往后进行搜索
                            curve_all, num_all = search(p_now, walked_set, p_walked, curve_all, curve_info, num_now, 'backward')
                            # s_temp = Curves.segmentize([curve_all], rdp=False)
                            # Segments.plot(s_temp, marked=True)
                    # 曲线抽稀
                    curve_new, i_start = Curve.rdp(curve_all, epsilon_make)
                    curve_new = Curve(curve_new, hold=False)
                    curve_new.raw = curve_all  # 原始点
                    # 计算有哪些附加点是在曲线上的关键点
                    p_walked_add = list(addpoint_set & set(list_tuplize(curve_all)))
                    # 计算关键点在曲线中的位置
                    keypoint_index = []
                    curve_seg = []
                    for curve_part in curve_info:
                        i_n1 = (len(curve_seg) - i_start) % len(curve_all)
                        i_n2 = (len(curve_seg) + len(curve_part) - 1 - i_start) % len(curve_all)
                        if i_n1 not in keypoint_index:
                            keypoint_index.append(i_n1)
                        if i_n2 not in keypoint_index:
                            keypoint_index.append(i_n2)
                        curve_seg.extend(curve_part[:-1])
                    curve_new.keypoint_index = keypoint_index
                    curve_new.p_walked = p_walked
                    curve_new.p_walked_add = p_walked_add
                    curve_group.append(curve_new)
        # 计算所拥有的分叉点
        p_walked_list = []
        for i_curve in range(len(curve_group)):
            curve_now = curve_group[i_curve]
            p_walked_list.extend(curve_now.p_walked)
        p_walked_set = set(p_walked_list)
        return curve_group, p_walked_set


# 平行线段
class Parallel:
    # 平行线类（基础是线段类segment）

    def __init__(self, pp_segment):
        # pp_segment是两个Segment的对象
        if pp_segment[0].locate > pp_segment[1].locate:
            # 按照LOCATE从小到大对线进行排序
            pp_segment = [pp_segment[1], pp_segment[0]]
        self.segment = pp_segment
        self.locate_min = pp_segment[0].locate  # locate最小值
        self.locate_max = pp_segment[1].locate  # locate最大值
        self.length = pp_segment[0].length  # 传进来的线段必须是相等长度的
        self.direction = pp_segment[0].direction
        self.mid = 0  # 结构中线的locate值
        self.p_min = pp_segment[0].p_min
        self.p_max = pp_segment[0].p_max
        self.ind = 0
        self.flag = 0
        self.p11 = pp_segment[0].p1
        self.p12 = pp_segment[0].p2
        self.p21 = pp_segment[1].p1
        self.p22 = pp_segment[1].p2

    def point_extract(self):
        # 返回平行线段端点的顺时针排序点列表
        p_list = [self.p11, self.p12, self.p21, self.p22]
        return Points.sorted(p_list, method='mean')

    def skeletonize(self):
        '''
        将平行线段骨架化，返回平行线的中线
        :return:
        '''
        p11 = self.p11
        p12 = self.p12
        p21 = self.p21
        p22 = self.p22
        p1_mid = Point.combination(p11, p21, 0.5)
        p2_mid = Point.combination(p12, p22, 0.5)
        s = Segment([p1_mid, p2_mid])
        return s

    @list_join
    @list_strize
    def __str__(self):
        return [self.locate_min, self.locate_max, self.direction, self.p_min, self.p_max]

    __repr__ = __str__


# 平行线段集合
class Parallels:

    @staticmethod
    def plot(pp_list, color=[1, 0, 0], fig=[], texted=False):
        if fig == []:
            fig = plt.figure()
        segment_list = []
        for pp in pp_list:
            segment_list.extend(pp.segment)
        Segments.plot(segment_list, color=color, fig=fig, texted=texted)

    @staticmethod
    @time_logger('将平行线段集合骨架化')
    def skeletonize(pp_list):
        '''
        将平行线段集合骨架化
        :param pp_list: 平行线段集合
        :return:
        '''
        skeleton_list = [pp.skeletonize() for pp in pp_list]
        Segments.reset(skeleton_list)
        return skeleton_list


# 多边形
class Polygen(Points):
    # 多边形类

    def __init__(self, p_list=(), hold=False):
        flag = Polygen.area(p_list, 'direction')
        if flag == 0:
            # 逆时针
            p_list = p_list[::-1]
        self.update(p_list, hold)
        self.s_list = self.segmentize()
        self.x_min = 0
        self.y_min = 0
        self.weight = 0  # 多边形的权重，用来做比较、排序使用

    def area(self, output='area'):
        '''
        计算多边形的方向（使用面积法，起点为（0，0））
        :param output: area 输出面积
                       direction 输出方向  1 顺时针 0 逆时针
        '''
        area = 0
        for i, p in enumerate(self):
            x1, y1 = self[i - 1]
            x2, y2 = p
            area += x1 * y2 - x2 * y1
        if output == 'area':
            return math.fabs(area) / 2
        elif output == 'direction':
            if area < 0:
                # 顺时针
                return 1
            else:
                # 逆时针
                return 0

    def clip(shape, border, method):
        '''
        Weiler-Atherton裁剪算法
        :param shape: 被裁剪多边形
        :param border: 裁剪窗口
        :param method: 方法 inner:内裁剪 diff外裁剪
        :return:
        shape = Polygen([(0,0),(0,3),(2,3),(2,1),(1,1),(1,0)])
        border = Polygen([(1,0), (1,2), (3,2), (3,0)])
        '''

        def list_combine(shape, point_s):
            # 将点插入形状当中，按照顺时针顺序
            seq_s = []
            for ind_p, p in enumerate(shape):
                if ind_p in point_s:
                    p_all = point_s[ind_p]
                    d_list = []
                    for p_a in p_all:
                        d_list.append(Point.distance(p, p_a))
                    for i in np.argsort(d_list)[::-1]:  # 距离远的先放
                        # 如果交点不在端点上，才能放入序列当中
                        seq_s.append(p_all[i])
                seq_s.append(p)
            return seq_s

        # fig = Polygens.plot([shape], marked=True)
        # Polygens.plot([border], color=[0,0,1], fig=fig, marked=True)

        # 由于底层误差的影响，必须对位数进行截断
        for poly in [shape, border]:
            for p in poly:
                Point.float_ban(p)

        shape.collapse()
        border.collapse()
        border_v = Polygen.segmentize(border)

        # 计算入点和出点序列
        in_list = []
        out_list = []
        point_s_tmp = {}  # 临时存储
        point_b = {}  # 真正存储
        point_s = {}
        if Point.in_polygen(shape[-1], border) == 0:
            flag = 0  # 表示点在外侧
        else:
            flag = 1  # 表示点在内侧
        path = [shape[-1]]
        i_inter = 0  # 交点索引
        p_inter_list = []
        for ind_s, point in enumerate(shape):
            # shape的边遍历一次
            point = shape[ind_s]
            last_point = shape[ind_s - 1]
            # 跟所有边计算交点
            for ind_b, vector in enumerate(border_v):
                flag1 = Point.in_segment(vector, point)
                flag2 = Point.in_segment(vector, last_point)
                if flag1 == 0 and flag2 == 0:
                    # 只有两个点都不在线段上，才能表示这两个点所形成的线段有可能穿过线段
                    p_inter = Segment.interpoint(Segment.build([last_point, point]), vector)
                    if p_inter != -1:
                        p_inter = Point(p_inter)
                        p_inter_list.append(p_inter)
                        p_inter.type = 'inter'
                        p_inter.ind = i_inter
                        i_inter += 1
                        dict_add(point_s_tmp, ind_s, [p_inter, ind_b], 'append', False)
            # 若有交点，则对交点进行排序
            if ind_s in point_s_tmp:
                info_list = point_s_tmp[ind_s]
                for info in info_list:
                    p_inter = info[0]
                    info.insert(0, Point.distance(point, p_inter))
                info_list.sort(key=lambda info:info[0], reverse=True)  # 距离当前点远的先放
                # 对该边上的所有点判断一次
                for _, info in enumerate(info_list):
                    _, p, ind_b = info
                    flag_inner = Point.in_polygen(point, border)
                    if flag == 1:
                        # 从内侧穿出外侧
                        # print('添加出点', p)
                        if flag_inner == 0:
                            out_list.append(p)
                            dict_add(point_b, ind_b, p, 'append', False)
                            dict_add(point_s, ind_s, p, 'append', False)
                            flag = 0
                    else:
                        # 从外侧穿入内侧
                        # print('添加入点', p)
                        if flag_inner == 1:
                            in_list.append(p)
                            dict_add(point_b, ind_b, p, 'append', False)
                            dict_add(point_s, ind_s, p, 'append', False)
                            flag = 1
                    path.append(p)

        # 按顺时针顺序向序列插入点
        seq_s = list_combine(shape, point_s)
        seq_b = list_combine(border, point_b)
        seq_list = [seq_s, seq_b]

        # 计算两个序列相同交点的所在位置
        ind_dict = {}
        for i_seq, seq in enumerate(seq_list):
            for site, p in enumerate(seq):
                if p.ind not in ind_dict:
                    ind_dict[p.ind] = {}
                if p.type == 'inter':
                    dict_add(ind_dict[p.ind], i_seq, site, 'append')

        # 输出裁剪序列
        if method == 'inner':
            # 内裁剪
            rst = []
            while len(in_list) > 0:
                p = in_list.pop(0)
                p_start = p
                p_now = -1
                i_seq = 0  # 从被裁剪多边形开始
                seq_now = seq_list[i_seq]
                site = ind_dict[p.ind][i_seq][0]
                flag = 'in'
                output = [p_start]
                while p_now != p_start:
                    site += 1
                    if site == len(seq_now):
                        site = 0
                    p_now = seq_now[site]
                    output.append(p_now)
                    ind = p_now.ind
                    if flag == 'in':
                        if p_now in out_list:
                            # 出点，跳到裁剪窗口列表
                            out_list.remove(p_now)
                            i_seq = 1 - i_seq
                            seq_now = seq_list[i_seq]
                            site = ind_dict[ind][i_seq][0]
                            flag = 'out'
                    elif flag == 'out':
                        if p_now in in_list:
                            in_list.remove(p_now)
                            # 入点，跳到被裁剪形状列表
                            i_seq = 1 - i_seq
                            seq_now = seq_list[i_seq]
                            site = ind_dict[ind][i_seq][0]
                            flag = 'in'
                rst.append(output[:-1])
        elif method == 'outer':
            # 外裁剪
            rst = []
            while len(out_list) > 0:
                p = out_list.pop(0)
                p_start = p
                p_now = -1
                i_seq = 0  # 从被裁剪多边形开始
                seq_now = seq_list[i_seq]  # 从被裁剪多边形开始
                site = ind_dict[p.ind][i_seq][0]
                sign = 1  # sign为1时，沿顺时针方向，为-1时，沿逆时针方向
                flag = 'out'  # 当前点是入点还是出点
                output = [p_start]
                while p_now != p_start:
                    site += sign
                    if site == len(seq_now):
                        site = 0
                    p_now = seq_now[site]
                    output.append(p_now)
                    ind = p_now.ind
                    if flag == 'out':
                        if p_now in in_list:
                            in_list.remove(p_now)
                            # 入点，跳到裁剪窗口列表
                            seq_now = seq_b
                            i_seq = 1 - i_seq
                            site = ind_dict[ind][i_seq][0]
                            sign = -1
                            flag = 'in'
                    elif flag == 'in':
                        if p_now in out_list:
                            # 出点，跳到被裁剪形状列表
                            out_list.remove(p_now)
                            seq_now = seq_s
                            i_seq = 1 - i_seq
                            site = ind_dict[ind][i_seq][0]
                            sign = 1
                            flag = 'out'
                rst.append(output[:-1])
            if len(rst) == 0:
                # 没有产生任何裁剪，则判断
                rst = [shape]
        elif method == 'union':
            # 并集
            rst = []
            while len(in_list) > 0:
                p = in_list.pop(0)
                p_start = p
                p_now = -1
                i_seq = 1  # 从裁剪窗口开始
                seq_now = seq_list[i_seq]
                site = ind_dict[p.ind][i_seq][0]
                flag = 'in'
                output = [p_start]
                while p_now != p_start:
                    site += 1
                    if site == len(seq_now):
                        site = 0
                    p_now = seq_now[site]
                    output.append(p_now)
                    ind = p_now.ind
                    if flag == 'in':
                        if p_now in out_list:
                            # 出点，跳到裁剪窗口列表
                            out_list.remove(p_now)
                            i_seq = 1 - i_seq
                            seq_now = seq_list[i_seq]
                            site = ind_dict[ind][i_seq][0]
                            flag = 'out'
                    elif flag == 'out':
                        if p_now in in_list:
                            in_list.remove(p_now)
                            # 入点，跳到被裁剪形状列表
                            i_seq = 1 - i_seq
                            seq_now = seq_list[i_seq]
                            site = ind_dict[ind][i_seq][0]
                            flag = 'in'
                rst.append(output[:-1])
        else:
            raise NotImplementedError
        # 清理冗余点
        rst = Polygens(rst)
        output = []
        for ind_r, r in enumerate(rst):
            r.collapse()
            if len(r) > 0:
                output.append(r)
        # Polygens.plot(output, marked=True)
        return output

    def coin_clear(self):
        # 如果存在点重合的情形，则去掉短的冗余边
        # self = [(0,0), (1,0), (1,1), (2, 1), (1, 2), (1,1),(0,1)]
        shape = list_tuplize(self)
        shape_unique = list_unique(shape)
        if len(shape) == len(shape_unique):
            # 正常
            pass
        else:
            # 存在点重合
            while True:
                info_list = [[p, shape.count(p)] for p in shape]
                info_list.sort(key=lambda info: info[1], reverse=True)
                p_now, count_now = info_list[0]
                if count_now >= 2:
                    # 计算存在多少条边
                    index_list = [i for i, p in enumerate(shape) if p == p_now]
                    i_head = index_list[0]
                    index_list += [len(shape) + i_head]
                    diff_info_list = []
                    for i in range(len(index_list) - 1):
                        diff = index_list[i + 1] - index_list[i]
                        diff_info_list.append([diff, index_list[i], index_list[i + 1]])
                    diff_info_list.sort(key=lambda info: info[0])
                    # 进行清理
                    _, i_start, i_end = diff_info_list[0]
                    delete_list = [i_now%len(shape) for i_now in list(range(i_start, i_end))]
                    delete_list.sort(reverse=True)
                    for i_delete in delete_list:
                        del shape[i_delete]
                        del self[i_delete]
                else:
                    break

    def collapse(self):
        # 退化，即清除多边形上的多余点
        # 清理在同一个点上的点
        ind_delete = []
        for i, p in enumerate(self):
            p_last = self[i - 1]
            if p == p_last:
                ind_delete.append(i)
        ind_delete = sorted(ind_delete, reverse=True)
        for ind in ind_delete:
            del self[ind]
        # 清除多边形上的同方向点
        ind_delete = []
        for ind_p, p in enumerate(self):
            p_last = self[ind_p - 1]
            p_next = self[(ind_p + 1)%len(self)]
            angle1 = Vector.angle([p_last, p])
            angle2 = Vector.angle([p, p_next])
            if angle1 == angle2:
                ind_delete.append(ind_p)
        ind_delete = sorted(ind_delete, reverse=True)
        for ind in ind_delete:
            del self[ind]

    def edge_adjust(self, direction_main_list):
        '''
        对多边形各条边进行调整，调整至主方向
        :param direction_main_list: 主方向列表
        :return:
        '''
        abc_list = []
        for i_p, p in enumerate(self):
            p_last = self[i_p - 1]
            angle = Vector.angle([p_last, p]) % math.pi
            diff_list = []
            for direction in direction_main_list:
                diff_list.append(Angle.diff_real(angle, direction, math.pi))
            # 计算角度和中心点
            l_direction = direction_main_list[diff_list.index(min(diff_list))]
            l_center = Point.combination(p, p_last, 0.5)
            abc_list.append(Line.transform([l_center, l_direction], 'pseta_ABC'))
        n = len(abc_list)
        i_now = 0
        self_origin = self[:]
        for ind_l, l in enumerate(abc_list):
            i_next = index_move(ind_l, n, 1)
            l_next = abc_list[i_next]
            interpoint = Line.interpoint(l, l_next)
            flag_ok = 0
            if interpoint != -1:
                if Point.distance(interpoint, self_origin[ind_l], 'direct') < 10:
                    # 如果生成的交点在10像素范围内
                    self[i_now] = Point.float_cut(interpoint)
                    i_now += 1
                    flag_ok = 1
            if flag_ok == 0:
                # 没有产生交点，将列表缩短一位
                del(self[-1])
                abc_list[i_next] = l

    def pack(self, MaxPackDepth=8, MaxPackWidth=8):
        # 对多边形进行自动填补
        for i in range(3):
            for method in ['flat', 'sidehole', 'v_sidehole', 'vertexhole', 'v_vertexhole']:
                self.pack_single(method, MaxPackDepth, MaxPackWidth)

    def pack_single(self, method, MaxPackDepth=8, MaxPackWidth=8):
        '''
        对空间空缺部分进行填补（在一定情况下是裁剪）
        :param method: flat:对开口平齐的洞进行处理
                       sidehole:对边洞进行处理
                       v_sidehole:对v型边洞进行处理
                       vertexhole:对角洞进行处理
                       v_vertexhole:对v型角洞进行处理
                       slope:对斜坡进行处理
        :param MaxPackDepth 最大处理深度
        :param MaxPackWidth 最大填补宽度
        '''

        # 先进行flat填补，因为flat填补与unflat填补的作用范围有重叠
        vectors = Polygen.vetorize(self)
        len_space = len(self)
        sub_list = []  # 替换列表（如果为空即删除）
        lock_ind_list = []  # 如果之前被处理过，则不能再进行处理
        for ind_p, p in enumerate(self):
            i_n3 = index_move(ind_p, len_space, -3)
            i_n2 = index_move(ind_p, len_space, -2)
            i_n1 = index_move(ind_p, len_space, -1)
            i_p1 = index_move(ind_p, len_space, 1)
            i_p2 = index_move(ind_p, len_space, 2)
            i_p3 = index_move(ind_p, len_space, 3)
            i_p4 = index_move(ind_p, len_space, 4)
            vector_last0 = vectors[ind_p]
            vector_last3 = vectors[i_n3]
            vector_last2 = vectors[i_n2]
            vector_last1 = vectors[i_n1]
            vector_next1 = vectors[i_p1]
            vector_next3 = vectors[i_p3]
            vector_next4 = vectors[i_p4]
            if method == 'flat':
                if Point.is_RtInSide(vector_last3, p) == 0:
                    # 填补对象为开口平齐，面积较小的平行区域
                    # 计算坑深度
                    if len({i_n3, i_n2, i_n1, ind_p} & set(lock_ind_list)) == 0:
                        p1, p2, p3, p4 = [self[i_n3], self[i_n2], self[i_n1], self[ind_p]]
                        hole_depth = min([Point.distance(p1, p2), Point.distance(p3, p4)])
                        hole_width = max([Point.distance(p1, p4), Point.distance(p2, p3)])  # 边洞宽度
                        if hole_depth <= MaxPackDepth and hole_width <= MaxPackWidth:
                            sub_list.extend([[i_n2, []], [i_n1, []]])
                            lock_ind_list.extend([i_n3, i_n2, i_n1, ind_p])
            else:
                # 填补对象为作一条平行线，产生一个开口
                l_last0 = Line.transform(vector_last0)
                l_last0_p1_v = Line.transform(self[i_p1], Angle.normal_calc(Vector.angle(vector_last0)))
                l_last1 = Line.transform(vector_last1)
                l_last2 = Line.transform(vector_last2)
                l_last3 = Line.transform(vector_last3)
                l_next1 = Line.transform(vector_next1)
                l_next1_n1_v = Line.transform(self[i_n1], Angle.normal_calc(Vector.angle(vector_next1)))
                l_next3 = Line.transform(vector_next3)
                l_next4 = Line.transform(vector_next4)
                interpoint_pack1 = Line.interpoint(l_last0, l_last3)
                interpoint_pack2 = Line.interpoint(l_next1, l_next4)
                interpoint_pack3 = Line.interpoint(l_next1, l_last2)
                interpoint_pack4 = Line.interpoint(l_next1, l_last1)
                interpoint_pack5 = Line.interpoint(l_last0, l_last2)
                interpoint_pack6 = Line.interpoint(l_next1, l_next3)
                interpoint_pack7 = Line.interpoint(l_next1_n1_v, l_next1)
                interpoint_pack8 = Line.interpoint(l_last0, l_last0_p1_v)
                if method == 'sidehole':
                    if interpoint_pack1 != -1:
                        # 起始点出发逆时针边洞填补
                        interpoint_pack1 = Point.float_cut(interpoint_pack1)
                        if Point.in_segment(vector_last0, interpoint_pack1) == 1:
                            if len({i_n3, i_n2, i_n1} & set(lock_ind_list)) == 0:
                                p1, p2, p3, p4 = [self[i_n3], self[i_n2], self[i_n1], interpoint_pack1]
                                hole_depth = min(
                                    [Point.distance(p1, p2), Point.distance(p3, p4)])  # 边洞深度
                                hole_width = max(
                                    [Point.distance(p1, p4), Point.distance(p2, p3)])  # 边洞宽度
                                if hole_depth <= MaxPackDepth:
                                    if hole_width <= MaxPackWidth:
                                        sub_list.extend([[i_n2, []], [i_n1, interpoint_pack1]])
                                        lock_ind_list.extend([i_n3, i_n2, i_n1])
                    if interpoint_pack2 != -1:
                        # 起始点出发顺时针边洞填补
                        interpoint_pack2 = Point.float_cut(interpoint_pack2)
                        if Point.in_segment(vector_next1, interpoint_pack2) == 1:
                            if len({i_p1, i_p2, i_p3} & set(lock_ind_list)) == 0:
                                p1, p2, p3, p4 = [interpoint_pack2, self[i_p1], self[i_p2], self[i_p3]]
                                hole_depth = min([Point.distance(p1, p2), Point.distance(p3, p4)])
                                hole_width = max(
                                    [Point.distance(p1, p4), Point.distance(p2, p3)])  # 边洞宽度
                                if hole_depth <= MaxPackDepth:
                                    if hole_width <= MaxPackWidth:
                                        sub_list.extend([[i_p2, []], [i_p1, interpoint_pack2]])
                                        lock_ind_list.extend([i_p1, i_p2, i_p3])
                if method == 'vertexhole':
                    if interpoint_pack3 != -1:
                        interpoint_pack3 = Point.float_cut(interpoint_pack3)
                        if Point.in_segment(vector_next1, interpoint_pack3) == 0 and Point.in_segment(vector_last2,
                                                                                                      interpoint_pack3) == 0:
                            # 起始点出发逆时针角洞填补
                            if len({i_n2, i_n1, ind_p} & set(lock_ind_list)) == 0:
                                p1, p2, p3, p4 = [interpoint_pack3, self[i_n2], self[i_n1], self[ind_p]]
                                hole_depth = min([Point.distance(p2, p3), Point.distance(p4, p3)])
                                hole_width = max(
                                    [Point.distance(p2, p3), Point.distance(p4, p3)])  # 边洞宽度
                                if hole_depth <= MaxPackDepth:
                                    if hole_width <= MaxPackWidth:
                                        sub_list.extend([[i_n2, interpoint_pack3], [i_n1, []], [ind_p, []]])
                                        lock_ind_list.extend([i_n2, i_n1, ind_p])
                if method == 'v_vertexhole':
                    if interpoint_pack4 != -1:
                        interpoint_pack4 = Point.float_cut(interpoint_pack4)
                        if Point.in_segment(vector_next1, interpoint_pack4) == 0 and Point.in_segment(vector_last1,
                                                                                                      interpoint_pack4) == 0:
                            # 起始点出发逆时针角洞填补
                            if len({i_n1, ind_p} & set(lock_ind_list)) == 0:
                                p1, p2, p3 = [self[i_n1], interpoint_pack4, self[ind_p]]
                                hole_depth = max([Point.distance(p1, p2), Point.distance(p3, p2)])
                                if hole_depth <= MaxPackDepth:
                                    sub_list.extend([[i_n1, []], [ind_p, interpoint_pack4]])
                                    lock_ind_list.extend([i_n1, ind_p])
                if method == 'v_sidehole':
                    # 解决V型边洞的填充
                    if interpoint_pack5 != -1:
                        # 起始点出发逆时针边洞填补
                        interpoint_pack5 = Point.float_cut(interpoint_pack5)
                        if Point.in_segment(vector_last0, interpoint_pack5) == 1:
                            if len({i_n2, i_n1} & set(lock_ind_list)) == 0:
                                p1, p2, p3 = [self[i_n2], self[i_n1], interpoint_pack5]
                                hole_depth = min(
                                    [Point.distance(p1, p2), Point.distance(p2, p3)])  # 边洞深度
                                hole_width = Point.distance(p1, p3)  # 边洞宽度
                                if hole_depth <= MaxPackDepth:
                                    if hole_width <= MaxPackWidth:
                                        sub_list.extend([[i_n1, interpoint_pack5]])
                                        lock_ind_list.extend([i_n2, i_n1])
                    if interpoint_pack6 != -1:
                        # 起始点出发顺时针边洞填补
                        interpoint_pack6 = Point.float_cut(interpoint_pack6)
                        if Point.in_segment(vector_next1, interpoint_pack6) == 1:
                            if len({i_p2, i_p1} & set(lock_ind_list)) == 0:
                                p1, p2, p3 = [self[i_p2], self[i_p1], interpoint_pack6]
                                hole_depth = min(
                                    [Point.distance(p1, p2), Point.distance(p2, p3)])  # 边洞深度
                                hole_width = Point.distance(p1, p3)  # 边洞宽度
                                if hole_depth <= MaxPackDepth:
                                    if hole_width <= MaxPackWidth:
                                        sub_list.extend([[i_p1, interpoint_pack6]])
                                        lock_ind_list.extend([i_p2, i_p1])
                    if method == 'slope':
                        # 解决斜坡的逆时针填充
                        if interpoint_pack7 != -1:
                            interpoint_pack7 = Point.float_cut(interpoint_pack7)
                            if Point.in_segment(vector_next1, interpoint_pack7) == 0:
                                if len({i_n1, ind_p} & set(lock_ind_list)) == 0:
                                    p1, p2, p3 = [self[i_n1], interpoint_pack7, self[ind_p]]
                                    hole_depth = min(
                                        [Point.distance(p1, p2), Point.distance(p2, p3)])  # 斜坡深度
                                    if hole_depth <= MaxPackDepth:
                                        sub_list.extend([[ind_p, interpoint_pack7]])
                                        lock_ind_list.extend([i_n1, ind_p])
                        # 解决斜坡的顺时针填充
                        if interpoint_pack8 != -1:
                            interpoint_pack8 = Point.float_cut(interpoint_pack8)
                            if Point.in_segment(vector_last0, interpoint_pack8) == 0:
                                if len({ind_p, i_p1} & set(lock_ind_list)) == 0:
                                    p1, p2, p3 = [self[ind_p], interpoint_pack8, self[i_p1]]
                                    hole_depth = min(
                                        [Point.distance(p1, p2), Point.distance(p2, p3)])  # 斜坡深度
                                    if hole_depth <= MaxPackDepth:
                                        sub_list.extend([[i_p1, interpoint_pack8]])
                                        lock_ind_list.extend([ind_p, i_p1])

        sub_list = sorted(sub_list, key=itemgetter(0), reverse=True)
        for term in sub_list:
            i, content = term
            if content == []:
                del self[i]
            else:
                self[i] = content
        # 退化
        self.collapse()

    def plot(self, color=[1, 0, 0], fig=[], colortype='', linewidth=1, marked=False, text=''):
        # 画单个多边形
        if fig == []:
            fig = plt.figure()
        vectors = Polygen.vetorize(self)
        if colortype == 'gradually':
            color = [c * 0.7 for c in color]
        elif colortype == 'ramdom':
            color = [(random.randint(0, 100)) / 200 for i in range(3)]
        for v in vectors:
            x1, y1 = v[0]
            x2, y2 = v[1]
            if marked:
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, marker="o", markersize=5)
            else:
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)
        if len(text) > 0:
            x_c, y_c = Polygen.gravity(self)
            plt.text(x_c, y_c, text, fontsize=20)
        return fig

    def rotate(self, seta):
        '''
        对多边形做坐标轴旋转，逆时针旋转seta角度。
        :param seta: 顺时针旋转的角度
        :return:
        '''
        x_min = 0
        y_min = 0
        for i_p, p in enumerate(self):
            x_p, y_p = p
            x_n = x_p * math.cos(seta) + y_p * math.sin(seta)
            y_n = y_p * math.cos(seta) - x_p * math.sin(seta)
            self[i_p] = Point.float_cut([x_n, y_n])
            if x_n < x_min:
                x_min = x_n
            if y_n < y_min:
                y_min = y_n
        self.x_min = x_min
        self.y_min = y_min

    def segmentize(self):
        # 返回多边形的线段形式
        s_list = []
        for ind, _ in enumerate(self):
            s = Segment.build([self[ind - 1], self[ind]])
            s_list.append(s)
        return s_list

    def tuplify(self):
        # 将多边形元组化
        rst = []
        for p in self:
            rst.append(tuple(p))
        return tuple(rst)

    def update(self, p_list=(), hold=False):
        # 更新几何数据
        super().__init__([])
        for p in p_list:
            if hold:
                # 不新建点，保留点的信息
                self.append(p)
            else:
                # 直接新建点
                self.append(Point(p))

    def vetorize(poly):
        '''
        将多边形的边界点集变成向量形式，输入必须保证是顺时针
        :param poly: 二维点集
        :return:
        '''
        return [[poly[ind - 1], poly[ind]] for ind, _ in enumerate(poly)]

    # 指标计算
    @staticmethod
    def build(p_list, type=''):
        # 多边形生成器，保留点的属性
        poly = Polygen(p_list, hold=True)
        poly.type = type
        return poly

    @staticmethod
    def clear_single_edge(poly):
        # 清理多边形中所有悬空结构的线
        flag_end = 0
        while flag_end == 0:
            s_list = poly.s_list
            delete_list = []
            for i, s in enumerate(s_list):
                s_last = s_list[i - 1]
                s1 = Segment(s)
                s2 = Segment(s_last)
                if tuple(s1) == tuple(s2):
                    delete_list.extend([i - 1, i])
            if len(delete_list) == 0:
                flag_end = 1
            else:
                delete_list = sorted(delete_list, reverse=True)
                for i in delete_list:
                    del poly[i]
                poly = Polygen(poly)
        return poly

    # @staticmethod
    # def difference(poly1, poly2):
    #     # 计算poly1减掉poly2的效果
    #     shape1 = SPolygon(poly1)
    #     shape2 = SPolygon(poly2)
    #     diff = shape1.difference(shape2)
    #     if isinstance(diff, SPolygon):
    #         return Polygen.from_spoly(diff)
    #     else:
    #         return []

    @staticmethod
    def fence(shape):
        # 计算多边形周长
        acc_c = 0
        for ind_p, p in enumerate(shape):
            p_last = shape[ind_p - 1]
            acc_c += Point.distance(p, p_last, 'direct')
        return acc_c

    @staticmethod
    def from_multi_spoly(multi_spoly):
        spoly_list = []
        for spoly in multi_spoly:
            spoly_list.append(Polygen.from_spoly(spoly))
        return spoly_list

    @staticmethod
    def from_spoly(spoly):
        x_list, y_list = spoly.exterior.xy
        poly = []
        for x, y in zip(x_list, y_list):
            poly.append((x, y))
        del poly[-1]
        return poly

    @staticmethod
    @lru_cache(None)
    def generator(p_list):
        # 多边形生成器（不保存点的信息）
        poly = Polygen(p_list)
        return poly

    # @staticmethod
    # def intersection(poly1, poly2):
    #     # 计算intersection的效果
    #     shape1 = SPolygon(poly1)
    #     shape2 = SPolygon(poly2)
    #     inter = shape1.intersection(shape2)
    #     if isinstance(inter, SPolygon):
    #         return Polygen.from_spoly(inter)
    #     else:
    #         return []

    # @staticmethod
    # def union(poly1, poly2):
    #     # 计算intersection的效果
    #     shape1 = SPolygon(poly1)
    #     shape2 = SPolygon(poly2)
    #     union = shape1.union(shape2)
    #     poly_list = []
    #     if isinstance(union, SPolygon):
    #         p_list = Polygen.from_spoly(union)
    #         poly_list.append(p_list)
    #     else:
    #         for spoly in union:
    #             p_list = Polygen.from_spoly(spoly)
    #             poly_list.append(p_list)
    #     return poly_list


class Polygens(list):
    # 多边形列表类

    def __init__(self, poly_list, hold=False):
        super().__init__([])
        for poly in poly_list:
            self.append(Polygen(poly, hold))
        self.reindex()

    def area(self):
        # 计算总面积
        return sum([poly.area() for poly in self])

    def clear_strange(self):
        # 将边数小于3的多边形去掉
        delete_list = []
        for i, poly in enumerate(self):
            if len(poly) < 3:
                delete_list.append(i)
        delete_list.sort(reverse=True)
        for i in delete_list:
            del self[i]

    def data_geom(self):
        # 输出多边形的集合数据
        return [list(poly) for poly in self]

    def plot(self, color=[1, 0, 0], fig=[], colortype='', linewidth=1, texted=False, marked=False, interval=0, inverted=False):
        # 画多个多边形
        if fig == []:
            fig = plt.figure()
        for ind_s, shape in enumerate(self):
            if texted:
                text = r'w' + str(shape.ind)
            else:
                text = ''
            shape.plot(color=color, fig=fig, colortype=colortype, linewidth=linewidth, marked=marked, text=text)
            if interval > 0:
                print(ind_s)
                plt.pause(interval)
        if inverted:
            fig.gca().invert_yaxis()
        plt.axis('scaled')
        return fig

    def reindex(self):
        '''
        重置列表索引
        :return:
        '''

        for i, l in enumerate(self):
            l.ind = i

    @staticmethod
    def clear_single_edge(poly_list):
        # 清理多边形列表中所有多边形的悬空结构的线
        for i, poly in enumerate(poly_list):
            poly_list[i] = Polygen.clear_single_edge(poly)
        return poly_list

    @staticmethod
    def reset(room_list):
        '''
        重置直线列表
        :param segment_list: 线段列表
        :return:
        '''

        for i, room in enumerate(room_list):
            room.ind = i

    @staticmethod
    @time_logger('线段房间分类')
    def room_edge_classify(room_list):
        '''
        对空间多边形的边进行空间分类，直接利用字典进行（注意这种空间多边形是经过segments.find_polygen计算后的结果）
        :param room_list: 空间多边形
        :return:
        '''
        segment_room_dict = {}  # 线段所属房间信息
        for i, room in enumerate(room_list):
            s_list = room.s_list
            for s in s_list:
                p1, p2 = s
                s_n = Segment.build([p1, p2])
                dict_add(segment_room_dict, s_n.tuplify(), i, 'append')
        return segment_room_dict

    @staticmethod
    def skeletonize(poly_list):
        # 将多边形拆成骨架。如果重复的骨架只算一条
        s_list = []
        for poly in poly_list:
            poly_slist = poly.s_list
            for s in poly_slist:
                p1, p2 = s
                s_list.append(Segment.build([p1, p2]))
        # 减少边的数量，形式一样的骨架只保留一条
        ss_dict = Skeletons.transform(s_list, method='sl_ssdict')
        s_list = Skeletons.transform(ss_dict, method='ssdict_sl')
        return s_list

    @staticmethod
    def sort_for_weight(room_list, reverse=False):
        room_list_new = sorted(room_list, key=lambda x: x.weight, reverse=reverse)
        Polygens.reset(room_list_new)
        return room_list_new

    @staticmethod
    def transform_to_contour(polygen_list):
        # 将多边形转化为cv2.drawcontours可以绘图的形式
        polygen_list_new = []
        for polygen in polygen_list:
            poly = []
            for p in polygen:
                poly.append([p])
            polygen_list_new.append(poly)
        contours = [np.array(poly, 'int32') for poly in polygen_list]
        return contours


# 矩形
class Rect(Polygen):

    @staticmethod
    def rotation_occupy(high, width, angle):
        '''
        计算矩形旋转后占据的长和宽
        旋转角度α，矩形高度h，宽度w
        width = w*cosα + h*sinα
        height = h*cosα + w*sinα
        :return:
        '''
        angle_mod = angle % 180
        if angle_mod < 90:
            v_angle = Angle.to_value(angle_mod)
            width_new = width * math.cos(v_angle) + high * math.sin(v_angle)
            high_new = high * math.cos(v_angle) + width * math.sin(v_angle)
        else:
            v_angle = Angle.to_value(angle_mod - 90)
            width_new = high * math.cos(v_angle) + width * math.sin(v_angle)
            high_new = width * math.cos(v_angle) + high * math.sin(v_angle)
        return int(high_new), int(width_new)