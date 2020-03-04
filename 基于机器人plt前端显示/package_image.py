# 图像处理库
from package_geometry2 import *
import cv2
import base64
import PIL
from PIL import Image
import io
import json
from queue import PriorityQueue
from scipy import ndimage as ndi


class Figure(np.ndarray):
    '''
    图像类，继承numpy库的np.ndarray类
    '''

    def __init__(self, shape, dtype, buffer):
        super().__init__()
        if len(self) > 1:
            high, width, channel = self.__img_size()
            self.high = high
            self.width = width
            self.channel = channel

    @time_logger('astar')
    def astar(self, p_start, p_target):
        # 利用A*算法，计算两个点之间的最短路径
        # self = np.ones((6, 10))
        # p_start = (2, 2)
        # p_target = (8, 4)
        # self[1:5, 5] = 0
        # self[1, 4] = 0
        # self[4, 4] = 0
        high, width = self.shape
        p_start = Point(p_start)
        p_start.f = Point.distance(p_start, p_target, 'manhattan')
        p_start.g = 0
        p_start.children = []
        p_dict = {tuple(p_start): p_start}
        open_list = [p_start]
        close_list = []
        flag_end = False
        flag_over = False
        j = 0
        while len(open_list) > 0 and flag_end is False:
            j += 1
            if j > 10000:
                # 步数过长
                p_list = []
                status = False
                flag_over = True
                raise Exception
            p_now = open_list.pop(0)
            close_list.append(p_now)
            for p_next in Point.neighbor(p_now, '8-dire'):
                x, y = p_next
                if 0 <= x < width and 0 <= y < high and self[y, x] > 0:
                    # 这里f值考虑地图的权重
                    g_new = p_dict[tuple(p_now)].g + Point.distance(p_now, p_next) * self[y, x]
                    f_new = g_new + Point.distance(p_next, p_target, 'manhattan')
                    # 更新数据
                    if tuple(p_next) not in p_dict:
                        p = Point(p_next)
                        p.f = f_new
                        p.g = g_new
                        p.father = p_now
                        p.children = []
                        p_now.children.append(p)
                        p_dict[tuple(p_next)] = p
                    else:
                        p = p_dict[tuple(p_next)]
                        if f_new < p.f and p not in close_list:
                            p.f = f_new
                            p.g = g_new
                            p_old = p.father
                            del p_old.children[p_old.children.index(p)]
                            p.father = p_now
                            p_now.children.append(p)
                            # 更新数值
                            change_list = [p]
                            k = 0
                            while len(change_list) > 0:
                                k = k + 1
                                if k > 1000:
                                    print('change_list')
                                    raise Exception
                                p_father = change_list.pop()
                                x, y = p_father
                                for child in p_father.children:
                                    child.g = p_father.g + Point.distance(p_father, child) * self[y, x]
                                    child.f = child.g + Point.distance(child, p_target, 'manhattan')
                                    change_list.extend(child.children)
                    # open列表补充
                    if p_dict[tuple(p_next)] not in close_list:
                        # 没有定下来点需要进行计算
                        if p_dict[tuple(p_next)] not in open_list:
                            open_list.append(p_dict[tuple(p_next)])
            # 在open列表中找到f值最低的点进入close列表
            open_list.sort(key=lambda x: x.f)  # 如果用了这个，当地图大的时候不是最优解
            if tuple(p_target) in p_dict:
                flag_end = True

        # high, width = self.shape
        # f_matrix = np.zeros(self.shape)
        # for i in range(0, width):
        #     for j in range(0, high):
        #         if (i, j) in p_dict:
        #             f_matrix[j, i] = p_dict[(i, j)].f
        # Figure.plot(f_matrix)

        if flag_over is False:
            # 计算最短路径
            p_list = [tuple(p_target)]
            flag_end = False
            i = 0
            while flag_end is False:
                i = i + 1
                if i > 1000:
                    raise Exception
                p_now = p_list[-1]
                if p_now == tuple(p_start):
                    flag_end = True
                    p_list = p_list[::-1]
                    status = True
                else:
                    if p_now in p_dict:
                        p_list.append(tuple(p_dict[p_now].father))
                    else:
                        # 说明与p_start没有连接上
                        flag_end = True
                        p_list = []
                        status = False
        # fig = Figure.plot(self)
        # Points.plot(p_list, fig=fig)
        # Point.plot(p_start, fig=fig, color=[0,0,1])
        # Point.plot(p_target, fig=fig, color=[0, 1, 0])
        return p_list, status

    @classmethod
    def build(cls, img):
        '''
        建造函数，直接可以构造figure
        :param img:
        :return:
        '''
        shape = img.shape
        dtype = img.dtype
        buffer = img.reshape((1,-1))
        return cls(shape=shape, dtype=dtype, buffer=buffer)

    def assigner(self, point, value):
        '''
        图像赋值器，对单个点进行赋值
        :param point: 位置(x,y)
        :param value: 数值
        :return:
        '''
        x, y = point
        self[y, x] = value

    def bgr2gray(self):
        '''
        对图像进行灰度化
        :param img: 图像
        :return:
        '''
        return Figure.build(cv2.cvtColor(self, cv2.COLOR_BGR2GRAY))

    def indexer(self, point, value=0):
        '''
        图像索引器
        :param point: 点
        :param value: 返回值
        :return:
        '''

        x, y = point
        if x < 0 or x > self.width - 1 or y < 0 or y > self.high - 1:
            # 计算越界做法
            return value
        else:
            return self[y, x]

    def reverse(self, method):
        '''
        对图像进行逆序
        :param method:  y_axis: 对y轴进行逆序
                        x_axis：对x轴进行逆序
        :return:
        '''
        img_copy = copy.deepcopy(self)
        if method == 'y_axis':
            for j in range(0, self.high):
                self[j, :] = img_copy[self.high - 1 - j, :]
        elif method == 'x_axis':
            for i in range(0, self.width):
                self[:, i] = img_copy[:, self.width - 1 - i]

    def shrink(self, threshold=500, type='MaxEdgeLength', method='opencv'):
        '''
        图像过大要进行适当的缩小，将长边缩小到MaxEdgeLength长度
        MaxEdgeLength: 长边的最终长度
        :return:
        '''
        side_length = max([self.high, self.width])
        resolution = self.high * self.width
        if type == 'MaxEdgeLength':
            if side_length > threshold:
                factor_shrink = threshold/side_length
            else:
                factor_shrink = 1
        elif type == 'MaxResolution':
            if resolution > threshold:
                factor_shrink = threshold / resolution
            else:
                factor_shrink = 1
        if method == 'opencv':
            if factor_shrink < 1:
                return Figure.build(cv2.resize(self, (0, 0), fx=factor_shrink, fy=factor_shrink, interpolation=cv2.INTER_AREA)), factor_shrink
            else:
                return self, 1
        elif method == 'pillow':
            if factor_shrink < 1:
                image = Image.fromarray(self)
                high_new = int(self.high * factor_shrink)
                width_new = int(self.width * factor_shrink)
                im_resized = image.resize((width_new, high_new), Image.ANTIALIAS)
                image_new = np.array(im_resized)
                return Figure.build(image_new), factor_shrink
            else:
                return self, 1
        else:
            return self, 1

    def slope(self):
        # 根据arcGIS的坡度定义进行计算
        # 计算坡度
        u, d, l, r, ul, ur, dl, dr = Figure.neighbor(self, value=1)
        # x方向的梯度
        slope_x = ((ur + r * 2 + dr) - (ul + l * 2 + dl)) / 8
        # y方向的梯度
        slope_y = ((dl + d * 2 + dr) - (ul + u * 2 + ur)) / 8
        # 计算梯度
        slope = np.sqrt(slope_x * slope_x + slope_y * slope_y)
        return slope

    def to_locate_dict(self, direction='hori'):
        '''
        将存在图像的坐标提取出来（大于0的数值为语义）
        :param direction:
        :return:
        '''
        y_list, x_list = np.where(self > 0)
        if direction == 'hori':
            dict_hori = {}
            for x, y in zip(x_list, y_list):
                dict_add(dict_hori, y, x, 'append')
            dict_value_sort(dict_hori)
            return dict_hori
        elif direction == 'vert':
            dict_vert = {}
            for x, y in zip(x_list, y_list):
                dict_add(dict_vert, x, y, 'append')
            dict_value_sort(dict_vert)
            return dict_vert

    def __img_size(self):
        # 输出图像的高和宽
        shape = self.shape
        high = shape[0]
        width = shape[1]
        if len(shape) == 2:
            channel = 1
        else:
            channel = shape[2]
        return high, width, channel

    @staticmethod
    def clear_single_noise(img_p, bv=0):
        '''
        去除单点噪声
        :param img_p: 图像
        :param bv: 背景值
        :return:
        '''
        u, d, l, r, ul, ur, dl, dr = Figure.neighbor(img_p)
        img_p[(u == bv) & (d == bv) & (l == bv) & (r == bv) &
              (ul == bv) & (ur == bv) & (dl == bv) & (dr == bv)] = bv

    @staticmethod
    def edt(img_now):
        # 欧氏距离变换
        edt, inds = ndi.distance_transform_edt(img_now, return_indices=True)
        return edt, inds

    @staticmethod
    def edt_practice(img_now):
        # 欧氏距离变换（自己写得，结果与scipy中的npimage.distance_transform_edt完全一致
        # 前景是1，背景是0
        # 方向列表，逆时针顺序
        # img_now = np.zeros((5, 5))
        # img_now[1:5, 1:5] = 1
        dir_list = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        idx_list = list(range(len(dir_list)))
        n4_idx_list = set([0, 2, 4, 6])
        diag_idx_list = set([1, 3, 5, 7])
        n4_list = set([dir_list[i] for i in n4_idx_list])
        nd_list = set([dir_list[i] for i in diag_idx_list])
        high, width = img_now.shape
        # 定义一个数据结构
        Pinfo = type('Pinfo', (object, ), dict(d=np.inf, d_min=0, dx1=0, dy1=0, dx2=-1, dy2=-1))


        def edge_trace(img_now):
            # 找到图像的从左到右，从上到下搜索的第一个边缘，同时将边缘的地方设为-2
            # 输出图像的第一个边缘序列
            y_list, x_list = np.where(img_now > 0)
            if len(y_list) > 0:
                y_min = min(y_list)
                x_min = min(x_list[y_list == y_min])
                p_now = (x_min, y_min)
                dir_now = 5
                p_list = [p_now]
                end_flag = 0
                # 方向清单
                while end_flag == 0:
                    x_now, y_now = p_now
                    flag_changed = 0
                    for i in idx_list[dir_now:] + idx_list[:dir_now]:
                        p_delta = dir_list[i]
                        x_delta, y_delta = p_delta
                        x_n = x_now + x_delta
                        y_n = y_now + y_delta
                        if 0 <= x_n < width and 0 <= y_n < high:
                            if img_now[y_n, x_n] == 1:
                                # 进行标记
                                img_now[y_n, x_n] = -2
                                flag_changed = 1
                                if (x_n, y_n) == p_list[0]:
                                    end_flag = 1
                                    break
                                else:
                                    p_list.append((x_n, y_n))
                                    p_now = (x_n, y_n)
                                    # 下一次搜索角度
                                    if i in diag_idx_list:
                                        # 斜角方向
                                        dir_now = (i + 4 + 2) % 8
                                    else:
                                        dir_now = (i + 4 + 3) % 8
                                    break
                    if flag_changed == 0:
                        # 表示只有单独一个点
                        end_flag = 1
                return p_list
            else:
                return []

        # 主程序
        end_flag = 0
        dinfo = {}
        while end_flag == 0:
            p_list = edge_trace(img_now)
            if len(p_list) > 0:
                for i, p in enumerate(p_list):
                    print(i, p)
                    x_self, y_self = p
                    # 4.1\4.2 先判断N4\N8邻域像素
                    # 假设出现0
                    pinfo_self = Pinfo()
                    flag_changed = 0
                    for p_delta in dir_list:
                        x_delta, y_delta = p_delta
                        x_n = x_self + x_delta
                        y_n = y_self + y_delta
                        if 0 <= x_n < width and 0 <= y_n < high:
                            if img_now[y_n, x_n] == 0:
                                if p_delta in n4_list:
                                    pinfo_self.d = min([pinfo_self.d, 1])
                                    flag_changed = 1
                                else:
                                    pinfo_self.d = min([pinfo_self.d, np.sqrt(2)])
                                    flag_changed = 1
                    if flag_changed == 1:
                        dinfo[p] = pinfo_self
                    # 邻域当中没有0
                    if p not in dinfo:
                        pinfo_self = Pinfo()
                        d_list = []
                        for p_delta in dir_list:
                            x_delta, y_delta = p_delta
                            x_n = x_self + x_delta
                            y_n = y_self + y_delta
                            if 0 <= x_n < width and 0 <= y_n < high:
                                if img_now[y_n, x_n] < 0:
                                    # 找-1和-2这些点
                                    if (x_n, y_n) in dinfo:
                                        # 有距离信息的情况下算。有的时候-2边缘还没计算，所以没有信息
                                        pinfo = dinfo[(x_n, y_n)]
                                        d = pinfo.d
                                        if p_delta in n4_list:
                                            d_new = d + 1
                                        else:
                                            d_new = d + np.sqrt(2)
                                        d_list.append(d_new)
                        # 记录距离变换值
                        pinfo_self.d = min(d_list)
                        dinfo[p] = pinfo_self
                    # 4.3 传播更新
                    dt_list = [p]
                    while len(dt_list) > 0:
                        x_now, y_now = dt_list.pop()
                        pinfo_now = dinfo[(x_now, y_now)]
                        for p_delta in dir_list:
                            x_delta, y_delta = p_delta
                            x_n = x_now + x_delta
                            y_n = y_now + y_delta
                            if 0 <= x_n < width and 0 <= y_n < high:
                                if img_now[y_n, x_n] < 0:
                                    if (x_n, y_n) in dinfo:
                                        pinfo_next = dinfo[(x_n, y_n)]
                                        if p_delta in n4_list:
                                            d_add = 1
                                        else:
                                            d_add = np.sqrt(2)
                                        if pinfo_now.d + d_add < pinfo_next.d:
                                            pinfo_next.d = pinfo_now.d + d_add
                                            dt_list.append((x_n, y_n))
                # 跟踪变换完成，边界点序列设为-1
                for i, p in enumerate(p_list):
                    x_self, y_self = p
                    img_now[y_self, x_self] = -1
            else:
                end_flag = 1

        canvas = np.zeros(img_now.shape)
        for key in dinfo:
            x, y = key
            canvas[y, x] = dinfo[key].d
        Figure.plot(canvas)

    @staticmethod
    def extract_curve(canvas):
        # 从图中寻找8邻域相连的曲线，连续的点组成一个列表
        y_list, x_list = np.where(canvas > 0)
        p_set = set(map(lambda x, y: (x, y), x_list, y_list))
        p_list = list(p_set)
        now_point = []  # 当前可能延展的点集合
        line_list = []
        for p in p_list:
            if p in p_set:
                line_now = []
                now_point.append(p)
                line_now.append(p)
                p_set.remove(p)
                while len(now_point) > 0:
                    # 开始进行搜索
                    p_now = now_point[0]
                    neighbor_list = list_tuplize(Point.neighbor(p_now))
                    for p_next in neighbor_list:
                        if p_next in p_set:
                            now_point.append(p_next)
                            line_now.append(p_next)
                            p_set.remove(p_next)
                    # 当前点完成搜索任务
                    now_point.remove(p_now)
                # 进行排序并保存
                line_list.append(Points.sorted(line_now, method='line'))
        return line_list

    @staticmethod
    def gaussian_matrix(r=2):
        '''
        计算高斯分布矩阵，用于滤波模板
        :param r:  半径
        :return:
        '''
        x = np.arange(-r, r + 1)
        y = np.arange(-r, r + 1)
        grid = np.meshgrid(x, y)
        x1, y1 = grid
        matrix = np.exp(-(np.sqrt(pow(x1, 2) + pow(y1, 2))))
        shape = matrix.shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                matrix[i][j] = round(matrix[i][j], 6)
        return matrix

    @staticmethod
    def get_frame_color(img):
        # 获取最外一层边框的主要颜色
        high, width, _ = img.shape
        img_array = copy.deepcopy(img)
        img_array[1:-1, 1:-1] = 255
        img_clean = Image.fromarray(img_array)
        totalSize = high * width
        colors = img_clean.getcolors(totalSize)
        colors = sorted(colors, key=lambda x: x[0], reverse=True)
        # 清除(255, 255, 255)的数据
        colors = list_listize(colors)
        colors[0][0] = colors[0][0] - (high - 2) * (width - 2)
        colors = sorted(colors, key=lambda x:x[0], reverse=True)
        return colors[0][1]

    @staticmethod
    def hilditch(img):

        def func_nc8(b):
            # connectivity detection for each point
            n_odd = [1, 3, 5, 7]  # odd-number neighbors
            d = [0] * 10
            for i in range(0, 10):
                j = i
                if i == 9:
                    j = 1
                if abs(b[j]) == 1:
                    d[i] = 1
            sum_v = 0
            for i in range(0, 4):
                j = n_odd[i]
                sum_v = sum_v + d[j] - d[j] * d[j + 1] * d[j + 2]
            return sum_v

        BLACK = 1
        GRAY = 128
        WHITE = 0
        offset = [[0, 0], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]]  # offsets for neighbors
        n_odd = [1, 3, 5, 7]  # odd-number neighbors
        print("Hilditch's thinning starts now.")
        height, width = img.shape
        counter = 1
        round = 0
        b = [-1] * 9
        # processing starts
        while counter > 0:
            round += 1
            counter = 0
            index_y, index_x = np.where(img == BLACK)
            for ind in range(0, len(index_x)):
                y = index_y[ind]
                x = index_x[ind]
                # substitution of 9-neighbor gray values
                for i in range(0, 9):
                    b[i] = 0
                    px = x + offset[i][0]
                    py = y + offset[i][1]
                    if 0 <= px < width and 0 <= py < height:
                        if img[py][px] == BLACK:
                            b[i] = 1
                        elif img[py][px] == GRAY:
                            b[i] = -1
                condition = [0] * 6

                # condition 1: figure point
                condition[0] = 1

                # condition 2: boundary point
                sum_v = 0
                for i in range(0, 4):
                    sum_v = sum_v + 1 - abs(b[n_odd[i]])
                if sum_v >= 1:
                    condition[1] = 1

                    # condition 3: endpoint conservation
                    sum_v = 0
                    for i in range(1, 9):
                        sum_v = sum_v + abs(b[i])
                    if sum_v >= 2:
                        condition[2] = 1

                        # condition 4: isolated point conservation
                        sum_v = 0
                        for i in range(1, 9):
                            if b[i] == 1:
                                sum_v += 1
                        if sum_v >= 1:
                            condition[3] = 1

                            # condition 5: connectivity conservation
                            if func_nc8(b) == 1:
                                condition[4] = 1

                                # condition 6: one-side elimination for line-width of two
                                sum_v = 0
                                for i in range(1, 9):
                                    if b[i] != -1:
                                        sum_v += 1
                                    else:
                                        copy_v = b[i]
                                        b[i] = 0
                                        if func_nc8(b) == 1:
                                            sum_v += 1
                                        b[i] = copy_v
                                if sum_v == 8:
                                    condition[5] = 1

                                    # final decision
                                    img[y][x] = GRAY  # equals -1
                                    counter += 1

            if counter > 0:
                img[img == GRAY] = WHITE
            print('round:', round, 'counter:', counter)

    @staticmethod
    def k3m(img_now):
        # k3m算法，提取骨架

        @time_logger('点的value值计算')
        def point_value_calc(p_dict):
            # 计算每个点的value值
            for p in p_dict:
                neighbor_list = list_tuplize(Point.neighbor(p))
                for p_next, value in zip(neighbor_list, [4, 2, 1, 128, 64, 32, 16, 8]):
                    if p_next in p_dict:
                        p_dict[p] += value

        def point_clear(p_dict, clear_list, border):
            # 清除value值在clear_list里面的点
            key_list = sorted(list(p_dict.keys()))
            for p in key_list:
                if p in p_dict:
                    if p_dict[p] in clear_list and p in border:
                        neighbor_list = list_tuplize(Point.neighbor(p))
                        for p_next, value in zip(neighbor_list, [64, 32, 16, 8, 4, 2, 1, 128]):
                            if p_next in p_dict:
                                p_dict[p_next] -= value
                        del p_dict[p]

        A0 = {3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120, 124, 126, 127, 129, 131, 135,
              143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
              251, 252, 253, 254}
        A1 = {7, 14, 28, 56, 112, 131, 193, 224}
        A2 = {7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135, 193, 195, 224, 225, 240}
        A3 = {7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120, 124, 131, 135, 143, 193, 195, 199, 224, 225, 227, 240,
              241, 248}
        A4 = {7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143, 159, 193, 195, 199, 207,
              224, 225, 227, 231, 240, 241, 243, 248, 249, 252}
        A5 = {7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143, 159, 191, 193, 195, 199,
              207, 224, 225, 227, 231, 239, 240, 241, 243, 248, 249, 251, 252, 254}
        A1pix = {3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120, 124, 126, 127, 129, 131, 135,
                 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
                 251, 252, 253, 254}

        y_list, x_list = np.where(img_now > 0)
        p_dict = {}
        for x, y in zip(x_list, y_list):
            p_dict[(x, y)] = 0
        # 计算每个点的value值
        point_value_calc(p_dict)
        end_flag = 0
        while end_flag == 0:
            total = len(p_dict)
            # 将border找出来
            border = set([p for p in p_dict if p_dict[p] in A0])
            point_clear(p_dict, A1, border)
            point_clear(p_dict, A2, border)
            point_clear(p_dict, A3, border)
            point_clear(p_dict, A4, border)
            point_clear(p_dict, A5, border)
            total_now = len(p_dict)
            print('total:', total, 'total_now', total_now)
            if total == total_now:
                end_flag = 1

        # 得到骨架
        canvas = np.zeros(img_now.shape)
        for p in p_dict:
            x, y = p
            canvas[y, x] = 1
        Figure.plot(canvas)
        return canvas

    @staticmethod
    def kernel(shape, radius):
        # 生成开闭运算（膨胀与腐蚀）的kernel
        ksize = int(radius * 2 - 1)
        if shape == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))  # 矩形结构
        if shape == 'ellipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))  # 椭圆结构
        if shape == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))  # 十字形结构

    @staticmethod
    def median_cut(img, colorNum=8, vBalance=8):
        '''
        中位切割主题色提取法。每次切分前对颜色空间进行排序，选择rank最前的进行切分
        rank = pixelSum * (v)^vBalance
        pixelSum ：像素数量，v：体积，vBalance：体积影响系数（非负）
        像素越多，体积越大的颜色空间则需要被切割
        :param img:  numpy.array格式的图像
        :param colorNum: 需要提取的颜色种类
        :param vBalance:  体积系数
        :return:
        '''

        class ColorBox:
            def __init__(self, colorRange, pixelSum, pixelSet):
                self.colorRange = colorRange
                self.pixelSum = pixelSum
                self.pixelSet = pixelSet
                self.v = (colorRange[0][1] - colorRange[0][0]) * (colorRange[1][1] - colorRange[1][0]) * (
                        colorRange[2][1] - colorRange[2][0]) >> 8  # 减少计算规模。否则这个数很大
                self.rank = (-1) * pixelSum * (self.v) ** vBalance

            def __lt__(self, other):
                return self.rank < other.rank

        def colorRange(imageColors):
            # 计算颜色范围
            rMax = imageColors[0][1][0]
            gMax = imageColors[0][1][1]
            bMax = imageColors[0][1][2]
            rMin = imageColors[0][1][0]
            gMin = imageColors[0][1][1]
            bMin = imageColors[0][1][2]
            for _, color in imageColors:
                if len(color) == 3:
                    r, g, b = color
                elif len(color) == 4:
                    r, g, b, a = color
                if r > rMax:
                    rMax = r
                elif r < rMin:
                    rMin = r
                if g > gMax:
                    gMax = g
                elif g < gMin:
                    gMin = g
                if b > bMax:
                    bMax = b
                elif b < bMin:
                    bMin = b
            colorMin = [rMin, gMin, bMin]
            colorMax = [rMax, gMax, bMax]
            return colorMin, colorMax

        def getCutSide(colorRange):  # 获取切割边:{r:0, g:1, b:2}
            vSize = [0] * 3
            for i in range(3):
                vSize[i] = colorRange[i][1] - colorRange[i][0]
            return vSize.index(max(vSize))

        def cutRange(colorRange, cutSide, cutValue):  # 将颜色范围一切为二
            ret0 = copy.deepcopy(colorRange)
            ret1 = copy.deepcopy(colorRange)
            ret0[cutSide][1] = cutValue
            ret1[cutSide][0] = cutValue
            return (ret0, ret1)

        def colorCut(colorBox):  # 颜色切分
            cutValue = 0
            colorRange = colorBox.colorRange
            cutSide = getCutSide(colorRange)  # 分割边
            pixelCount = 0  # 当前像素累加数
            sourceList = colorBox.pixelSet  # 像素集合
            pixelSum = colorBox.pixelSum
            cutPoint = 0  # 切分点
            sourceList.sort(key=lambda x: x[1][cutSide])
            for pixelPoint in sourceList:  # pixelPoint:(count, (r, g, b))
                pixelCount += pixelPoint[0]
                cutPoint += 1
                if pixelCount * ((pixelPoint[1][cutSide] - colorRange[cutSide][0])) ** vBalance > (
                        pixelSum - pixelCount) * (
                (colorRange[cutSide][1] - pixelPoint[1][cutSide])) ** vBalance:  # 达到一半
                    cutValue = pixelPoint[1][cutSide]
                    break
            if cutPoint == len(sourceList):  # 到最后一个才触发，最后一个作为独立元素（修改了）
                newRange = cutRange(colorRange, cutSide, sourceList[cutPoint - 1][1][cutSide])
                box0 = ColorBox(newRange[0], pixelCount - sourceList[cutPoint - 1][0], sourceList[:(cutPoint - 1)])
                r_v, g_v, b_v = sourceList[cutPoint - 1][1]
                box1 = ColorBox([[r_v, r_v], [g_v, g_v], [b_v, b_v]], sourceList[cutPoint - 1][0],
                                [sourceList[cutPoint - 1]])
                if box0.pixelSum == 0:
                    return [box1]
                else:
                    return [box0, box1]
            else:
                newRange = cutRange(colorRange, cutSide, cutValue)
                box0 = ColorBox(newRange[0], pixelCount, sourceList[0:cutPoint])  # 不带上超过50%的那个元素
                box1 = ColorBox(newRange[1], colorBox.pixelSum -
                                pixelCount, sourceList[cutPoint:])
                return [box0, box1]

        def doCut(queue):  # 递归切分
            if queue.qsize() < colorNum:
                box = queue.get()[1]  # 获取rank第一的box
                c = colorCut(box)
                for vbox in c:
                    # print("Rank:" + str(vbox.rank))
                    queue.put((vbox.rank, vbox))
                return doCut(queue)
            else:
                return queue

        def sumColor(colorList):  # 颜色求和
            sumList = [0] * 4
            for count, color in colorList:
                if len(color) == 3:
                    r, g, b = color
                elif len(color) == 4:
                    r, g, b, a = color
                sumList[0] += count
                sumList[1] += r * count
                sumList[2] += g * count
                sumList[3] += b * count
            if sumList[0] != 0:
                sumList[1] = int(round(sumList[1] / sumList[0], 0))
                sumList[2] = int(round(sumList[2] / sumList[0], 0))
                sumList[3] = int(round(sumList[3] / sumList[0], 0))
            var_sum = 0
            for count, color in colorList:
                if len(color) == 3:
                    r, g, b = color
                elif len(color) == 4:
                    r, g, b, a = color
                var_sum += count * ((np.abs(r - sumList[1]) + np.abs(g - sumList[2]) + np.abs(b - sumList[3])) / 3) ** 2
            var_v = (var_sum / sumList[0]) ** (1 / 2)
            return sumList, var_v

        def getMainColor(queue, number):  # 根据box计算主色调
            colorList = []
            rangeList = []
            rankList = []
            pixelsumList = []
            vList = []
            varList = []
            for i in range(number):
                box = queue.get()[1]
                sumList, var_v = sumColor(box.pixelSet)
                colorList.append(sumList)
                rangeList.append(box.colorRange)
                rankList.append(box.rank)
                pixelsumList.append(box.pixelSum)
                vList.append(box.v)
                varList.append(floor_fun(var_v, 2))
            return colorList, rangeList, rankList, pixelsumList, vList, varList

        def main():
            # 读取图片
            image = Image.fromarray(img)
            high, width = image.size
            totalSize = high * width
            imageColors = image.getcolors(totalSize)
            # 计算颜色范围
            colorMin, colorMax = colorRange(imageColors)
            [rMin, gMin, bMin] = colorMin
            [rMax, gMax, bMax] = colorMax
            # 初始化
            initRange = [[rMin, rMax], [gMin, gMax], [bMin, bMax]]
            initBox = ColorBox(initRange, totalSize, imageColors)
            initQueue = PriorityQueue()
            initQueue.put((initBox.rank, initBox))
            resQueue = doCut(initQueue)
            mainColor, rangeList, rankList, pixelsumList, vList, varList = getMainColor(resQueue, colorNum)
            # mainColor.sort(key=lambda x: x[0], reverse=True)
            return mainColor, rangeList, vList, varList

        # 执行
        return main()

    @staticmethod
    def neighbor(img_trace, var='', value=0):
        # 邻域对比，注意是二值图像，边缘为1，背景为0
        # 一共可以输出八个方向的图像
        # 比如以var='l'为例，得到的图像相当于每个点的左边的点
        height, width = np.shape(img_trace)
        if var == '':
            u = np.concatenate((np.ones((1, width)) * value, img_trace[:-1, :]), axis=0)
            d = np.concatenate((img_trace[1:, :], np.ones((1, width)) * value), axis=0)
            l = np.concatenate((np.ones((height, 1)) * value, img_trace[:, :-1]), axis=1)
            r = np.concatenate((img_trace[:, 1:], np.ones((height, 1)) * value), axis=1)
            dl = np.concatenate((np.ones((height, 1)) * value, d[:, :-1]), axis=1)
            dr = np.concatenate((d[:, 1:], np.ones((height, 1)) * value), axis=1)
            ul = np.concatenate((np.ones((height, 1)) * value, u[:, :-1]), axis=1)
            ur = np.concatenate((u[:, 1:], np.ones((height, 1)) * value), axis=1)
            return u, d, l, r, ul, ur, dl, dr
        if var == 'u':
            u = np.concatenate((np.ones((1, width)) * value, img_trace[:-1, :]), axis=0)
            return u
        if var == 'd':
            d = np.concatenate((img_trace[1:, :], np.ones((1, width)) * value), axis=0)
            return d
        if var == 'l':
            l = np.concatenate((np.ones((height, 1)) * value, img_trace[:, :-1]), axis=1)
            return l
        if var == 'r':
            r = np.concatenate((img_trace[:, 1:], np.ones((height, 1)) * value), axis=1)
            return r
        if var == 'dl':
            d = np.concatenate((img_trace[1:, :], np.ones((1, width)) * value), axis=0)
            dl = np.concatenate((np.ones((height, 1)) * value, d[:, :-1]), axis=1)
            return dl
        if var == 'dr':
            d = np.concatenate((img_trace[1:, :], np.ones((1, width)) * value), axis=0)
            dr = np.concatenate((d[:, 1:], np.ones((height, 1)) * value), axis=1)
            return dr
        if var == 'ul':
            u = np.concatenate((np.ones((1, width)) * value, img_trace[:-1, :]), axis=0)
            ul = np.concatenate((np.ones((height, 1)) * value, u[:, :-1]), axis=1)
            return ul
        if var == 'ur':
            u = np.concatenate((np.ones((1, width)) * value, img_trace[:-1, :]), axis=0)
            ur = np.concatenate((u[:, 1:], np.ones((height, 1)) * value), axis=1)
            return ur

    @staticmethod
    def octree(img, maxColors=16):
        # 八叉树主题色提取法
        class OctreeNode:
            # 八叉树节点
            def __init__(self):
                self.isLeaf = False
                self.pixelCount = 0
                self.red = 0
                self.green = 0
                self.blue = 0
                self.children = [None] * 8
                self.ancestor = None
                self.idx = None
                self.next = None
                self.level = None

        def createNode(idx, level):
            # 建立节点
            node = OctreeNode()
            node.idx = idx
            node.level = level
            if level == 7:
                node.isLeaf = True
                leafNum[0] += 1
            else:
                # 将其丢到第 level 层的 reducible 链表中
                node.next = reducible[level]
                reducible[level] = node
            return node

        def addColor(node, color, pixelCount, level):
            # 添加颜色
            if node.isLeaf:
                node.pixelCount += pixelCount
                node.red += color[0] * pixelCount
                node.green += color[1] * pixelCount
                node.blue += color[2] * pixelCount
            else:
                # 变成二进制
                r = bin(color[0])[2:]
                g = bin(color[1])[2:]
                b = bin(color[2])[2:]
                r = '0' * (8 - len(r)) + r
                g = '0' * (8 - len(g)) + g
                b = '0' * (8 - len(b)) + b
                string = r[level] + g[level] + b[level]
                # 从二进制变回十进制
                idx = int(string, 2)
                if node.children[idx] is None:
                    node.children[idx] = createNode(idx, level + 1)
                    node.children[idx].ancestor = node
                # 继续下一层的叠加
                addColor(node.children[idx], color, pixelCount, level + 1)

        def reduceTree():
            # 找到最深层次的并且有可合并节点的链表
            lv = 6
            while reducible[lv] is None:
                lv -= 1
            # 取出链表头并将其从链表中移除
            node = reducible[lv]
            reducible[lv] = node.next
            # 合并子节点
            r = 0
            g = 0
            b = 0
            count = 0
            for i in range(8):
                if node.children[i] is None:
                    continue
                else:
                    r += node.children[i].red
                    g += node.children[i].green
                    b += node.children[i].blue
                    count += node.children[i].pixelCount
                    leafNum[0] -= 1
            # 赋值
            node.isLeaf = True
            node.red = r
            node.green = g
            node.blue = b
            node.pixelCount = count
            leafNum[0] += 1

        def buildOctree(img, maxColors):
            # 建树
            image = Image.fromarray(img)
            high, width = image.size
            totalSize = high * width
            imageColors = image.getcolors(totalSize)
            for i, term in enumerate(imageColors):
                # 添加颜色
                pixelCount, color = term
                addColor(root, color, pixelCount, 0)
                # 合并叶子节点
                while leafNum[0] > maxColors:
                    reduceTree()

        def colorsStats(node, rst_dict):
            # 深度优先的方法找到叶子节点
            if node.isLeaf:
                # 用十六进制表示键
                # r = hex(int(node.red / node.pixelCount))[2:]
                # g = hex(int(node.green / node.pixelCount))[2:]
                # b = hex(int(node.blue / node.pixelCount))[2:]
                # r = '0' * (2 - len(r)) + r
                # g = '0' * (2 - len(g)) + g
                # b = '0' * (2 - len(b)) + b
                # color = r + g + b
                # 用元组表示键
                r = int(node.red / node.pixelCount)
                g = int(node.green / node.pixelCount)
                b = int(node.blue / node.pixelCount)
                color = (r, g, b)
                ancestor_str = ''  # 祖先字符串
                node_now = node
                while node_now.level > 0:
                    ancestor_str = str(node_now.idx) + ancestor_str
                    node_now = node_now.ancestor
                rst_dict[color] = [node.pixelCount, ancestor_str, node]
                return

            for i in range(8):
                if node.children[i] is not None:
                    colorsStats(node.children[i], rst_dict)

        reducible = [None] * 8
        leafNum = [0]
        root = createNode(-1, 0)
        buildOctree(img, maxColors)
        rst_dict = {}
        colorsStats(root, rst_dict)
        return rst_dict

    @staticmethod
    def plot(img, cmap='gray', fig=[]):
        if fig == []:
            fig = plt.figure()
        plt.imshow(img, cmap=cmap)
        fig.gca().invert_yaxis()

    @staticmethod
    def plot_polygen_filled(contours, canvas_all):
        # 画被填充的多边形
        # canvas是全零矩阵
        for i, _ in enumerate(contours):
            canvas = np.zeros(canvas_all.shape)
            cv2.drawContours(canvas, contours, i, [1, 1, 1], cv2.FILLED)
            canvas_all += canvas
        canvas_all[canvas_all > 0] = 1

    @staticmethod
    def point_radius_square(p, radius, high, width):
        # 在已知中心点的情况下，计算往左右各延展radius距离后的范围，可以考虑到图像的实际大小。一般是正方形范围。
        x, y = p
        x_min = max([0, x - radius])
        x_max = min([width - 1, x + radius])
        y_min = max([0, y - radius])
        y_max = min([high - 1, y + radius])
        return x_min, x_max, y_min, y_max

    @staticmethod
    def shrink_pil(img, threshold, type='MaxResolution'):
        width, high = img.size
        resolution = high * width
        if type == 'MaxResolution':
            if resolution > threshold:
                factor_shrink = threshold / resolution
            else:
                factor_shrink = 1
        if factor_shrink < 1:
            width_new = math.floor(width * factor_shrink)
            high_new = math.floor(high * factor_shrink)
            img = img.resize((width_new, high_new), Image.ANTIALIAS)
        return img

    @staticmethod
    def skeleton_clear_each_side(img):
        '''
        对各个方向依次进行消除，前提是不破坏原来的连通性
        从上往下看，我们当前要搜索的模式是：
        * 0 *
        * 1 *
        * 1 *
        我们对*的位置进行编号：
        0 * 3
        1 * 4
        2 * 5
        一共有64中可能，需要将所有的可能组合写一次，这样就不需要计算了。
        要确定去掉中间的点以后，组合当中的连接关系是否还存在
        '''
        segmentation = (img > 0) * 1
        # 模式的可行性判断
        mode_list = [[[0, 0, 0, 0, 0, 0], True],
                     [[0, 0, 0, 0, 0, 1], True],
                     [[0, 0, 0, 0, 1, 0], True],
                     [[0, 0, 0, 0, 1, 1], True],
                     [[0, 0, 0, 1, 0, 0], False],
                     [[0, 0, 0, 1, 0, 1], False],
                     [[0, 0, 0, 1, 1, 0], True],
                     [[0, 0, 0, 1, 1, 1], True],
                     [[0, 0, 1, 0, 0, 0], True],
                     [[0, 0, 1, 0, 0, 1], True],
                     [[0, 0, 1, 0, 1, 0], True],
                     [[0, 0, 1, 0, 1, 1], True],
                     [[0, 0, 1, 1, 0, 0], False],
                     [[0, 0, 1, 1, 0, 1], False],
                     [[0, 0, 1, 1, 1, 0], True],
                     [[0, 0, 1, 1, 1, 1], True],
                     [[0, 1, 0, 0, 0, 0], True],
                     [[0, 1, 0, 0, 0, 1], True],
                     [[0, 1, 0, 0, 1, 0], True],
                     [[0, 1, 0, 0, 1, 1], True],
                     [[0, 1, 0, 1, 0, 0], False],
                     [[0, 1, 0, 1, 0, 1], False],
                     [[0, 1, 0, 1, 1, 0], True],
                     [[0, 1, 0, 1, 1, 1], True],
                     [[0, 1, 1, 0, 0, 0], True],
                     [[0, 1, 1, 0, 0, 1], True],
                     [[0, 1, 1, 0, 1, 0], True],
                     [[0, 1, 1, 0, 1, 1], True],
                     [[0, 1, 1, 1, 0, 0], False],
                     [[0, 1, 1, 1, 0, 1], False],
                     [[0, 1, 1, 1, 1, 0], True],
                     [[0, 1, 1, 1, 1, 1], True],
                     [[1, 0, 0, 0, 0, 0], False],
                     [[1, 0, 0, 0, 0, 1], False],
                     [[1, 0, 0, 0, 1, 0], False],
                     [[1, 0, 0, 0, 1, 1], False],
                     [[1, 0, 0, 1, 0, 0], False],
                     [[1, 0, 0, 1, 0, 1], False],
                     [[1, 0, 0, 1, 1, 0], False],
                     [[1, 0, 0, 1, 1, 1], False],
                     [[1, 0, 1, 0, 0, 0], False],
                     [[1, 0, 1, 0, 0, 1], False],
                     [[1, 0, 1, 0, 1, 0], False],
                     [[1, 0, 1, 0, 1, 1], False],
                     [[1, 0, 1, 1, 0, 0], False],
                     [[1, 0, 1, 1, 0, 1], False],
                     [[1, 0, 1, 1, 1, 0], False],
                     [[1, 0, 1, 1, 1, 1], False],
                     [[1, 1, 0, 0, 0, 0], True],
                     [[1, 1, 0, 0, 0, 1], True],
                     [[1, 1, 0, 0, 1, 0], True],
                     [[1, 1, 0, 0, 1, 1], True],
                     [[1, 1, 0, 1, 0, 0], False],
                     [[1, 1, 0, 1, 0, 1], False],
                     [[1, 1, 0, 1, 1, 0], True],
                     [[1, 1, 0, 1, 1, 1], True],
                     [[1, 1, 1, 0, 0, 0], True],
                     [[1, 1, 1, 0, 0, 1], True],
                     [[1, 1, 1, 0, 1, 0], True],
                     [[1, 1, 1, 0, 1, 1], True],
                     [[1, 1, 1, 1, 0, 0], False],
                     [[1, 1, 1, 1, 0, 1], False],
                     [[1, 1, 1, 1, 1, 0], True],
                     [[1, 1, 1, 1, 1, 1], True],
                     ]
        # 计算四个方向各自的清除列表
        # 从上往下的可以清除列表
        clear_mode_dict = {}
        for direction in ['up_down', 'left_right', 'down_up', 'right_left']:
            clear_mode_list = []
            for mode_now in mode_list:
                m0, m1, m2, m3, m4, m5 = mode_now[0]
                if direction == 'up_down':
                    # 注意只有8个数字，正中不用计算
                    mode_end = [m0, 0, m3, m1, m4, m2, 1, m5]
                elif direction == 'left_right':
                    mode_end = [m3, m4, m5, 0, 1, m0, m1, m2]
                elif direction == 'down_up':
                    mode_end = [m5, 1, m2, m4, m1, m3, 0, m0]
                elif direction == 'right_left':
                    mode_end = [m2, m1, m0, 1, 0, m5, m4, m3]
                else:
                    raise NotImplementedError
                if mode_now[1]:
                    clear_mode_list.append(mode_end)
            clear_mode_dict[direction] = clear_mode_list
        # 开始进行删除，删除顺序是up_down, left_right, down_up, right_left
        sum_old = np.sum(segmentation)
        while True:
            print('clear')
            for direction in ['up_down', 'left_right', 'down_up', 'right_left']:
                # 寻找符合的位置
                clear_mode_list = clear_mode_dict[direction]
                delta_list = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                if direction == 'up_down':
                    u = Figure.neighbor(segmentation, 'u')
                    d = Figure.neighbor(segmentation, 'd')
                    y_list, x_list = np.where((u == 0) & (segmentation == 1) & (d == 1))
                elif direction == 'left_right':
                    l = Figure.neighbor(segmentation, 'l')
                    r = Figure.neighbor(segmentation, 'r')
                    y_list, x_list = np.where((l == 0) & (segmentation == 1) & (r == 1))
                elif direction == 'down_up':
                    u = Figure.neighbor(segmentation, 'u')
                    d = Figure.neighbor(segmentation, 'd')
                    y_list, x_list = np.where((d == 0) & (segmentation == 1) & (u == 1))
                elif direction == 'right_left':
                    l = Figure.neighbor(segmentation, 'l')
                    r = Figure.neighbor(segmentation, 'r')
                    y_list, x_list = np.where((r == 0) & (segmentation == 1) & (l == 1))
                else:
                    raise NotImplementedError
                img_clear = np.zeros(segmentation.shape)
                for x, y in zip(x_list, y_list):
                    # 计算每个点的情形
                    # 组织点的位置
                    mode_now = []
                    for delta in delta_list:
                        x_delta, y_delta = delta
                        x_now = x + x_delta
                        y_now = y + y_delta
                        mode_now.append(segmentation[y_now, x_now])
                    if mode_now in clear_mode_list:
                        img_clear[y, x] = 1
                segmentation[img_clear == 1] = 0
            sum_now = np.sum(segmentation)
            if sum_now == sum_old:
                break
            else:
                sum_old = sum_now
        return segmentation

    @staticmethod
    @time_logger('基于薄化提取骨架')
    def skeleton_hit_or_miss(img):
        # 利用击中变换找到需要薄化的位置，经过循环迭代，得到单像素骨架
        # 算法参考《一种保形的快速图象形态细化算法_盛业华》
        img[img > 0] = 1
        # 元素列表，固定元素+不固定元素
        element_dict = {'D': [[[6, 8], [3, 7, 9]], [[4, 8], [1, 7, 9]],
                              [[2, 4], [1, 3, 7]], [[2, 6], [1, 3, 9]]],
                        'E': [[[7, 8, 9], [4, 6]], [[1, 4, 7], [2, 8]],
                              [[1, 2, 3], [4, 6]], [[3, 6, 9], [2, 8]]],
                        'L': [[[4, 6, 8], [1, 3, 7, 9]], [[2, 4, 8], [1, 3, 7, 9]],
                              [[2, 4, 6], [1, 3, 7, 9]], [[2, 6, 8], [1, 3, 7, 9]]]}
        # 生成结构元素
        struct_dict = {}
        for key in element_dict:
            element_list = element_dict[key]
            for i, element in enumerate(element_list):
                fixed_element, float_element = element
                # 元素组合可能
                com_list = permutation_combination(float_element) + [[]]
                for com_now in com_list:
                    unit_kernel = com_now + fixed_element + [5]
                    dict_add(struct_dict, key + str(i), unit_kernel, 'append')
        # 计算顺序列表
        calc_list = []
        for i in range(4):
            calc_list.append(['D' + str(i), 'D' + str((i + 1) % 4), 'E' + str(i), 'L' + str(i)])
        # 计算
        high, width = img.shape
        img_clear = np.zeros((high, width))
        kernel = np.zeros((3, 3), 'uint8')
        # 提取骨架
        while True:
            # 骨架图像
            flag_changed = False
            for calc_term in calc_list:
                img_clear[:] = 0
                for key in calc_term:
                    for unit_kernel in struct_dict[key]:
                        kernel[:] = 0
                        for idx in unit_kernel:
                            # 转化索引
                            column = (idx - 1) % 3
                            row = (idx - 1) // 3
                            kernel[row, column] = 1
                        # 图像腐蚀T^1
                        img_erode = cv2.erode(img, kernel, borderValue=1)
                        # 图像腐蚀T^1的补集
                        kernel_complement = (1 - kernel).astype('uint8')
                        img_erode2 = cv2.erode((1 - img).astype('uint8'), kernel_complement, borderValue=1)
                        img_clear[(img_erode == 1) & (img_erode2 == 1)] = 1
                # 判断是否结束
                if np.sum(img_clear) > 0:
                    # 清理
                    img[img_clear == 1] = 0
                    flag_changed = True
            if flag_changed is False:
                break

    @staticmethod
    @time_logger('中轴骨架提取算法')
    def skeleton_medial_axis(img_now):
        # 中轴算法
        # img_now = np.zeros((50, 50))
        # img_now[5:45, 5:10] = 1
        # img_now[5:45, 40:45] = 1
        # img_now[5:10, 5:45] = 1
        # img_now[40:45, 5:45] = 1
        high, width = img_now.shape
        # edt, inds = Figure.edt(img_now)
        cdt, inds = ndi.distance_transform_cdt(img_now, metric='chessboard', return_indices=True)
        y_inds, x_inds = inds
        y_list, x_list = np.where(cdt > 0)
        # 对每一个点计算其中轴距离（半径）
        median_d = np.zeros(img_now.shape)
        for x, y in zip(x_list, y_list):
            n_list = [(x, y)]  # 邻居列表
            if x == width - 1:
                n_list.append((x - 1, y))
            else:
                n_list.append((x + 1, y))
            if y == high - 1:
                n_list.append((x, y - 1))
            else:
                n_list.append((x, y + 1))
            # 直径
            p_list = [(x_inds[y_next, x_next], y_inds[y_next, x_next]) for x_next, y_next in n_list]
            a, b, c = [Point.distance(p_list[0], p_list[1], 'direct'),
                       Point.distance(p_list[0], p_list[2], 'direct'),
                       Point.distance(p_list[1], p_list[2], 'direct')]
            p = (a + b + c)/2
            s = np.sqrt(p*(p-a)*(p-b)*(p-c))
            if s == 0:
                r = p
            else:
                r = (a*b*c)/(4*s)
            median_d[y, x] = r
            # median_d[y, x] = max([a, b, c])


# 图像处理
def getimage(filepath):
    # path为需要读取图片的路径
    img = Image.open(filepath)
    M, N = img.size
    r, g, b = img.split()
    rd = np.asarray(r)
    gd = np.asarray(g)
    bd = np.asarray(b)
    return rd, gd, bd


def img_indexer(img, point, value=0):
    '''
    图像索引器
    :param img: 图像
    :param point: 点
    :param value: 返回值
    :return:
    '''

    x, y = point
    high, width, channel = img_size(img)
    if x < 0 or x > width - 1 or y < 0 or y > high - 1:
        # 计算越界做法
        return value
    else:
        return img[y, x]


def img_assigner(img, point, value):
    '''
    图像赋值器，对单个点进行赋值
    :param img: 图像
    :param point: 位置(x,y)
    :param value: 数值
    :return:
    '''
    x, y = point
    img[y, x] = value


def img_negative_process(img_now, value=0):
    # 将图像小于0的数值进行处理
    img_now[img_now < 0] = value
    return img_now


def img_size(img):
    # 输出图像的高和宽
    shape = img.shape
    high = shape[0]
    width = shape[1]
    if len(shape) == 2:
        channel = 1
    else:
        channel = shape[2]
    return high, width, channel


def img_main_color(color_list, ratio=0.9):
    # 提取图中的主要颜色
    acc = 0
    main_color = []
    total_size = sum([term[0] for term in color_list])
    for i in range(len(color_list)):
        color_size = color_list[i][0]
        acc += color_size
        main_color.append(color_list[i])
        if acc / total_size > ratio:
            break
    return main_color


def shapen_process(img, flag, method=''):
    # 进行滤波操作

    def filp_diag(matrix):
        rst = copy.deepcopy(matrix)
        l = len(matrix)
        for i in range(0, l):
            for j in range(0, l):
                rst[i, j] = matrix[j, i]
                rst[j, i] = matrix[i, j]
        return rst

    def filp_lr(matrix):
        return matrix[:, ::-1]

    def filp_ud(matrix):
        return matrix[::-1, :]

    def model_rotate(left_model, direction):
        # 以左方向模板为基准模板，生成四个方向的模板，宗旨是旋转方向
        if direction == 'l':
            kernel = left_model
        elif direction == 'r':
            kernel = filp_ud(filp_lr(left_model))
        elif direction == 'u':
            kernel = filp_lr(filp_diag(left_model))
        elif direction == 'd':
            kernel = filp_ud(filp_diag(left_model))
        return kernel

    def sword_templete(r):
        # 生成sword模板
        tmp = np.zeros((r*2+1, r*2+1))
        for i in range(0, r+1):
            for j in range(0, i+1):
                tmp[i][j] = -1
        tmp[r][r] = -np.sum(tmp) - 1
        return tmp

    def hom_templete(r):
        # 生成hom模板
        tmp = sword_templete(r) + filp_ud(sword_templete(r))
        tmp[tmp < 0] = -1
        tmp[tmp > 0] = -1
        tmp[r][r] = -np.sum(tmp) - 1
        return tmp

    def window_templete(r):
        # 创造window模板
        tmp = np.zeros((r*2+1, r*2+1))
        for i in range(0, r+1):
            tmp[i][r - 1] = -1
            tmp[i][r] = 1
        return tmp

    # 均值滤波
    meanblur = np.array([[1/9, 1/9, 1/9],
                          [1/9, 1/9, 1/9],
                          [1/9, 1/9, 1/9]])
    # wujwus3模板的作用是，仅计算单点与邻居单点的梯度
    wujwus3 = np.array([[0, 0, 0],
                          [-1, 1, 0],
                          [0, 0, 0]])
    wujwusn3 = np.array([[0, 0, 0],
                          [1, -1, 0],
                          [0, 0, 0]])
    # wujwu51模板的作用是，计算单点与周边5个点的梯度，可以适应圆弧以及直线的情形
    wujwu3 = np.array([[-1, 0, 0],
                        [-2, 4, 0],
                        [-1, 0, 0]])
    wujwun3 = np.array([[1, 0, 0],
                        [2, -4, 0],
                        [1, 0, 0]])
    wujwu51 = np.array([[0, -1, 0, 0, 0],
                           [0, -2, 0, 0, 0],
                           [0, -4, 10, 0, 0],
                           [0, -2, 0, 0, 0],
                           [0, -1, 0, 0, 0]])
    wujwun51 = np.array([[0, 1, 0, 0, 0],
                          [0, 2, 0, 0, 0],
                          [0, 4, -10, 0, 0],
                          [0, 2, 0, 0, 0],
                          [0, 1, 0, 0, 0]])
    # 模仿sobel模板
    sobel3 = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobel5 = np.array([[-1, -1, 0, 1, 1],
                       [-2, -2, 0, 2, 2],
                       [-4, -4, 0, 4, 4],
                       [-2, -2, 0, 2, 2],
                       [-1, -1, 0, 1, 1]])
    # 模仿sobel模板（3，10，3）
    sobelc3 = np.array([[-3, 3, 0],
                        [-10, 10, 0],
                        [-3, 3, 0]])
    sobelc51 = np.array([[-1, -3, 4, 0, 0],
                           [-3, -9, 12, 0, 0],
                           [-10, -30, 40, 0, 0],
                           [-3, -9, 12, 0, 0],
                           [-1, -3, 4, 0, 0]])
    # straight模板
    straight51 = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, -4, 0, 4, 0],
                           [0, -2, 0, 2, 0],
                           [0, -1, 0, 1, 0]])
    # wujwu51ave模板的作用是，分散中心点的权重，以周围的权重来补充（圆弧会挂）
    wujwu3ave = np.array([[-1, 1, 0],
                          [-2, 2, 0],
                          [-1, 1, 0]])
    wujwun3ave = np.array([[1, -1, 0],
                           [2, -2, 0],
                           [1, -1, 0]])
    wujwu51ave = np.array([[0, -1, 1, 0, 0],
                           [0, -1, 1, 0, 0],
                           [0, -1, 1, 0, 0],
                           [0, -1, 1, 0, 0],
                           [0, -1, 1, 0, 0]])
    # sword模板的作用是，只检测单个方向上的梯度，不需要检测周围的情况
    swordu = sword_templete(2)
    swordd = filp_ud(swordu)
    # hom模板的作用是，只检测单个方向上的全部梯度
    hom = hom_templete(3)
    # point模板的作用是，直接计算点与点之间的梯度
    pointn = np.array([[0, 0, 0],
                        [-1, 1, 0],
                        [0, 0, 0]])
    pointd = np.array([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    # wujwu51u模板的作用是，在wujwu_51模板进行升级，针对边缘情况进行处理，防止实心强的影响
    wujwu51u = np.array([[0, -1, 0, 0, 0],
                           [0, -2, -1, 0, 0],
                           [0, -4, 9, 0, 0],
                           [0, 0, -1, 0, 0],
                           [0, 0, 0, 0, 0]])
    wujwu51d = filp_ud(wujwu51u)
    # window模板的作用，针对窗口线进行提取
    windowu = window_templete(3)
    windowd = filp_ud(windowu)
    model, direction = flag.split('_')
    kernel = eval("model_rotate(" + model + ", '" + direction + "')")
    dst = cv2.filter2D(img, cv2.CV_64F, kernel)
    if method == 'nega_clear':
        # 将负数统一海拔（因为负数的多少没有意义）
        dst = img_negative_process(dst, 0)
    return dst


def shapen(img, flag, method=''):
    l = shapen_process(img, flag + '_l', method=method)
    r = shapen_process(img, flag + '_r', method=method)
    u = shapen_process(img, flag + '_u', method=method)
    d = shapen_process(img, flag + '_d', method=method)
    return l, r, u, d


def img_split_first_moment(img):
    # 计算图像分量的一阶矩
    return img.mean()


def img_split_second_moment(img):
    # 计算图像分量的二阶矩
    return img.std()


def img_split_third_moment(img):
    # 计算图像分量的三阶矩
    mid = np.mean(((img - img.mean()) ** 3))
    return np.sign(mid) * abs(mid) ** (1/3)


def img_first_moment(img):
    # 计算图像的一阶矩
    return np.mean(img)


def img_second_moment(img):
    # 计算图像的二阶矩
    split_moment = [img_split_first_moment(img[:, :, i]) for i in range(3)]
    diff = [(img[:, :, i] - split_moment[i]) for i in range(3)]
    diff_max = np.max(diff, axis=0)
    return np.mean(pow(diff_max, 2)) ** (1/2)


def img_moments(img):
    # 计算图的颜色矩
    first_moment = [img_split_first_moment(img[:, :, i]) for i in range(3)]
    second_moment = [img_split_second_moment(img[:, :, i]) for i in range(3)]
    third_moment = [img_split_third_moment(img[:, :, i]) for i in range(3)]
    return [first_moment, second_moment, third_moment]




