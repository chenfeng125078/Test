# 基础函数库
import time
import glob
import shutil
import random
import math
import copy
from operator import itemgetter
import re
import os
import numpy as np
from functools import wraps
import yaml
import ctypes
import matplotlib.pyplot as plt
from functools import reduce
import json
import hashlib
import logging
import inspect
import pickle


# 计算方法
def product(d_list):
    # 连乘
    return reduce(lambda x, y: x * y, d_list)


def mode(d_list):
    # 计算非负数的众数
    return int(np.argmax(np.bincount(d_list)))


# 式子处理
def fs_process(fs):
    '''
    从范围表达式中提取信息
    :param fs: 范围表达式
    :return: low: 下限
             high: 上限
             low_type：下限类型
             high_type：上限类型
    '''
    input = re.findall(r'-{0,1}\d+', fs)
    low_type = 0
    high_type = 0
    if len(input) == 2:
        low = float(input[0])
        high = float(input[1])
        if fs.find('[') != -1:
            low_type = 1
        if fs.find(']') != -1:
            high_type = 1
    elif len(input) == 1:
        if fs.find('(') != -1:
            low = float(input[0])
            high = np.inf
        elif fs.find('[') != -1:
            low = float(input[0])
            high = np.inf
            low_type = 1
        elif fs.find(')') != -1:
            low = -np.inf
            high = float(input[0])
        elif fs.find(']') != -1:
            low = -np.inf
            high = float(input[0])
            high_type = 1
        else:
            # 没有输入合法的式子
            raise NotImplementedError
    else:
        raise NotImplementedError
    return low, high, low_type, high_type


# 修饰器
def str_header(str_info=''):
    '''
    定义一个字符串修饰器，可以在前面添加信息
    :param str_info: 添加在字符串头的信息
    :return:
    '''

    def str_processer(func):

        @wraps(func)
        def wrapper(*args):
            rst = func(*args)
            return str_info + rst

        return wrapper

    return str_processer


def str_ender(str_info=''):
    '''
    定义一个字符串修饰器，可以在后面添加信息
    :param str_info: 添加在字符串头的信息
    :return:
    '''

    def str_processer(func):

        @wraps(func)
        def wrapper(*args):
            rst = func(*args)
            return rst + str_info

        return wrapper

    return str_processer


def list_join(func):
    # 让列表变成字符串为列表的形式

    def wrapper(*args):
        rst = func(*args)
        return '[' + ', '.join(rst) + ']'

    return wrapper


def list_strize(func):
    # 将列表中的项目字符串化

    def wrapper(*args):
        rst = func(*args)
        return [str(term) for term in rst]

    return wrapper


def time_logger(str_info=''):
    '''
    定义一个计时修饰器
    :param str_info: 时间显示
    :return:
    '''

    def show_time(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            rst = func(*args, **kwargs)
            end_time = time.time()
            print(str_info + ' 耗时：%s s 函数：%s' % (floor_fun(end_time - start_time, 2), func.__qualname__))
            return rst

        return wrapper

    return show_time


# 运算函数
def fsign(v):
    # 计算数值的正负符号
    if v < 0:
        return -1
    elif v > 0:
        return 1
    else:
        return 0


def floor_fun(num, digit):
    if digit > 0:
        return np.floor(num*pow(10, digit)) / pow(10, digit)
    else:
        return np.floor(num)


def ceil_fun(num, digit):
    if digit > 0:
        return np.ceil(num*pow(10, digit)) / pow(10, digit)
    else:
        return np.ceil(num)


def cycle_difference(i, j, n):
    # 在模n的情形下，计算从i到j的距离
    if i <= j:
        return j - i
    else:
        return j + n - i


def cycle_difference_real(i, j, n):
    # 计算i和j的最短循环距离
    i = i % n
    j = j % n
    if i < j:
        return min([j - i, i + n - j])
    else:
        return min([i - j, j + n - i])


# 列表处理函数
class List(list):

    def filter_sub(self, index_out=[]):
        # 取出子集
        sub_list = []
        for i, s in enumerate(self):
            if i not in index_out:
                sub_list.append(s)
        return sub_list


def list_del_index(d_list, ind_list):
    # 批量根据索引删除列表中的数据
    ind_list = list_unique(ind_list)
    ind_list = sorted(ind_list, reverse=True)  # 从高到低进行删除
    list_new = []
    for i in range(0, len(d_list)):
        if i not in ind_list:
            list_new.append(d_list[i])
    return list_new


def list_id(d_list):
    # 将列表里面的信息对象化
    id_list = []
    for d in d_list:
        id_list.append(id(d))
    return id_list


def list_listize(data_list):
    # 将列表当中的第二层元素变成列表
    return [list(term) for term in data_list]


def list_selector(d_list, fs, method='select', dimstr=''):
    '''
    list数据选择器，针对二维列表，形如
    [[a1, b1],
     [a2, b2],
     [a3, b3]]
    :param d_list: 列表数据
    :param fs: 符号表达式（目前只能解决全并或者全或的情形）
    :param method: select:正选 unselect:反选（相当于删除））
    :return:
    '''
    if re.search('&', fs) is not None:
        formula = fs.split('&')
        selector_type = 'and'
    elif re.search('\|', fs) is not None:
        formula = fs.split('|')
        selector_type = 'or'
    else:
        # 只有一个查找条件
        formula = [fs]
        selector_type = 'and'
    idx_list = np.arange(0, len(d_list)).tolist()
    if selector_type == 'and':
        idx_res = []
        d_res = d_list
        for f in formula:
            column, value, sign = str_sign_judge(f)
            d_list_new = []
            for i, d in enumerate(d_res):
                if eval('d' + dimstr + '[' + column + ']' + sign + value):
                    d_list_new.append(d)
                    idx_res.append(idx_list[i])
            d_res = d_list_new
            idx_list = idx_res
            idx_res = []
        if method == 'select':
            return d_res
        elif method == 'unselect':
            return list_del_index(d_list, idx_list)
    elif selector_type == 'or':
        d_res = []
        d_list_new = []
        for i, d in enumerate(d_list):
            flag_select = 0
            for f in formula:
                column, value, sign = str_sign_judge(f)
                if eval('d' + dimstr + '[' + column + ']' + sign + value):
                    flag_select = 1
            if flag_select == 1:
                d_list_new.append(d)
            elif flag_select == 0:
                d_res.append(d)
        if method == 'select':
            return d_list_new
        elif method == 'unselect':
            return d_res


def list_tuplize(data_list):
    # 将点列表当中的点变成元组
    # 元组形式的点嵌套在列表当中可以进行集合运算，但是列表形式的点不可以
    return [tuple(term) for term in data_list]


def list_str(data_list):
    return [str(term) for term in data_list]


def list_unique(list_now):
    # 简化一个列表
    return list(set(list_now))


def sum_quantity(value_list, i, j):
    # 计算列表总和
    if i < j:
        return sum(value_list[i:j+1])
    else:
        return sum(value_list[i:] + value_list[:j+1])


def permutation_combination(list_now):
    # 对列表元素进行排列组合
    length = len(list_now)
    bin_num = length
    rst = []
    for i in range(pow(2, bin_num)):
        if i > 0:
            bin_str = bin(i).replace('0b', '')
            # 补充位数
            bin_str = '0' * (bin_num - len(bin_str)) + bin_str
            temp = [list_now[j] for j, string in enumerate(bin_str) if string == '1']
            temp.sort()
            rst.append(temp)
    return rst


# 字典处理函数
def dict_add(d, key, content, method='append', is_unique=True):
    # 向字典特定键值下的列表添加内容
    # 字典值是列表结构
    if method == 'extend':
        if key not in d:
            d[key] = []
        d[key].extend(content)
        if is_unique:
            d[key] = list_unique(d[key])
    elif method == 'append':
        if key not in d:
            d[key] = []
        if is_unique:
            if content not in d[key]:
                d[key].append(content)
        else:
            d[key].append(content)


def dict_search_del(d, content):
    # 在字典所有键值下的列表清除内容
    for key in d:
        while content in d[key]:
            d[key].remove(content)


def dict_swap_key_value(dict_old):
    # 字典的键值互换
    dict_new = {}
    for key in dict_old.keys():
        if isinstance(dict_old[key], list):
            # 列表类型的字典
            for v in dict_old[key]:
                if v not in dict_new.keys():
                    dict_new[v] = [key]
                else:
                    dict_new[v].append(key)
        else:
            # 单纯值的字典
            dict_new[dict_old[key]] = key
    return dict_new


def dict_value_sort(d):
    # 对字典的键值进行排序（键值以列表形式存储）
    for key in d:
        d[key] = sorted(d[key])


# 索引处理函数
def index_move(idx, n, move):
    '''
    已知索引边界情况下，计算索引增减一定量时的实际位置
    :param idx:  当前索引
    :param n:  数据总长度
    :param move:  移动单位
    :return:
    '''
    # idx_new = idx + move
    # if idx_new < 0:
    #     idx_new += n
    # while idx_new >= n:
    #     idx_new -= n
    # return idx_new
    return (idx + move) % n


def index_list_move(idx_start, move_num, n):
    # 生成一段序列，模为n，长度为move_num
    # index_list = []
    # if move_num > 0:
    #     for i in range(0, move_num):
    #         index_list.append(index_move(i, n, idx_start))
    # elif move_num < 0:
    #     for i in range(0, move_num, -1):
    #         index_list.append(index_move(i, n, idx_start))
    # return index_list
    if move_num > 0:
        return [index_move(idx_start, n, d) for d in range(0, move_num)]
    elif move_num < 0:
        return [index_move(idx_start, n, d) for d in range(0, move_num, -1)]


def index_generate(idx_start, idx_end, n):
    # 生成idx_start到Idx_end的索引，专门处理跨越了最大值的索引
    if idx_end < idx_start:
        idx_end += n
    diff = idx_end - idx_start
    return index_list_move(idx_start, diff, n)


# 时间处理函数
def time_str(string='Now time: ', type='yyyy-mm-dd HH:MM:SS'):
    '''
    返回当前时间的字符串
    :return:
    '''
    if type == 'yyyy-mm-dd HH:MM:SS':
        strf = "%Y-%m-%d %H:%M:%S"
    else:
        strf = "%Y%m%d%H%M%S"
    return string + time.strftime(strf, time.localtime(time.time()))


# 字符串处理
def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False


def str_sign_judge(f):
    # 符号表达式的符号判断
    f_ext = re.search('<=|>=', f)
    if f_ext is not None:
        sign = f_ext.group()
    else:
        sign = re.search('=|>|<', f).group()
    column, value = f.split(sign)
    if sign == '=':  # 等号简写方便输入
        sign = '=='
    return column, value, sign


@str_ender('\n' + '='*50)
@str_header('='*50 + '\n')
def specified_content(string, *args):
    '''
    打印内容
    :param string:  
    :param kwargs: 
    :return: 
    '''
    return string.format(*args)


# 文件处理
def filepath_split(filepath):
    # 将带路径的文件名分割成路径+文件名+扩展名
    path, filename = os.path.split(filepath)
    basename, extension = os.path.splitext(filename)
    return path, basename, extension


def file_create(file_name):
    # 新建文件
    folder_build(file_name)
    with open(file_name, 'w', encoding='gbk') as f:
        pass


def folder_build(file_name):
    # 建立路径上的文件夹
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def folder_transfer(folderpath, foldernewpath, start_num):
    # 将某文件夹下文件转移新的文件夹，并且以数字命名
    # folderpath的书写规则类似：'./data_wujwu/floorplan_image/business2'
    # foldernewpath的书写规则类似：'./data_wujwu/floorplan_image/temp'
    if not os.path.exists(foldernewpath):
        os.makedirs(foldernewpath)
    filePaths = glob.glob(folderpath + '/*')
    for i, filepath in enumerate(filePaths):
        name = str(i + start_num)
        name = str(0) * (4 - len(name)) + name
        _, _, extension = filepath_split(filepath)
        filename = name + extension
        shutil.copyfile(filepath, os.path.join(foldernewpath, filename))


def json_load(filepath):
    # 读取json文件
    with open(filepath, 'r', encoding='gbk') as file_object:
        data = json.load(file_object)
    return data


def json_save(data, filepath):
    # 将数据保存为json文件，使用于字典或者列表
    with open(filepath, 'w', encoding='gbk') as file_object:
        json.dump(data, file_object, indent=2)


# 对象处理
def id_to_object(id_value):
    # 通过id查到id对应的对象
    return ctypes.cast(id_value, ctypes.py_object).value


def attr_transfer(obj_now, obj_old):
    # 对象属性迁移，将obj_old有而obj_now没有的属性转移给obj_now
    now_dict = obj_now.__dict__
    old_dict = obj_old.__dict__
    for attr in old_dict:
        if attr not in now_dict:
            obj_now.__dict__[attr] = obj_old.__dict__[attr]


def save_variable(variable):
    # 保存变量
    with open('temp.data', 'wb') as f:
        pickle.dump(variable, f)


def load_variable():
    # 读取变量
    with open('temp.data', 'rb') as f:
        variable = pickle.load(f)
    return variable


# 作图
def hist_graph(data, bins, range, times=1, flag='ungraph'):
    # 画直方图
    cnts, bins = np.histogram(data, bins=bins, range=range)
    bins1 = np.array((bins[:-1] + bins[1:]) / 2)
    if flag == 'graph':
        fig = plt.figure()
        plt.plot(bins1, cnts*times, marker="o", markersize=5)
    return cnts, bins1


# 分组器
class Grouper:
    # 作用：划分分组号

    def __init__(self):
        self.data = list()  # 分组数据

    def add(self, set_add):
        '''
        添加属于一组的号码集合
        :param set_add: 属于一组的号码集合
        '''
        group_finded = []
        for i, group in enumerate(self.data):
            if len(set_add & group):
                group.update(set_add)
                group_finded.append(i)
        # 将找到的组直接合成一起
        if len(group_finded) > 1:
            # 进行组合并，并且提取出需要删除的组
            merge = set()
            for i in group_finded:
                merge = merge | self.data[i]
            self.data.append(merge)
            # 删除已合并的组
            delete_list = sorted(group_finded, reverse=True)
            for idx in delete_list:
                del self.data[idx]
        elif len(group_finded) == 0:
            # 没有找到一个组，则自己建一个组
            self.data.append(set_add)


# 储存器
class Storer(dict):

    def __init__(self, dict={}):
        super().__init__(dict)

    def gen(self, data):
        # 储存器，如果已经存在索引，则直接提取。否则保存起来
        key = tuple(data)
        if key in self:
            if tuple(self[key]) != key:
                # 如果存在索引，且数据不一致，说明数据有变动，则保存下来
                self[key] = data
                return data
            else:
                # 如果存在索引，而且数据一致，则直接返回原来保存的结果
                return self[key]
        else:
            # 如果不存在索引，那么则保存下来
            self[key] = data
            return data


# 分组变量器
class Variable(dict):

    def __init__(self):
        super().__init__({})

    def add(self, key, change):
        # 记录变动量。初始默认值为0
        if key not in self:
            self[key] = 0
        self[key] += change

    def sort(self, reverse=False):
        return sorted(self.items(), key=lambda x: x[1], reverse=reverse)


# 离散处理
def hash_str(string):
    # 将字符串转化为hash值形式的字符串
    return hashlib.md5(string.encode(encoding='utf-8')).hexdigest()


# 日志
def getLogger(name):
    '''
    运用logging模块写日志
    :param name: logger的名称
    :return: 返回一个Logger对象
    添加日志的方法
    logger.info("service is run....")  # 返回级别为info的日志
    logger.warning("service is warning....")  # 返回级别为
    日志的严重程度：critical > error > warning > info > debug
    '''
    logger = logging.getLogger(name)


    this_file = inspect.getfile(inspect.currentframe())
    dirpath = os.path.abspath(os.path.dirname(this_file))
    handler = logging.FileHandler(os.path.join(dirpath, "log/" + name + ".log"))

    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


# 随机颜色
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color