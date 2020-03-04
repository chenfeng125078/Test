import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class BarBrowser(object):

    def __init__(self):
        self.xy = 0
        self.width = 0
        self.height = 0
        # 这里是为了让鼠标选中画板中的矩形，用来进行变色的。需要注意visible参数
        # visible是设置图形是否直接在画板进行显示的，如果设置为True，运行程序，
        # 则会直接进行显示。
        self.selected = ax.add_artist(Rectangle(rects[0].get_xy(),
                                                rects[0].get_width(),
                                                rects[0].get_height(),
                                                color='g', visible=False))

    def enter_axes(self, event):
        if not event.artist:
            return True

        self.xy = event.artist.xy
        self.width = event.artist.get_width()
        self.height = event.artist.get_height()

        self.selected.set_visible(True)
        self.selected.set_xy(self.xy)
        self.selected.set_height(self.height)
        self.selected.set_alpha(0.7)
        fig.canvas.draw()


if __name__ == '__main__':
    # 设置字体为SimHei显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 下面的两个变量是画板上要展示的数据
    theyear = ['1月', '2月', '3月', '4月', '5月', '6月',
               '7月', '8月', '9月', '10月', '11月', '12月']
    peoplesum = [855, 770, 808, 793, 850, 795, 887, 909, 824, 879, 802, 827]

    xno = np.arange(len(theyear))

    # 绘制柱形图。这里需要注意picker属性，它决定了'pick_event'是否被激活
    # 取值可以是浮点型、布尔值或者函数。这里我们设置为True。
    rects = ax.bar(xno, peoplesum, picker=True)
    # print(rects)
    plt.xticks(xno, theyear)  # 设置x轴坐标

    browser = BarBrowser()
    fig.canvas.mpl_connect('pick_event', browser.enter_axes)

    plt.show()
