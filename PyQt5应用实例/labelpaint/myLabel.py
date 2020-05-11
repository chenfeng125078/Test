
from PyQt5 import QtGui,QtWidgets,QtCore
from PyQt5.QtCore import QObject,pyqtSignal
# from Label import Ui_widget
from oldlabel import Ui_widget
import sys
import glob
import os
import numpy as np
import matplotlib.image as mpimg
import pyqtgraph as pg
import shutil
import cv2

pg.setConfigOptions(imageAxisOrder='row-major')


class kxImageItem(pg.ImageItem):
    def __init__(self, image=None, **kargs):
        super(kxImageItem, self).__init__(image, **kargs)
        self.maskimg = None

    def setImagekx(self, image=None, autoLevels=None, **kargs):
        self.maskimg = np.copy(image)
        self.setImage(np.flipud(image), autoLevels, **kargs)

    def drawAt(self, pos, ev=None):
        self._setpixcelvalue(self.image, int(pos.x()), int(pos.y()))
        self.setImage(self.image, autoLevels=False)

    def _setpixcelvalue(self, img, x, y):
        kernelwidth = self.drawKernel.shape[0]
        # print (type(img), type(self.maskimg))
        if img is not None and self.maskimg is not None:
            nleft = int(max(0, x - kernelwidth / 2))
            nright = int(min(self.image.shape[1] - 1, x + kernelwidth / 2))
            ntop = int(max(0, y - kernelwidth / 2))
            nbot = int(min(self.image.shape[0] - 1, y + kernelwidth / 2))
            if len(img.shape) == 3:
                img[ntop:nbot, nleft:nright] = [255, 255, 255]
                self.maskimg[ntop:nbot, nleft:nright] = [255, 255, 255]
            else:
                img[ntop:nbot, nleft:nright] = 255
                self.maskimg[ntop:nbot, nleft:nright] = 255


class Window(QtWidgets.QWidget,Ui_widget):
    def __init__(self):
        super(Window,self).__init__()
        self.setupUi(self)
        self.tableWidget.clicked.connect(self.slot2)
        self.pushButton_2.clicked.connect(self.getLabel)
        self.pushButton_3.App.connect(self.slot3)
        self.pushButton_4.App.connect(self.slot3)
        self.pushButton_5.App.connect(self.slot3)
        self.pushButton_6.App.connect(self.slot3)

    """
    实现槽函数
    """
    def getclik(self,connect):
        print(connect)
        for a in connect:
            print(a)
            self.layout_1.addWidget(mypushbutton(a))
            mypushbutton(a).clicked.connect(self.slot3)

    def slot1(self):
        path = str((QtGui.QFileDialog.getExistingDirectory(self,
                                                     "打开文件夹",
                                                     "./")) + "\\")
        if path == "\\" or path == None or path == "":
            return
        wsi_mask_paths = glob.glob(os.path.join(path, '*.bmp'))
        for i, j in zip(range(len(wsi_mask_paths)), wsi_mask_paths):
            newItem = QtWidgets.QTableWidgetItem(j)
            self.tableWidget.setItem(i, 0, newItem)


    def slot2(self,index):
        table_column=0
        table_row = index.row()
        print (table_row, table_column)
        current_item = self.tableWidget.item(table_row, table_column).text()

        # self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        photo = np.array(mpimg.imread(current_item))
        # photo = cv2.imread(current_item, 1)
        self.img.clear()
        self.img.setImagekx(photo, True, autoDownsample=False)

    def slot3(self, label_name):
        print(label_name)
        w2=self.tableWidget.currentRow()#获取当前行数
        text=QtWidgets.QTableWidgetItem(label_name)
        # print(text)
        self.tableWidget.setItem(w2,1,text)

    def getLabel(self):
        # list1=[]
        list_filename = []
        list_labelname = []
        for j in range(12):
            list_filename.append(self.tableWidget.item(j,0).text())
            if self.tableWidget.item(j,1)==None:
                continue
            list_labelname.append(self.tableWidget.item(j,1).text())

        set_labelname = list(set(list_labelname))
        dict_files = {s_label1 : [] for s_label1 in set_labelname}
        # print(dict_files)

        for i, s_label in enumerate(list_labelname):
            dict_files[s_label].append(list_filename[i])

        path="D:\标签"
        folder = os.path.exists(path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)

        for ph,s_file in dict_files.items():
            folder=os.path.exists(path+'/'+ph)
            if not folder:   # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(path+'/'+ph)
            for s in s_file:
                shutil.copy(s,path+'/'+ph)


if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    myapp = Window()
    myapp.show()
    sys.exit(app.exec_())