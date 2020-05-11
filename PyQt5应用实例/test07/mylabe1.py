
from PyQt5 import QtGui,QtWidgets,QtCore
from PyQt5.QtCore import QObject,pyqtSignal
# from Label import Ui_widget
from labe1 import Ui_widget,mywindow
import sys
import glob
import os
import numpy as np
import matplotlib.image as mpimg
import pyqtgraph as pg
import shutil
from myButton import mypushbutton

class mywindow_2(QtWidgets.QWidget,mywindow):
    #自定义信号
    mySigal=pyqtSignal(list)
    def __init__(self):
        super(mywindow_2,self).__init__()
        self.setupUi(self)
        self.button_1.clicked.connect(self.sendEditContent)

    def files(self):
        list_filenames = []
        for j in range(10):
            if self.tableWidget_2.item(j, 0) == None:
                continue
            list_filenames.append(self.tableWidget_2.item(j, 0).text())

        # print(list_filenames)
        return list_filenames

    def sendEditContent(self):
        content=self.files()
        self.mySigal.emit(content)#发射信号
        print("---1---")

class Window(QtWidgets.QWidget,Ui_widget):
    def __init__(self):
        super(Window,self).__init__()
        self.setupUi(self)
        self.tableWidget.clicked.connect(self.slot2)
        self.pushButton_2.clicked.connect(self.getLabel)
        self.list_name=['1','2']


    """
    实现槽函数
    """
    def getclik(self,connect):
        self.list_name=connect
        print(connect)
        for name in connect:
            self.layout_1.addWidget(mypushbutton(name))
            # mypushbutton.App.connect(self.slot3)
            # button[i] = QtWidgets.QPushButton(str(name))
            # self.layout_1.addWidget((button[i]))

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
        current_item = self.tableWidget.item(table_row, table_column).text()

        # self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        photo = np.array(mpimg.imread(current_item))
        self.img.clear()
        self.img.setImage(photo)

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
        print(dict_files)

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
    mapp = mywindow_2()
    btn = myapp.pushButton_7
    btn1=mapp.button_1
    btn.clicked.connect(mapp.show)
    # btn1.clicked.connect(mapp.files.list_filenames.clear)
    btn1.clicked.connect(mapp.hide)
    mapp.mySigal.connect(myapp.getclik)
    myapp.show()
    sys.exit(app.exec_())