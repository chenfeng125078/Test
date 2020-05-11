# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Label.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from myButton import mypushbutton


class Ui_widget(object):
    def setupUi(self, widget):
        widget.setObjectName("widget")
        widget.resize(1120, 860)
        self.widget_2 = QtWidgets.QWidget(widget)
        self.widget_2.setGeometry(QtCore.QRect(90, 140, 611, 631))
        self.widget_2.setObjectName("widget_2")
        self.imv = pg.GraphicsLayoutWidget(self.widget_2)
        # self.imv.context
        self.imv.setGeometry(QtCore.QRect(0, 0, 611, 631))
        self.view = self.imv.addViewBox()

        # # 创建图片项
        self.img = pg.ImageItem(border='w')

        self.widget_3 = QtWidgets.QWidget(widget)
        self.widget_3.setGeometry(QtCore.QRect(810, 160, 261, 591))
        self.widget_3.setStyleSheet("background-color: rgba(255, 255, 255, 60%);")
        self.widget_3.setObjectName("widget_3")

        # WSI_MASK_PATH = 'res\\'  # 存放图片的文件夹路径
        # wsi_mask_paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.bmp'))
        self.conLayout = QtWidgets.QHBoxLayout()
        # # 设置行和列的大小
        self.tableWidget = QtWidgets.QTableWidget(12, 2)
        self.conLayout.addWidget(self.tableWidget)
        self.tableWidget.setHorizontalHeaderLabels(['文件名', '标签'])
        self.widget_3.setLayout(self.conLayout)

        self.pushButton = QtWidgets.QPushButton(widget)
        self.pushButton.setGeometry(QtCore.QRect(840, 70, 180, 61))
        self.pushButton.setStyleSheet("\n"
"font: 75 14pt \"Arial\";")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(widget)
        self.pushButton_2.setGeometry(QtCore.QRect(840, 770, 180, 61))
        self.pushButton_2.setStyleSheet("font: 75 14pt \"Arial\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.widget_4 = QtWidgets.QWidget(widget)
        self.widget_4.setGeometry(QtCore.QRect(40, 50, 550, 40))
        self.widget_4.setStyleSheet("font: 75 11pt \"Arial\";")
        self.widget_4.setObjectName("widget_4")
        self.layout_1=QtWidgets.QHBoxLayout()
        # self.layout_1.addWidget(self.widget_4)
        self.widget_4.setLayout(self.layout_1)


        self.pushButton_7 = QtWidgets.QPushButton(widget)
        self.pushButton_7.setGeometry(QtCore.QRect(90, 790, 180, 50))
        self.pushButton_7.setStyleSheet("font: 12pt \"Arial\";")
        self.pushButton_7.setObjectName("pushButton_7")

        self.retranslateUi(widget)
        self.pushButton.clicked.connect(widget.slot1)

        QtCore.QMetaObject.connectSlotsByName(widget)

    def retranslateUi(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "Form"))
        self.pushButton.setText(_translate("widget", "导入图片"))
        self.pushButton_2.setText(_translate("widget", "导出图片"))
        self.pushButton_7.setText(_translate("widget", "标签设置"))
class mywindow(object):
    def setupUi(self,widget):
        widget.setObjectName("widget")
        widget.resize(300, 250)
        self.widget_5 = QtWidgets.QWidget(widget)
        # self.widget_2.setGeometry(QtCore.QRect(90, 140, 611, 631))
        self.widget_5.setObjectName("widget_5")
        self.layout_1 = QtWidgets.QHBoxLayout()
        # # 设置行和列的大小
        self.tableWidget_2 = QtWidgets.QTableWidget(10, 1)
        self.layout_1.addWidget(self.tableWidget_2)
        self.tableWidget_2.setHorizontalHeaderLabels(['标签'])
        self.widget_5.setLayout(self.layout_1)
        self.button_1=QtWidgets.QPushButton(widget)
        self.button_1.setGeometry(QtCore.QRect(10, 210, 70, 30))
        self.button_1.setText("保存")
        self.button_2=QtWidgets.QPushButton(widget)
        self.button_2.setGeometry(QtCore.QRect(200, 210, 70, 30))
        self.button_2.setText("取消")


