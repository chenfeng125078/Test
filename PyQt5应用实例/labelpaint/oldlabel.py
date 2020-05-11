# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Label.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from myButton import mypushbutton
import numpy as np

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
    #
    #
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
    #


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

        # 创建图片项
        self.img = kxImageItem(border='w')
        kern = np.ones((3, 3))
        self.img.setDrawKernel(kern, mask=kern, center=(kern.shape[0]/2, kern.shape[1]/2), mode='set')

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
        # for i, j in zip(range(len(wsi_mask_paths)), wsi_mask_paths):
        #     newItem = QtWidgets.QTableWidgetItem(j)
        #     tableWidget.setItem(i, 0, newItem)
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
        self.pushButton_3 = mypushbutton(widget)
        self.pushButton_3.setGeometry(QtCore.QRect(40, 50, 93, 35))
        self.pushButton_3.setStyleSheet("font: 75 11pt \"Arial\";")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = mypushbutton(widget)
        self.pushButton_4.setGeometry(QtCore.QRect(210, 50, 93, 35))
        self.pushButton_4.setStyleSheet("font: 75 11pt \"Arial\";")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = mypushbutton(widget)
        self.pushButton_5.setGeometry(QtCore.QRect(380, 50, 93, 35))
        self.pushButton_5.setStyleSheet("font: 75 11pt \"Arial\";")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = mypushbutton(widget)
        self.pushButton_6.setGeometry(QtCore.QRect(550, 50, 93, 35))
        self.pushButton_6.setStyleSheet("font: 75 11pt \"Arial\";")
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(widget)
        self.pushButton_7.setGeometry(QtCore.QRect(90, 790, 180, 50))
        self.pushButton_7.setStyleSheet("font: 12pt \"Arial\";")
        self.pushButton_7.setObjectName("pushButton_7")
        self.label = QtWidgets.QLabel(widget)
        self.label.setGeometry(QtCore.QRect(330, 800, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Aharoni")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.spinBox = QtWidgets.QSpinBox(widget)
        self.spinBox.setGeometry(QtCore.QRect(460, 800, 61, 22))
        font = QtGui.QFont()
        font.setFamily("Andalus")
        font.setPointSize(14)
        self.spinBox.setFont(font)
        self.spinBox.setObjectName("spinBox")


        self.retranslateUi(widget)
        self.pushButton.clicked.connect(widget.slot1)

        QtCore.QMetaObject.connectSlotsByName(widget)



    def retranslateUi(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "Form"))
        self.pushButton.setText(_translate("widget", "导入图片"))
        self.pushButton_2.setText(_translate("widget", "导出图片"))
        self.pushButton_3.setText(_translate("widget", "标签1"))
        self.pushButton_4.setText(_translate("widget", "标签2"))
        self.pushButton_5.setText(_translate("widget", "标签3"))
        self.pushButton_6.setText(_translate("widget", "标签4"))
        self.pushButton_7.setText(_translate("widget", "标签设置"))
        self.label.setText(_translate("widget", "掩膜核大小："))


