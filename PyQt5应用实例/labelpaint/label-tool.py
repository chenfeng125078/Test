#coding:utf-8

"""
标注工具：
将缺陷分类

"""
from PyQt5 import QtGui, QtWidgets, QtCore
from Label import Ui_Labelwidget
import pyqtgraph as pg
import glob
import os
from io import StringIO, BytesIO
import numpy as np
import DefectInfoData_pb2, pbjson
from pyqtgraph import ROI
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
    # def mouseClickEvent(self, ev):
    #     pg.ImageItem.mouseClickEvent(self, ev)
    #     print ("here")

class Tabwidgetkx(QtWidgets.QTableWidget):
    signalrow = QtCore.pyqtSignal(int)
    def __init__(self, *__args):
        super(Tabwidgetkx, self).__init__(*__args)

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        QtWidgets.QTableWidget.keyPressEvent(self, a0)
        self.signalrow.emit(self.currentRow())


class LabelTool(QtWidgets.QWidget, Ui_Labelwidget):
    def __init__(self):
        super(LabelTool, self).__init__()
        self.setupUi(self)
        self._completeUi()
        self._initconnection()
        self.list_bufimg = []
        self.list_bufsrc = []
        self.list_rect = []
        self.rectitem = None
        self.rectitem1 = None

    def _completeUi(self):
        self.verlayout = QtWidgets.QHBoxLayout(self.widget_2)
        self.graphview = pg.GraphicsView(self)
        self.verlayout.addWidget(self.graphview)
        self.view = pg.ViewBox(invertY=True)
        self.graphview.setCentralItem(self.view)
        self.imgitem = kxImageItem()
        self.view.addItem(self.imgitem)

        self.verlayout1 = QtWidgets.QHBoxLayout(self.widget_5)
        self.graphview1 = pg.GraphicsView(self)
        self.verlayout1.addWidget(self.graphview1)
        self.view1 = pg.ViewBox(invertY=True)
        self.graphview1.setCentralItem(self.view1)
        self.imgitem1 = kxImageItem()
        self.view1.addItem(self.imgitem1)

        self.conLayout = QtWidgets.QHBoxLayout(self.widget_3)
        self.tableWidget = Tabwidgetkx(0,1)
        self.conLayout.addWidget(self.tableWidget)
        self.tableWidget.setHorizontalHeaderLabels(['标签'])


    def _initconnection(self):
        self.pushButton_loadimg.clicked.connect(self.slot_loaddat)
        self.tableWidget.clicked.connect(self.slot_chooseimg)
        self.tableWidget.signalrow.connect(self.move)
        self.pushButton_export.clicked.connect(self.slot_splitimg)

    def move(self, nindex):
        self.slot_chooseimg(nindex)


    def clear(self):
        self.list_bufimg = []
        self.list_bufsrc = []
        self.list_rect = []
        self._removeitem()
        self.rectitem = None
        self.rectitem1 = None
        self.tableWidget.clear()

    def _removeitem(self):
        if self.rectitem is not None:
            self.graphview.removeItem(self.rectitem)
        if self.rectitem1 is not None:
            self.graphview1.removeItem(self.rectitem1)

    def slot_loaddat(self):
        fileName, readtype = QtWidgets.QFileDialog.getOpenFileName(None, u"导入文件..", "./", "DAT(*.dat)")
        # print (fileName)
        if fileName:
            self.clear()
            fileName = fileName
            # print "start load"
            with open(fileName, 'rb') as fd:
                data = fd.read()
            reelDefect = DefectInfoData_pb2.ReelDefectData()
            reelDefect.ParseFromString(data)
            reelDefectState = pbjson.pb2dict(reelDefect)
            nimagenum = len(reelDefectState['defectDataList'])
            print (nimagenum)
            nindex = 0
            self.tableWidget.setRowCount(nimagenum)
            for defect in reelDefectState['defectDataList']:
                self.list_bufimg.append(defect['realTimeImg']['buf'])
                self.list_bufsrc.append(defect['standardImg']['buf'])
                featureDict = {}
                for feature in defect['featureData']:
                    featureDict[feature['featurename']] = feature['featureVal']
                self.list_rect.append([int(featureDict[u'X坐标']), int(featureDict[u'Y坐标']),
                                       int(featureDict[u'缺陷宽']), int(featureDict[u'缺陷高'])])
                newItem = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(nindex, 0, newItem)
                nindex +=1

            # self.tableWidget.setRowCount(nimagenum)
            self.tableWidget.setHorizontalHeaderLabels(['标签'])

        # for i, j in zip(range(len(wsi_mask_paths)), wsi_mask_paths):
        #     newItem = QtWidgets.QTableWidgetItem(j)
        #     self.tableWidget.setItem(i, 0, newItem)

    def slot_chooseimg(self, nindex):
        # print (nindex)
        if isinstance(nindex, int):
            nindex = nindex
        else:
            nindex = nindex.row()
        self._removeitem()
        currect = self.list_rect[nindex]
        print (currect)
        self.rectitem = ROI(QtCore.QPoint(currect[0], currect[1]), [currect[2], currect[3]])
        self.rectitem1 = ROI(QtCore.QPoint(currect[0], currect[1]), [currect[2], currect[3]])

        image = self._ioimage(self.list_bufimg[nindex])
        self.imgitem.setImagekx(image, autoLevels=False)
        self.view.addItem(self.rectitem)

        image1 = self._ioimage(self.list_bufsrc[nindex])
        self.imgitem1.setImagekx(image1, autoLevels=False)
        self.view1.addItem(self.rectitem1)

    def _ioimage(self, imgbuf):
        """
        2020.04.17 因为一次性读入所有图像耗时，所以改成单次解析
        :param imgbuf:
        :return:
        """
        from PIL import Image

        imghandle = BytesIO(imgbuf)
        im = Image.open(imghandle)
        realTimeImg = np.asarray(im)
        return realTimeImg

    def slot_splitimg(self):
        # self.list_result = []
        dict_result = {}
        for nindex in range(self.tableWidget.rowCount()):
            # print (nindex)
            curtext = self.tableWidget.item(nindex, 0).text()
            print (curtext)
            if curtext != "":
                if curtext in dict_result:
                    dict_result[curtext].append(self._ioimage(self.list_bufimg[nindex]))
                else:
                    dict_result[curtext] = [self._ioimage(self.list_bufimg[nindex])]

        basepath = "d:\\data\\"
        if not os.path.isdir(basepath):
            os.mkdir(basepath)

        for name in dict_result:
            if not os.path.isdir(basepath + name):
                os.mkdir(basepath + name)
            list_name = os.listdir(basepath + name)
            nsaveimgname = len(list_name) + 1
            for img in dict_result[name]:
                cv2.imwrite(basepath + name + "\\" + str(nsaveimgname) + ".bmp", img)
                nsaveimgname += 1


if __name__ == "__main__":
    a = QtWidgets.QApplication([])
    w = LabelTool()
    w.show()
    a.exec_()
