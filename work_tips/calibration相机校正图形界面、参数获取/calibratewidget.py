# coding:UTF-8
import sys
import os
import copy
from PIL import Image
from ui_calibration import  Ui_Calibration
from camera_class import CameraCorrect
from PyQt4 import QtGui, QtCore
import cv2
import pyqtgraph as pg
from lxml import etree
import numpy as np
import time
import struct
from project.mainwindow.kxpackage.KxImageBuf import KxImageBuf
pg.setConfigOptions(imageAxisOrder='row-major')


class CalibrateWidget(QtGui.QWidget, Ui_Calibration):
    def __init__(self, hparent):
        super(CalibrateWidget, self).__init__()
        self.h_parent = hparent
        self.sharpness_case, self.brightness_case = False, False
        self.camera_instance = None
        self.setupUi(self)
        self._initconnection()
        # 文件当前路径
        self.xml_path = "./standard.xml"
        # 初始化图像显示模块
        self.view = pg.ViewBox(invertY=True, enableMenu=False)
        self.h_gVShowRealImg.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.img = pg.ImageItem()
        self.view.addItem(self.img)

        if os.path.exists(self.xml_path):
            self._read_standard_xml()
            self.sharpness_case, self.brightness_case = True, True

    # 擦除标准文本
    def _clear_standard_label(self):
        self.standard_angle_x.clear()
        self.standard_angle_y.clear()
        self.standard_angle_z.clear()
        # self.standard_sharpness.clear()
        self.standard_coordinate_x.clear()
        self.standard_coordinate_y.clear()
        # self.standard_brightness.clear()

    def _clear_rate_score(self):
        self.sharpness_rate.clear()
        self.brightness_rate.clear()

    # 擦除当前文本
    def _clear_actual_label(self):
        self.actual_angle_x.clear()
        self.actual_angle_y.clear()
        self.actual_angle_z.clear()
        self.actual_coordinate_x.clear()
        self.actual_coordinate_y.clear()
        self.x_resolution.clear()
        self.y_resolution.clear()

    # 初始化连接槽函数
    def _initconnection(self):
        self.pushButton_grabImg.clicked.connect(self.slot_collect_image)
        self.pushButton_save.clicked.connect(self.slot_set_standard)

    # 采集按钮槽函数
    def slot_collect_image(self):
    #     if self.pushButton_grabImg.isChecked():
    #         if hasattr(self.h_parent, 'getStatus'):
    #             if self.h_parent.getStatus():
    #                 if hasattr(self.h_parent, 'calibrateSendmsg'):
    #                     self.h_parent.calibrateSendmsg()
    #                     self.pushButton_grabImg.setText(u"停止采集")
    #
    #     else:
    #         self.pushButton_grabImg.setText(u"开始采集")
    #         if hasattr(self.h_parent, 'stopgrap'):
    #             self.h_parent.stopgrap()
    #     return
        file_name = QtGui.QFileDialog.getOpenFileName(self, "open file dialog", "D://card",
                                                      "bmp files(*.bmp)")

        if file_name != '':
            file_name = file_name.toUtf8().data().decode('utf-8')
            with open(file_name, 'rb') as curfp:
                self.imagepil = copy.copy(Image.open(curfp))
                self.image = np.array(self.imagepil)

                start_time = time.time()
                self._clear_actual_label()
                self._clear_rate_score()
                self._updateUi(self.image)
                self.img.clear()
                self.img.setImage(self.image)
                end_time = time.time()
                print(end_time - start_time)

    # 设置标准槽函数，写入xml
    def slot_set_standard(self):
        if self.camera_instance.ret:
            # print("将实际换成标准、写入xml")
            self._clear_standard_label()
            self.standard_angle_x.setText(str(self.camera_instance.x_angle))
            self.standard_angle_y.setText(str(self.camera_instance.y_angle))
            self.standard_angle_z.setText(str(self.camera_instance.z_angle))
            # self.standard_sharpness.setText(str(self.camera_instance.sharpness_score))
            self.standard_coordinate_x.setText(str(self.camera_instance.center_point[0]))
            self.standard_coordinate_y.setText(str(self.camera_instance.center_point[1]))
            self.sharpness = self.camera_instance.sharpness_score
            self.brightness = self.camera_instance.brightness_score
            self._clear_rate_score()
            self.sharpness_rate.setText("100.0%")
            self.brightness_rate.setText("100.0%")
            self._write_standrard_xml()

    # 参数显示
    def _updateUi(self, Img):
        self.camera_instance = CameraCorrect(Img)
        self.actual_angle_x.setText(str(self.camera_instance.x_angle))
        self.actual_angle_y.setText(str(self.camera_instance.y_angle))
        self.actual_angle_z.setText(str(self.camera_instance.z_angle))
        if self.sharpness_case and self.camera_instance.ret:
            sharpness_score = np.round(self.camera_instance.sharpness_score / self.sharpness * 100, 1)
            self.sharpness_rate.setText(str(sharpness_score) + "%")
        else:
            self.sharpness_rate.setText("")
        self.actual_coordinate_x.setText(str(self.camera_instance.center_point[0]))
        self.actual_coordinate_y.setText(str(self.camera_instance.center_point[1]))
        if self.brightness_case and self.camera_instance.ret:
            brightness = np.round(self.camera_instance.brightness_score / self.brightness * 100, 1)
            self.brightness_rate.setText(str(brightness) + "%")
        else:
            self.brightness_rate.setText("")
        self.x_resolution.setText(str(self.camera_instance.x_resolution))
        self.y_resolution.setText(str(self.camera_instance.y_resolution))

    # 写入xml
    def _write_standrard_xml(self):
        """

        :return:
        """
        camera = etree.Element("model")
        parameters = etree.SubElement(camera, "parameters")
        parameters.set("angle_x", str(self.camera_instance.x_angle))
        parameters.set("angle_y", str(self.camera_instance.y_angle))
        parameters.set("angle_z", str(self.camera_instance.z_angle))
        parameters.set("sharpness", str(self.camera_instance.sharpness_score))
        parameters.set("coordinate_x", str(self.camera_instance.center_point[0]))
        parameters.set("coordinate_y", str(self.camera_instance.center_point[1]))
        parameters.set("brightness", str(self.camera_instance.brightness_score))
        parameters.set("resolution_x", str(self.camera_instance.x_resolution))
        parameters.set("resolution_y", str(self.camera_instance.y_resolution))
        tree = etree.ElementTree(camera)
        tree.write(self.xml_path, pretty_print=True, xml_declaration=True, encoding='utf-8', standalone='yes')

    # 读取xml并以此为标准显示
    def _read_standard_xml(self):
        xml_file = etree.parse(self.xml_path)
        root_node = xml_file.getroot()
        parameters_node = root_node[0]
        # node节点通过 .attrib["name"] 获取name对应的值
        # name_value = parameters_node.attrib["angle_x"]
        # xml里为 str 类型
        self.standard_angle_x.setText(parameters_node.attrib["angle_x"])
        self.standard_angle_y.setText(parameters_node.attrib["angle_y"])
        self.standard_angle_z.setText(parameters_node.attrib["angle_z"])
        self.sharpness = float(parameters_node.attrib["sharpness"])
        self.standard_coordinate_x.setText(parameters_node.attrib["coordinate_x"])
        self.standard_coordinate_y.setText(parameters_node.attrib["coordinate_y"])
        self.brightness = float(parameters_node.attrib["brightness"])


    def recmsg(self, n_stationid, n_msgtype, tuple_data):
        data = tuple_data[1]
        nOffset, nP, nh = struct.unpack('I2i', data[:struct.calcsize('I2i')])
        # 解析读图路径
        data = data[struct.calcsize('I2i'):]
        str_readImagePath = struct.unpack('256s', data)[0].split('\x00')[0]
        data = data[struct.calcsize('256s'):]

        if os.path.isfile(str_readImagePath):
            self.h_readImgHandle = open(str_readImagePath, "rb")
            nReadlen = nP * nh + 20
            try:
                self.h_readImgHandle.seek(nOffset)
            except IOError as e:
                strTmp = 'nOffset:' + str(nOffset) + 'IOError:' + str(e)
                return
            self.data = self.h_readImgHandle.read(nReadlen)
            self.saveimg = KxImageBuf()
            self.saveimg.unpack(self.data)
            self.image = self.saveimg.Kximage2npArr()
            self.img.setImage(self.image, autoLevels=False)
            start_time = time.time()
            self._clear_actual_label()
            self._clear_rate_score()
            self._updateUi(self.image)
            self.img.setImage(self.image)
            end_time = time.time()
            print(end_time - start_time)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myWin = CalibrateWidget()
    myWin.show()
    sys.exit(app.exec_())
