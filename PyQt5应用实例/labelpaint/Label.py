# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Label.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Labelwidget(object):
    def setupUi(self, Labelwidget):
        Labelwidget.setObjectName("Labelwidget")
        Labelwidget.resize(926, 860)
        self.widget_2 = QtWidgets.QWidget(Labelwidget)
        self.widget_2.setGeometry(QtCore.QRect(200, 80, 431, 371))
        self.widget_2.setObjectName("widget_2")
        # 标签窗口
        self.widget_3 = QtWidgets.QWidget(Labelwidget)
        self.widget_3.setGeometry(QtCore.QRect(650, 80, 261, 751))
        self.widget_3.setStyleSheet("background-color: rgba(255, 255, 255, 60%);")
        self.widget_3.setObjectName("widget_3")
        self.pushButton_export = QtWidgets.QPushButton(Labelwidget)
        self.pushButton_export.setGeometry(QtCore.QRect(20, 10, 261, 61))
        self.pushButton_export.setStyleSheet("font: 75 14pt \"Arial\";")
        self.pushButton_export.setObjectName("pushButton_export")
        self.pushButton_loadimg = QtWidgets.QPushButton(Labelwidget)
        self.pushButton_loadimg.setGeometry(QtCore.QRect(650, 10, 261, 61))
        self.pushButton_loadimg.setStyleSheet("font: 75 14pt \"Arial\";")
        self.pushButton_loadimg.setObjectName("pushButton_loadimg")
        self.widget_5 = QtWidgets.QWidget(Labelwidget)
        self.widget_5.setGeometry(QtCore.QRect(200, 460, 431, 371))
        self.widget_5.setObjectName("widget_5")
        self.widget = QtWidgets.QWidget(Labelwidget)
        self.widget.setGeometry(QtCore.QRect(40, 140, 151, 471))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Andalus")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_3 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Andalus")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Andalus")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Andalus")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Andalus")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)

        self.retranslateUi(Labelwidget)
        QtCore.QMetaObject.connectSlotsByName(Labelwidget)

    def retranslateUi(self, Labelwidget):
        _translate = QtCore.QCoreApplication.translate
        Labelwidget.setWindowTitle(_translate("Labelwidget", "Form"))
        self.pushButton_export.setText(_translate("Labelwidget", "导出图片"))
        self.pushButton_loadimg.setText(_translate("Labelwidget", "导入数据包"))
        self.label.setText(_translate("Labelwidget", "缺版 ： 1"))
        self.label_3.setText(_translate("Labelwidget", "污渍 ： 2"))
        self.label_4.setText(_translate("Labelwidget", "折痕 ： 3"))
        self.label_5.setText(_translate("Labelwidget", "漏洞 ： 4"))
        self.label_6.setText(_translate("Labelwidget", "晶典 ： 5"))
