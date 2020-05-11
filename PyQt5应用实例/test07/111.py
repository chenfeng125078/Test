from PyQt5.QtWidgets import QWidget, QApplication, QPushButton
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sip

class mywidget(QWidget):
    def __init__(self,parent=None):
        super(QWidget,self).__init__(parent)
        self.widget_1=QWidget()
        self.horizontalLayout=QHBoxLayout(self.widget_1)
        addButton1=QPushButton(u"添加控件")
        addButton2=QPushButton(u"删除控件")
        self.horizontalLayout.addWidget(addButton1)
        self.horizontalLayout.addWidget(addButton2)
        addButton3=QPushButton("控件")

        self.layout=QGridLayout()
        self.layout.addWidget(self.widget_1,0,0,2,3)
        self.setLayout(self.layout)
        self.layout.addWidget(addButton3)
        # self.add()
        addButton1.clicked.connect(self.add)
        addButton2.clicked.connect(self.deleteWidget)
    def add(self):
        self.button = {}
        for i in range(1, 3):
            self.button[i] = QPushButton(str(i))
            self.layout.addWidget((self.button[i]))
        # self.button=QPushButton("1")
        # self.layout.addWidget(self.button)
    def deleteWidget(self):
        # self.button.deleteLater()#第一种删除
        # self.layout.removeWidget(self.button)#第二种删除
        # sip.delete(self.button)
        for i in range(1,3):
            self.button[i].deleteLater()
            # self.layout.removeWidget(self.button[i])
            # sip.delete(self.button[i])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = mywidget()
    form.show()
    app.exec_()

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.add
    def initUI(self):
        self.resize(1000, 300)
        self.setWindowTitle('动态删除增加控件测试')
        self.CreatUI()

    def CreatUI(self):
        self.lb1 = QPushButton("按键1", self)
        self.lb1.setGeometry(100, 200, 100, 50)
        self.lb2 = QPushButton("按键1", self)
        self.lb2.setGeometry(280, 200, 100, 50)
        self.bt1 = QPushButton('删除', self)
        self.bt2 = QPushButton('新建', self)

        self.bt1.move(100, 20)
        self.bt2.move(280, 20)

        self.bt1.clicked.connect(self.deleteWidget)
        self.bt2.clicked.connect(self.addWidget)
    def add(self):
        self.button={}
        for i in range(1,8):
            self.button[i]=QPushButton(str(i))
            self.layout.addWidget((self.button[i]))

    def deleteWidget(self):
        self.lb1.deleteLater()
        # self.lb2.deleteLater()
        # self.bt1.deleteLater()

    def addWidget(self):
        self.CreatUI()
        self.showWidget()

    def closeWidget(self):
        self.lb1.hide()
        self.lb2.hide()
        self.bt1.hide()

    def showWidget(self):
        self.lb1.show()
        self.lb2.show()
        self.lb3.show()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = Example()
#     ex.show()
#     sys.exit(app.exec_())