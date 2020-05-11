from PyQt5 import QtGui,QtWidgets
from PyQt5.QtCore import QObject,pyqtSignal

class mypushbutton(QtWidgets.QPushButton):
    App = pyqtSignal(str)

    def __init__(self, parent):
        super(mypushbutton, self).__init__(parent)
        self.clicked.connect(self.Emit)

    def Emit(self):
        self.App.emit(self.text())
